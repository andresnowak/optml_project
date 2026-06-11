"""Shared training loop, logging, validation, and noise hook."""

from __future__ import annotations

import math
import os
import signal
import time
from contextlib import nullcontext

import torch

from .config import pick_device
from .data import get_token_batch, iter_microbatches, load_bin, validation_offsets
from .models import GPT, GPTConfig
from .models.gpt import Linear
from .optimizers import build_optimizers
from .optimizers.dynmuon import DynMuonRoute, _svd
from .optimizers.input_muon import InputMuon, input_basis_from_activations
from .optimizers.relmuon import RELMUON_WEIGHT_ONLY_MODES, relmuon_weight_scales


_TERMINATE_REQUESTED = False


class MemoryLogger:
    def __init__(self) -> None:
        self.history: dict[str, list[tuple[int, float]]] = {}

    def log(self, payload: dict, step: int) -> None:
        for k, v in payload.items():
            self.history.setdefault(k, []).append((step, float(v)))


class WandbLogger:
    def __init__(self, cfg: dict) -> None:
        import wandb
        self._wandb = wandb
        settings = wandb.Settings(init_timeout=cfg.get("wandb_init_timeout", 180))
        self._run = wandb.init(
            project=cfg.get("wandb_project", "dynmuon-route"),
            entity=cfg.get("wandb_entity"),
            name=cfg.get("run_name"),
            group=cfg.get("wandb_group"),
            id=cfg.get("wandb_run_id"),
            resume="allow" if cfg.get("wandb_run_id") else None,
            config=cfg,
            reinit=True,
            settings=settings,
        )

    def log(self, payload: dict, step: int) -> None:
        self._wandb.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()

    @property
    def run_id(self) -> str:
        return self._run.id


class TeeLogger:
    def __init__(self, loggers: list) -> None:
        self.loggers = loggers

    def log(self, payload: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log(payload, step=step)

    def finish(self) -> None:
        for logger in self.loggers:
            finish = getattr(logger, "finish", None)
            if finish is not None:
                finish()


class InputMuonBasisCollector:
    """Collect linear-layer inputs and register active-subspace bases."""

    def __init__(self, model: GPT, optimizer: InputMuon, cfg: dict):
        self.optimizer = optimizer
        self.rank = int(cfg.get("input_muon_rank", 64))
        self.update_every = int(cfg.get("input_muon_update_every", 1))
        self.basis_max_tokens = int(cfg.get("input_muon_basis_max_tokens", 4096))
        self.center = bool(cfg.get("input_muon_center", False))
        if self.rank <= 0:
            raise ValueError(f"input_muon_rank must be positive, got {self.rank}")
        if self.update_every <= 0:
            raise ValueError(f"input_muon_update_every must be positive, got {self.update_every}")
        if self.basis_max_tokens <= 0:
            raise ValueError(
                "input_muon_basis_max_tokens must be positive, "
                f"got {self.basis_max_tokens}"
            )

        opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self._params: dict[Linear, torch.nn.Parameter] = {}
        self._buffers: dict[torch.nn.Parameter, list[torch.Tensor]] = {}
        self._counts: dict[torch.nn.Parameter, int] = {}
        self._handles = []
        self.enabled = False

        for _, module in model.named_modules():
            if isinstance(module, Linear) and id(module.weight) in opt_params:
                self._params[module] = module.weight
                self._handles.append(module.register_forward_pre_hook(self._hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self, step: int) -> None:
        self.enabled = step % self.update_every == 0
        self._buffers.clear()
        self._counts.clear()

    @torch.no_grad()
    def _hook(self, module: Linear, inputs) -> None:
        if not self.enabled or not module.training:
            return
        if not inputs:
            return
        param = self._params.get(module)
        if param is None:
            return
        remaining = self.basis_max_tokens - self._counts.get(param, 0)
        if remaining <= 0:
            return

        x = inputs[0].detach()
        if x.ndim < 2:
            return
        rows = x.reshape(-1, x.size(-1))
        take = min(remaining, rows.size(0))
        if take < rows.size(0):
            idx = torch.linspace(0, rows.size(0) - 1, steps=take, device=rows.device).long()
            rows = rows.index_select(0, idx)
        self._buffers.setdefault(param, []).append(rows.contiguous())
        self._counts[param] = self._counts.get(param, 0) + take

    @torch.no_grad()
    def end_step(self) -> None:
        if not self.enabled:
            return
        for param, chunks in self._buffers.items():
            if not chunks:
                continue
            X = torch.cat(chunks, dim=0)
            Q = input_basis_from_activations(X, self.rank, center=self.center)
            if Q.numel() > 0:
                self.optimizer.set_input_basis(param, Q)
                self.optimizer.state[param]["last_basis_rank"] = Q.size(1)
        self.enabled = False
        self._buffers.clear()
        self._counts.clear()


def build_arm_logger(cfg: dict, use_wandb: bool, run_name: str, group: str):
    mem = MemoryLogger()
    if not use_wandb:
        return mem, mem, None
    wb = WandbLogger({**cfg, "run_name": run_name, "wandb_group": group})
    return TeeLogger([mem, wb]), mem, wb


def make_noise_hook(lam: float, generator: torch.Generator):
    """Build the anisotropic momentum-noise hook used by Exp 2."""
    def hook(M2: torch.Tensor, state: dict) -> torch.Tensor:
        U, S, Vh = _svd(M2)
        z = torch.randn(1, generator=generator).item()
        return M2 + (lam * S[0] * z) * torch.outer(U[:, 0], Vh[0, :])
    return hook


def lr_factor(step: int, warmup_steps: int, train_steps: int, min_lr_ratio: float) -> float:
    """LR multiplier: linear warmup, then cosine decay.

        eta = (step + 1) / warmup_steps                         during warmup
        eta = min_lr_ratio + cosine_decay * (1 - min_lr_ratio)  after warmup

    The last optimizer update (``step == train_steps - 1``) reaches
    ``min_lr_ratio`` exactly.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)

    decay_steps = max(1, train_steps - warmup_steps - 1)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + cosine * (1 - min_lr_ratio)


def _model_config(cfg: dict) -> GPTConfig:
    """Translate the flat YAML config into a GPTConfig."""
    return GPTConfig(
        sequence_length=cfg.get("sequence_length", GPTConfig.sequence_length),
        vocab_size=cfg.get("vocab_size", GPTConfig.vocab_size),
        n_layer=cfg.get("n_layer", GPTConfig.n_layer),
        n_head=cfg.get("n_head", GPTConfig.n_head),
        n_embd=cfg.get("model_dim", cfg.get("n_embd", GPTConfig.n_embd)),
        head_dim=cfg.get("head_dim", GPTConfig.head_dim),
    )


def _validate_batching(cfg: dict) -> None:
    """Validate batching invariants.

    ``batch_size`` is sequences per optimizer step and ``mbs`` is sequences per
    microbatch. Token counts are derived as ``batch_size * sequence_length``.
    """
    batch_size = cfg["batch_size"]
    mbs = cfg["mbs"]
    if batch_size % mbs != 0:
        raise ValueError("batch_size must be divisible by mbs")


def _checkpoint_path(cfg: dict) -> str:
    base_dir = cfg.get("checkpoint_dir") or os.environ.get("CHECKPOINT_DIR") or "~/checkpoints/optml_project"
    base_dir = os.path.expanduser(base_dir)
    run_name = cfg.get("run_name") or cfg.get("model") or "run"
    safe_run_name = "".join(c if c.isalnum() or c in "._=-" else "_" for c in run_name)
    return os.path.join(base_dir, safe_run_name, "latest.pt")


def _load_checkpoint(path: str, device: torch.device) -> dict | None:
    del device
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_checkpoint(
    path: str,
    *,
    model: GPT,
    dynmuon,
    adamw,
    step: int,
    training_time: float,
    cfg: dict,
    logger=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "training_time": training_time,
        "cfg": cfg,
        "model": model.state_dict(),
        "dynmuon": dynmuon.state_dict() if dynmuon is not None else None,
        "adamw": adamw.state_dict() if adamw is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "wandb_run_id": _wandb_run_id(logger),
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    print(f"saved checkpoint {path} at step {step}")


def _restore_rng_state(ckpt: dict) -> None:
    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"].detach().cpu().to(torch.uint8))
    if ckpt.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        cuda_states = [state.detach().cpu().to(torch.uint8) for state in ckpt["cuda_rng_state_all"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _wandb_run_id(logger) -> str | None:
    if logger is None:
        return None
    run_id = getattr(logger, "run_id", None)
    if run_id is not None:
        return run_id
    for child in getattr(logger, "loggers", ()):
        run_id = _wandb_run_id(child)
        if run_id is not None:
            return run_id
    return None


def _finish_logger(logger) -> None:
    if logger is None:
        return
    finish = getattr(logger, "finish", None)
    if finish is not None:
        finish()


def _install_signal_handlers() -> None:
    def request_terminate(signum, frame) -> None:
        del signum, frame
        global _TERMINATE_REQUESTED
        _TERMINATE_REQUESTED = True
        print("termination requested; will checkpoint at the next safe point")

    signal.signal(signal.SIGTERM, request_terminate)


def log_routing(model: GPT, dynmuon: DynMuonRoute | None, step: int, logger) -> None:
    """Log per-parameter routing diagnostics cached by DynMuonRoute."""
    if logger is None or dynmuon is None:
        return
    name_of = {id(p): n for n, p in model.named_parameters()}
    payload = {}

    # Log global schedule p_t if active (reference for deviation plots).
    if dynmuon.param_groups:
        g = dynmuon.param_groups[0]
        if g.get("routing_mode") in ("global_schedule", "schedule_modulated"):
            payload["route/p_global"] = dynmuon._p_schedule(g)

    for group in dynmuon.param_groups:
        group_name = group.get("name", "matrix")
        for p in group["params"]:
            st = dynmuon.state.get(p)
            if not st or "last_p" not in st:
                continue
            n = name_of.get(id(p), "?")
            # NaN means "undefined this step" (zero-momentum layers have no
            # exponent and no proxies); skip rather than log gaps as NaN.
            if not math.isnan(st["last_p"]):
                payload[f"route/group/{group_name}/p/{n}"] = st["last_p"]
                payload[f"route/p/{n}"] = st["last_p"]
            for key, label in (("last_sr", "sr"), ("last_gamma", "gamma"),
                               ("last_gamma_ema", "gamma_ema"), ("last_alpha", "alpha")):
                value = st.get(key, float("nan"))
                if not math.isnan(value):
                    payload[f"route/{label}/{n}"] = value
    logger.log(payload, step=step)


def log_input_muon(model: GPT, optimizer: InputMuon | None, step: int, logger) -> None:
    """Log InputMuon basis and projection diagnostics."""
    if logger is None or not isinstance(optimizer, InputMuon):
        return
    by_type: dict[str, dict[str, list[float]]] = {}
    for name, p in model.named_parameters():
        layer_type = _matrix_layer_type(name)
        if layer_type is None:
            continue
        st = optimizer.state.get(p)
        if not st:
            continue
        bucket = by_type.setdefault(layer_type, {
            "basis_rank": [],
            "projected_grad_fraction": [],
        })
        if "last_basis_rank" in st:
            bucket["basis_rank"].append(float(st["last_basis_rank"]))
        if "last_projected_grad_fraction" in st:
            bucket["projected_grad_fraction"].append(float(st["last_projected_grad_fraction"]))

    payload = {}
    all_values: dict[str, list[float]] = {}
    for layer_type, stats in by_type.items():
        for key, values in stats.items():
            if not values:
                continue
            payload[f"input_muon/{layer_type}/{key}/mean"] = float(sum(values) / len(values))
            all_values.setdefault(key, []).extend(values)
    for key, values in all_values.items():
        payload[f"input_muon/all/{key}/mean"] = float(sum(values) / len(values))
    if payload:
        logger.log(payload, step=step)


def _matrix_layer_type(name: str) -> str | None:
    """Map GPT matrix parameter names to compact layer-type labels."""
    for suffix, layer_type in (
        ("attn.q.weight", "attn.q"),
        ("attn.k.weight", "attn.k"),
        ("attn.v.weight", "attn.v"),
        ("attn.proj.weight", "attn.proj"),
        ("mlp.fc.weight", "mlp.fc"),
        ("mlp.proj.weight", "mlp.proj"),
    ):
        if name.endswith(suffix):
            return layer_type
    return None


def _matrix_param_snapshots(model: GPT) -> dict[str, torch.Tensor]:
    """Clone matrix weights whose updates we want to diagnose."""
    snapshots = {}
    for name, p in model.named_parameters():
        if p.ndim == 2 and _matrix_layer_type(name) is not None:
            snapshots[name] = p.detach().float().clone()
    return snapshots


@torch.no_grad()
def log_matrix_update_ratios(
    model: GPT,
    before: dict[str, torch.Tensor],
    step: int,
    logger,
    *,
    eps: float = 1e-12,
) -> None:
    """Log relative matrix update sizes grouped by layer type."""
    if logger is None or not before:
        return
    by_type: dict[str, dict[str, list[float]]] = {}
    for name, p in model.named_parameters():
        old = before.get(name)
        layer_type = _matrix_layer_type(name)
        if old is None or layer_type is None:
            continue
        new = p.detach().float()
        old = old.to(device=new.device)
        delta = new - old
        weight_fro = torch.linalg.norm(old).item()
        update_fro = torch.linalg.norm(delta).item()
        weight_op = torch.linalg.svdvals(old).amax().item()
        update_op = torch.linalg.svdvals(delta).amax().item()
        bucket = by_type.setdefault(layer_type, {
            "fro_ratio": [],
            "op_ratio": [],
            "update_fro": [],
            "weight_fro": [],
            "update_op": [],
            "weight_op": [],
        })
        bucket["fro_ratio"].append(update_fro / (weight_fro + eps))
        bucket["op_ratio"].append(update_op / (weight_op + eps))
        bucket["update_fro"].append(update_fro)
        bucket["weight_fro"].append(weight_fro)
        bucket["update_op"].append(update_op)
        bucket["weight_op"].append(weight_op)

    payload = {}
    all_values: dict[str, list[float]] = {}
    for layer_type, stats in by_type.items():
        for key, values in stats.items():
            value = float(sum(values) / max(1, len(values)))
            payload[f"weight_update/{layer_type}/{key}/mean"] = value
            all_values.setdefault(key, []).extend(values)
    for key, values in all_values.items():
        payload[f"weight_update/all/{key}/mean"] = float(sum(values) / max(1, len(values)))
    if payload:
        logger.log(payload, step=step)


@torch.no_grad()
def log_matrix_weight_spectra(
    model: GPT,
    step: int,
    logger,
    *,
    log_relmuon_scales: bool = False,
    relmuon_scale_mode: str = "log1p",
    eps: float = 1e-8,
) -> None:
    """Log matrix weight spectrum diagnostics grouped by layer type.

    Raw singular-value summaries show the current weight spectra. For RelMuon,
    scale summaries show how far its update spectrum moves away from Muon's
    all-ones update spectrum.
    """
    if logger is None:
        return
    sv_by_type: dict[str, list[torch.Tensor]] = {}
    scale_by_type: dict[str, list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        layer_type = _matrix_layer_type(name)
        if layer_type is None:
            continue
        sv = torch.linalg.svdvals(p.detach().float())
        sv_by_type.setdefault(layer_type, []).append(sv)
        # Aligned scales depend on the update direction, not just the weight,
        # so they cannot be reproduced here; log weight-only modes.
        if log_relmuon_scales and relmuon_scale_mode in RELMUON_WEIGHT_ONLY_MODES:
            scales = relmuon_weight_scales(p.detach(), scale_mode=relmuon_scale_mode, eps=eps)
            scale_by_type.setdefault(layer_type, []).append(scales)

    payload = {}
    all_sv = []
    for layer_type, chunks in sv_by_type.items():
        sv = torch.cat(chunks)
        all_sv.append(sv)
        q = torch.quantile(sv, torch.tensor([0.1, 0.5, 0.9], device=sv.device))
        prefix = f"weight_svd/{layer_type}/sv"
        payload[f"{prefix}/mean"] = sv.mean().item()
        payload[f"{prefix}/p10"] = q[0].item()
        payload[f"{prefix}/median"] = q[1].item()
        payload[f"{prefix}/p90"] = q[2].item()
        payload[f"{prefix}/max"] = sv.max().item()

    if all_sv:
        sv = torch.cat(all_sv)
        q = torch.quantile(sv, torch.tensor([0.1, 0.5, 0.9], device=sv.device))
        payload["weight_svd/all/sv/mean"] = sv.mean().item()
        payload["weight_svd/all/sv/p10"] = q[0].item()
        payload["weight_svd/all/sv/median"] = q[1].item()
        payload["weight_svd/all/sv/p90"] = q[2].item()
        payload["weight_svd/all/sv/max"] = sv.max().item()

    all_scales = []
    for layer_type, chunks in scale_by_type.items():
        scales = torch.cat(chunks)
        all_scales.append(scales)
        q = torch.quantile(scales, torch.tensor([0.1, 0.5, 0.9], device=scales.device))
        prefix = f"relmuon/{layer_type}/scale"
        payload[f"{prefix}/mean_abs_delta_from_one"] = (scales - 1.0).abs().mean().item()
        payload[f"{prefix}/p10"] = q[0].item()
        payload[f"{prefix}/median"] = q[1].item()
        payload[f"{prefix}/p90"] = q[2].item()

    if all_scales:
        scales = torch.cat(all_scales)
        q = torch.quantile(scales, torch.tensor([0.1, 0.5, 0.9], device=scales.device))
        payload["relmuon/all/scale/mean_abs_delta_from_one"] = (scales - 1.0).abs().mean().item()
        payload["relmuon/all/scale/p10"] = q[0].item()
        payload["relmuon/all/scale/median"] = q[1].item()
        payload["relmuon/all/scale/p90"] = q[2].item()

    if payload:
        logger.log(payload, step=step)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, amp_ctx) -> tuple[float, int]:
    """Evaluate deterministic fixed-token validation loss."""
    model.eval()
    sequence_length = cfg["sequence_length"]
    train_batch_tokens = cfg["batch_size"] * sequence_length
    val_batch_tokens = cfg.get("val_batch_size", cfg["mbs"]) * sequence_length
    requested = min(cfg["val_tokens"], max(0, len(data) - 1))
    actual = (requested // sequence_length) * sequence_length
    if actual <= 0:
        raise ValueError("validation data is too small")
    batch_tokens = min(train_batch_tokens, val_batch_tokens, actual)
    batch_tokens = (batch_tokens // sequence_length) * sequence_length
    weighted_loss = 0.0
    weighted_tokens = 0
    for offset in validation_offsets(actual, batch_tokens):
        tokens = min(batch_tokens, actual - offset)
        if tokens <= 0:
            continue
        x, y = get_token_batch(data, tokens, sequence_length, device, offset=offset)
        with amp_ctx:
            _, loss = model(x, y)
        weighted_loss += loss.item() * tokens
        weighted_tokens += tokens
    model.train()
    return weighted_loss / max(1, weighted_tokens), actual


def train(cfg: dict, logger=None) -> tuple[GPT, object | None]:
    global _TERMINATE_REQUESTED
    _TERMINATE_REQUESTED = False
    _install_signal_handlers()

    _validate_batching(cfg)
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))

    model_cfg = _model_config(cfg)
    model = GPT(model_cfg).to(device)
    compile_model = bool(cfg.get("compile", True))
    if cfg.get("matrix_optimizer") == "input_muon" and compile_model:
        print("input_muon uses forward hooks for input bases; disabling torch.compile")
        compile_model = False
        cfg["compile"] = False
    if compile_model:
        model.compile(dynamic=False)

    print(f"model: {model.num_params() / 1e6:.1f}M non-embedding params")

    data_dir = cfg.get("data_dir", "data/wikitext103")
    train_data = load_bin(f"{data_dir}/{cfg.get('train_bin', 'train.bin')}")
    val_data = load_bin(f"{data_dir}/{cfg.get('val_bin', 'val.bin')}")

    dynmuon, adamw = build_optimizers(model, cfg)
    input_muon_collector = (
        InputMuonBasisCollector(model, dynmuon, cfg)
        if isinstance(dynmuon, InputMuon)
        else None
    )

    checkpoint_enabled = bool(cfg.get("checkpoint_enabled", False))
    checkpoint_path = _checkpoint_path(cfg)
    start_step = 0
    training_time = 0.0
    if checkpoint_enabled and cfg.get("checkpoint_resume", True):
        ckpt = _load_checkpoint(checkpoint_path, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            if dynmuon is not None and ckpt.get("dynmuon") is not None:
                dynmuon.load_state_dict(ckpt["dynmuon"])
            if adamw is not None and ckpt.get("adamw") is not None:
                adamw.load_state_dict(ckpt["adamw"])
            _restore_rng_state(ckpt)
            if ckpt.get("wandb_run_id") and not cfg.get("wandb_run_id"):
                cfg["wandb_run_id"] = ckpt["wandb_run_id"]
            start_step = int(ckpt.get("step", 0))
            training_time = float(ckpt.get("training_time", 0.0))
            print(f"resumed checkpoint {checkpoint_path} from step {start_step}")

    if logger is None and cfg.get("wandb", False):
        logger = WandbLogger(cfg)

    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    noise_hook = None
    if cfg.get("noise_lambda", 0.0) > 0:
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        noise_hook = make_noise_hook(cfg["noise_lambda"], gen)

    train_steps = cfg["train_steps"]
    batch_size = cfg["batch_size"]
    batch_tokens = batch_size * cfg["sequence_length"]
    mbs = cfg["mbs"]
    val_loss_every = cfg.get("val_loss_every", 0)
    checkpoint_every = cfg.get("checkpoint_every", val_loss_every)
    warmup_steps = cfg.get("warmup_steps", max(1, train_steps // 50))
    min_lr_ratio = cfg.get("min_lr_ratio", 0.1)
    clip = cfg.get("grad_clip", 1.0)
    base_muon, base_adam = cfg["muon_lr"], cfg["adam_lr"]

    model.train()
    t0 = time.perf_counter()
    for step in range(start_step, train_steps + 1):
        should_validate = (
            step == 0
            or step == train_steps
            or (val_loss_every and step % val_loss_every == 0)
        )
        if should_validate:
            training_time += time.perf_counter() - t0
            vloss, actual_val_tokens = estimate_loss(model, val_data, cfg, device, amp_ctx)
            print(f"step {step:5d}/{train_steps} | val {vloss:.4f} | train_time {training_time:.2f}s")
            if logger is not None:
                logger.log({
                    "val/loss": vloss,
                    "val/tokens": actual_val_tokens,
                    "tokens/train": step * batch_tokens,
                    "time/train_seconds": training_time,
                }, step=step)
            if checkpoint_enabled and step > 0 and (step == train_steps or not checkpoint_every or step % checkpoint_every == 0):
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    dynmuon=dynmuon,
                    adamw=adamw,
                    step=step,
                    training_time=training_time,
                    cfg=cfg,
                    logger=logger,
                )
            if _TERMINATE_REQUESTED:
                if input_muon_collector is not None:
                    input_muon_collector.close()
                _finish_logger(logger)
                return model, logger
            t0 = time.perf_counter()
        if step == train_steps:
            break

        f = lr_factor(step, warmup_steps, train_steps, min_lr_ratio)
        if dynmuon is not None:
            for g in dynmuon.param_groups:
                g["lr"] = base_muon * f
            dynmuon.zero_grad(set_to_none=True)
        if adamw is not None:
            for g in adamw.param_groups:
                g["lr"] = g.get("initial_lr", base_adam) * f
            adamw.zero_grad(set_to_none=True)

        x, y = get_token_batch(train_data, batch_tokens, cfg["sequence_length"], device)
        losses: list[float] = []
        if input_muon_collector is not None:
            input_muon_collector.begin_step(step)
        for xb, yb in iter_microbatches(x, y, mbs):
            with amp_ctx:
                _, loss = model(xb, yb)
                loss = loss / (len(x) // mbs)
            loss.backward()
            losses.append(loss.item())
        if input_muon_collector is not None:
            input_muon_collector.end_step()

        should_log_step = step % cfg.get("log_every", 10) == 0
        weight_update_before = None
        if should_log_step and logger is not None and cfg.get("log_weight_update_ratio", False):
            weight_update_before = _matrix_param_snapshots(model)

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        primary_optimizer = dynmuon if dynmuon is not None else adamw
        lr_used = primary_optimizer.param_groups[0]["lr"] if primary_optimizer is not None else 0.0
        if dynmuon is not None:
            dynmuon.step(noise_hook=noise_hook)
        if adamw is not None:
            adamw.step()

        completed_step = step + 1

        if should_log_step:
            train_loss = float(sum(losses))
            print(f"step {completed_step:5d}/{train_steps} | loss {train_loss:.4f} | lr {lr_used:.2e}")
            if logger is not None:
                logger.log({
                    "train/loss": train_loss,
                    "lr": lr_used,
                    "tokens/train": completed_step * batch_tokens,
                }, step=completed_step)
            if cfg.get("log_weight_svd", False) and cfg.get("matrix_optimizer") in (
                "muon", "relmuon", "dynmuon", "input_muon", "kaon",
            ):
                log_matrix_weight_spectra(
                    model,
                    completed_step,
                    logger,
                    log_relmuon_scales=cfg.get("matrix_optimizer") == "relmuon",
                    relmuon_scale_mode=cfg.get("relmuon_scale_mode", "log1p"),
                    eps=cfg.get("relmuon_eps", 1e-8),
                )
            if cfg.get("log_weight_update_ratio", False) and weight_update_before is not None:
                log_matrix_update_ratios(model, weight_update_before, completed_step, logger)
            log_input_muon(model, dynmuon, completed_step, logger)
            log_routing(model, dynmuon, completed_step, logger)

        if _TERMINATE_REQUESTED:
            training_time += time.perf_counter() - t0
            if checkpoint_enabled:
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    dynmuon=dynmuon,
                    adamw=adamw,
                    step=completed_step,
                    training_time=training_time,
                    cfg=cfg,
                    logger=logger,
                )
            if input_muon_collector is not None:
                input_muon_collector.close()
            _finish_logger(logger)
            return model, logger

    if input_muon_collector is not None:
        input_muon_collector.close()
    _finish_logger(logger)
    return model, logger
