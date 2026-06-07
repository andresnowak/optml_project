"""Shared training loop, logging, validation, and noise hook."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch

from .config import pick_device
from .data import get_token_batch, iter_microbatches, load_bin, validation_offsets
from .models import GPT, GPTConfig
from .optimizers import build_optimizers
from .optimizers.dynmuon import DynMuonRoute, _svd


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
        self._run = wandb.init(
            project=cfg.get("wandb_project", "dynmuon-route"),
            entity=cfg.get("wandb_entity"),
            name=cfg.get("run_name"),
            group=cfg.get("wandb_group"),
            config=cfg,
            reinit=True,
        )

    def log(self, payload: dict, step: int) -> None:
        self._wandb.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


class TeeLogger:
    def __init__(self, loggers: list) -> None:
        self.loggers = loggers

    def log(self, payload: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log(payload, step=step)


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

    The final multiplier approaches ``min_lr_ratio`` rather than zero.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
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


def log_routing(model: GPT, dynmuon: DynMuonRoute | None, step: int, logger) -> None:
    """Log per-parameter routing diagnostics cached by DynMuonRoute."""
    if logger is None or dynmuon is None:
        return
    name_of = {id(p): n for n, p in model.named_parameters()}
    payload = {}
    for group in dynmuon.param_groups:
        group_name = group.get("name", "matrix")
        for p in group["params"]:
            st = dynmuon.state.get(p)
            if not st or "last_p" not in st:
                continue
            n = name_of.get(id(p), "?")
            payload[f"route/group/{group_name}/p/{n}"] = st["last_p"]
            payload[f"route/p/{n}"] = st["last_p"]
            payload[f"route/sr/{n}"] = st["last_sr"]
            payload[f"route/gamma/{n}"] = st["last_gamma"]
            payload[f"route/alpha/{n}"] = st["last_alpha"]
    logger.log(payload, step=step)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, amp_ctx) -> tuple[float, int]:
    """Evaluate deterministic fixed-token validation loss."""
    model.eval()
    sequence_length = cfg["sequence_length"]
    train_batch_tokens = cfg["batch_size"] * sequence_length
    requested = min(cfg["val_tokens"], max(0, len(data) - 1))
    actual = (requested // sequence_length) * sequence_length
    if actual <= 0:
        raise ValueError("validation data is too small")
    batch_tokens = min(train_batch_tokens, actual)
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
    _validate_batching(cfg)
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))

    model_cfg = _model_config(cfg)
    model = GPT(model_cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M non-embedding params")

    data_dir = cfg.get("data_dir", "data/wikitext103")
    train_data = load_bin(f"{data_dir}/{cfg.get('train_bin', 'train.bin')}")
    val_data = load_bin(f"{data_dir}/{cfg.get('val_bin', 'val.bin')}")

    dynmuon, adamw = build_optimizers(model, cfg)
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
    warmup_steps = cfg.get("warmup_steps", max(1, train_steps // 50))
    min_lr_ratio = cfg.get("min_lr_ratio", 0.1)
    clip = cfg.get("grad_clip", 1.0)
    base_muon, base_adam = cfg["muon_lr"], cfg["adam_lr"]

    model.train()
    training_time = 0.0
    t0 = time.perf_counter()
    for step in range(train_steps + 1):
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
        for xb, yb in iter_microbatches(x, y, mbs):
            with amp_ctx:
                _, loss = model(xb, yb)
                loss = loss / (len(x) // mbs)
            loss.backward()
            losses.append(loss.item())

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if dynmuon is not None:
            dynmuon.step(noise_hook=noise_hook)
        if adamw is not None:
            adamw.step()

        if step % cfg.get("log_every", 10) == 0:
            train_loss = float(sum(losses))
            lr_now = (base_muon if dynmuon is not None else base_adam) * f
            print(f"step {step + 1:5d}/{train_steps} | loss {train_loss:.4f} | lr {lr_now:.2e}")
            if logger is not None:
                logger.log({
                    "train/loss": train_loss,
                    "lr": lr_now,
                    "tokens/train": (step + 1) * batch_tokens,
                }, step=step + 1)
            log_routing(model, dynmuon, step + 1, logger)

    return model, logger
