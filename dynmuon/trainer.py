"""Training loop, optimizer wiring, routing logging and the noise-injection hook.

Matrix weights (attn / mlp / other 2-D) go to ``DynMuonRoute`` in one param group
per layer type so each gets its own routing logistic; embeddings, norms and biases
go to AdamW. Per-layer routing diagnostics (``p_{t,l}``, stable rank, γ, α) are read
from the optimizer state every ``log_every`` steps — the trajectories the experiments
plot. Everything is driven by the config dict (see ``configs/*.yaml``).
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import numpy as np
import torch

from .config import pick_device
from .data import get_batch, load_bin
from .models import GPT, GPTConfig
from .optimizer import DynMuonRoute, _svd


class MemoryLogger:
    """Records scalar series in memory (used by the experiment scripts)."""

    def __init__(self) -> None:
        self.history: dict[str, list[tuple[int, float]]] = {}

    def log(self, payload: dict, step: int) -> None:
        for k, v in payload.items():
            self.history.setdefault(k, []).append((step, float(v)))


class WandbLogger:
    def __init__(self, cfg: dict) -> None:
        import wandb
        self._wandb = wandb
        wandb.init(project=cfg.get("wandb_project", "dynmuon-route"),
                   name=cfg.get("run_name"), config=cfg)

    def log(self, payload: dict, step: int) -> None:
        self._wandb.log(payload, step=step)


# -- param grouping --------------------------------------------------------

def _layer_type(name: str) -> str:
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def build_optimizers(model: GPT, cfg: dict):
    """Return (dynmuon, adamw).

    ``matrix_optimizer="dynmuon"`` (default): matrices -> DynMuonRoute grouped by
    layer type; embeddings/norms/biases -> AdamW (dynmuon may be None only if the
    model has no matrices). ``matrix_optimizer="adamw"`` (the AdamW baseline):
    every parameter -> AdamW and dynmuon is None. Tied weights are deduped by id.
    """
    seen: set[int] = set()
    groups: dict[str, list] = {"attn": [], "mlp": [], "other": []}
    adam_params: list = []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embedding = name.endswith("wte.weight") or name.endswith("wpe.weight")
        (adam_params if (p.ndim < 2 or is_embedding) else groups[_layer_type(name)]).append(p)

    if cfg.get("matrix_optimizer", "dynmuon") == "adamw":
        adam_params += [p for ps in groups.values() for p in ps]
        groups = {"attn": [], "mlp": [], "other": []}

    routing_mode = cfg["routing_mode"]
    # Per-(routing_mode, layer_type) logistic params; routers only.
    route_mode = cfg.get("route", {}).get(routing_mode, {})
    fallback = route_mode.get("default", {"mu": 0.0, "omega": 1.0})
    param_groups = [
        {"params": params,
         "mu": route_mode.get(lt, fallback)["mu"],
         "omega": route_mode.get(lt, fallback)["omega"]}
        for lt, params in groups.items() if params
    ]
    dynmuon = DynMuonRoute(
        param_groups,
        lr=cfg["muon_lr"],
        momentum=cfg.get("momentum", 0.95),
        nesterov=cfg.get("nesterov", True),
        routing_mode=routing_mode,
        compute_mode=cfg["compute_mode"],
        ns_variant=cfg.get("ns_variant", "quintic"),
        ns_steps=cfg.get("ns_steps", 5),
        adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        fixed_p=cfg.get("fixed_p", 0.0),
        tau_ratio=cfg.get("tau_ratio", 0.04),
        width_ratio=cfg.get("width_ratio", 0.04),
        total_steps=cfg["max_steps"] if routing_mode == "global_schedule" else None,
    ) if param_groups else None
    adamw = torch.optim.AdamW(
        adam_params, lr=cfg["adam_lr"], betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.1),
    ) if adam_params else None
    return dynmuon, adamw


# -- noise injection (Experiment 2) ----------------------------------------

def make_noise_hook(lam: float, generator: torch.Generator):
    """Inject anisotropic noise aligned with the top singular direction of M:
    ``M += lam * sigma_1 * z * u_1 v_1ᵀ`` with ``z ~ N(0, 1)``."""
    def hook(M2: torch.Tensor, state: dict) -> torch.Tensor:
        U, S, Vh = _svd(M2)
        z = torch.randn(1, generator=generator).item()
        return M2 + (lam * S[0] * z) * torch.outer(U[:, 0], Vh[0, :])
    return hook


# -- schedule / logging ----------------------------------------------------

def lr_factor(step: int, warmup: int, max_steps: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def log_routing(model: GPT, dynmuon: DynMuonRoute, step: int, logger) -> None:
    if logger is None:
        return
    name_of = {id(p): n for n, p in model.named_parameters()}
    payload = {}
    for group in dynmuon.param_groups:
        for p in group["params"]:
            st = dynmuon.state.get(p)
            if not st or "last_p" not in st:
                continue
            n = name_of.get(id(p), "?")
            payload[f"route/p/{n}"] = st["last_p"]
            payload[f"route/sr/{n}"] = st["last_sr"]
            payload[f"route/gamma/{n}"] = st["last_gamma"]
            payload[f"route/alpha/{n}"] = st["last_alpha"]
    logger.log(payload, step=step)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, amp_ctx, iters) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# -- train -----------------------------------------------------------------

def train(cfg: dict, logger=None) -> tuple[GPT, object | None]:
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg.get("seed", 0))

    model_cfg = GPTConfig.small() if cfg.get("model") == "small" else GPTConfig.gpt124m()
    model_cfg.mlp = cfg.get("mlp", model_cfg.mlp)
    model = GPT(model_cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M non-embedding params, mlp={model_cfg.mlp}")

    data_dir = cfg.get("data_dir", "data/wikitext103")
    train_data = load_bin(f"{data_dir}/train.bin")
    val_data = load_bin(f"{data_dir}/val.bin")

    dynmuon, adamw = build_optimizers(model, cfg)
    if logger is None and cfg.get("wandb", False):
        logger = WandbLogger(cfg)

    use_amp = device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    noise_hook = None
    if cfg.get("noise_lambda", 0.0) > 0:
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        noise_hook = make_noise_hook(cfg["noise_lambda"], gen)

    block_size = model_cfg.block_size
    batch_size = cfg.get("batch_size", 12)
    grad_accum = cfg.get("grad_accum", 1)
    max_steps = cfg["max_steps"]
    warmup = cfg.get("warmup_steps", max(1, max_steps // 50))
    clip = cfg.get("grad_clip", 1.0)
    base_muon, base_adam = cfg["muon_lr"], cfg["adam_lr"]

    model.train()
    for step in range(max_steps):
        t0 = time.perf_counter()
        f = lr_factor(step, warmup, max_steps, cfg.get("min_lr_ratio", 0.1))
        if dynmuon is not None:
            for g in dynmuon.param_groups:
                g["lr"] = base_muon * f
            dynmuon.zero_grad(set_to_none=True)
        if adamw is not None:
            for g in adamw.param_groups:
                g["lr"] = base_adam * f
            adamw.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = get_batch(train_data, block_size, batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if dynmuon is not None:
            dynmuon.step(noise_hook=noise_hook)
        if adamw is not None:
            adamw.step()

        dt = time.perf_counter() - t0
        if step % cfg.get("log_every", 10) == 0 or step == max_steps - 1:
            lr_now = (base_muon if dynmuon is not None else base_adam) * f
            print(f"step {step:5d} | loss {loss_accum:.4f} | lr {lr_now:.2e} | {dt * 1e3:.0f}ms")
            if logger is not None:
                logger.log({"train/loss": loss_accum, "lr": lr_now}, step=step)
            if dynmuon is not None:
                log_routing(model, dynmuon, step, logger)

        if cfg.get("eval_every") and (step % cfg["eval_every"] == 0 or step == max_steps - 1):
            vloss = estimate_loss(model, val_data, block_size, batch_size, device, amp_ctx,
                                  cfg.get("eval_iters", 20))
            print(f"  val loss {vloss:.4f}")
            if logger is not None:
                logger.log({"val/loss": vloss}, step=step)

    return model, logger
