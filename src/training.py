from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.experiments import EXPERIMENTS
from src.logger import BaseLogger
from src.optimizers import SpecMuon, build_optimizer
from src.utils import filtered_experiment_kwargs

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """All knobs needed to run one training trajectory.

    `experiment_kwargs` is filtered against the experiment class signature at
    `train()` time, so the same dict can be reused across experiments.
    `opt_kwargs` is filtered against the chosen optimizer's accepted kwargs by
    `build_optimizer`.
    """

    experiment_name: str
    optimizer_name: str
    device: torch.device
    steps: int
    lr: float
    weight_decay: float = 0.0
    batch_size: int = 128
    log_every: int = 25
    experiment_kwargs: Mapping[str, Any] = field(default_factory=dict)
    opt_kwargs: Mapping[str, Any] = field(default_factory=dict)
    log_grad_svd: bool = False
    log_grad_norms: bool = False
    log_weight_norms: bool = False
    log_sav_r: bool = False
    svd_every: int | None = None
    checkpoint_dir: str | None = None
    checkpoint_every: int | None = None  # None = end-of-training only

    def with_(self, **changes: Any) -> "TrainConfig":
        return replace(self, **changes)


def _split_params(
    model: nn.Module,
    *,
    accept_high_rank: bool = True,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(matrix_params, other_params)``.

    Embedding/LayerNorm/BatchNorm weights and any 1-D parameter (biases, scales)
    always go to AdamW; 2-D weights always go to the matrix optimizer
    (Muon/SpecMuon).

    For 4-D Conv2d weights and other higher-rank tensors, ``accept_high_rank``
    decides: SpecMuon flattens internally to ``(out, in*…)`` and handles them,
    so it passes ``True``. Stock ``torch.optim.Muon`` rejects anything but 2-D,
    so it passes ``False`` to send those to AdamW instead.
    """
    matrix_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    norm_or_emb = (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if isinstance(module, norm_or_emb) or p.ndim < 2:
                other_params.append(p)
            elif p.ndim == 2 or accept_high_rank:
                matrix_params.append(p)
            else:
                other_params.append(p)
    return matrix_params, other_params


def train(
    config: TrainConfig,
    *,
    log_sink: BaseLogger | None = None,
    run_name: str | None = None,
) -> list[float]:
    """Run one training trajectory and return the per-step loss list."""
    cls = EXPERIMENTS[config.experiment_name]
    exp_kwargs = filtered_experiment_kwargs(config.experiment_name, config.experiment_kwargs)
    experiment = cls(device=config.device, batch_size=config.batch_size, **exp_kwargs)

    model = experiment.build_model()
    if config.optimizer_name in ("muon", "specmuon"):
        matrix_params, other_params = _split_params(
            model, accept_high_rank=(config.optimizer_name == "specmuon"),
        )
        optimizer = build_optimizer(
            config.optimizer_name, matrix_params, config.lr, config.weight_decay, **config.opt_kwargs
        )
        adam_optimizer: torch.optim.Optimizer | None = (
            torch.optim.AdamW(other_params, lr=config.lr, weight_decay=config.weight_decay)
            if other_params else None
        )
    else:
        optimizer = build_optimizer(
            config.optimizer_name, model.parameters(), config.lr, config.weight_decay, **config.opt_kwargs
        )
        adam_optimizer = None

    prefix = f"{run_name}/" if run_name else ""
    svd_every = config.svd_every or config.log_every

    print(
        f"experiment={config.experiment_name} optimizer={config.optimizer_name} "
        f"device={config.device.type} steps={config.steps}"
    )

    ckpt_dir: Path | None = None
    if config.checkpoint_dir is not None:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"checkpoints -> {ckpt_dir}")

    def _save_ckpt(step: int, final: bool = False) -> None:
        if ckpt_dir is None:
            return
        tag = "final" if final else f"step{step:06d}"
        path = ckpt_dir / f"{config.experiment_name}_{config.optimizer_name}_seed{run_name or 'run'}_{tag}.pt"
        payload: dict = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": losses[-1] if losses else None,
            "config": {
                "experiment": config.experiment_name,
                "optimizer": config.optimizer_name,
                "lr": config.lr,
                "steps": config.steps,
                "opt_kwargs": dict(config.opt_kwargs),
            },
        }
        if adam_optimizer is not None:
            payload["adam_optimizer"] = adam_optimizer.state_dict()
        torch.save(payload, path)
        print(f"  saved {path.name}")

    losses: list[float] = []
    for step in range(1, config.steps + 1):
        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if adam_optimizer is not None:
            adam_optimizer.zero_grad(set_to_none=True)
        batch = experiment.next_batch()
        loss = experiment.loss(model, batch)
        loss.backward()

        if log_sink is not None and config.log_grad_svd and (step == 1 or step % svd_every == 0 or step == config.steps):
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.dim() >= 2:
                    g = param.grad.detach().reshape(param.grad.shape[0], -1).cpu().float()
                    svs = torch.linalg.svdvals(g)
                    log_sink.log({f"{prefix}grad_svd/{name}": svs}, step)

        if isinstance(optimizer, SpecMuon):
            optimizer.step(loss=loss)
        else:
            optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()

        if (log_sink is not None and config.log_sav_r and isinstance(optimizer, SpecMuon)
                and (step == 1 or step % svd_every == 0 or step == config.steps)):
            for name, param in model.named_parameters():
                pstate = optimizer.state.get(param)
                if pstate is not None and "r" in pstate:
                    log_sink.log({f"{prefix}sav_r/{name}": pstate["r"].detach()}, step)
                    if "last_iota" in pstate:
                        log_sink.log({
                            f"{prefix}sav_iota/{name}": pstate["last_iota"],
                            f"{prefix}sav_iota_w/{name}": pstate["last_iota_w"],
                        }, step)

        step_dt = time.perf_counter() - step_t0

        losses.append(loss.item())

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            current_lr = optimizer.param_groups[0]["lr"]
            extra = experiment.metric(model)
            extra_str = " " + " ".join(f"{k}={v:.4f}" for k, v in extra.items()) if extra else ""
            print(f"step={step:04d} loss={loss.item():.6f} lr={current_lr:.2e} time(s)={step_dt:.2f}{extra_str}")
            if log_sink is not None:
                payload: dict[str, Any] = {
                    f"{prefix}loss": loss.item(),
                    f"{prefix}lr": current_lr,
                    f"{prefix}time(s)": step_dt,
                    f"{prefix}step_seconds": step_dt,
                }
                for k, v in extra.items():
                    payload[f"{prefix}{k}"] = v
                log_sink.log(payload, step)
                if config.log_grad_norms:
                    grad_norms = {
                        f"{prefix}grad_norm/{n}": p.grad.detach().norm().item()
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    }
                    if grad_norms:
                        total_grad_norm = torch.linalg.vector_norm(
                            torch.stack([
                                p.grad.detach().norm()
                                for p in model.parameters()
                                if p.grad is not None
                            ])
                        ).item()
                        grad_norms[f"{prefix}grad_norm/total"] = total_grad_norm
                        log_sink.log(grad_norms, step)
                if config.log_weight_norms:
                    log_sink.log(
                        {f"{prefix}weight_norm/{n}": p.detach().norm().item()
                         for n, p in model.named_parameters()},
                        step,
                    )

        if (config.checkpoint_every is not None
                and step % config.checkpoint_every == 0
                and step != config.steps):
            _save_ckpt(step)

    _save_ckpt(config.steps, final=True)
    return losses
