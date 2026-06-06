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
    specmuon_target: str = "all"          # "all" | "mlp" | "attention"
    scheduler: str = "linear"             # "none" | "linear" | "cosine"
    min_lr: float = 0.0

    def with_(self, **changes: Any) -> "TrainConfig":
        return replace(self, **changes)


def _is_attention_param(name: str) -> bool:
    """Heuristic: parameter belongs to a self-attention submodule.

    Matches the naming conventions used in `src/experiments/shakespeare.py`
    (`blocks.<i>.attn.qkv.weight`, `blocks.<i>.attn.proj.weight`). Also matches
    the more general `.attention.` token used in other transformer codebases.
    """
    return ".attn." in name or ".attention." in name


def _is_mlp_param(name: str) -> bool:
    """Heuristic: parameter belongs to a feed-forward / MLP submodule.

    Matches `blocks.<i>.mlp.fc{1,2}.weight` from the shakespeare model and
    common synonyms (`.ffn.`, `.feedforward.`). The output `head` and the
    embeddings are intentionally NOT classified as MLPs.
    """
    return ".mlp." in name or ".ffn." in name or ".feedforward." in name


def _split_matrix_by_target(model: nn.Module, target: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split 2-D+ matrix weights into (targeted, untargeted) for the
    selective-SAV ablation. Non-matrix params (embeddings, LayerNorms,
    biases) are handled separately by ``_split_params`` and not touched here.

    ``target ∈ {"mlp", "attention"}``. The classification uses parameter
    names — keep it consistent with the model's nn.Module naming or expect
    the wrong routing. Output `head` and any other matrix that doesn't match
    the selected predicate land in ``untargeted`` (they'll receive the
    paper-tail SpecMuon update with ``top_k=0``).
    """
    if target not in ("mlp", "attention"):
        raise ValueError(f"specmuon_target must be 'mlp' or 'attention', got {target!r}")
    is_target = _is_mlp_param if target == "mlp" else _is_attention_param
    name_of = {id(p): n for n, p in model.named_parameters()}
    norm_or_emb = (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)
    targeted: list[nn.Parameter] = []
    untargeted: list[nn.Parameter] = []
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if isinstance(module, norm_or_emb) or p.ndim < 2:
                continue
            name = name_of.get(id(p), "")
            (targeted if is_target(name) else untargeted).append(p)
    return targeted, untargeted


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


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer | None,
    scheduler: str,
    steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if optimizer is None or scheduler == "none":
        return None
    schedule_iters = max(1, steps - 1)
    base_lr = optimizer.param_groups[0]["lr"]
    if min_lr < 0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}")
    if min_lr > base_lr:
        raise ValueError(f"min_lr ({min_lr}) must be <= lr ({base_lr})")
    if scheduler == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / base_lr if base_lr > 0 else 0.0,
            total_iters=schedule_iters,
        )
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=schedule_iters,
            eta_min=min_lr,
        )
    raise ValueError(f"unsupported scheduler {scheduler!r}")


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
    total_params = sum(p.numel() for p in model.parameters())
    if log_sink is not None:
        log_sink.update_config({
            "model_params_total": total_params,
            "model_params_total_m": total_params / 1_000_000,
        })
    # ``aux_specmuon`` is the second SpecMuon used by the selective-SAV
    # ablation: when ``specmuon_target ∈ {"mlp","attention"}`` it carries the
    # *untargeted* matrix params with ``top_k=0`` (paper-tail-only update),
    # while the primary ``optimizer`` carries the targeted ones with the
    # user's full opt_kwargs. ``aux_specmuon`` is None in every other case.
    aux_specmuon: torch.optim.Optimizer | None = None
    if config.optimizer_name in ("muon", "specmuon"):
        matrix_params, other_params = _split_params(
            model, accept_high_rank=(config.optimizer_name == "specmuon"),
        )
        if config.optimizer_name == "specmuon" and config.specmuon_target != "all":
            sav_params, tail_params = _split_matrix_by_target(model, config.specmuon_target)
            if not sav_params:
                raise ValueError(
                    f"--specmuon-target={config.specmuon_target!r} matched no parameters; "
                    f"check the model's nn.Module names against _is_{config.specmuon_target}_param."
                )
            optimizer = build_optimizer(
                "specmuon", sav_params, config.lr, config.weight_decay, **config.opt_kwargs
            )
            if tail_params:
                tail_kwargs = {**config.opt_kwargs, "top_k": 0}
                aux_specmuon = build_optimizer(
                    "specmuon", tail_params, config.lr, config.weight_decay, **tail_kwargs
                )
        else:
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

    lr_schedulers = [
        scheduler
        for scheduler in (
            _build_lr_scheduler(optimizer, config.scheduler, config.steps, config.min_lr),
            _build_lr_scheduler(aux_specmuon, config.scheduler, config.steps, config.min_lr),
            _build_lr_scheduler(adam_optimizer, config.scheduler, config.steps, config.min_lr),
        )
        if scheduler is not None
    ]

    prefix = f"{run_name}/" if run_name else ""
    svd_every = config.svd_every or config.log_every

    print(
        f"experiment={config.experiment_name} optimizer={config.optimizer_name} "
        f"device={config.device.type} steps={config.steps} "
        f"scheduler={config.scheduler} min_lr={config.min_lr:.2e}"
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
                "scheduler": config.scheduler,
                "min_lr": config.min_lr,
                "opt_kwargs": dict(config.opt_kwargs),
            },
        }
        if adam_optimizer is not None:
            payload["adam_optimizer"] = adam_optimizer.state_dict()
        if aux_specmuon is not None:
            payload["aux_specmuon"] = aux_specmuon.state_dict()
        if lr_schedulers:
            payload["lr_schedulers"] = [scheduler.state_dict() for scheduler in lr_schedulers]
        torch.save(payload, path)
        print(f"  saved {path.name}")

    losses: list[float] = []
    for step in range(1, config.steps + 1):
        step_t0 = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        if aux_specmuon is not None:
            aux_specmuon.zero_grad(set_to_none=True)
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
        if aux_specmuon is not None:
            # aux_specmuon is always a SpecMuon (top_k=0 paper-tail variant).
            aux_specmuon.step(loss=loss)
        if adam_optimizer is not None:
            adam_optimizer.step()
        for scheduler in lr_schedulers:
            scheduler.step()

        if (log_sink is not None and config.log_sav_r and isinstance(optimizer, SpecMuon)
                and (step == 1 or step % svd_every == 0 or step == config.steps)):
            # Look up state across BOTH SpecMuon optimizers — under the
            # selective-SAV ablation each param lives in exactly one of them.
            spec_opts = [optimizer] + ([aux_specmuon] if isinstance(aux_specmuon, SpecMuon) else [])
            for name, param in model.named_parameters():
                for opt in spec_opts:
                    pstate = opt.state.get(param)
                    if pstate is not None and "r" in pstate:
                        log_sink.log({f"{prefix}sav_r/{name}": pstate["r"].detach()}, step)
                        if "last_sav_scale" in pstate:
                            log_sink.log({f"{prefix}sav_sigma_scale/{name}": pstate["last_sav_scale"]}, step)
                        if "last_iota" in pstate:
                            log_sink.log({
                                f"{prefix}sav_iota/{name}": pstate["last_iota"],
                                f"{prefix}sav_iota_w/{name}": pstate["last_iota_w"],
                                f"{prefix}sav_sigma_min/{name}": pstate.get("last_sigma_min", 0.0),
                                f"{prefix}sav_sigma_max/{name}": pstate.get("last_sigma_max", 0.0),
                            }, step)
                        break

        step_dt = time.perf_counter() - step_t0

        losses.append(loss.item())

        if step == 1 or step % config.log_every == 0 or step == config.steps:
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
