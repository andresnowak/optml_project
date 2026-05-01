from __future__ import annotations

import torch
from torch import nn

from src.experiments import EXPERIMENTS
from src.logger import BaseLogger
from src.optimizers import SpecMuon, build_optimizer


def _split_params(model: nn.Module):
    """Return (matrix_params, other_params).

    Embeddings and LayerNorm weights are excluded from matrix_params because
    Muon/SpecMuon expect proper weight matrices; everything else goes to Adam.
    """
    matrix_params, other_params = [], []
    for module in model.modules():
        for p in module.parameters(recurse=False):
            if isinstance(module, (nn.Embedding, nn.LayerNorm)) or p.ndim < 2:
                other_params.append(p)
            else:
                matrix_params.append(p)
    return matrix_params, other_params


def train(
    experiment_name: str,
    optimizer_name: str,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    log_every: int,
    feature_dim: int,
    output_dim: int,
    samples: int,
    matrix_rows: int,
    matrix_cols: int,
    rank: int,
    block_size: int = 128,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    opt_kwargs: dict | None = None,
    logger: BaseLogger | None = None,
    run_name: str | None = None,
    log_grad_svd: bool = False,
    svd_every: int | None = None,
) -> list[float]:
    experiment = EXPERIMENTS[experiment_name](
        device=device,
        batch_size=batch_size,
        feature_dim=feature_dim,
        output_dim=output_dim,
        samples=samples,
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        rank=rank,
        block_size=block_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    model = experiment.build_model()
    if optimizer_name in ("muon", "specmuon"):
        matrix_params, other_params = _split_params(model)
        optimizer = build_optimizer(optimizer_name, matrix_params, lr, weight_decay, **(opt_kwargs or {}))
        adam_optimizer: torch.optim.Optimizer | None = (
            torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay) if other_params else None
        )
    else:
        optimizer = build_optimizer(optimizer_name, model.parameters(), lr, weight_decay, **(opt_kwargs or {}))
        adam_optimizer = None

    prefix = f"{run_name}/" if run_name else ""
    _svd_every = svd_every or log_every

    print(f"experiment={experiment_name} optimizer={optimizer_name} device={device.type} steps={steps}")

    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if adam_optimizer is not None:
            adam_optimizer.zero_grad(set_to_none=True)
        batch = experiment.next_batch()
        loss = experiment.loss(model, batch)
        loss.backward()

        if logger is not None and log_grad_svd and (step == 1 or step % _svd_every == 0 or step == steps):
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.dim() >= 2:
                    g = param.grad.detach().reshape(param.grad.shape[0], -1).cpu().float()
                    svs = torch.linalg.svdvals(g)
                    logger.log({f"{prefix}grad_svd/{name}": svs}, step)

        if isinstance(optimizer, SpecMuon):
            optimizer.step(loss=loss)
        else:
            optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % log_every == 0 or step == steps:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:04d} loss={loss.item():.6f} lr={current_lr:.2e}")
            if logger is not None:
                logger.log({f"{prefix}loss": loss.item(), f"{prefix}lr": current_lr}, step)

    return losses
