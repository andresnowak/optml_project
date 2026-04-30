from __future__ import annotations

import torch

from src.experiments import EXPERIMENTS
from src.logger import BaseLogger
from src.optimizers import SpecMuon, build_optimizer


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
    samples: int,
    matrix_rows: int,
    matrix_cols: int,
    rank: int,
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
        samples=samples,
        matrix_rows=matrix_rows,
        matrix_cols=matrix_cols,
        rank=rank,
    )
    model = experiment.build_model()
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr, weight_decay, **(opt_kwargs or {}))

    prefix = f"{run_name}/" if run_name else ""
    _svd_every = svd_every or log_every

    print(f"experiment={experiment_name} optimizer={optimizer_name} device={device.type} steps={steps}")

    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
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
        losses.append(loss.item())

        if step == 1 or step % log_every == 0 or step == steps:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step={step:04d} loss={loss.item():.6f} lr={current_lr:.2e}")
            if logger is not None:
                logger.log({f"{prefix}loss": loss.item(), f"{prefix}lr": current_lr}, step)

    return losses
