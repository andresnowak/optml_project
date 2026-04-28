from __future__ import annotations

import torch

from optml_project.experiments import EXPERIMENTS
from optml_project.optimizers import build_optimizer


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

    print(f"experiment={experiment_name} optimizer={optimizer_name} device={device.type} steps={steps}")

    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = experiment.next_batch()
        loss = experiment.loss(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step={step:04d} loss={loss.item():.6f}")

    return losses
