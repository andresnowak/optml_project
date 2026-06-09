"""RelMuon-log1p optimizer for matrix parameters.

RelMuon keeps Muon's gradient-facing singular vectors, but replaces Muon's flat
update spectrum with normalized ``log(1 + singular_value(weight))`` scales.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


@torch.compile
def relmuon_log1p_update(
    grad: Tensor,
    weight: Tensor,
    momentum: Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Build the log1p RelMuon matrix update."""
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    update_f = update.float()
    weight_f = weight.float()

    U, _, Vh = torch.linalg.svd(update_f, full_matrices=False)
    scales = torch.log1p(torch.linalg.svdvals(weight_f).clamp(min=0.0))
    rms = torch.sqrt(torch.mean(scales.square()))
    scales = (scales + eps) / (rms + eps)

    rank = min(U.size(-1), Vh.size(-2), scales.numel())
    shaped = (U[:, :rank] * scales[:rank].to(U.dtype)) @ Vh[:rank, :]
    return shaped.to(dtype=grad.dtype)


class RelMuon(torch.optim.Optimizer):
    """Log1p RelMuon for 2D matrix parameters.

    Non-matrix parameters should be optimized by the auxiliary AdamW path.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-8,
        adjust_lr_fn: str | None = None,
    ):
        if adjust_lr_fn not in (None, "none"):
            raise ValueError(f"RelMuon only supports adjust_lr_fn=None for now, got {adjust_lr_fn!r}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov,
            eps=eps, adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one RelMuon step.

        ``noise_hook`` is accepted for trainer API compatibility with
        ``DynMuonRoute`` and intentionally ignored.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError(f"RelMuon expects 2D gradients, got {tuple(p.grad.shape)}")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                update = relmuon_log1p_update(
                    p.grad,
                    p,
                    state["momentum"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    eps=group["eps"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss
