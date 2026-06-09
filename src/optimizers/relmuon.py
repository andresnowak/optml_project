"""RelMuon optimizer for matrix parameters.

RelMuon keeps Muon's gradient-facing singular vectors, but replaces Muon's flat
update spectrum with scales derived from the current weight singular values.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


RELMUON_SCALE_MODES = ("log1p", "rms", "complete")


def _validate_scale_mode(scale_mode: str) -> None:
    if scale_mode not in RELMUON_SCALE_MODES:
        modes = ", ".join(RELMUON_SCALE_MODES)
        raise ValueError(f"Unknown RelMuon scale mode {scale_mode!r}; expected one of: {modes}")


def relmuon_weight_scales(weight: Tensor, scale_mode: str = "log1p", eps: float = 1e-8) -> Tensor:
    """Return the singular scales RelMuon will use for a weight matrix."""
    _validate_scale_mode(scale_mode)
    sv = torch.linalg.svdvals(weight.float()).clamp(min=0.0)
    if scale_mode == "complete":
        if torch.sqrt(torch.mean(sv.square())) <= eps:
            return torch.ones_like(sv) # Muon like if the Weights are initialized to 0 (or just to small values)
        return sv
    if scale_mode == "rms":
        rms = torch.sqrt(torch.mean(sv.square()))
        return (sv + eps) / (rms + eps) # Muon like if the Weights are 0.
    if scale_mode == "log1p":
        log_scales = torch.log1p(sv)
        rms = torch.sqrt(torch.mean(log_scales.square()))
        return (log_scales + eps) / (rms + eps)
    raise AssertionError(f"Unhandled RelMuon scale mode: {scale_mode}")


@torch.compile
def relmuon_update(
    grad: Tensor,
    weight: Tensor,
    momentum: Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    eps: float = 1e-8,
    scale_mode: str = "log1p",
) -> Tensor:
    """Build a RelMuon matrix update."""
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    update_f = update.float()

    U, _, Vh = torch.linalg.svd(update_f, full_matrices=False)
    scales = relmuon_weight_scales(weight, scale_mode=scale_mode, eps=eps)

    rank = min(U.size(-1), Vh.size(-2), scales.numel())
    shaped = (U[:, :rank] * scales[:rank].to(U.dtype)) @ Vh[:rank, :]
    return shaped.to(dtype=grad.dtype)


relmuon_log1p_update = relmuon_update


class RelMuon(torch.optim.Optimizer):
    """RelMuon for 2D matrix parameters.

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
        scale_mode: str = "log1p",
    ):
        if adjust_lr_fn not in (None, "none"):
            raise ValueError(f"RelMuon only supports adjust_lr_fn=None for now, got {adjust_lr_fn!r}")
        _validate_scale_mode(scale_mode)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov,
            eps=eps, adjust_lr_fn=adjust_lr_fn, scale_mode=scale_mode,
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
                update = relmuon_update(
                    p.grad,
                    p,
                    state["momentum"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    eps=group["eps"],
                    scale_mode=group["scale_mode"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss
