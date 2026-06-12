"""Track-3-style Muon optimizer.

This mirrors the compact Muon implementation used in modded-nanogpt's Track 3
training script: a compiled Newton-Schulz polar update, Nesterov momentum, Muon
shape scaling, and decoupled weight decay for matrix parameters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

MUON_ORTHOGONALIZE_MODES = ("ns", "svd")


def zeropower_via_newtonschulz5(G: Tensor, ns_steps: int = 12) -> Tensor:
    """Approximate the zeroth power / polar factor with Newton-Schulz steps."""
    if G.ndim < 2:
        raise ValueError(f"Muon expects matrix-like gradients, got shape {tuple(G.shape)}")
    X = G.bfloat16()  # Faster Newton-Schulz; empirically close to fp32 for Muon.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def zeropower_via_svd(G: Tensor) -> Tensor:
    """Exact polar factor ``U Vᵀ`` from an SVD.

    This sets every numerically live singular value to one. Rank-deficient
    directions remain absent because ``full_matrices=False`` returns only the
    compact singular-vector factors.
    """
    if G.ndim < 2:
        raise ValueError(f"Muon expects matrix-like gradients, got shape {tuple(G.shape)}")
    U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return (U @ Vh).to(dtype=G.dtype)


def muon_update(
    grad: Tensor,
    momentum: Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 12,
    orthogonalize: str = "ns",
) -> Tensor:
    """Build a Muon matrix update.

    The buffer is stored in EMA-scaled form. With ``nesterov=True`` the
    pre-orthogonalization matrix is ``(1-mu)`` times the usual sum-form
    Nesterov direction ``g + mu * (mu * B + g)``. The polar step is
    scale-invariant, so this matches the reference direction for
    ``orthogonalize="ns"`` and the exact SVD-polar ablation.
    """
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    if orthogonalize == "ns":
        return zeropower_via_newtonschulz5(update, ns_steps)
    if orthogonalize == "svd":
        return zeropower_via_svd(update)
    raise ValueError(f"orthogonalize must be one of {MUON_ORTHOGONALIZE_MODES}, got {orthogonalize!r}")


def _shape_lr_scale(fan_out: int, fan_in: int, adjust_lr_fn: str | None) -> float:
    if adjust_lr_fn in (None, "none"):
        return 1.0
    if adjust_lr_fn == "spectral_norm":
        return float(math.sqrt(fan_out / fan_in))
    if adjust_lr_fn == "keller_jordan":
        return float(max(1.0, fan_out / fan_in) ** 0.5)
    raise ValueError(f"unsupported Muon adjust_lr_fn: {adjust_lr_fn}")


class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters, with AdamW expected for non-matrix params."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 12,
        orthogonalize: str = "ns",
        adjust_lr_fn: str | None = "spectral_norm",
    ):
        if orthogonalize not in MUON_ORTHOGONALIZE_MODES:
            raise ValueError(
                f"orthogonalize must be one of {MUON_ORTHOGONALIZE_MODES}, got {orthogonalize!r}"
            )
        if adjust_lr_fn not in (None, "none", "spectral_norm", "keller_jordan"):
            raise ValueError(
                "adjust_lr_fn must be one of None, 'none', 'spectral_norm', "
                f"or 'keller_jordan', got {adjust_lr_fn!r}"
            )
        defaults = dict(
            lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov,
            ns_steps=ns_steps, orthogonalize=orthogonalize, adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one Muon step.

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
                    raise ValueError(f"Muon expects 2D gradients, got {tuple(p.grad.shape)}")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    ns_steps=group["ns_steps"],
                    orthogonalize=group["orthogonalize"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(p.grad.size(-2), p.grad.size(-1), group["adjust_lr_fn"])
                p.add_(update, alpha=-group["lr"] * lr_scale)
        return loss
