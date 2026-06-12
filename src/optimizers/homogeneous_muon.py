"""HomogeneousMuon optimizer.

This is a Frobenius-normalized fixed-power spectral shaping variant:

    X = M / ||M||_F = U diag(sigma) V^T,    D = U diag(sigma^p) V^T.

It uses the same EMA-scaled Nesterov momentum convention, shape-aware learning
rate scaling, and decoupled matrix weight decay as ``Muon`` in this repo. The
    default ``p=0.25`` partially flattens the normalized momentum spectrum. The
    supported range is ``0 < p <= 1``: ``p=1`` is Frobenius-normalized momentum,
    while smaller positive values make the spectrum more homogeneous without
    using the singular p=0 polar endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .muon import _shape_lr_scale


def _check_homogeneous_p(p: float) -> None:
    if not math.isfinite(p) or not (0.0 < p <= 1.0):
        raise ValueError(f"HomogeneousMuon requires 0 < p <= 1, got {p}")


def power_spectrum_via_svd(G: Tensor, p: float = 0.25, eps: float = 1e-7) -> Tensor:
    """Return ``U diag(sigma^p) V^T`` from the SVD of ``G / ||G||_F``.

    Since ``0 < p <= 1``, exact zero and exact one are fixed points of the map:
    ``0^p = 0`` and ``1^p = 1``.
    """
    if G.ndim < 2:
        raise ValueError(f"HomogeneousMuon expects matrix-like gradients, got shape {tuple(G.shape)}")
    _check_homogeneous_p(p)

    X = G.float()
    X = X / (torch.linalg.norm(X) + eps)

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    D = (U * S.pow(p)) @ Vh

    if transposed:
        D = D.mT
    return D.to(dtype=G.dtype)


def homogeneous_muon_update(
    grad: Tensor,
    momentum: Tensor,
    *,
    p: float = 0.25,
    mu: float = 0.95,
    nesterov: bool = True,
) -> Tensor:
    """Build one HomogeneousMuon matrix update.

    The momentum buffer is stored in the same EMA-scaled form as ``Muon``.
    With ``nesterov=True``, the shaped matrix is built from
    ``grad.lerp(momentum, mu)`` after the EMA update.
    """
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    return power_spectrum_via_svd(update, p=p)


class HomogeneousMuon(torch.optim.Optimizer):
    """Fixed-power SVD spectral shaping for 2D matrix parameters."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        p: float = 0.25,
        mu: float = 0.95,
        nesterov: bool = True,
        adjust_lr_fn: str | None = "spectral_norm",
    ):
        _check_homogeneous_p(p)
        if adjust_lr_fn not in (None, "none", "spectral_norm", "keller_jordan"):
            raise ValueError(
                "adjust_lr_fn must be one of None, 'none', 'spectral_norm', "
                f"or 'keller_jordan', got {adjust_lr_fn!r}"
            )
        defaults = dict(
            lr=lr, weight_decay=weight_decay, p=p, mu=mu, nesterov=nesterov,
            adjust_lr_fn=adjust_lr_fn, routing_mode="fixed",
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one HomogeneousMuon step.

        ``noise_hook`` is accepted for trainer API compatibility with
        ``DynMuonRoute`` and intentionally ignored.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.ndim != 2:
                    raise ValueError(f"HomogeneousMuon expects 2D gradients, got {tuple(param.grad.shape)}")
                state = self.state[param]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(param)
                update = homogeneous_muon_update(
                    param.grad,
                    state["momentum"],
                    p=group["p"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                )
                if group["weight_decay"]:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(
                    param.grad.size(-2), param.grad.size(-1), group["adjust_lr_fn"]
                )
                param.add_(update, alpha=-group["lr"] * lr_scale)
                state["last_p"] = float(group["p"])
        return loss
