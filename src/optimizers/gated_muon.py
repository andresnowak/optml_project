"""GatedMuon optimizer.

GatedMuon is ordinary Muon momentum followed by a guarded polar step. The guard
suppresses near-null singular directions before Newton-Schulz can inflate them
into full update energy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .muon import _shape_lr_scale


GATED_MUON_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def gated_zeropower_via_newtonschulz5(
    G: Tensor,
    gate_tau: float = 1e-5,
    ns_steps: int = 5,
) -> Tensor:
    """Approximate ``U g_tau(Sigma) V^T`` for the normalized input matrix.

    ``gate_tau`` is measured against the singular values of
    ``G / (||G||_F + eps)``:

        g_tau(sigma) = sigma^2 / (sigma^2 + tau^2).
    """
    if G.ndim < 2:
        raise ValueError(f"GatedMuon expects matrix-like gradients, got shape {tuple(G.shape)}")
    if gate_tau <= 0.0:
        raise ValueError(f"gate_tau must be positive, got {gate_tau}")
    if not 1 <= ns_steps <= len(GATED_MUON_NS_COEFFS):
        raise ValueError(f"ns_steps must be in [1, {len(GATED_MUON_NS_COEFFS)}], got {ns_steps}")

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    Y = X
    for a, b, c in GATED_MUON_NS_COEFFS[:ns_steps]:
        A = Y @ Y.mT
        B = b * A + c * (A @ A)
        Y = a * Y + B @ Y

    A0 = X.float() @ X.float().mT
    evals, Q = torch.linalg.eigh(A0)
    evals = evals.clamp(min=0.0)
    gates = evals / (evals + gate_tau * gate_tau)
    Y = ((Q * gates) @ Q.mT) @ Y.float()

    if transposed:
        Y = Y.mT
    return Y


def gated_muon_update(
    grad: Tensor,
    momentum: Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    gate_tau: float = 1e-5,
) -> Tensor:
    """Build a GatedMuon matrix update."""
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    return gated_zeropower_via_newtonschulz5(update, gate_tau=gate_tau, ns_steps=ns_steps).to(
        dtype=grad.dtype
    )


class GatedMuon(torch.optim.Optimizer):
    """Muon with a small-singular-value gate in the polar step."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        gate_tau: float = 1e-5,
        adjust_lr_fn: str | None = "spectral_norm",
    ):
        if gate_tau <= 0.0:
            raise ValueError(f"gate_tau must be positive, got {gate_tau}")
        if adjust_lr_fn not in (None, "none", "spectral_norm", "keller_jordan"):
            raise ValueError(
                "adjust_lr_fn must be one of None, 'none', 'spectral_norm', "
                f"or 'keller_jordan', got {adjust_lr_fn!r}"
            )
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            nesterov=nesterov,
            ns_steps=ns_steps,
            gate_tau=gate_tau,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one GatedMuon step.

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
                    raise ValueError(f"GatedMuon expects 2D gradients, got {tuple(p.grad.shape)}")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                update = gated_muon_update(
                    p.grad,
                    state["momentum"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    ns_steps=group["ns_steps"],
                    gate_tau=group["gate_tau"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(p.grad.size(-2), p.grad.size(-1), group["adjust_lr_fn"])
                p.add_(update, alpha=-group["lr"] * lr_scale)
        return loss
