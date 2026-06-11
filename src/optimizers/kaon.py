"""Kaon optimizer: chaotic Newton-Schulz-style spectral shaping.

Kaon follows the chaotic polynomial iteration from Shumaylov et al.,
"Muon is Not That Special: Random or Inverted Spectra Work Just as Well"
(arXiv:2605.11181). It uses the same momentum, Nesterov, shape scaling, and
decoupled weight decay conventions as this repo's Muon baseline, but replaces
Muon's polar iteration with the chaotic map
``X <- lambda * (I - X X^T)^2 X``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .muon import _shape_lr_scale


def kaon_chaos_map(
    G: Tensor,
    *,
    chaos_steps: int = 5,
    chaos_lambda: float = 4.1,
    output_scale: float = 1.175,
    eps: float = 1e-7,
) -> Tensor:
    """Apply the paper's Kaon chaotic spectral iteration.

    With ``X = U diag(s) V^T``, the matrix update applies
    ``s <- lambda * s * (1 - s^2)^2`` at each iteration. The final division by
    ``1.175`` matches the paper pseudocode's stationary-support normalization.
    """
    if G.ndim < 2:
        raise ValueError(f"Kaon expects matrix-like gradients, got shape {tuple(G.shape)}")
    if chaos_steps <= 0:
        raise ValueError(f"chaos_steps must be positive, got {chaos_steps}")
    if output_scale <= 0:
        raise ValueError(f"output_scale must be positive, got {output_scale}")

    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(chaos_steps):
        A = X @ X.mT
        I = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
        B = I - A
        X = chaos_lambda * ((B @ B) @ X)
    X = X / output_scale
    if transposed:
        X = X.mT
    return X


def kaon_update(
    grad: Tensor,
    momentum: Tensor,
    *,
    mu: float = 0.95,
    nesterov: bool = True,
    chaos_steps: int = 5,
    chaos_lambda: float = 4.1,
    output_scale: float = 1.175,
    eps: float = 1e-7,
) -> Tensor:
    """Build a Kaon matrix update with momentum and chaotic spectral shaping.

    Uses the same EMA-scaled lookahead convention as Muon. Kaon's map
    normalizes its input, so the positive scale relative to sum-form Nesterov
    does not affect the shaped direction.
    """
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    return kaon_chaos_map(
        update,
        chaos_steps=chaos_steps,
        chaos_lambda=chaos_lambda,
        output_scale=output_scale,
        eps=eps,
    ).to(dtype=grad.dtype)


class Kaon(torch.optim.Optimizer):
    """Kaon for 2D matrix parameters, with AdamW expected for non-matrix params."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        adjust_lr_fn: str | None = "spectral_norm",
        chaos_steps: int = 5,
        chaos_lambda: float = 4.1,
        output_scale: float = 1.175,
        eps: float = 1e-7,
    ):
        if adjust_lr_fn not in (None, "none", "spectral_norm", "keller_jordan"):
            raise ValueError(
                "adjust_lr_fn must be one of None, 'none', 'spectral_norm', "
                f"or 'keller_jordan', got {adjust_lr_fn!r}"
            )
        if chaos_steps <= 0:
            raise ValueError(f"chaos_steps must be positive, got {chaos_steps}")
        if output_scale <= 0:
            raise ValueError(f"output_scale must be positive, got {output_scale}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov,
            adjust_lr_fn=adjust_lr_fn, chaos_steps=chaos_steps,
            chaos_lambda=chaos_lambda, output_scale=output_scale, eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one Kaon step.

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
                    raise ValueError(f"Kaon expects 2D gradients, got {tuple(p.grad.shape)}")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                update = kaon_update(
                    p.grad,
                    state["momentum"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    chaos_steps=group["chaos_steps"],
                    chaos_lambda=group["chaos_lambda"],
                    output_scale=group["output_scale"],
                    eps=group["eps"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(p.grad.size(-2), p.grad.size(-1), group["adjust_lr_fn"])
                p.add_(update, alpha=-group["lr"] * lr_scale)
        return loss
