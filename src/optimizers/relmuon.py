"""RelMuon optimizer for matrix parameters.

RelMuon keeps Muon's gradient-facing singular vectors, but replaces Muon's flat
update spectrum with scales derived from the current weight matrix ``W``.

Scale modes (``scale_mode``):

  * ``log1p`` (default) — ``s_i = log(1 + σ_i(W))``, RMS-normalized. Pairs the
    i-th largest weight singular value with the i-th strongest update
    direction (an *ordinal* pairing: the two singular bases are unrelated).
  * ``rms``      — ``s_i = σ_i(W) / RMS(σ(W))`` (ordinal pairing, linear).
  * ``complete`` — ``s_i = σ_i(W)`` raw (ordinal pairing, unnormalized; the
    spectral analogue of LARS-style relative updates — the update spectrum is
    proportional to the weight spectrum, which is multiplicative dynamics and
    needs ``scale_cap`` or a small LR to stay stable).
  * ``log1p_aligned`` — ``s_i = log(1 + ‖W v_i‖₂)``, RMS-normalized, where
    ``v_i`` is the i-th *right singular vector of the update*. This removes
    the arbitrary ordinal pairing: each update direction is scaled by how
    strongly the weight matrix actually acts along that direction.

All normalized modes degrade gracefully to Muon (all scales = 1) when the
weight matrix is zero (e.g. zero-initialized projection layers).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


RELMUON_SCALE_MODES = ("log1p", "rms", "complete", "log1p_aligned")
# Modes whose scales depend only on the weight matrix (loggable without an update).
RELMUON_WEIGHT_ONLY_MODES = ("log1p", "rms", "complete")


def _validate_scale_mode(scale_mode: str) -> None:
    if scale_mode not in RELMUON_SCALE_MODES:
        modes = ", ".join(RELMUON_SCALE_MODES)
        raise ValueError(f"Unknown RelMuon scale mode {scale_mode!r}; expected one of: {modes}")


def _normalize_log1p(values: Tensor, eps: float) -> Tensor:
    """``log1p`` then RMS normalization (RMS of the scales is ~1)."""
    log_scales = torch.log1p(values)
    rms = torch.sqrt(torch.mean(log_scales.square()))
    return (log_scales + eps) / (rms + eps)


def relmuon_weight_scales(weight: Tensor, scale_mode: str = "log1p", eps: float = 1e-8) -> Tensor:
    """Return the singular scales RelMuon uses for a weight-only scale mode."""
    _validate_scale_mode(scale_mode)
    if scale_mode not in RELMUON_WEIGHT_ONLY_MODES:
        raise ValueError(
            f"scale mode {scale_mode!r} depends on the update direction; "
            "use relmuon_aligned_scales instead"
        )
    sv = torch.linalg.svdvals(weight.float()).clamp(min=0.0)
    if scale_mode == "complete":
        if torch.sqrt(torch.mean(sv.square())) <= eps:
            return torch.ones_like(sv)  # Muon-like for zero-initialized weights
        return sv
    if scale_mode == "rms":
        rms = torch.sqrt(torch.mean(sv.square()))
        return (sv + eps) / (rms + eps)  # Muon-like for zero weights
    return _normalize_log1p(sv, eps)     # log1p


def relmuon_aligned_scales(weight: Tensor, Vh: Tensor, eps: float = 1e-8) -> Tensor:
    """Scales for ``log1p_aligned``: ``s_i = log1p(‖W v_i‖₂)``, RMS-normalized.

    ``Vh`` holds the update's right singular vectors as rows (k, n); ``W v_i``
    measures the weight's action along the i-th update direction, so the
    pairing between weight spectrum and update direction is geometric instead
    of ordinal. Zero weights give all-ones scales (Muon-like).
    """
    action = torch.linalg.norm(weight.float() @ Vh.float().mT, dim=-2)  # (k,)
    return _normalize_log1p(action, eps)


def _shape_lr_scale(fan_out: int, fan_in: int, adjust_lr_fn: str | None) -> float:
    if adjust_lr_fn in (None, "none"):
        return 1.0
    if adjust_lr_fn == "spectral_norm":
        return float(math.sqrt(fan_out / fan_in))
    if adjust_lr_fn == "rms_norm":
        return float(0.2 * math.sqrt(max(fan_out, fan_in)))
    raise ValueError(f"unsupported RelMuon adjust_lr_fn: {adjust_lr_fn}")


@torch.compile
def relmuon_update(
    grad: Tensor,
    weight: Tensor,
    momentum: Tensor,
    mu: float = 0.95,
    nesterov: bool = True,
    eps: float = 1e-8,
    scale_mode: str = "log1p",
    scale_cap: float | None = None,
) -> Tensor:
    """Build a RelMuon matrix update.

    ``scale_cap`` (trust cap) clamps the final scales from above; it bounds the
    update's spectral norm by ``scale_cap`` and is the stability guard for the
    unnormalized ``complete`` mode. The momentum buffer is stored in the same
    EMA-scaled convention as Muon; with ``nesterov=True`` it has the same
    singular vectors as the sum-form Nesterov direction.
    """
    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    update_f = update.float()

    U, _, Vh = torch.linalg.svd(update_f, full_matrices=False)
    if scale_mode == "log1p_aligned":
        scales = relmuon_aligned_scales(weight, Vh, eps=eps)
    else:
        scales = relmuon_weight_scales(weight, scale_mode=scale_mode, eps=eps)
    if scale_cap is not None:
        scales = scales.clamp(max=scale_cap)

    rank = min(U.size(-1), Vh.size(-2), scales.numel())
    shaped = (U[:, :rank] * scales[:rank].to(U.dtype)) @ Vh[:rank, :]
    return shaped.to(dtype=grad.dtype)


relmuon_log1p_update = relmuon_update


class RelMuon(torch.optim.Optimizer):
    """RelMuon for 2D matrix parameters.

    Non-matrix parameters should be optimized by the auxiliary AdamW path.
    Weight decay is decoupled (AdamW-style).
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
        scale_cap: float | None = None,
    ):
        if adjust_lr_fn not in (None, "none", "spectral_norm", "rms_norm"):
            raise ValueError(
                "RelMuon adjust_lr_fn must be None, 'none', 'spectral_norm', "
                f"or 'rms_norm' (got {adjust_lr_fn!r})"
            )
        _validate_scale_mode(scale_mode)
        if scale_cap is not None and scale_cap <= 0:
            raise ValueError(f"scale_cap must be positive, got {scale_cap}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, mu=mu, nesterov=nesterov,
            eps=eps, adjust_lr_fn=adjust_lr_fn, scale_mode=scale_mode,
            scale_cap=scale_cap,
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
                    scale_cap=group["scale_cap"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(p.grad.size(-2), p.grad.size(-1), group["adjust_lr_fn"])
                p.add_(update, alpha=-group["lr"] * lr_scale)
        return loss
