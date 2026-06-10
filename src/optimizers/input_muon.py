"""InputMuon optimizer: Muon restricted to an empirical input subspace.

For a linear layer ``y = W x``, standard Muon applies the polar factor of the
matrix gradient/momentum in the full input space. InputMuon instead accepts an
orthonormal basis ``Q`` for the active input subspace and applies Muon to
``G Q`` before lifting the update back with ``Qᵀ``:

    ``Δ = polar(G Q) Qᵀ``.

If no basis has been registered for a parameter, InputMuon falls back to the
standard Muon update. The activation-hook machinery that estimates ``Q`` can be
kept outside the optimizer; this module owns only the optimizer math and state.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from .muon import _shape_lr_scale, muon_update, zeropower_via_newtonschulz5


def input_basis_from_activations(
    activations: Tensor,
    rank: int,
    *,
    center: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """Return a top-``rank`` orthonormal input basis from layer activations.

    ``activations`` may have any leading dimensions, but its final dimension is
    interpreted as ``d_in``. For transformer blocks this means flattening
    ``(batch, time, d_in)`` to ``(batch*time, d_in)``. The returned basis has
    shape ``(d_in, k)`` with ``k <= rank``.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if activations.ndim < 2:
        raise ValueError(f"activations must have at least 2 dims, got {tuple(activations.shape)}")

    X = activations.detach().float().reshape(-1, activations.size(-1))
    if X.numel() == 0:
        raise ValueError("activations must be non-empty")
    if center:
        X = X - X.mean(dim=0, keepdim=True)

    # SVD on X directly is stable and naturally returns the active right
    # singular directions. It is also cheap for the small-model experiments this
    # optimizer is meant to prototype.
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    active = int((S > eps).sum().item())
    k = min(rank, active, Vh.size(0))
    if k == 0:
        return torch.empty(X.size(-1), 0, dtype=X.dtype, device=X.device)
    return Vh[:k].mT.contiguous()


def _validate_input_basis(input_basis: Tensor, grad: Tensor) -> Tensor:
    if input_basis.ndim != 2:
        raise ValueError(f"input_basis must be 2D, got {tuple(input_basis.shape)}")
    if input_basis.size(0) != grad.size(1):
        raise ValueError(
            "input_basis first dimension must match matrix input dimension: "
            f"got basis {tuple(input_basis.shape)} for grad {tuple(grad.shape)}"
        )
    if input_basis.size(1) == 0:
        raise ValueError("input_basis must have at least one column")
    return input_basis.to(device=grad.device, dtype=torch.float32)


def input_muon_update(
    grad: Tensor,
    momentum: Tensor,
    input_basis: Tensor | None = None,
    *,
    mu: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 12,
) -> Tensor:
    """Build an InputMuon matrix update.

    With an orthonormal basis ``Q`` for active input directions, the update is
    ``polar(M Q) Qᵀ``, where ``M`` is the momentum/Nesterov gradient estimate.
    Without ``Q`` this is exactly the repo's standard Muon update.

    This can be read as a right-side input preconditioner / projector. Standard
    Muon uses a full-space spectral constraint on ``M`` and therefore treats all
    input directions as equally relevant. InputMuon first restricts the right
    side to the empirical input subspace, applies the Muon polar step there,
    then lifts the update back to the original parameter shape. Equivalently,
    it optimizes the matrix only along directions that recent activations
    actually used, while leaving the orthogonal input complement untouched.
    """
    if input_basis is None:
        return muon_update(grad, momentum, mu=mu, nesterov=nesterov, ns_steps=ns_steps)

    momentum.lerp_(grad, 1.0 - mu)
    update = grad.lerp(momentum, mu) if nesterov else momentum
    Q = _validate_input_basis(input_basis, update)
    projected = update.float() @ Q
    subspace_update = zeropower_via_newtonschulz5(projected, ns_steps)
    lifted = subspace_update.float() @ Q.mT
    return lifted.to(dtype=grad.dtype)


class InputMuon(torch.optim.Optimizer):
    """Muon restricted to a registered input subspace for each matrix parameter.

    External code can call ``set_input_basis(param, Q)`` with an orthonormal
    ``Q`` of shape ``(fan_in, rank)``. Missing bases fall back to standard Muon,
    which keeps the optimizer usable before activation-basis hooks are wired in.
    Non-matrix parameters should be optimized by the auxiliary AdamW path.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 12,
        adjust_lr_fn: str | None = "spectral_norm",
    ):
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
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def set_input_basis(self, param: torch.nn.Parameter, input_basis: Tensor | None) -> None:
        """Register or clear the active input basis for a parameter."""
        state = self.state[param]
        if input_basis is None:
            state.pop("input_basis", None)
            return
        _validate_input_basis(input_basis, param)
        state["input_basis"] = input_basis.detach().float().contiguous()

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """Apply one InputMuon step.

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
                    raise ValueError(f"InputMuon expects 2D gradients, got {tuple(p.grad.shape)}")
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                update = input_muon_update(
                    p.grad,
                    state["momentum"],
                    state.get("input_basis"),
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    ns_steps=group["ns_steps"],
                )
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                lr_scale = _shape_lr_scale(p.grad.size(-2), p.grad.size(-1), group["adjust_lr_fn"])
                p.add_(update, alpha=-group["lr"] * lr_scale)
        return loss
