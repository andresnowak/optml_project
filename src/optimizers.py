from __future__ import annotations

from torch.optim import Adam, AdamW, SGD
from torch.optim import Muon as _TorchMuon
from torch.optim._muon import muon as _torch_muon_functional
import torch
import metalcore

metalcore.enable_pytorch_overrides(activations=False, embedding_bag=False, normalization=False, softmax=False, optimizers=False, linalg=True)


def _keller_jordan_shape_scale(shape: torch.Size) -> float:
    """Keller/Jordan Muon shape scaling: sqrt(max(1, rows / cols))."""
    rows, cols = shape[:2]
    return float(max(1.0, rows / cols) ** 0.5)


class Muon(_TorchMuon):
    """Muon with repo-controlled learning-rate shape scaling.

    PyTorch's Muon applies its own shape-dependent LR adjustment even when
    ``adjust_lr_fn`` is None. This subclass deliberately disables that internal
    adjustment in the functional kernel and applies only the scaling selected
    here, so experiments know exactly which scaling rule is active.
    """

    _NO_TORCH_LR_ADJUST = "_optml_no_torch_lr_adjust"

    @staticmethod
    def _lr_scale(shape: torch.Size, adjust_lr_fn: str | None) -> float:
        if adjust_lr_fn is None:
            return 1.0
        if adjust_lr_fn == "shape_scaling":
            return _keller_jordan_shape_scale(shape)
        raise ValueError(f"Unsupported Muon adjust_lr_fn: {adjust_lr_fn}")

    def __init__(self, params, *args, adjust_lr_fn: str | None = None, **kwargs):
        if adjust_lr_fn not in (None, "shape_scaling"):
            raise ValueError(f"Unsupported Muon adjust_lr_fn: {adjust_lr_fn}")
        # Keep torch's own adjust_lr_fn disabled. We store our selected mode
        # separately and apply it in step().
        super().__init__(params, *args, adjust_lr_fn=None, **kwargs)
        for group in self.param_groups:
            group["optml_adjust_lr_fn"] = adjust_lr_fn

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            base_lr = group["lr"]
            adjust_lr_fn: str | None = group["optml_adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                lr = base_lr * self._lr_scale(p.shape, adjust_lr_fn)
                _torch_muon_functional(
                    [p],
                    [p.grad],
                    [state["momentum_buffer"]],
                    lr=lr,
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    nesterov=group["nesterov"],
                    ns_coefficients=group["ns_coefficients"],
                    ns_steps=group["ns_steps"],
                    eps=group["eps"],
                    adjust_lr_fn=self._NO_TORCH_LR_ADJUST,
                    has_complex=torch.is_complex(p),
                )

        return loss


class SpecMuon(torch.optim.Optimizer):
    """Muon with SAV (Scalar Auxiliary Variable) adaptive scaling for top-k singular directions.

    For the top-k singular components the step size is modulated by a scalar
    auxiliary variable r_j that tracks a smoothed energy proxy; the remaining
    components receive the standard Muon (orthogonalized gradient) update.
    Momentum is then applied on top of the combined update direction.

    Args:
        params:         parameters to optimize
        lr:             base learning rate η
        momentum:       Nesterov-style momentum coefficient μ  (default 0.95)
        top_k:          number of singular directions to treat with SAV (default 5)
        sav_smooth:     SAV smoothing factor ξ ∈ [0, 1]  (default 0.1)
        eps:            numerical stability ε  (default 1e-8)
        adjust_lr_fn:      lr scaling mode. None = no scaling, "shape_scaling" = sqrt(max(1, rows / cols)).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.95,
        top_k: int = 5,
        sav_smooth: float = 0.1,
        eps: float = 1e-8,
        adjust_lr_fn: str | None = None,
    ):
        defaults = dict(lr=lr, momentum=momentum, top_k=top_k, sav_smooth=sav_smooth, eps=eps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, loss: torch.Tensor | None = None):  # type: ignore[override]
        """Perform one optimisation step.

        Either pass a ``closure`` (called here with grad enabled) or supply
        the already-computed ``loss`` tensor directly.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None:
            raise ValueError(
                "SpecMuon.step() requires the current loss. "
                "Pass a closure or use step(loss=<tensor>)."
            )

        sqrt_loss = float(loss.item()) ** 0.5

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            k: int = group["top_k"]
            xi: float = group["sav_smooth"]
            eps: float = group["eps"]
            adjust_lr_fn: str | None = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                G = p.grad
                orig_shape = G.shape

                # Reshape to 2-D for SVD: (rows, cols)'
                # Things that aren't matrices should be treated by Adam, only matrices get SpecMuon update.
                if G.dim() != 2:
                    raise ValueError(f"SpecMuon only supports 2-D parameter gradients for SVD-based updates, but got shape {G.shape} for parameter with shape {p.shape}. Consider using a different optimizer for this parameter.")

                # ── Step 4-5: normalise gradient ──────────────────────────────
                G_hat = G / (torch.linalg.norm(G) + eps) # Frobenius norm normalization with stability eps (to have singular values [0, 1])

                # ── Step 6: full SVD of normalised gradient ───────────────────
                U, S, Vh = torch.linalg.svd(G_hat, full_matrices=False) # S \in [0, 1]
                # U: (m, r), S: (r,), Vh: (r, n)  where r = min(m, n)

                # ── Initialise per-parameter state ────────────────────────────
                state = self.state[p]
                if not state:
                    k_act = min(k, S.shape[0]) # actual number of top singular directions to treat with SAV (can't be more than rank)
                    state["momentum_buffer"] = torch.zeros_like(G)
                    # r initialised to √L₀ for each of the k directions
                    state["r"] = torch.full(
                        (k_act,), sqrt_loss, dtype=G.dtype, device=G.device
                    )
                    state["k_act"] = k_act

                B: torch.Tensor = state["momentum_buffer"]
                r: torch.Tensor = state["r"]
                k_act: int = state["k_act"]

                O = torch.zeros_like(G) # (rows, cols)

                # ── Steps 10-21: SAV (Scalar Auxiliary Variable) update for top-k directions ──────────────
                s_k = S[:k_act]                                           # (k_act,)
                eta_prime = lr / (s_k + eps)                              # inverse scaling by singular value
                d_g_norm = s_k / (sqrt_loss + eps)                        # ‖d_g‖_F per direction
                r_new = r / (1.0 + 0.5 * eta_prime * d_g_norm)           # SAV variable update

                scale = r_new / (sqrt_loss + eps)
                O.addmm_(U[:, :k_act] * scale.unsqueeze(0), Vh[:k_act, :])  # Σ_j scale_j * u_j ⊗ v_j

                # SAV state update
                T = ((1.0 - xi) * r_new ** 2
                     + xi * r ** 2
                     + (1.0 - xi) * (r_new - r) ** 2).clamp(min=0.0)    # smoothed energy proxy
                sqrt_T = T.sqrt()
                denom = sqrt_loss - r_new + eps
                chi = ((sqrt_loss - sqrt_T) / denom).clamp(0.0, 1.0)     # blending factor χ_j
                state["r"] = chi * r_new + (1.0 - chi) * sqrt_loss

                # ── Steps 24-25: standard Muon update for remaining directions ─
                if k_act < S.shape[0]:
                    U_rest = U[:, k_act:]    # (m, rest)
                    S_rest = S[k_act:]       # (rest,)
                    Vh_rest = Vh[k_act:, :]  # (rest, n)
                    # O += U_rest @ diag(S_rest) @ Vh_rest, ones in the diagonal as in Muon
                    # O.addmm_(U_rest * S_rest.unsqueeze(0), Vh_rest)
                    O.addmm_(U_rest, Vh_rest)

                # ── Steps 28-29: momentum + parameter update ──────────────────
                lr_scale = _keller_jordan_shape_scale(p.shape) if adjust_lr_fn == "shape_scaling" else 1.0
                B_new = mu * B + O
                state["momentum_buffer"] = B_new
                p.add_(B_new.reshape(orig_shape), alpha=-lr * lr_scale)

        return loss


OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "muon": Muon,
    "sgd": SGD,
    "specmuon": SpecMuon,
}


def build_optimizer(name, params, lr, weight_decay, **kwargs):
    """Build an optimizer, forwarding only the kwargs each one understands."""
    if name == "muon":
        kw = {k: kwargs[k] for k in ("momentum", "ns_steps", "adjust_lr_fn") if kwargs.get(k) is not None}
        return Muon(params, lr=lr, **kw)
    if name == "sgd":
        kw = {k: kwargs[k] for k in ("momentum",) if kwargs.get(k) is not None}
        return SGD(params, lr=lr, weight_decay=weight_decay, **kw)
    if name in ("adam", "adamw"):
        kw = {}
        if kwargs.get("betas") is not None:
            kw["betas"] = kwargs["betas"]
        if kwargs.get("eps") is not None:
            kw["eps"] = kwargs["eps"]
        return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay, **kw)
    if name == "specmuon":
        kw = {}
        for key in ("momentum", "top_k", "sav_smooth", "eps", "adjust_lr_fn"):
            if kwargs.get(key) is not None:
                kw[key] = kwargs[key]
        return SpecMuon(params, lr=lr, **kw)
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)
