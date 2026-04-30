from __future__ import annotations

from torch.optim import Adam, AdamW, SGD
from torch.optim import Muon as _TorchMuon
import torch
import metalcore

metalcore.enable_pytorch_overrides(activations=False, embedding_bag=False, normalization=False, softmax=False, optimizers=False, linalg=True)


class Muon(_TorchMuon):
    """Our Muon subclass — adjust_lr_fn shape-scaling functions will be added here."""
    pass


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
        adjust_lr_fn:      lr scaling mode. None = no scaling, "shape_scaling" = sqrt(max(m, n)).
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
                if G.dim() == 0:
                    continue  # scalar param — skip
                elif G.dim() == 1:
                    G2 = G.unsqueeze(1)           # (n, 1)
                else:
                    G2 = G.reshape(G.shape[0], -1)  # (m, n*…)

                # ── Step 4-5: normalise gradient ──────────────────────────────
                G_hat = G2 / (torch.linalg.norm(G2) + eps) # Frobenius norm normalization with stability eps (to have singular values [0, 1])

                # ── Step 6: full SVD of normalised gradient ───────────────────
                U, S, Vh = torch.linalg.svd(G_hat, full_matrices=False)
                # U: (m, r), S: (r,), Vh: (r, n)  where r = min(m, n)

                # ── Initialise per-parameter state ────────────────────────────
                state = self.state[p]
                if not state:
                    k_act = min(k, S.shape[0]) # actual number of top singular directions to treat with SAV (can't be more than rank)
                    state["momentum_buffer"] = torch.zeros_like(G2)
                    # r initialised to √L₀ for each of the k directions
                    state["r"] = torch.full(
                        (k_act,), sqrt_loss, dtype=G2.dtype, device=G2.device
                    )
                    state["k_act"] = k_act

                B: torch.Tensor = state["momentum_buffer"]
                r: torch.Tensor = state["r"]
                k_act: int = state["k_act"]

                O = torch.zeros_like(G2) # (rows, cols)

                # ── Steps 10-21: SAV (Scalar Auxiliary Variable) update for top-k directions ──────────────
                # for j in range(k_act):
                #     u_j = U[:, j]        # (m,)
                #     s_j = S[j].item()
                #     v_j = Vh[j, :]       # (n,)
                #     r_prev_j = r[j].item()

                #     eta_prime_j = lr / (s_j + eps) # inverse scaling by singular value (with stability eps)
                #     # ‖d_g‖_F = s_j / (√L + ε)  because ‖u v^T‖_F = 1
                #     d_g_norm = s_j / (sqrt_loss + eps)

                #     r_new_j = r_prev_j / (1.0 + 0.5 * eta_prime_j * d_g_norm) # update rule for SAV variable r_j

                #     # O += (r_new_j / (√L + ε)) · u_j v_j^T
                #     scale = r_new_j / (sqrt_loss + eps)
                #     O.addmm_(u_j.unsqueeze(1), v_j.unsqueeze(0), alpha=scale)

                #     # SAV state update
                #     T = (
                #         (1.0 - xi) * r_new_j ** 2
                #         + xi * r_prev_j ** 2
                #         + (1.0 - xi) * (r_new_j - r_prev_j) ** 2
                #     ) # smoothed energy proxy T_j for the j-th direction, combining current and previous r_j values with smoothing factor xi
                #     sqrt_T = max(T, 0.0) ** 0.5
                #     denom = sqrt_loss - r_new_j + eps
                #     chi = float(torch.clamp(
                #         torch.tensor((sqrt_loss - sqrt_T) / denom), 0.0, 1.0
                #     )) # blending factor χ_j (chi distribution) for smoothing the update of r_j, based on how close the energy proxy T_j is to the current loss sqrt_loss
                #     r_next[j] = chi * r_new_j + (1.0 - chi) * sqrt_loss

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
                lr_scale = max(G2.shape) ** 0.5 if adjust_lr_fn == "shape_scaling" else 1.0
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
        kw = {k: kwargs[k] for k in ("momentum", "ns_steps") if kwargs.get(k) is not None}
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
        for key in ("momentum", "top_k", "sav_smooth", "eps", "adjust_lr_fn_fn"):
            if kwargs.get(key) is not None:
                kw[key] = kwargs[key]
        return SpecMuon(params, lr=lr, **kw)
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)
