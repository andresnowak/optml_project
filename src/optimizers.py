from __future__ import annotations

import collections
from collections.abc import Iterable
from typing import Any

from torch.optim import Adam, AdamW, SGD
from torch.optim import Muon as _TorchMuon
from torch.optim._muon import muon as _torch_muon_functional
import torch

# Optional metalcore import: enables a CPU fallback for `torch.linalg.svd` on
# MPS (Mac) backends. Not needed on cuda; absent in slim cluster images.
try:
    import metalcore
    metalcore.enable_pytorch_overrides(activations=False, embedding_bag=False, normalization=False, softmax=False, optimizers=False, linalg=True)
except ImportError:
    pass


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
    """SpecMuon — Algorithm 1 of Lu, Zhang & Lin, *Muon with Spectral Guidance*
    (arXiv:2602.16167, Feb 2026). Defaults match the paper's PINN/Burgers
    grid-search winners (μ=0.9, k=6, ξ=0.2).

    Paper-faithful args: ``lr``, ``momentum``, ``top_k``, ``sav_smooth``, ``eps``,
    ``kappa`` (loss shift; non-negative losses can leave it 0).

    Extensions (opt-in, not in Algorithm 1):
      - ``adjust_lr_fn="shape_scaling"`` — Keller/Jordan sqrt(max(1, rows/cols))
        scaling (matches the Muon subclass above).
      - ``sigma_mode``:
          ``baseline`` = paper's η/(σ+ε);
          ``power`` = η/(σ^β+ε), β=``power_beta`` (β=1 is baseline);
          ``sqrt`` = β=0.5; ``clip`` = η/(σ+sigma_clip);
          ``truncate`` = drop σ < sigma_truncate·σ_max;
          ``energy`` = dynamic k by cumulative spectral energy ≥ ``energy_threshold``.
      - ``tail_mode``:
          ``gradient`` = paper Algorithm 1 tail, U diag(S) V^T;
          ``muon`` = true Muon tail, U V^T.
      - SAV-gating (``gate_window``/``gate_threshold``) bypasses SAV when the
        loss is dropping fast over the rolling window — collapses to the
        tail-only (paper-Muon) update; default ``gate_threshold=0`` disables.

    Higher-rank gradients (e.g. Conv2d) flatten to ``(out, in·kH·kW)`` for the
    SVD and reshape back. The paper's algorithm assumes matrix gradients.
    """

    _VALID_SIGMA_MODES = ("power", "clip", "truncate", "energy")
    _VALID_TAIL_MODES = ("gradient", "muon")
    # Legacy aliases collapsed into ``power``: η/(σ+ε) ≡ power(β=1); η/(√σ+ε) ≡ power(β=0.5).
    _SIGMA_ALIASES = {"baseline": ("power", 1.0), "sqrt": ("power", 0.5)}

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        top_k: int = 6,
        sav_smooth: float = 0.2,
        eps: float = 1e-8,
        kappa: float = 0.0,
        adjust_lr_fn: str | None = None,
        sigma_mode: str = "baseline",
        sigma_clip: float = 0.05,
        sigma_truncate: float = 0.05,
        power_beta: float = 1.0,
        energy_threshold: float = 0.9,
        gate_window: int = 10,
        gate_threshold: float = 0.0,
        tail_mode: str = "gradient",
    ):
        if sigma_mode in self._SIGMA_ALIASES:
            sigma_mode, power_beta = self._SIGMA_ALIASES[sigma_mode]
        if sigma_mode not in self._VALID_SIGMA_MODES:
            valid = self._VALID_SIGMA_MODES + tuple(self._SIGMA_ALIASES)
            raise ValueError(f"sigma_mode must be one of {valid}, got {sigma_mode}")
        if tail_mode not in self._VALID_TAIL_MODES:
            raise ValueError(f"tail_mode must be one of {self._VALID_TAIL_MODES}, got {tail_mode}")
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0 (paper §2.1), got {kappa}")
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError(f"energy_threshold must be in (0, 1], got {energy_threshold}")
        if gate_window < 2:
            raise ValueError(f"gate_window must be >= 2, got {gate_window}")
        if gate_threshold < 0:
            raise ValueError(f"gate_threshold must be >= 0 (0 disables the gate), got {gate_threshold}")
        defaults = dict(lr=lr, momentum=momentum, top_k=top_k, sav_smooth=sav_smooth,
                        eps=eps, kappa=kappa,
                        adjust_lr_fn=adjust_lr_fn, sigma_mode=sigma_mode,
                        sigma_clip=sigma_clip, sigma_truncate=sigma_truncate,
                        power_beta=power_beta, energy_threshold=energy_threshold,
                        gate_window=gate_window, gate_threshold=gate_threshold,
                        tail_mode=tail_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, loss: torch.Tensor | None = None):  # type: ignore[override]
        """Perform one optimisation step (Algorithm 1 of arXiv:2602.16167).

        Pass ``closure`` (re-executed under enable_grad) or ``loss`` directly.
        SpecMuon needs L_t to update the SAV state, hence the loss argument
        — unlike standard PyTorch optimizers.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if loss is None:
            raise ValueError(
                "SpecMuon.step() requires the current loss. "
                "Pass a closure or use step(loss=<tensor>)."
            )

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            k: int = group["top_k"]
            xi: float = group["sav_smooth"]
            eps: float = group["eps"]
            kappa: float = group["kappa"]
            adjust_lr_fn: str | None = group["adjust_lr_fn"]
            sigma_mode: str = group["sigma_mode"]
            sigma_clip: float = group["sigma_clip"]
            sigma_truncate: float = group["sigma_truncate"]
            power_beta: float = group["power_beta"]
            energy_threshold: float = group["energy_threshold"]
            gate_window: int = group["gate_window"]
            gate_threshold: float = group["gate_threshold"]
            tail_mode: str = group["tail_mode"]

            # Paper §2.1: E(Θ) := f(Θ) + κ ; Algorithm 1 writes √L_t for √E_t.
            energy = float(loss.item()) + kappa
            if energy <= 0:
                raise ValueError(
                    f"loss + kappa must be positive (paper §2.1), got {energy}. "
                    f"Pass a kappa large enough that f(Θ) + κ > 0."
                )
            sqrt_loss = energy ** 0.5

            for p in group["params"]:
                if p.grad is None:
                    continue

                G = p.grad                                          # paper line 4
                orig_shape = G.shape
                if G.dim() < 2:
                    raise ValueError(
                        f"SpecMuon requires gradients with at least 2 dimensions; "
                        f"got shape {tuple(G.shape)}. Route this parameter through AdamW instead."
                    )
                # Higher-rank tensors (e.g. Conv2d (out, in, kH, kW)) flatten to
                # 2-D for the SVD, then reshape back. This is an extension —
                # the paper's Algorithm 1 is defined for matrix gradients only.
                G2 = G.reshape(G.shape[0], -1) if G.dim() > 2 else G

                G_hat = G2 / (torch.linalg.norm(G2) + eps)          # paper line 5

                U, S, Vh = torch.linalg.svd(G_hat, full_matrices=False)   # paper line 6
                # U: (m, r), S: (r,), Vh: (r, n)  where r = min(m, n).

                state = self.state[p]
                if not state:                                        # paper lines 1-2
                    k_act = min(k, S.shape[0])
                    state["momentum_buffer"] = torch.zeros_like(G2)  # B_0 ← 0
                    # r_0 ← √L_0 · 1  — the first call to step() lands here.
                    state["r"] = torch.full(
                        (k_act,), sqrt_loss, dtype=G2.dtype, device=G2.device
                    )
                    state["k_act"] = k_act
                    # SAV-gating state: rolling loss window + last gate state.
                    # Default gate_threshold=0 ⇒ gate disabled ⇒ always active (paper default).
                    state["loss_window"] = collections.deque(maxlen=gate_window)
                    state["sav_active_prev"] = True

                B: torch.Tensor = state["momentum_buffer"]
                r: torch.Tensor = state["r"]
                k_act: int = state["k_act"]

                # SAV gating: bypass SAV when loss is dropping fast over the
                # rolling window (§4.2 regime where Muon-tail wins); engage
                # SAV on plateaus. gate_threshold=0 ⇒ always-on (paper default).
                state["loss_window"].append(energy)
                if gate_threshold > 0.0 and len(state["loss_window"]) == gate_window:
                    L0 = state["loss_window"][0]
                    L1 = state["loss_window"][-1]
                    rel_drop = max(0.0, (L0 - L1)) / max(L0, eps)
                    sav_active = rel_drop < gate_threshold
                else:
                    sav_active = (gate_threshold == 0.0)
                # Re-seed r_j on inactive→active transition: the cached r is
                # stale (loss has moved while it was frozen), and using it
                # directly would inject a spurious large deviation on the
                # gate-on step.
                if sav_active and not state["sav_active_prev"]:
                    state["r"].fill_(sqrt_loss)
                    r = state["r"]
                state["sav_active_prev"] = sav_active
                state["gate_active"] = sav_active

                O = torch.zeros_like(G2)                             # paper line 7

                # Decide how many directions enter the SAV branch. Default
                # (paper-faithful) is k_act; "truncate" / "energy" modes shrink it.
                if sigma_mode == "truncate":
                    k_step = int((S[:k_act] > sigma_truncate * S[0]).sum().item())
                elif sigma_mode == "energy":
                    # Smallest k such that Σ_{j<k} σ_j² / Σ σ_j² ≥ τ.
                    s_top = S[:k_act]
                    total = (s_top * s_top).sum().clamp(min=eps)
                    cum = (s_top * s_top).cumsum(0) / total
                    mask = cum >= energy_threshold
                    if mask.any():
                        k_step = int(mask.nonzero(as_tuple=False)[0].item()) + 1
                    else:
                        k_step = k_act
                else:
                    k_step = k_act

                # Gate override: when SAV is inactive, collapse to tail-only
                # (k_step=0 ⇒ SAV branch skipped, tail covers all directions
                # using the selected tail mode).
                if not sav_active:
                    k_step = 0

                state["last_sigma"] = S[:k_step].detach().clone()
                state["last_sav_scale"] = torch.empty_like(state["last_sigma"])
                state["last_sigma_min"] = float(S.min().item()) if S.numel() else 0.0
                state["last_sigma_max"] = float(S.max().item()) if S.numel() else 0.0

                # paper line 11: η'_j ← η/(σ_j^β + ϵ).  β=1 (default) is the
                # paper's baseline; β=0.5 ≡ legacy "sqrt"; β=0 decouples the
                # step from σ. ``clip`` is the only mode that does NOT use σ^β.
                s_k = S[:k_step]
                if sigma_mode == "clip":
                    eta_prime = lr / (s_k + sigma_clip)
                else:
                    eta_prime = lr / (s_k.pow(power_beta) + eps)

                # paper line 12-13: ‖d_g‖_F = σ_j / (√L + ϵ)  (since ‖u v^T‖_F = 1).
                d_g_norm = s_k / (sqrt_loss + eps)
                r_slice = r[:k_step]
                r_new = r_slice / (1.0 + 0.5 * eta_prime * d_g_norm)        # paper line 13

                # ι_t = mean_j |r_j^new/√L_t − σ_j| — per-step magnitude of
                # the SAV intervention. The no-SAV (paper-tail) contribution
                # for direction j is σ_j·u_j v_jᵀ; the SAV contribution is
                # (r_j^new/√L)·u_j v_jᵀ. Their Frobenius difference (since
                # u_j v_jᵀ is unit-norm) is |r_j^new/√L − σ_j|. ≈0 ⇒ SAV
                # mechanically inactive; ≫0 ⇒ SAV is firing.
                state["last_iota"] = 0.0
                state["last_iota_w"] = 0.0
                if k_step > 0:
                    sav_scale = r_new / (sqrt_loss + eps)
                    state["last_sav_scale"] = sav_scale.detach().clone()
                    dev = (sav_scale - s_k).abs()
                    state["last_iota"] = float(dev.mean().item())
                    w_denom = s_k.sum() + eps
                    state["last_iota_w"] = float(((s_k * dev).sum() / w_denom).item())
                    # paper line 14: O += (r_j^new / (√L + ϵ)) · u_j v_j^T
                    scale = r_new / (sqrt_loss + eps)
                    O.addmm_(U[:, :k_step] * scale.unsqueeze(0), Vh[:k_step, :])

                    # paper line 16: T ← (1-ξ)(r^new)² + ξ(r_prev)² + (1-ξ)(r^new-r_prev)²
                    T = ((1.0 - xi) * r_new ** 2
                         + xi * r_slice ** 2
                         + (1.0 - xi) * (r_new - r_slice) ** 2).clamp(min=0.0)
                    sqrt_T = T.sqrt()
                    # paper line 17: χ ← (√L - √T) / (√L - r^new + ϵ)
                    denom = sqrt_loss - r_new + eps
                    chi = ((sqrt_loss - sqrt_T) / denom).clamp(0.0, 1.0)
                    # paper line 18: r_{t,j} ← clamp(χ,0,1)·r^new + (1-clamp(χ,0,1))·√L
                    r[:k_step] = chi * r_new + (1.0 - chi) * sqrt_loss

                # paper line 22: O_t ← O_t + U_{k:} diag(S_{k:}) V^T_{k:}.
                # ``tail_mode="muon"`` tests the true orthogonalized Muon tail
                # U_{k:} V^T_{k:}, which the paper labels as Muon but does not
                # write in Algorithm 1.
                if k_step < S.shape[0]:
                    if tail_mode == "gradient":
                        O.addmm_(U[:, k_step:] * S[k_step:].unsqueeze(0), Vh[k_step:, :])
                    elif tail_mode == "muon":
                        O.addmm_(U[:, k_step:], Vh[k_step:, :])
                    else:
                        raise RuntimeError(f"unvalidated tail_mode: {tail_mode}")

                # paper lines 24-25 (with optional shape_scaling extension —
                # uses the same Keller/Jordan rule as the Muon subclass above).
                lr_scale = (
                    _keller_jordan_shape_scale(G2.shape) if adjust_lr_fn == "shape_scaling" else 1.0
                )
                B_new = mu * B + O                                   # paper line 24
                state["momentum_buffer"] = B_new
                p.add_(B_new.reshape(orig_shape), alpha=-lr * lr_scale)

        return loss


OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": Adam,
    "adamw": AdamW,
    "muon": Muon,
    "sgd": SGD,
    "specmuon": SpecMuon,
}


# Single source of truth for which extra kwargs each optimizer accepts.
# Used both by build_optimizer (to filter) and by cli.py (to validate flags).
OPTIMIZER_KWARGS: dict[str, frozenset[str]] = {
    "adam":     frozenset({"betas", "eps"}),
    "adamw":    frozenset({"betas", "eps"}),
    "sgd":      frozenset({"momentum"}),
    "muon":     frozenset({"momentum", "ns_steps", "adjust_lr_fn"}),
    "specmuon": frozenset({"momentum", "top_k", "sav_smooth", "eps", "kappa",
                           "adjust_lr_fn", "sigma_mode", "sigma_clip", "sigma_truncate",
                           "power_beta", "energy_threshold",
                           "gate_window", "gate_threshold", "tail_mode"}),
}

# Optimizers that don't accept the standard ``weight_decay`` argument.
# Torch Muon supports weight decay, but these comparisons keep Muon-family
# matrix updates at zero weight decay; AdamW fallback params still use the
# configured value.
_NO_WEIGHT_DECAY = frozenset({"muon", "specmuon"})


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Construct an optimizer, forwarding only the kwargs it understands."""
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}'; expected one of {sorted(OPTIMIZERS)}")
    accepted = OPTIMIZER_KWARGS[name]
    extra = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
    cls = OPTIMIZERS[name]
    if name in _NO_WEIGHT_DECAY:
        return cls(params, lr=lr, **extra)
    return cls(params, lr=lr, weight_decay=weight_decay, **extra)
