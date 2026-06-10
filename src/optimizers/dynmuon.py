"""DynMuon-Route — dynamic layer-wise spectral-exponent routing for Muon.

Baseline (real DynMuon, github.com/fzwark/DynMuon): shape the momentum-averaged
gradient ``M = U Σ Vᵀ`` with ``D(p) = U Σ^p Vᵀ`` (p=1 → SGD, p=0 → Muon polar
factor, p=-0.25 → late-stage outlier suppression) where a single *global* logistic
time schedule drives ``p_t : 1 → -0.25`` for every layer:

    q_t = step / total_steps
    p_t = p_min + (p_max - p_min) / (1 + exp((q_t - tau_ratio) / width_ratio))

(defaults ``p_max=1.0, p_min=-0.25, tau_ratio=0.04, width_ratio=0.04`` — exactly
the values in the reference repo's ``get_p``).

DynMuon-Route (this work) replaces the global schedule with a *local* per-layer
proxy mapped through a per-layer-type logistic to a parameter-specific exponent
``p_{t,l}``.  Three proxies: gradient stable rank (default), an SNR proxy γ, and a
weight/gradient alignment proxy α (see ``ROUTING_MODES``).

Newton-Schulz follows the reference repo's tuned quintic iteration (5 steps,
``QUINTIC_NS_COEFFS``); a plain cubic iteration (``newton_schulz``) is also kept
for the Part-1.1 math validation.

Compute backends for ``D(p)``:
  * ``compute_mode="svd"`` — exact ``U Σ^p Vᵀ`` via SVD (validation / debugging).
  * ``compute_mode="ns"``  — fast path using the equivalent factorization
    ``U Σ^p Vᵀ = A^{p/2} Y_μ`` where ``Y_μ`` is the Newton-Schulz polar factor and
    ``A = X_n X_nᵀ`` is the small Gram matrix whose symmetric eigendecomposition
    gives ``A^{p/2}`` (and ``λ_max`` for the stable rank, free of charge).

Per-parameter diagnostics (``last_p``, ``last_sr``, ``last_gamma``, ``last_alpha``)
are cached in ``self.state[p]`` so the trainer can log the ``p_{t,l}`` trajectories.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

ROUTING_MODES = (
    "fixed", "global_schedule", "schedule_modulated", "stable_rank", "snr", "alignment",
)
PROXY_METRICS = ("stable_rank", "snr", "alignment")
COMPUTE_MODES = ("svd", "ns")
NS_VARIANTS = ("quintic", "cubic")
SPECTRUM_MODES = ("power", "relmuon", "relmuon_log1p", "random_uniform", "inverted")

# Tuned quintic Newton-Schulz coefficients from the reference DynMuon/Dion repo.
# Each row (a, b, c) applies X <- a X + b (X Xᵀ) X + c (X Xᵀ)² X for one iteration.
QUINTIC_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def _eigh(A: torch.Tensor):
    """Symmetric eigendecomposition with a CPU fallback (MPS lacks eigh)."""
    try:
        return torch.linalg.eigh(A)
    except (NotImplementedError, RuntimeError):
        evals, Q = torch.linalg.eigh(A.cpu())
        return evals.to(A.device), Q.to(A.device)


def _svd(X: torch.Tensor):
    """Thin SVD with a CPU fallback for backends without device SVD."""
    try:
        return torch.linalg.svd(X, full_matrices=False)
    except (NotImplementedError, RuntimeError):
        U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        return U.to(X.device), S.to(X.device), Vh.to(X.device)


def _shape_lr_scale(fan_out: int, fan_in: int, adjust_lr_fn: str | None) -> float:
    """Spectral-norm LR scaling ``sqrt(fan_out / fan_in)`` (reference DynMuon)."""
    if adjust_lr_fn in (None, "none"):
        return 1.0
    return float(math.sqrt(fan_out / fan_in))


def newton_schulz(X: torch.Tensor, steps: int) -> torch.Tensor:
    """Plain cubic polar-factor iteration ``X_{k+1} = 1.5 X - 0.5 X Xᵀ X``.

    Converges to the orthogonal polar factor ``U Vᵀ`` when every singular value
    is in ``(0, √3)``; the caller Frobenius-normalizes so ``‖X‖_2 ≤ 1``. Used by
    the Part-1.1 math validation. The cubic term is grouped as ``(X Xᵀ) X`` so the
    intermediate is the small ``m × m`` Gram (``X`` is ``m × n`` with ``m ≤ n``).
    """
    Y = X
    for _ in range(steps):
        Y = 1.5 * Y - 0.5 * ((Y @ Y.transpose(-2, -1)) @ Y)
    return Y


def quintic_newton_schulz(X: torch.Tensor) -> torch.Tensor:
    """Tuned 5-step quintic polar-factor iteration (reference DynMuon)."""
    Y = X
    for a, b, c in QUINTIC_NS_COEFFS:
        G = Y @ Y.transpose(-2, -1)          # (m, m) Gram
        Y = a * Y + b * (G @ Y) + c * (G @ (G @ Y))
    return Y


def logistic_route(x: float, p_min: float, p_max: float, mu: float, omega: float) -> float:
    """Parameterized logistic map of proxy ``x`` to exponent ``p``:

        p = p_min + (p_max - p_min) / (1 + exp(-(x - mu) / omega))

    Sign of ``omega`` sets orientation: ``omega > 0`` => p increases with x
    (stable-rank, alignment); ``omega < 0`` => p decreases (SNR).
    """
    z = max(-60.0, min(60.0, (x - mu) / omega))
    return p_min + (p_max - p_min) / (1.0 + math.exp(-z))


class DynMuonRoute(torch.optim.Optimizer):
    """Muon with dynamic layer-wise spectral-exponent routing.

    Routing modes (set ``routing_mode``):
      * ``fixed``              — constant ``fixed_p`` (p=0 Muon, p=1 SGD).
      * ``global_schedule``    — the reference DynMuon logistic time schedule
                                 ``p_t : 1 → -0.25``, identical for every layer.
      * ``schedule_modulated`` — **the router.** Follow the same global time
                                 schedule, but nudge each layer by how its
                                 gradient geometry deviates from a typical value:
                                     p_{t,l} = clip( p_t  +  beta·(proxy_l − ref_l) )
                                 With the stable-rank proxy, a layer whose gradient
                                 is *more anisotropic than typical* (proxy < ref) is
                                 pushed toward negative p (suppress outliers); a
                                 *more isotropic* layer is pushed up. ``ref`` is
                                 per-layer-type; ``beta`` is the (shared) gain.
      * ``stable_rank`` / ``snr`` / ``alignment`` — map the proxy straight to p
        through a per-layer-type logistic (``mu``, ``omega``); no time schedule.

    Per-param-group knobs (``mu``, ``omega``, ``ref``) let Attention, MLP and other
    matrices carry distinct routing; the trainer builds one group per layer type.
    Biases, norm gains, and embeddings must NOT be passed here — route them through
    AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        routing_mode: str = "stable_rank",
        spectrum_mode: str = "power",
        compute_mode: str = "ns",
        ns_variant: str = "quintic",
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn: str | None = "spectral_norm",
        p_min: float = -0.25,
        p_max: float = 1.0,
        mu: float = 0.0,
        omega: float = 1.0,
        ref: float = 0.0,
        beta: float = 0.1,
        dynamic_ref: bool = False,
        ref_decay: float = 0.9,
        modulate_metric: str = "stable_rank",
        fixed_p: float = 0.0,
        tau_ratio: float = 0.04,
        width_ratio: float = 0.04,
        total_steps: int | None = None,
    ):
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}, got {routing_mode}")
        if spectrum_mode not in SPECTRUM_MODES:
            raise ValueError(f"spectrum_mode must be one of {SPECTRUM_MODES}, got {spectrum_mode}")
        if compute_mode not in COMPUTE_MODES:
            raise ValueError(f"compute_mode must be one of {COMPUTE_MODES}, got {compute_mode}")
        if ns_variant not in NS_VARIANTS:
            raise ValueError(f"ns_variant must be one of {NS_VARIANTS}, got {ns_variant}")
        if modulate_metric not in PROXY_METRICS:
            raise ValueError(f"modulate_metric must be one of {PROXY_METRICS}, got {modulate_metric}")
        if adjust_lr_fn not in (None, "none", "spectral_norm"):
            raise ValueError(
                "DynMuon adjust_lr_fn must be None, 'none', or 'spectral_norm' "
                f"(got {adjust_lr_fn!r})"
            )
        if routing_mode in ("global_schedule", "schedule_modulated") and not total_steps:
            raise ValueError(f"routing_mode={routing_mode!r} requires total_steps > 0")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, routing_mode=routing_mode,
            spectrum_mode=spectrum_mode, compute_mode=compute_mode,
            ns_variant=ns_variant, ns_steps=ns_steps, eps=eps,
            adjust_lr_fn=adjust_lr_fn, p_min=p_min, p_max=p_max, mu=mu, omega=omega,
            ref=ref, beta=beta, dynamic_ref=dynamic_ref, ref_decay=ref_decay,
            modulate_metric=modulate_metric,
            fixed_p=fixed_p, tau_ratio=tau_ratio, width_ratio=width_ratio,
            total_steps=total_steps,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        # Running cross-layer mean of the routing proxy (for dynamic_ref): lets
        # the schedule own the global/temporal trend while the router responds
        # only to each layer's deviation from the network average.
        self._proxy_ema: float | None = None

    def _p_schedule(self, group: dict) -> float:
        """Reference DynMuon logistic time schedule p_t : p_max → p_min."""
        q_t = self._step_count / max(1, group["total_steps"])
        u = (q_t - group["tau_ratio"]) / max(group["width_ratio"], 1e-8)
        anneal = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, u))))
        return group["p_min"] + (group["p_max"] - group["p_min"]) * anneal

    def _select_p(self, group: dict, x: float | None) -> float:
        """Map the active routing mode (and proxy ``x``) to a spectral exponent."""
        mode = group["routing_mode"]
        if mode == "fixed":
            return group["fixed_p"]              # 0.0 = Muon, 1.0 = SGD
        if mode == "global_schedule":
            return self._p_schedule(group)
        if mode == "schedule_modulated":
            # Global time arc + per-layer geometry nudge, clipped to [p_min, p_max].
            # dynamic_ref: nudge relative to the running cross-layer mean (removes
            # the global temporal trend, leaving the per-layer deviation).
            ref = (self._proxy_ema if (group["dynamic_ref"] and self._proxy_ema is not None)
                   else group["ref"])
            p = self._p_schedule(group) + group["beta"] * (x - ref)
            return max(group["p_min"], min(group["p_max"], p))
        return logistic_route(x, group["p_min"], group["p_max"], group["mu"], group["omega"])

    @torch.no_grad()
    def step(self, closure=None, noise_hook=None):  # type: ignore[override]
        """One optimization step.

        ``noise_hook`` (Experiment 2) is an optional callable
        ``hook(M2, state) -> M2`` applied to the oriented momentum matrix before
        shaping, so the router and baseline see identical injected noise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        step_proxies = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    x = self._update_param(p, group, noise_hook)
                    if x is not None:
                        step_proxies.append(x)
        # Update the running cross-layer proxy mean (used by dynamic_ref next step).
        if step_proxies:
            m = sum(step_proxies) / len(step_proxies)
            decay = self.param_groups[0]["ref_decay"]
            self._proxy_ema = m if self._proxy_ema is None else decay * self._proxy_ema + (1 - decay) * m
        self._step_count += 1
        return loss

    def _update_param(self, p, group, noise_hook) -> float | None:
        """Apply the shaped update to ``p``; return the routing proxy value (for
        the dynamic cross-layer mean), or None when no proxy is used."""
        eps = group["eps"]
        G = p.grad
        orig_shape = G.shape
        G2 = G.reshape(G.shape[0], -1) if G.dim() != 2 else G
        fan_out, fan_in = G2.shape          # original orientation (for LR scaling)
        transposed = G2.shape[0] > G2.shape[1]
        if transposed:
            G2 = G2.transpose(0, 1)

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G2)
        B = state["momentum_buffer"]
        B.mul_(group["momentum"]).add_(G2)
        M2 = G2.add(B, alpha=group["momentum"]) if group["nesterov"] else B

        W2 = p.reshape(p.shape[0], -1) if p.dim() != 2 else p
        if transposed:
            W2 = W2.transpose(0, 1)

        if noise_hook is not None:
            M2 = noise_hook(M2, state)

        fro_M = torch.linalg.norm(M2)
        if fro_M <= eps:
            state.update(last_p=float("nan"), last_sr=float("nan"),
                         last_gamma=float("nan"), last_alpha=float("nan"))
            return None
        X_n = M2 / fro_M

        spectrum_mode = group["spectrum_mode"]

        # Spectral decomposition (needed for both the proxy and the shaping).
        # Non-power spectra need explicit singular vectors/values, so they force
        # the exact SVD path even when compute_mode is configured as "ns".
        if spectrum_mode != "power":
            U, S, Vh = _svd(M2)
            lam_max = float(((S[0] / fro_M) ** 2).item())
        elif group["compute_mode"] == "svd":
            U, S, Vh = _svd(X_n)
            lam_max = float((S[0] ** 2).item())
        else:
            A = X_n @ X_n.transpose(-2, -1)               # (rows, rows) PSD Gram
            evals, Q = _eigh(A)
            evals = evals.clamp(min=0.0)
            lam_max = float(evals[-1].item())
            Y_mu = (quintic_newton_schulz(X_n) if group["ns_variant"] == "quintic"
                    else newton_schulz(X_n, group["ns_steps"]))

        # -- routing proxies (raw, layer-local) ----------------------------
        sr = 1.0 / (lam_max + eps)                        # ‖M‖_F²/σ_max² = 1/λ_max(A) ∈ [1, d]
        gamma = float((fro_M / (torch.linalg.norm(G2 - M2) + eps)).item())
        alpha = float((torch.sum(W2 * M2).abs() / (torch.linalg.norm(W2) * fro_M + eps)).item())
        proxies = {"stable_rank": sr, "snr": gamma, "alignment": alpha}

        # The proxy that feeds routing: the chosen metric for schedule_modulated,
        # else the metric named by the routing mode (fixed/global ignore it).
        mode = group["routing_mode"]
        metric = group["modulate_metric"] if mode == "schedule_modulated" else mode
        x = proxies.get(metric)
        p_exp = self._select_p(group, x)

        # -- shape D(p) = U Σ^p Vᵀ -----------------------------------------
        if spectrum_mode == "power" and group["compute_mode"] == "svd":
            D = (U * S.clamp(min=eps).pow(p_exp)) @ Vh
        elif spectrum_mode == "power":
            A_p2 = (Q * evals.clamp(min=eps).pow(p_exp / 2.0)) @ Q.transpose(-2, -1)
            D = A_p2 @ Y_mu
        elif spectrum_mode == "relmuon":
            _, S_w, _ = _svd(W2)
            r = min(S.numel(), S_w.numel())
            S_hat = S_w[:r]
            S_hat = S_hat / (torch.sqrt(torch.mean(S_hat.square())) + 1e-8)
            D = (U[:, :r] * S_hat.to(dtype=U.dtype, device=U.device)) @ Vh[:r, :]
        elif spectrum_mode == "random_uniform":
            S_hat = torch.rand_like(S)
            S_hat = S_hat / (torch.sqrt(torch.mean(S_hat.square())) + 1e-8)
            D = (U * S_hat) @ Vh
        elif spectrum_mode == "inverted":
            S_hat = torch.flip(S, dims=[0])
            S_hat = S_hat / (torch.sqrt(torch.mean(S_hat.square())) + 1e-8)
            D = (U * S_hat) @ Vh
        else:
            raise RuntimeError(f"unsupported spectrum_mode: {spectrum_mode}")

        if transposed:
            D = D.transpose(0, 1)
        lr_scale = _shape_lr_scale(fan_out, fan_in, group["adjust_lr_fn"])
        p.add_(D.reshape(orig_shape), alpha=-group["lr"] * lr_scale)

        state.update(last_p=float(p_exp), last_sr=float(sr),
                     last_gamma=float(gamma), last_alpha=float(alpha))
        return x
          D = D.transpose(0, 1)
        lr_scale = _shape_lr_scale(fan_out, fan_in, group["adjust_lr_fn"])
        p.add_(D.reshape(orig_shape), alpha=-group["lr"] * lr_scale)

        state.update(last_p=float(p_exp), last_sr=float(sr),
                     last_gamma=float(gamma), last_alpha=float(alpha))
        return x
