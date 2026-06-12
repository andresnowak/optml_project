"""DynMuon-Route — dynamic layer-wise spectral-exponent routing for Muon.

Baseline (DynMuon, github.com/fzwark/DynMuon, arXiv:2605.17109): replace the
momentum-averaged gradient ``M = U Σ Vᵀ`` with the spectrally shaped update
``D(p) = U Σ^p Vᵀ`` (p=1 → SGD-momentum, p=0 → Muon polar factor, p=-0.25 →
late-stage outlier suppression), where a single *global* logistic time schedule
drives ``p_t : p_max → p_min`` for every layer:

    q_t = step / total_steps                      (step starts at 1)
    p_t = p_min + (p_max - p_min) / (1 + exp((q_t - tau_ratio) / width_ratio))

The released reference implements ``D(p)`` as a **three-phase transform**
(``dynmuon_spectral_transform``), NOT as a continuous power for every p:

    p >= 0.25      ->  raw momentum M (plain SGD-momentum step)
    0 <= p < 0.25  ->  Newton-Schulz polar factor (plain Muon)
    p < 0          ->  ``fast_spectral``: second-order Taylor correction
                       C ≈ A^{p/2} applied to the polar factor of the
                       Frobenius-normalized momentum, multiplied back by
                       ``‖M‖_F^p`` so the *raw* spectrum is shaped.

``compute_mode="reference"`` (default) reproduces that transform exactly,
including the bfloat16 Newton-Schulz and the bfloat16 update quantization.
``compute_mode="svd"`` and ``compute_mode="ns"`` compute the *continuous*
family ``D(p) = U Σ^p Vᵀ`` on the raw momentum spectrum in float32 — "svd"
exactly, "ns" through the Gram-eigendecomposition identity

    U Σ^p Vᵀ = ‖M‖_F^p · A_n^{p/2} (U Vᵀ),   A_n = X_n X_nᵀ,  X_n = M/‖M‖_F

with the polar factor approximated by Newton-Schulz. Singular values below
the standard numerical-rank tolerance ``σ_max · max(m,n) · ε_fp32`` are
treated as exact zeros (pseudo-power convention), so negative exponents never
amplify null directions.

DynMuon-Route (this work) replaces the global schedule with a *local*
per-layer proxy mapped to a parameter-specific exponent ``p_{t,l}``. Proxies:
gradient stable rank (default), two SNR proxies (instantaneous ``snr`` and the
bias-corrected EMA estimator ``snr_ema``), and a weight/momentum alignment
proxy (see ``ROUTING_MODES``).

Orthogonal experiment knobs (both deviate from the reference and exist to
isolate *why* spectral shaping helps; see math.tex §6):

  * ``magnitude="polar_fro"`` — rescale every update to the Frobenius norm of
    an exact polar factor, ``‖D‖_F = sqrt(min(m,n))``, so the exponent p (or a
    spectrum control) changes only the *shape* of the update spectrum, never
    its size. Without this, ``‖D(p)‖_F`` varies by ~sqrt(min(m,n)) as p moves
    from 1 to 0, i.e. the p-schedule doubles as an implicit LR schedule.
  * ``spectrum="random" | "inverted"`` — Kaon-style controls (arXiv:2605.11181)
    that replace ``Σ^p`` with a Frobenius-norm-preserving random or
    order-reversed spectrum (``compute_mode="svd"`` only).

Per-parameter diagnostics (``last_p``, ``last_sr``, ``last_gamma``,
``last_gamma_ema``, ``last_alpha``) are cached in ``self.state[param]`` so the
trainer can log the ``p_{t,l}`` trajectories. The schedule step counter, the
cross-layer proxy EMA, and the spectrum-control RNG are persisted through
``state_dict``/``load_state_dict`` so checkpoint resume is exact.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

ROUTING_MODES = (
    "fixed", "global_schedule", "schedule_modulated",
    "stable_rank", "snr", "snr_ema", "alignment",
)
PROXY_METRICS = ("stable_rank", "snr", "snr_ema", "alignment")
COMPUTE_MODES = ("reference", "svd", "ns")
NS_VARIANTS = ("quintic", "cubic")
SPECTRUM_MODES = ("power", "relmuon", "relmuon_log1p", "random_uniform", "random", "inverted")
MAGNITUDE_MODES = ("none", "polar_fro")

# Tuned quintic Newton-Schulz coefficients (reference DynMuon / Dion repo).
# Each row (a, b, c) applies X <- a X + b (X Xᵀ) X + c (X Xᵀ)² X.
QUINTIC_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

_FP32_EPS = torch.finfo(torch.float32).eps


def _eigh(A: torch.Tensor):
    """Symmetric eigendecomposition with a CPU fallback (MPS lacks eigh)."""
    try:
        return torch.linalg.eigh(A)
    except (NotImplementedError, RuntimeError):
        evals, Q = torch.linalg.eigh(A.cpu())
        return evals.to(A.device), Q.to(A.device)


def _eigvalsh(A: torch.Tensor) -> torch.Tensor:
    """Symmetric eigenvalues with a CPU fallback (MPS lacks eigvalsh)."""
    try:
        return torch.linalg.eigvalsh(A)
    except (NotImplementedError, RuntimeError):
        return torch.linalg.eigvalsh(A.cpu()).to(A.device)


def _svd(X: torch.Tensor):
    """Thin SVD with a CPU fallback for backends without device SVD."""
    try:
        return torch.linalg.svd(X, full_matrices=False)
    except (NotImplementedError, RuntimeError):
        U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        return U.to(X.device), S.to(X.device), Vh.to(X.device)


def _shape_lr_scale(fan_out: int, fan_in: int, adjust_lr_fn: str | None) -> float:
    """Shape-aware LR scaling (reference DynMuon ``adjust_lr_*``):

      * ``spectral_norm`` — ``sqrt(fan_out / fan_in)`` (arXiv:2310.17813).
      * ``rms_norm``      — ``0.2 * sqrt(max(fan_out, fan_in))`` (arXiv:2502.16982).
    """
    if adjust_lr_fn in (None, "none"):
        return 1.0
    if adjust_lr_fn == "spectral_norm":
        return float(math.sqrt(fan_out / fan_in))
    if adjust_lr_fn == "rms_norm":
        return float(0.2 * math.sqrt(max(fan_out, fan_in)))
    raise ValueError(f"unsupported DynMuon adjust_lr_fn: {adjust_lr_fn}")


# ---------------------------------------------------------------------------
# Reference DynMuon transform (mirrors dynmuon/dynmuon.py in fzwark/DynMuon
# operation-for-operation; verified numerically by validate_reference.py).
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Reference 5-step quintic Newton-Schulz polar approximation (bfloat16).

    Transposes internally so the iteration runs on the small Gram side and
    divides by the Frobenius norm so the spectral norm is at most 1 (the
    quintic coefficients require singular values in (0, 1]).
    """
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in QUINTIC_NS_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def fast_spectral(
    G: torch.Tensor,
    p: float,
    epsilon: float = 1e-7,
    order: int = 2,
) -> torch.Tensor:
    """Reference fast spectral shaping ``D(p) ≈ U Σ^p Vᵀ`` for the raw spectrum.

    Normalizes ``X_n = X / ‖X‖_F``, takes the Newton-Schulz polar factor
    ``Y_μ ≈ U Vᵀ``, approximates ``A_n^{p/2}`` (``A_n = X_n X_nᵀ``) by the
    Taylor polynomial of ``(I + E)^{p/2}`` around ``E = A_n - I = 0``

        order 1:  C = I + δE,                       δ = p/2
        order 2:  C = I + δE + ½ δ(δ-1) E²

    and multiplies back ``‖X‖_F^p`` so the *raw* singular values are shaped.
    The truncation implicitly caps the amplification of near-null directions
    (C(0) = 1 - δ + ½δ(δ-1) instead of the exact, divergent ``λ^δ → ∞``).
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    orig_dtype = G.dtype
    X = G.to(torch.float32)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    scale = X.norm(dim=(-2, -1), keepdim=True) + epsilon
    Xn = X / scale

    # Muon base on normalized input
    Y_mu = zeropower_via_newtonschulz5(Xn, epsilon=epsilon).to(torch.float32)

    # Pure Muon path
    if p == 0.0:
        U = Y_mu
        if transposed:
            U = U.mT
        return U.to(orig_dtype)

    A = Xn @ Xn.mT
    m = A.size(-1)
    I = torch.eye(m, device=A.device, dtype=A.dtype).view(
        (1,) * (A.ndim - 2) + (m, m)
    )

    # Polynomial correction
    delta = 0.5 * p
    E = A - I

    if order == 1:
        C = I + delta * E
    else:
        E2 = E @ E
        C = I + delta * E + 0.5 * delta * (delta - 1.0) * E2

    U = C @ Y_mu
    U = U * scale.pow(p)

    if transposed:
        U = U.mT
    return U.to(orig_dtype)


def dynmuon_spectral_transform(G: torch.Tensor, p: float, epsilon: float = 1e-8) -> torch.Tensor:
    """The reference DynMuon three-phase transform (global p made explicit).

    Mirrors ``dynmuon_spectral_transform`` in the reference repo, including its
    epsilon quirk: the optimizer epsilon reaches Newton-Schulz, while
    ``fast_spectral`` is invoked with its own internal default (1e-7).
    """
    if p >= 0.25:
        return G
    if p >= 0.0:
        return zeropower_via_newtonschulz5(G, epsilon=epsilon)
    return fast_spectral(G, p=p)


def logistic_schedule_p(
    step: int,
    total_steps: int,
    p_min: float,
    p_max: float,
    tau_ratio: float,
    width_ratio: float,
) -> float:
    """Reference DynMuon logistic time schedule ``p_t : p_max → p_min``.

    ``step`` starts at 1 on the first optimizer step (the reference increments
    the per-group step counter before scheduling).
    """
    q_t = step / float(total_steps)
    u = (q_t - tau_ratio) / max(width_ratio, 1e-8)
    anneal = 1.0 / (1.0 + math.exp(min(u, 700.0)))  # guard math.exp overflow only
    return p_min + (p_max - p_min) * anneal


# ---------------------------------------------------------------------------
# Continuous exact shaping (validation / ablation paths)
# ---------------------------------------------------------------------------

def newton_schulz(X: torch.Tensor, steps: int) -> torch.Tensor:
    """Plain cubic polar-factor iteration ``X_{k+1} = 1.5 X - 0.5 X Xᵀ X``.

    Converges to the orthogonal polar factor ``U Vᵀ`` when every singular value
    is in ``(0, √3)``; the caller Frobenius-normalizes so ``‖X‖_2 ≤ 1``. The
    cubic term is grouped as ``(X Xᵀ) X`` so the intermediate is the small
    ``m × m`` Gram (``X`` is ``m × n`` with ``m ≤ n``).
    """
    Y = X
    for _ in range(steps):
        Y = 1.5 * Y - 0.5 * ((Y @ Y.transpose(-2, -1)) @ Y)
    return Y


def quintic_newton_schulz(X: torch.Tensor) -> torch.Tensor:
    """Tuned 5-step quintic polar-factor iteration in the caller's dtype.

    Same coefficients as the reference kernel; the caller orients (rows ≤
    cols) and Frobenius-normalizes the input.
    """
    Y = X
    for a, b, c in QUINTIC_NS_COEFFS:
        G = Y @ Y.transpose(-2, -1)          # (m, m) Gram
        Y = a * Y + b * (G @ Y) + c * (G @ (G @ Y))
    return Y


def _pseudo_power_spectrum(S: torch.Tensor, p: float, rows: int, cols: int) -> torch.Tensor:
    """``σ_i^p`` with singular values below the numerical-rank tolerance
    ``σ_max · max(m,n) · ε_fp32`` treated as exact zeros (pseudo-inverse
    convention), so ``p ≤ 0`` never amplifies null directions."""
    tol = S.amax() * max(rows, cols) * _FP32_EPS
    powered = S.clamp(min=torch.finfo(S.dtype).tiny).pow(p)
    return torch.where(S > tol, powered, torch.zeros_like(S))


def shape_exact_svd(
    M: torch.Tensor,
    p: float,
    spectrum: str = "power",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Exact ``D = U f(Σ) Vᵀ`` of the raw momentum in float32.

    ``spectrum``:
      * ``power``    — ``f(Σ) = Σ^p`` (pseudo-power; see above).
      * ``inverted`` — ``f(Σ) = flip(Σ)``: the descending spectrum is assigned
        in reversed order (weakest direction gets the largest value); this is
        a Frobenius-norm-preserving permutation of the spectrum.
      * ``random``   — i.i.d. ``U(0,1)`` values rescaled so ``‖f(Σ)‖₂ = ‖Σ‖₂``
        (Frobenius-norm-preserving random spectrum, Kaon-style control).
    """
    X = M.to(torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    U, S, Vh = _svd(X)
    if spectrum == "power":
        s_new = _pseudo_power_spectrum(S, p, X.size(-2), X.size(-1))
    elif spectrum == "inverted":
        s_new = S.flip(-1)
    elif spectrum == "random":
        r = torch.rand(S.shape, generator=generator, dtype=torch.float32)
        r = r.to(S.device)
        s_norm = torch.linalg.norm(S)
        r_norm = torch.linalg.norm(r).clamp(min=torch.finfo(torch.float32).tiny)
        s_new = r * (s_norm / r_norm)
    else:
        raise ValueError(f"unknown spectrum mode: {spectrum}")
    D = (U * s_new) @ Vh
    if transposed:
        D = D.mT
    return D


def shape_exact_ns(
    M: torch.Tensor,
    p: float,
    ns_variant: str = "quintic",
    ns_steps: int = 5,
) -> torch.Tensor:
    """Continuous ``D = U Σ^p Vᵀ`` of the raw momentum via the Gram identity.

    With ``X_n = M/‖M‖_F`` (oriented rows ≤ cols), ``A_n = X_n X_nᵀ = U Σ_n² Uᵀ``
    and polar factor ``U Vᵀ``:

        U Σ^p Vᵀ = ‖M‖_F^p · (Q Λ^{p/2} Qᵀ) (U Vᵀ),  A_n = Q Λ Qᵀ.

    ``A_n^{p/2}`` is exact (float32 eigendecomposition with the pseudo-power
    rank tolerance); the polar factor is approximated by Newton-Schulz.
    """
    X = M.to(torch.float32)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    fro = torch.linalg.norm(X)
    if fro == 0:
        return M.to(torch.float32) * 0.0
    Xn = X / fro

    A = Xn @ Xn.transpose(-2, -1)
    evals, Q = _eigh(A)
    evals = evals.clamp(min=0.0)
    # Eigenvalues of A_n are squared singular values: the rank tolerance on
    # singular values (σ > σ_max·max(m,n)·ε) becomes λ > λ_max·(max(m,n)·ε)².
    lam_tol = evals[-1] * (max(X.size(-2), X.size(-1)) * _FP32_EPS) ** 2
    factors = torch.where(
        evals > lam_tol,
        evals.clamp(min=torch.finfo(torch.float32).tiny).pow(p / 2.0),
        torch.zeros_like(evals),
    )

    Y = quintic_newton_schulz(Xn) if ns_variant == "quintic" else newton_schulz(Xn, ns_steps)
    D = ((Q * factors) @ Q.transpose(-2, -1)) @ Y
    D = D * fro.pow(p)
    if transposed:
        D = D.mT
    return D


def logistic_route(x: float, p_min: float, p_max: float, mu: float, omega: float) -> float:
    """Parameterized logistic map of proxy ``x`` to exponent ``p``:

        p = p_min + (p_max - p_min) / (1 + exp(-(x - mu) / omega))

    Sign of ``omega`` sets orientation: ``omega > 0`` => p increases with x
    (stable-rank, alignment); ``omega < 0`` => p decreases with x (the SNR
    proxies: noisy layers, low x, are routed toward p_max).
    """
    z = max(-60.0, min(60.0, (x - mu) / omega))
    return p_min + (p_max - p_min) / (1.0 + math.exp(-z))


class DynMuonRoute(torch.optim.Optimizer):
    """Muon with dynamic layer-wise spectral-exponent routing.

    Routing modes (set ``routing_mode``):
      * ``fixed``              — constant ``fixed_p`` (p=0 Muon, p=1 SGD).
      * ``global_schedule``    — the reference DynMuon logistic time schedule
                                 ``p_t : p_max → p_min``, identical for every
                                 layer. With ``compute_mode="reference"`` this
                                 *is* the released DynMuon optimizer (verified
                                 numerically by validate_reference.py).
      * ``schedule_modulated`` — **the router.** Follow the same global time
                                 schedule, but nudge each layer by how its
                                 gradient geometry deviates from a reference:
                                     p_{t,l} = clip( p_t + lean_l )
                                 with ``lean_l = beta·(proxy_l − ref_l)`` and,
                                 when ``lean_max`` is set, ``lean_l`` clipped to
                                 ``[-lean_max, +lean_max]``. ``dynamic_ref=True``
                                 replaces ``ref_l`` with an EMA of the
                                 cross-layer mean proxy, so the schedule owns
                                 the global temporal trend and the router
                                 responds only to per-layer deviation.
                                 ``lean_norm="zscore"`` divides the deviation by
                                 an EMA of the cross-layer proxy std, making
                                 ``beta`` "p-units per standard deviation":
                                 heavy-tailed proxies (stable rank reaches
                                 50+ on early layers) then cannot saturate the
                                 router into a bang-bang controller pinned at
                                 the clip boundaries.
      * ``stable_rank`` / ``snr`` / ``snr_ema`` / ``alignment`` — map the proxy
        straight to p through a per-layer-type logistic (``mu``, ``omega``);
        no time schedule.

    Per-param-group knobs (``mu``, ``omega``, ``ref``) let Attention, MLP and
    other matrices carry distinct routing; the trainer builds one group per
    layer type. Biases, norm gains, and embeddings must NOT be passed here —
    route them through AdamW.

    Weight decay is decoupled (AdamW-style), applied with the *base* LR before
    the update, exactly as in the reference: ``W ← (1 - lr·λ) W - lr_adj · D``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        routing_mode: str = "stable_rank",
        spectrum_mode: str = "power",
        compute_mode: str = "ns",
        ns_variant: str = "quintic",
        ns_steps: int = 5,
        eps: float = 1e-8,
        adjust_lr_fn: str | None = "spectral_norm",
        p_min: float = -0.25,
        p_max: float = 1.0,
        mu: float = 0.0,
        omega: float = 1.0,
        ref: float = 0.0,
        beta: float = 0.1,
        dynamic_ref: bool = False,
        ref_decay: float = 0.9,
        lean_norm: str = "raw",
        lean_max: float | None = None,
        modulate_metric: str = "stable_rank",
        fixed_p: float = 0.0,
        tau_ratio: float = 0.04,
        width_ratio: float = 0.04,
        total_steps: int | None = None,
        magnitude: str = "none",
        spectrum: str = "power",
        spectrum_seed: int = 0,
        track_proxies: bool = True,
        snr_ema_decay: float = 0.95,
    ):
        # Resolve spectrum and spectrum_mode:
        if spectrum_mode == "power" and spectrum != "power":
            spectrum_mode = spectrum
        if spectrum_mode == "random":
            spectrum_mode = "random_uniform"

        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}, got {routing_mode}")
        if spectrum_mode not in SPECTRUM_MODES:
            raise ValueError(f"spectrum_mode must be one of {SPECTRUM_MODES}, got {spectrum_mode}")
        if compute_mode not in COMPUTE_MODES:
            raise ValueError(f"compute_mode must be one of {COMPUTE_MODES}, got {compute_mode}")
        if ns_variant not in NS_VARIANTS:
            raise ValueError(f"ns_variant must be one of {NS_VARIANTS}, got {ns_variant}")
        if magnitude not in MAGNITUDE_MODES:
            raise ValueError(f"magnitude must be one of {MAGNITUDE_MODES}, got {magnitude}")
        if spectrum not in SPECTRUM_MODES:
            raise ValueError(f"spectrum must be one of {SPECTRUM_MODES}, got {spectrum}")
        if spectrum_mode != "power" and compute_mode == "reference":
            raise ValueError(
                f"spectrum_mode={spectrum_mode!r} is not supported with compute_mode='reference'"
            )
        if modulate_metric not in PROXY_METRICS:
            raise ValueError(f"modulate_metric must be one of {PROXY_METRICS}, got {modulate_metric}")
        if adjust_lr_fn not in (None, "none", "spectral_norm", "rms_norm"):
            raise ValueError(
                "DynMuon adjust_lr_fn must be None, 'none', 'spectral_norm', or "
                f"'rms_norm' (got {adjust_lr_fn!r})"
            )
        if routing_mode in ("global_schedule", "schedule_modulated") and not total_steps:
            raise ValueError(f"routing_mode={routing_mode!r} requires total_steps > 0")
        if not 0.0 < snr_ema_decay < 1.0:
            raise ValueError(f"snr_ema_decay must be in (0, 1), got {snr_ema_decay}")
        if lean_norm not in ("raw", "zscore"):
            raise ValueError(f"lean_norm must be 'raw' or 'zscore', got {lean_norm!r}")
        if lean_max is not None and lean_max <= 0:
            raise ValueError(f"lean_max must be positive or None, got {lean_max}")
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
            routing_mode=routing_mode, spectrum_mode=spectrum_mode, compute_mode=compute_mode,
            ns_variant=ns_variant, ns_steps=ns_steps, eps=eps, adjust_lr_fn=adjust_lr_fn,
            p_min=p_min, p_max=p_max, mu=mu, omega=omega, ref=ref, beta=beta,
            dynamic_ref=dynamic_ref, ref_decay=ref_decay,
            lean_norm=lean_norm, lean_max=lean_max,
            modulate_metric=modulate_metric, fixed_p=fixed_p, tau_ratio=tau_ratio,
            width_ratio=width_ratio, total_steps=total_steps, magnitude=magnitude,
            spectrum=spectrum, track_proxies=track_proxies,
            snr_ema_decay=snr_ema_decay,
        )
        super().__init__(params, defaults)
        # Schedule step counter: incremented at the START of step(), so the
        # first scheduled step is 1 (reference DynMuon semantics).
        self._step_count = 0
        # Running cross-layer mean/std of the routing proxy (for dynamic_ref
        # and lean_norm="zscore"): lets the schedule own the global/temporal
        # trend while the router responds only to each layer's deviation from
        # the network average, measured in network-spread units.
        self._proxy_ema: float | None = None
        self._proxy_std_ema: float | None = None
        # CPU RNG for spectrum="random" so the control is reproducible and
        # device-independent.
        self._spectrum_generator = torch.Generator()
        self._spectrum_generator.manual_seed(spectrum_seed)

    # -- schedule state must survive checkpointing ---------------------------

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["dynmuon_extras"] = {
            "step_count": self._step_count,
            "proxy_ema": self._proxy_ema,
            "proxy_std_ema": self._proxy_std_ema,
            "spectrum_rng": self._spectrum_generator.get_state(),
        }
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        extras = state_dict.pop("dynmuon_extras", None)
        super().load_state_dict(state_dict)
        if extras is not None:
            self._step_count = int(extras["step_count"])
            self._proxy_ema = extras["proxy_ema"]
            self._proxy_std_ema = extras.get("proxy_std_ema")
            self._spectrum_generator.set_state(extras["spectrum_rng"])

    # -- routing -------------------------------------------------------------

    def _p_schedule(self, group: dict) -> float:
        return logistic_schedule_p(
            self._step_count, group["total_steps"], group["p_min"], group["p_max"],
            group["tau_ratio"], group["width_ratio"],
        )

    def _select_p(self, group: dict, x: float | None) -> float:
        """Map the active routing mode (and proxy ``x``) to a spectral exponent."""
        mode = group["routing_mode"]
        if mode == "fixed":
            return group["fixed_p"]              # 0.0 = Muon, 1.0 = SGD
        if mode == "global_schedule":
            return self._p_schedule(group)
        if mode == "schedule_modulated":
            # Global time arc + per-layer geometry lean, clipped to [p_min, p_max].
            # dynamic_ref: lean relative to the running cross-layer mean (removes
            # the global temporal trend, leaving the per-layer deviation).
            ref = (self._proxy_ema if (group["dynamic_ref"] and self._proxy_ema is not None)
                   else group["ref"])
            deviation = x - ref
            if group["lean_norm"] == "zscore":
                # Per-std units: heavy-tailed proxies cannot saturate the
                # router. Until the std EMA exists (first step), lean 0.
                std = self._proxy_std_ema
                deviation = 0.0 if not std else deviation / std
            lean = group["beta"] * deviation
            if group["lean_max"] is not None:
                lean = max(-group["lean_max"], min(group["lean_max"], lean))
            p = self._p_schedule(group) + lean
            return max(group["p_min"], min(group["p_max"], p))
        return logistic_route(x, group["p_min"], group["p_max"], group["mu"], group["omega"])

    @staticmethod
    def _routing_metric(group: dict) -> str | None:
        mode = group["routing_mode"]
        if mode == "schedule_modulated":
            return group["modulate_metric"]
        if mode in PROXY_METRICS:
            return mode
        return None

    def _snr_ema_proxy(self, state: dict, G2: torch.Tensor, decay: float, eps: float) -> float:
        """Bias-corrected EMA signal-to-noise estimate of the raw gradient:

            m_t = ρ m_{t-1} + (1-ρ) G_t          (matrix EMA, the signal)
            v_t = ρ v_{t-1} + (1-ρ) ‖G_t‖_F²     (scalar EMA, the energy)
            γ̂  = ‖m̂_t‖_F / sqrt(max(v̂_t − ‖m̂_t‖_F², 0) + eps)

        with ``m̂ = m/(1-ρ^t)``, ``v̂ = v/(1-ρ^t)``. ``v̂ − ‖m̂‖²`` estimates the
        total gradient variance across steps (E‖G‖² − ‖EG‖²).
        """
        if "snr_ema_grad" not in state:
            state["snr_ema_grad"] = torch.zeros_like(G2, dtype=torch.float32)
            state["snr_ema_sq"] = 0.0
            state["snr_ema_count"] = 0
        m = state["snr_ema_grad"]
        m.mul_(decay).add_(G2.to(torch.float32), alpha=1.0 - decay)
        state["snr_ema_sq"] = decay * state["snr_ema_sq"] + (1.0 - decay) * float(
            torch.linalg.norm(G2.to(torch.float32)) ** 2
        )
        state["snr_ema_count"] += 1
        bc = 1.0 - decay ** state["snr_ema_count"]
        signal_sq = float(torch.linalg.norm(m) ** 2) / (bc * bc)
        energy = state["snr_ema_sq"] / bc
        noise_var = max(energy - signal_sq, 0.0)
        return math.sqrt(signal_sq) / math.sqrt(noise_var + eps)

    # -- optimization --------------------------------------------------------

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
        self._step_count += 1
        step_proxies = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    x = self._update_param(param, group, noise_hook)
                    if x is not None:
                        step_proxies.append(x)
        # Update the running cross-layer proxy mean/std (used by dynamic_ref
        # and lean_norm="zscore" on the next step). Only finite samples may
        # enter: one NaN would latch the EMA to NaN permanently.
        step_proxies = [v for v in step_proxies if math.isfinite(v)]
        if step_proxies:
            m = sum(step_proxies) / len(step_proxies)
            var = sum((v - m) ** 2 for v in step_proxies) / len(step_proxies)
            std = math.sqrt(var)
            decay = self.param_groups[0]["ref_decay"]
            if self._proxy_ema is None:
                self._proxy_ema, self._proxy_std_ema = m, std
            else:
                self._proxy_ema = decay * self._proxy_ema + (1 - decay) * m
                self._proxy_std_ema = (std if self._proxy_std_ema is None
                                       else decay * self._proxy_std_ema + (1 - decay) * std)
        return loss

    def _update_param(self, param, group, noise_hook) -> float | None:
        """Apply the shaped update to ``param``; return the routing proxy value
        (for the dynamic cross-layer mean), or None when no proxy is routed."""
        eps = group["eps"]
        G = param.grad
        orig_shape = G.shape
        G2 = G.reshape(G.shape[0], -1) if G.dim() != 2 else G
        W2 = param.reshape(param.shape[0], -1) if param.dim() != 2 else param
        fan_out, fan_in = G2.shape

        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(G2)
        B = state["momentum_buffer"]
        B.mul_(group["momentum"]).add_(G2)
        # Reference op order (mul then add) so the float rounding matches the
        # reference pre-orthogonalization exactly.
        M2 = B.mul(group["momentum"]).add_(G2) if group["nesterov"] else B

        if noise_hook is not None:
            M2 = noise_hook(M2, state)

        # -- routing proxies (raw, layer-local, float32, update-independent) --
        metric = self._routing_metric(group)
        track = group["track_proxies"]
        fro_M = float(torch.linalg.norm(M2.to(torch.float32)))
        sr = gamma = alpha = gamma_ema = float("nan")
        if metric == "snr_ema":
            # The EMA estimator consumes every gradient sample, including zero
            # ones, so update it before the zero-momentum early exit.
            gamma_ema = self._snr_ema_proxy(state, G2, group["snr_ema_decay"], eps)
        if fro_M == 0.0:
            # Zero momentum (e.g. step-1 gradients blocked by zero-initialized
            # downstream weights): D(p) = 0 for every p, so the exponent is
            # undefined and there is no proxy sample. Apply weight decay (the
            # reference decays any param with a grad), skip the zero update,
            # and contribute nothing to the cross-layer proxy statistics —
            # a NaN here would latch the proxy EMA to NaN permanently.
            if group["weight_decay"]:
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
            state.update(last_p=float("nan"), last_sr=float("nan"),
                         last_gamma=float("nan"), last_gamma_ema=float(gamma_ema),
                         last_alpha=float("nan"))
            return None
        spectrum_mode = group["spectrum_mode"]
        use_svd = (group["compute_mode"] == "svd") or (spectrum_mode != "power")

        if use_svd:
            U, S, Vh = _svd(M2)
            lam_max = float(((S[0] / fro_M) ** 2).item())
        else:
            Xn = M2.to(torch.float32) / fro_M
            A = Xn @ Xn.mT if Xn.size(-2) <= Xn.size(-1) else Xn.mT @ Xn
            lam_max = float(_eigvalsh(A)[-1].clamp(min=0.0))

        if track or metric == "stable_rank":
            sr = 1.0 / (lam_max + eps)
        if track or metric == "snr":
            gamma = fro_M / (float(torch.linalg.norm((G2 - M2).to(torch.float32))) + eps)
        if track or metric == "alignment":
            num = float(torch.sum(W2.to(torch.float32) * M2.to(torch.float32)).abs())
            alpha = num / (float(torch.linalg.norm(W2.to(torch.float32))) * fro_M + eps)

        proxies = {"stable_rank": sr, "snr": gamma, "snr_ema": gamma_ema, "alignment": alpha}
        x = proxies.get(metric) if metric is not None else None
        p_exp = self._select_p(group, x)

        # -- shape update D ---------------------------------------------------
        if group["compute_mode"] == "reference":
            D = dynmuon_spectral_transform(M2.to(torch.bfloat16), p_exp, epsilon=eps)
        elif spectrum_mode in ("relmuon", "relmuon_log1p"):
            # U, S, Vh are already computed from M2 above since use_svd is True
            _, S_w, _ = _svd(W2)
            r = min(S.numel(), S_w.numel())
            if spectrum_mode == "relmuon":
                S_hat = S_w[:r]
            else:  # relmuon_log1p
                S_hat = torch.log1p(S_w[:r].clamp(min=0.0))
            S_hat = (S_hat + eps) / (torch.sqrt(torch.mean(S_hat.square())) + eps)
            D = (U[:, :r] * S_hat.to(dtype=U.dtype, device=U.device)) @ Vh[:r, :]
        elif use_svd:
            # compute_mode == "svd", or a non-power spectrum control that needs
            # the explicit SVD (power must NOT land here in ns mode, or the
            # Newton-Schulz path silently becomes the exact-SVD path).
            spec_arg = "random" if spectrum_mode == "random_uniform" else spectrum_mode
            D = shape_exact_svd(M2, p_exp, spectrum=spec_arg, generator=self._spectrum_generator)
        else:  # compute_mode == "ns" and spectrum_mode == "power"
            D = shape_exact_ns(M2, p_exp, ns_variant=group["ns_variant"], ns_steps=group["ns_steps"])

        if group["magnitude"] == "polar_fro":
            # Decouple magnitude from shape: every update carries the Frobenius
            # norm of an exact polar factor, ‖D‖_F = sqrt(min(m, n)).
            D = D.to(torch.float32)
            d_norm = torch.linalg.norm(D)
            if float(d_norm) > 0.0:
                D = D * (math.sqrt(min(fan_out, fan_in)) / d_norm)

        # -- decoupled weight decay + update (reference order) ----------------
        if group["weight_decay"]:
            param.mul_(1.0 - group["lr"] * group["weight_decay"])
        lr_eff = group["lr"] * _shape_lr_scale(fan_out, fan_in, group["adjust_lr_fn"])
        # For the reference path D is bfloat16; (D * lr_eff) stays bfloat16 and
        # is upcast during the subtraction — exactly the reference behavior.
        param.sub_((D * lr_eff).reshape(orig_shape))

        state.update(last_p=float(p_exp), last_sr=float(sr), last_gamma=float(gamma),
                     last_gamma_ema=float(gamma_ema), last_alpha=float(alpha))
        return x
