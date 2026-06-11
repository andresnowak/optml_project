"""Unit tests for the DynMuon-Route spectral math.

Validates that the equivalent-factorization identities of math.tex reproduce
the exact SVD spectral shaping ``D(p) = U Σ^p Vᵀ`` across random matrices and
fractional exponents, that Newton-Schulz recovers the polar factor, that the
optimizer's continuous compute paths agree with each other, and that the
magnitude / spectrum / routing extensions behave as specified.

Reference parity (against vendored code from the released DynMuon repo) lives
in validate_reference.py.

Run with:  pytest validate_math.py
"""

from __future__ import annotations

import math

import pytest
import torch

from src import DynMuonRoute, logistic_route, newton_schulz
from src.optimizers.dynmuon import (
    logistic_schedule_p,
    quintic_newton_schulz,
    shape_exact_svd,
)
from src.optimizers.relmuon import (
    relmuon_aligned_scales,
    relmuon_update,
    relmuon_weight_scales,
)
from src.trainer import lr_factor

torch.manual_seed(0)

SHAPES = [(8, 8), (5, 12), (12, 5), (16, 4), (3, 9)]
EXPONENTS = [-0.25, 0.0, 0.3, 0.5, 1.0]


def test_lr_factor_hits_min_ratio_on_last_update():
    """The cosine schedule should reach the configured floor on the final update."""
    train_steps = 1526
    warmup_steps = 150
    assert math.isclose(lr_factor(warmup_steps - 1, warmup_steps, train_steps, 0.0), 1.0)
    assert math.isclose(lr_factor(warmup_steps, warmup_steps, train_steps, 0.0), 1.0)
    assert math.isclose(lr_factor(train_steps - 1, warmup_steps, train_steps, 0.0), 0.0, abs_tol=1e-12)


def _sym_matrix_power(A: torch.Tensor, power: float) -> torch.Tensor:
    """``A^power`` for a symmetric PSD matrix via eigendecomposition."""
    evals, Q = torch.linalg.eigh(A)
    evals = evals.clamp(min=1e-30)
    return (Q * evals.pow(power)) @ Q.transpose(-2, -1)


def _exact_shaping(X: torch.Tensor, p: float) -> torch.Tensor:
    """Reference ``U Σ^p Vᵀ`` from a full SVD."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return (U * S.pow(p)) @ Vh


def _orient(X: torch.Tensor) -> torch.Tensor:
    """Match the optimizer's orientation (rows <= cols)."""
    return X.transpose(0, 1) if X.shape[0] > X.shape[1] else X


# ---------------------------------------------------------------------------
# core spectral identities (math.tex §3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", EXPONENTS)
def test_factorization_matches_svd(shape, p):
    """A^{(p-1)/2} X_n == A^{p/2} (A^{-1/2} X_n) == U Σ^p Vᵀ."""
    X = _orient(torch.randn(*shape, dtype=torch.float64))
    X_n = X / torch.linalg.norm(X)
    A = X_n @ X_n.transpose(0, 1)

    ref = _exact_shaping(X_n, p)
    form1 = _sym_matrix_power(A, (p - 1) / 2) @ X_n
    polar = _sym_matrix_power(A, -0.5) @ X_n
    form2 = _sym_matrix_power(A, p / 2) @ polar

    assert torch.allclose(form1, ref, atol=1e-8, rtol=1e-6)
    assert torch.allclose(form2, ref, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("p", EXPONENTS)
def test_raw_spectrum_backscale_identity(shape, p):
    """Shaping the normalized spectrum and multiplying back ``‖M‖_F^p`` equals
    shaping the raw spectrum: ``‖M‖^p · U (Σ/‖M‖)^p Vᵀ = U Σ^p Vᵀ``."""
    M = _orient(torch.randn(*shape, dtype=torch.float64)) * 3.7
    s = torch.linalg.norm(M)
    raw = _exact_shaping(M, p)
    normalized_then_scaled = s.pow(p) * _exact_shaping(M / s, p)
    assert torch.allclose(normalized_then_scaled, raw, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_newton_schulz_recovers_polar(shape):
    """NS(X_n) converges to the orthogonal polar factor U Vᵀ."""
    X = _orient(torch.randn(*shape, dtype=torch.float64))
    X_n = X / torch.linalg.norm(X)
    U, _, Vh = torch.linalg.svd(X_n, full_matrices=False)
    polar = U @ Vh
    Y = newton_schulz(X_n, steps=40)
    assert torch.allclose(Y, polar, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("shape", SHAPES)
def test_quintic_newton_schulz_approximates_polar(shape):
    """The tuned 5-step quintic iteration (reference coefficients) is a coarse
    but usable approximation of the polar factor."""
    X = _orient(torch.randn(*shape, dtype=torch.float64))
    X_n = X / torch.linalg.norm(X)
    U, _, Vh = torch.linalg.svd(X_n, full_matrices=False)
    polar = U @ Vh
    Y = quintic_newton_schulz(X_n)
    rel = torch.linalg.norm(Y - polar) / torch.linalg.norm(polar)
    assert rel < 0.1, f"quintic NS too far from polar: {rel}"


def test_stable_rank_identity():
    """sr = ‖M‖_F²/σ_max² equals 1/λ_max of the normalized Gram."""
    X = _orient(torch.randn(10, 14, dtype=torch.float64))
    sv = torch.linalg.svdvals(X)
    sr_def = float((sv.pow(2).sum() / sv[0].pow(2)))
    X_n = X / torch.linalg.norm(X)
    lam_max = float(torch.linalg.eigvalsh(X_n @ X_n.transpose(0, 1))[-1])
    assert math.isclose(sr_def, 1.0 / lam_max, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# continuous compute paths (svd vs ns) — both shape the RAW spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ns_variant,tol", [("cubic", 5e-3), ("quintic", 5e-2)])
def test_optimizer_ns_matches_svd(ns_variant, tol):
    """A single DynMuonRoute step in ns mode matches svd mode (global schedule
    fixes p so both paths shape the same exponent). The cubic iteration (with
    many steps) is near-exact; the tuned quintic is a coarser 5-step approx."""
    torch.manual_seed(1)
    g = torch.randn(20, 28)

    def run(mode):
        w = torch.zeros(20, 28, requires_grad=True)
        opt = DynMuonRoute([w], lr=0.1, momentum=0.0, nesterov=False,
                           routing_mode="global_schedule", compute_mode=mode,
                           ns_variant=ns_variant, ns_steps=30, adjust_lr_fn=None,
                           total_steps=4, track_proxies=False)
        w.grad = g.clone()
        opt.step()
        return w.detach().clone()

    d_svd, d_ns = run("svd"), run("ns")
    rel = torch.linalg.norm(d_ns - d_svd) / torch.linalg.norm(d_svd)
    assert rel < tol, f"ns({ns_variant}) vs svd relative error too large: {rel}"


@pytest.mark.parametrize("fixed_p,name", [(0.0, "muon"), (1.0, "sgd")])
def test_fixed_mode_constant_exponent(fixed_p, name):
    """routing_mode='fixed' with the exact svd path: p=0 is Muon (polar factor,
    all singular values 1); p=1 leaves the raw momentum unshaped, so with zero
    momentum the update IS the gradient (true SGD endpoint)."""
    torch.manual_seed(2)
    w = torch.zeros(8, 12, requires_grad=True)
    opt = DynMuonRoute([w], lr=0.1, momentum=0.0, nesterov=False,
                       routing_mode="fixed", fixed_p=fixed_p, compute_mode="svd",
                       adjust_lr_fn=None)
    g = torch.randn(8, 12)
    w.grad = g.clone()
    opt.step()
    assert opt.state[w]["last_p"] == fixed_p
    update = -w.detach() / 0.1                  # lr=0.1 -> update = D(p)
    if name == "muon":                          # D(0) = U Vᵀ -> singular values 1
        sv = torch.linalg.svdvals(update)
        assert torch.allclose(sv, torch.ones_like(sv), atol=1e-5)
    else:                                       # D(1) = M = G (raw spectrum!)
        assert torch.allclose(update, g, atol=1e-5, rtol=1e-5)


def test_pseudo_power_handles_rank_deficiency():
    """Negative exponents must not amplify numerically-null directions: for a
    rank-2 momentum, D(-0.25) keeps rank 2 and stays bounded."""
    torch.manual_seed(4)
    M = torch.outer(torch.randn(8), torch.randn(12)) + torch.outer(torch.randn(8), torch.randn(12))
    D = shape_exact_svd(M, p=-0.25)
    assert torch.isfinite(D).all()
    sv = torch.linalg.svdvals(D)
    assert int((sv > 1e-4 * sv.max()).sum()) == 2
    # the two live directions carry σ^p of the raw singular values
    sv_m = torch.linalg.svdvals(M)
    assert torch.allclose(sv[:2].sort(descending=True).values,
                          sv_m[:2].pow(-0.25).sort(descending=True).values, rtol=1e-4)


# ---------------------------------------------------------------------------
# magnitude decoupling and spectrum controls (math.tex §6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", EXPONENTS)
def test_polar_fro_magnitude_is_constant_in_p(p):
    """magnitude='polar_fro' pins every update to ‖D‖_F = sqrt(min(m,n)),
    so the exponent changes only the spectrum shape."""
    torch.manual_seed(6)
    w = torch.zeros(12, 20, requires_grad=True)
    opt = DynMuonRoute([w], lr=1.0, momentum=0.0, nesterov=False,
                       routing_mode="fixed", fixed_p=p, compute_mode="svd",
                       adjust_lr_fn=None, magnitude="polar_fro")
    w.grad = torch.randn(12, 20)
    opt.step()
    assert math.isclose(float(torch.linalg.norm(w.detach())), math.sqrt(12), rel_tol=1e-5)


def test_spectrum_inverted_reverses_singular_values():
    """spectrum='inverted' assigns the reversed spectrum (norm-preserving)."""
    torch.manual_seed(8)
    M = torch.randn(6, 9)
    D = shape_exact_svd(M, p=0.0, spectrum="inverted")
    sv_m = torch.linalg.svdvals(M)
    sv_d = torch.linalg.svdvals(D)
    assert torch.allclose(sv_d, sv_m, atol=1e-5)            # same multiset
    assert math.isclose(float(torch.linalg.norm(D)), float(torch.linalg.norm(M)), rel_tol=1e-6)
    # the top direction of M now carries the SMALLEST singular value
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    carried = float(U[:, 0] @ D @ Vh[0, :])
    assert math.isclose(carried, float(S[-1]), rel_tol=1e-4)


def test_spectrum_random_is_norm_preserving_and_reproducible():
    """spectrum='random' preserves ‖Σ‖₂ and is reproducible via spectrum_seed."""
    torch.manual_seed(9)
    g = torch.randn(7, 11)

    def run(seed):
        w = torch.zeros(7, 11, requires_grad=True)
        opt = DynMuonRoute([w], lr=1.0, momentum=0.0, nesterov=False,
                           routing_mode="fixed", fixed_p=0.0, compute_mode="svd",
                           adjust_lr_fn=None, spectrum="random", spectrum_seed=seed)
        w.grad = g.clone()
        opt.step()
        return w.detach().clone()

    a, b, c = run(0), run(0), run(1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert math.isclose(float(torch.linalg.norm(a)), float(torch.linalg.norm(g)), rel_tol=1e-5)


def test_weight_decay_is_decoupled():
    """W ← (1 − lr·λ) W − lr_adj · D, with weight decay at the base LR."""
    torch.manual_seed(10)
    init = torch.randn(8, 16)
    w = init.clone().requires_grad_(True)
    opt = DynMuonRoute([w], lr=0.1, momentum=0.0, nesterov=False, weight_decay=0.5,
                       routing_mode="fixed", fixed_p=0.0, compute_mode="svd",
                       adjust_lr_fn="spectral_norm")
    g = torch.randn(8, 16)
    w.grad = g.clone()
    opt.step()
    D = shape_exact_svd(g, p=0.0)
    expected = init * (1 - 0.1 * 0.5) - (0.1 * math.sqrt(8 / 16)) * D
    assert torch.allclose(w.detach(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_logistic_bounds_and_orientation():
    pmin, pmax = -0.25, 1.0
    # increasing in x for omega > 0
    lo = logistic_route(-1e6, pmin, pmax, mu=0.0, omega=1.0)
    hi = logistic_route(+1e6, pmin, pmax, mu=0.0, omega=1.0)
    assert math.isclose(lo, pmin, abs_tol=1e-6) and math.isclose(hi, pmax, abs_tol=1e-6)
    # decreasing in x for omega < 0 (SNR orientation: noisy/low x -> p_max)
    lo2 = logistic_route(-1e6, pmin, pmax, mu=0.0, omega=-1.0)
    hi2 = logistic_route(+1e6, pmin, pmax, mu=0.0, omega=-1.0)
    assert lo2 > hi2
    # always within [pmin, pmax]
    for x in torch.linspace(-50, 50, 200):
        p = logistic_route(float(x), pmin, pmax, mu=3.0, omega=2.0)
        assert pmin - 1e-9 <= p <= pmax + 1e-9


def test_schedule_modulated_reduces_to_schedule_when_beta_zero():
    """schedule_modulated with beta=0 equals the plain global schedule; with
    beta>0 it deviates per layer but stays within [p_min, p_max]."""
    torch.manual_seed(3)
    g = torch.randn(10, 16)

    def run_p(beta):
        w = torch.zeros(10, 16, requires_grad=True)
        opt = DynMuonRoute([w], routing_mode="schedule_modulated", compute_mode="reference",
                           beta=beta, ref=2.5, modulate_metric="stable_rank",
                           total_steps=8, adjust_lr_fn=None)
        ps = []
        for _ in range(4):
            w.grad = g.clone()
            opt.step()
            ps.append(opt.state[w]["last_p"])
        return ps

    sched_vals = [logistic_schedule_p(k, 8, -0.25, 1.0, 0.04, 0.04) for k in range(1, 5)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(run_p(0.0), sched_vals))   # beta=0 == schedule
    assert all(-0.25 - 1e-9 <= p <= 1.0 + 1e-9 for p in run_p(0.2))         # beta>0 in range


def test_zscore_lean_is_bounded_and_reduces_to_schedule():
    """With lean_norm='zscore' the lean is measured in cross-layer-std units
    and clipped to lean_max, so a heavy-tailed proxy cannot pin layers at the
    [p_min, p_max] boundaries (the marathon-run failure mode); beta=0 still
    reduces exactly to the global schedule."""
    torch.manual_seed(15)
    # Two params with wildly different stable ranks: near-rank-one vs identity-like.
    g_aniso = torch.outer(torch.randn(12), torch.randn(12)) + 0.01 * torch.randn(12, 12)
    g_iso = torch.eye(12) + 0.01 * torch.randn(12, 12)

    def run(beta, lean_max):
        wa = torch.zeros(12, 12, requires_grad=True)
        wb = torch.zeros(12, 12, requires_grad=True)
        opt = DynMuonRoute([wa, wb], routing_mode="schedule_modulated",
                           compute_mode="reference", beta=beta, dynamic_ref=True,
                           lean_norm="zscore", lean_max=lean_max,
                           modulate_metric="stable_rank", total_steps=8,
                           adjust_lr_fn=None)
        ps = []
        for _ in range(6):
            wa.grad, wb.grad = g_aniso.clone(), g_iso.clone()
            opt.step()
            ps.append((opt.state[wa]["last_p"], opt.state[wb]["last_p"]))
        return ps

    sched = [logistic_schedule_p(k, 8, -0.25, 1.0, 0.04, 0.04) for k in range(1, 7)]
    # beta=0: exactly the schedule for both layers.
    for (pa, pb), pt in zip(run(0.0, 0.25), sched):
        assert abs(pa - pt) < 1e-9 and abs(pb - pt) < 1e-9
    # beta>0: leans differ across layers but never exceed lean_max.
    routed = run(0.3, 0.2)
    for (pa, pb), pt in zip(routed, sched):
        assert abs(pa - pt) <= 0.2 + 1e-9
        assert abs(pb - pt) <= 0.2 + 1e-9
    assert any(pb > pa for (pa, pb) in routed[1:]), "isotropic layer should lean higher"


def test_zero_first_step_gradient_does_not_poison_proxy_ema():
    """Regression test for the sweep-run failure: with zero-initialized
    projection layers, q/k/v/fc matrices receive exactly zero gradients on
    step 1; their undefined proxy must NOT enter the cross-layer EMA (a single
    NaN latches the EMA to NaN forever, and the NaN lean then resolves to
    p = p_max through Python's min/max, pinning every layer at +1.0)."""
    torch.manual_seed(16)
    w_live = torch.zeros(8, 12, requires_grad=True)   # has gradient from step 1
    w_dead = torch.zeros(8, 12, requires_grad=True)   # zero gradient on step 1
    opt = DynMuonRoute([w_live, w_dead], lr=0.02, weight_decay=0.5,
                       routing_mode="schedule_modulated", compute_mode="reference",
                       beta=0.15, dynamic_ref=True, lean_norm="zscore",
                       lean_max=0.25, modulate_metric="stable_rank",
                       total_steps=20, adjust_lr_fn=None)
    g = torch.randn(8, 12)
    w_live.grad = g.clone()
    w_dead.grad = torch.zeros(8, 12)
    opt.step()
    # dead param: weight decay still applied, update zero, no proxy sample
    assert torch.equal(w_dead.detach(), torch.zeros(8, 12))  # was zero anyway
    assert math.isnan(opt.state[w_dead]["last_p"])
    assert math.isfinite(opt.state[w_live]["last_p"])
    assert opt._proxy_ema is not None and math.isfinite(opt._proxy_ema)

    # from step 2 the dead param has momentum; everyone follows the schedule
    for k in range(2, 21):
        w_live.grad = g.clone()
        w_dead.grad = g.clone()
        opt.step()
    assert math.isfinite(opt._proxy_ema)
    p_t = logistic_schedule_p(20, 20, -0.25, 1.0, 0.04, 0.04)
    for w in (w_live, w_dead):
        p_final = opt.state[w]["last_p"]
        assert abs(p_final - p_t) <= 0.25 + 1e-9, \
            f"p={p_final} should track the schedule {p_t:.3f} within lean_max"


def test_global_schedule_anneals():
    """p_t runs p_max -> p_min across total_steps (step counter starts at 1)."""
    w = torch.zeros(6, 8, requires_grad=True)
    opt = DynMuonRoute([w], routing_mode="global_schedule", compute_mode="reference",
                       total_steps=4, adjust_lr_fn=None)
    ps = []
    for _ in range(4):
        w.grad = torch.randn(6, 8)
        opt.step()
        ps.append(opt.state[w]["last_p"])
    assert ps[0] > ps[-1]
    assert ps[0] <= 1.0 + 1e-6 and ps[-1] >= -0.25 - 1e-6


def test_snr_ema_proxy_separates_clean_from_noisy_gradients():
    """The EMA SNR proxy is high for a repeated gradient and low for a
    sign-alternating one."""
    def final_gamma(flip):
        torch.manual_seed(12)
        w = torch.zeros(8, 12, requires_grad=True)
        opt = DynMuonRoute([w], lr=0.01, routing_mode="snr_ema",
                           compute_mode="reference", adjust_lr_fn=None,
                           snr_ema_decay=0.8, track_proxies=False)
        g = torch.randn(8, 12)
        for step in range(12):
            sign = -1.0 if (flip and step % 2 == 1) else 1.0
            w.grad = sign * g.clone()
            opt.step()
        return opt.state[w]["last_gamma_ema"]

    clean, noisy = final_gamma(flip=False), final_gamma(flip=True)
    assert clean > 5.0, f"clean gradient should have high EMA SNR, got {clean}"
    assert noisy < 1.0, f"alternating gradient should have low EMA SNR, got {noisy}"
    # routing orientation: omega < 0 maps the noisy (low) proxy toward p_max
    p_noisy = logistic_route(noisy, -0.25, 1.0, mu=1.0, omega=-0.5)
    p_clean = logistic_route(clean, -0.25, 1.0, mu=1.0, omega=-0.5)
    assert p_noisy > p_clean


# ---------------------------------------------------------------------------
# RelMuon scales
# ---------------------------------------------------------------------------

def test_relmuon_zero_weight_modes_are_muon_like():
    """All normalized scale modes give all-ones scales on zero weights."""
    W = torch.zeros(6, 10)
    for mode in ("log1p", "rms", "complete"):
        scales = relmuon_weight_scales(W, scale_mode=mode)
        assert torch.allclose(scales, torch.ones_like(scales))
    Vh = torch.linalg.svd(torch.randn(6, 10), full_matrices=False)[2]
    aligned = relmuon_aligned_scales(W, Vh)
    assert torch.allclose(aligned, torch.ones_like(aligned))


def test_relmuon_aligned_scales_measure_weight_action():
    """With Vh = I, the aligned scales are the RMS-normalized log1p of the
    weight's column norms (the weight's action along the coordinate axes)."""
    torch.manual_seed(13)
    W = torch.randn(5, 5)
    Vh = torch.eye(5)
    scales = relmuon_aligned_scales(W, Vh, eps=0.0)
    expected = torch.log1p(torch.linalg.norm(W, dim=0))
    expected = expected / torch.sqrt(torch.mean(expected.square()))
    assert torch.allclose(scales, expected, atol=1e-6)


def test_relmuon_scale_cap_bounds_update_spectral_norm():
    """scale_cap clamps the update's singular values (trust cap for the
    unnormalized 'complete' mode)."""
    torch.manual_seed(14)
    W = torch.randn(8, 8) * 5.0
    grad = torch.randn(8, 8)
    momentum = torch.zeros(8, 8)
    update = relmuon_update(grad, W, momentum, mu=0.0, nesterov=False,
                            scale_mode="complete", scale_cap=1.5)
    sv = torch.linalg.svdvals(update.float())
    assert float(sv.max()) <= 1.5 + 1e-4
