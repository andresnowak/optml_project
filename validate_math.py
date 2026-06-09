"""Unit tests for the DynMuon-Route spectral math.

Validates that the equivalent-factorization identities of Part 1.1 reproduce the
exact SVD spectral-shaping ``D(p) = U Σ^p Vᵀ`` across random matrices and
fractional exponents, that Newton-Schulz recovers the polar factor, and that the
optimizer's fast (``ns``) path matches its exact (``svd``) path.

Run with:  pytest validate_math.py
"""

from __future__ import annotations

import math

import pytest
import torch

from src import DynMuonRoute, logistic_route, newton_schulz
from src.optimizers.dynmuon import quintic_newton_schulz
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
    """The tuned 5-step quintic iteration (reference DynMuon) is a coarse but
    usable approximation of the polar factor."""
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


@pytest.mark.parametrize("ns_variant,tol", [("cubic", 5e-3), ("quintic", 5e-2)])
def test_optimizer_ns_matches_svd(ns_variant, tol):
    """A single DynMuonRoute step in ns mode matches svd mode (global schedule
    fixes p so both paths shape the same exponent). The cubic iteration (with
    many steps) is near-exact; the tuned quintic is a coarser 5-step approx."""
    torch.manual_seed(1)
    g = torch.randn(20, 28, dtype=torch.float64)

    def run(mode):
        w = torch.zeros(20, 28, dtype=torch.float64, requires_grad=True)
        opt = DynMuonRoute([w], lr=0.1, momentum=0.0, nesterov=False,
                           routing_mode="global_schedule", compute_mode=mode,
                           ns_variant=ns_variant, ns_steps=30, adjust_lr_fn=None,
                           total_steps=4)
        w.grad = g.clone()
        opt.step()
        return w.detach().clone()

    d_svd, d_ns = run("svd"), run("ns")
    rel = torch.linalg.norm(d_ns - d_svd) / torch.linalg.norm(d_svd)
    assert rel < tol, f"ns({ns_variant}) vs svd relative error too large: {rel}"


def test_logistic_bounds_and_orientation():
    pmin, pmax = -0.25, 1.0
    # increasing in x for omega > 0
    lo = logistic_route(-1e6, pmin, pmax, mu=0.0, omega=1.0)
    hi = logistic_route(+1e6, pmin, pmax, mu=0.0, omega=1.0)
    assert math.isclose(lo, pmin, abs_tol=1e-6) and math.isclose(hi, pmax, abs_tol=1e-6)
    # decreasing in x for omega < 0 (SNR orientation)
    lo2 = logistic_route(-1e6, pmin, pmax, mu=0.0, omega=-1.0)
    hi2 = logistic_route(+1e6, pmin, pmax, mu=0.0, omega=-1.0)
    assert lo2 > hi2
    # always within [pmin, pmax]
    for x in torch.linspace(-50, 50, 200):
        p = logistic_route(float(x), pmin, pmax, mu=3.0, omega=2.0)
        assert pmin - 1e-9 <= p <= pmax + 1e-9


@pytest.mark.parametrize("fixed_p,name", [(0.0, "muon"), (1.0, "sgd")])
def test_fixed_mode_constant_exponent(fixed_p, name):
    """routing_mode='fixed' shapes every step at the same exponent: p=0 is Muon
    (polar factor, singular values -> 1), p=1 leaves the gradient unshaped."""
    torch.manual_seed(2)
    w = torch.zeros(8, 12, dtype=torch.float64, requires_grad=True)
    opt = DynMuonRoute([w], lr=0.1, momentum=0.0, nesterov=False,
                       routing_mode="fixed", fixed_p=fixed_p, compute_mode="svd",
                       adjust_lr_fn=None)
    w.grad = torch.randn(8, 12, dtype=torch.float64)
    opt.step()
    assert opt.state[w]["last_p"] == fixed_p
    update = -w.detach()                     # lr=0.1 -> update = 0.1 * D(p)
    sv = torch.linalg.svdvals(update / 0.1)  # singular values of D(p)
    if name == "muon":                       # D(0) = U Vᵀ -> all singular values 1
        assert torch.allclose(sv, torch.ones_like(sv), atol=1e-6)


def test_schedule_modulated_reduces_to_schedule_when_beta_zero():
    """schedule_modulated with beta=0 equals the plain global schedule; with
    beta>0 it deviates per layer but stays within [p_min, p_max]."""
    torch.manual_seed(3)
    g = torch.randn(10, 16)

    def run_p(beta):
        w = torch.zeros(10, 16, requires_grad=True)
        opt = DynMuonRoute([w], routing_mode="schedule_modulated", compute_mode="ns",
                           beta=beta, ref=2.5, modulate_metric="stable_rank",
                           total_steps=8, adjust_lr_fn=None)
        ps = []
        for _ in range(4):
            w.grad = g.clone()
            opt.step()
            ps.append(opt.state[w]["last_p"])
        return ps

    # Reference schedule values for steps 0..3.
    sched = DynMuonRoute([torch.zeros(2, 2, requires_grad=True)],
                         routing_mode="global_schedule", total_steps=8)
    sched_vals = []
    for _ in range(4):
        sched_vals.append(sched._p_schedule(sched.param_groups[0]))
        sched._step_count += 1

    assert all(abs(a - b) < 1e-6 for a, b in zip(run_p(0.0), sched_vals))   # beta=0 == schedule
    assert all(-0.25 - 1e-9 <= p <= 1.0 + 1e-9 for p in run_p(0.2))         # beta>0 in range


def test_global_schedule_anneals():
    """p_t runs 1.0 -> -0.25 across total_steps."""
    w = torch.zeros(6, 8, requires_grad=True)
    opt = DynMuonRoute([w], routing_mode="global_schedule", compute_mode="ns",
                       total_steps=4, adjust_lr_fn=None)
    ps = []
    for _ in range(4):
        w.grad = torch.randn(6, 8)
        opt.step()
        ps.append(opt.state[w]["last_p"])
    assert ps[0] > ps[-1]
    assert ps[0] <= 1.0 + 1e-6 and ps[-1] >= -0.25 - 1e-6
