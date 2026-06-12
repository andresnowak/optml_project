"""Numerical parity tests against the released DynMuon reference.

The functions in the ``reference vendored code`` section below are copied
verbatim from github.com/fzwark/DynMuon (``dynmuon/dynmuon.py`` and
``dynmuon/newton_schulz_triton.py``), with exactly two mechanical adaptations:

  * ``newton_schulz_triton`` is replaced by the repo's own non-Triton
    equivalent ``zeropower_via_newtonschulz5`` (same coefficients, same dtype
    handling; the repo documents them as interchangeable), since Triton is
    CUDA-only.
  * the ``@torch.compile`` decorators are dropped (compilation does not change
    the math; these tests run eager on CPU).

``test_full_step_parity_*`` drives the vendored pre-orthogonalize → transform
→ post-orthogonalize pipeline exactly the way the reference optimizer does
(per-group step counter incremented before scheduling, momentum in parameter
dtype, bfloat16 transform input, tensor-scalar hyperparameters) and asserts
our ``DynMuonRoute(routing_mode="global_schedule", compute_mode="reference")``
produces the same parameter trajectories.

Run with:  pytest validate_reference.py
"""

from __future__ import annotations

import math

import pytest
import torch

from src.optimizers.dynmuon import (
    DynMuonRoute,
    dynmuon_spectral_transform,
    logistic_schedule_p,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# reference vendored code (github.com/fzwark/DynMuon) — do not edit
# ---------------------------------------------------------------------------

_GLOBAL_P = 1.0


def set_global_p(p: float):
    global _GLOBAL_P
    _GLOBAL_P = float(p)


def ref_zeropower_via_newtonschulz5(G: torch.Tensor, epsilon: float = 1e-7):
    """
    Reference implementation of Newton-Schulz without Triton.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def ref_fast_spectral(
    G: torch.Tensor,
    p: float,
    epsilon: float = 1e-7,
    order: int = 2,
):
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
    Y_mu = ref_zeropower_via_newtonschulz5(Xn, epsilon=epsilon).to(torch.float32)

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

    # Restore original shape
    if transposed:
        U = U.mT

    return U.to(orig_dtype)


def ref_dynmuon_spectral_transform(G: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    if _GLOBAL_P >= 0.25:
        return G

    if _GLOBAL_P >= 0.0:
        return ref_zeropower_via_newtonschulz5(G, epsilon=epsilon)

    return ref_fast_spectral(G, p=_GLOBAL_P)


class Logistic_Scheduler:
    def __init__(self, p_max=1.0, p_min=-0.25, tau_ratio=0.02, width_ratio=0.08):
        self.p_max = p_max
        self.p_min = p_min
        self.tau_ratio = tau_ratio
        self.width_ratio = width_ratio

    def get_p(self, step, total_steps=10000):
        q_t = step / float(total_steps)
        u = (q_t - self.tau_ratio) / max(self.width_ratio, 1e-8)
        anneal = 1.0 / (1.0 + math.exp(u))
        return self.p_min + (self.p_max - self.p_min) * anneal


def ref_muon_update_pre_orthogonalize(G, M, momentum, nesterov):
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


def ref_muon_update_post_orthogonalize(X, U, base_lr, adjusted_lr, weight_decay):
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    U = torch._foreach_mul(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def ref_adjust_lr_spectral_norm(lr, param_shape, flatten):
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr


# ---------------------------------------------------------------------------
# reference optimizer driver (mirrors DynMuon.step / _create_muon_tasks)
# ---------------------------------------------------------------------------

class ReferenceDynMuon:
    """Drives the vendored functions exactly like the reference optimizer:
    per-group step incremented before scheduling, momentum buffers in the
    parameter dtype, bfloat16 transform input, tensor-scalar hyperparameters.
    """

    def __init__(self, params, lr, mu, weight_decay, nesterov, total_steps,
                 p_max=1.0, p_min=-0.25, tau_ratio=0.04, width_ratio=0.04,
                 epsilon=1e-8):
        self.params = list(params)
        self.momenta = [torch.zeros_like(p) for p in self.params]
        self.lr = torch.tensor(lr)
        self.mu = torch.tensor(mu)
        self.weight_decay = torch.tensor(weight_decay)
        self.nesterov = nesterov
        self.epsilon = torch.tensor(epsilon)
        self.step_count = 0
        self.total_steps = total_steps
        self.scheduler = Logistic_Scheduler(p_max, p_min, tau_ratio, width_ratio)

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        p_t = self.scheduler.get_p(self.step_count, self.total_steps)
        set_global_p(p_t)
        for X, M in zip(self.params, self.momenta):
            U = ref_muon_update_pre_orthogonalize(
                [X.grad], [M], momentum=self.mu, nesterov=self.nesterov,
            )
            U[0] = ref_dynmuon_spectral_transform(U[0], epsilon=self.epsilon)
            adjusted_lr = ref_adjust_lr_spectral_norm(self.lr, X.shape, flatten=False)
            ref_muon_update_post_orthogonalize(
                [X], U, base_lr=self.lr, adjusted_lr=adjusted_lr,
                weight_decay=self.weight_decay,
            )


# ---------------------------------------------------------------------------
# parity tests
# ---------------------------------------------------------------------------

SCHEDULES = [
    dict(p_max=1.0, p_min=-0.25, tau_ratio=0.04, width_ratio=0.04),   # paper/README
    dict(p_max=1.0, p_min=-0.25, tau_ratio=0.02, width_ratio=0.08),   # class default
]


@pytest.mark.parametrize("sched", SCHEDULES)
def test_schedule_matches_reference(sched):
    """Our logistic_schedule_p equals the reference Logistic_Scheduler.get_p
    at every step (both use step counters that start at 1)."""
    ref = Logistic_Scheduler(**sched)
    total = 200
    for step in range(1, total + 1):
        ours = logistic_schedule_p(step, total, sched["p_min"], sched["p_max"],
                                   sched["tau_ratio"], sched["width_ratio"])
        assert ours == ref.get_p(step, total)


@pytest.mark.parametrize("shape", [(16, 16), (8, 24), (24, 8)])
@pytest.mark.parametrize("p", [1.0, 0.6, 0.3, 0.25, 0.1, 0.0, -0.1, -0.25])
def test_transform_matches_reference(shape, p):
    """Our explicit-p dynmuon_spectral_transform is bitwise identical to the
    reference global-p transform on bfloat16 inputs across all three phases."""
    torch.manual_seed(7)
    G = torch.randn(*shape).to(torch.bfloat16)
    set_global_p(p)
    ref = ref_dynmuon_spectral_transform(G.clone(), epsilon=1e-8)
    ours = dynmuon_spectral_transform(G.clone(), p, epsilon=1e-8)
    assert ref.dtype == ours.dtype
    assert torch.equal(ref, ours), f"transform mismatch at p={p}, shape={shape}"


def _run_pair(shapes, steps, *, lr, mu, weight_decay, tau_ratio, width_ratio,
              total_steps, seed=11):
    """Run ReferenceDynMuon and DynMuonRoute on identical params/grads."""
    torch.manual_seed(seed)
    inits = [torch.randn(*s) for s in shapes]
    grads = [[torch.randn(*s) for s in shapes] for _ in range(steps)]

    ref_params = [init.clone().requires_grad_(True) for init in inits]
    ref_opt = ReferenceDynMuon(ref_params, lr=lr, mu=mu, weight_decay=weight_decay,
                               nesterov=True, total_steps=total_steps,
                               tau_ratio=tau_ratio, width_ratio=width_ratio)

    our_params = [init.clone().requires_grad_(True) for init in inits]
    our_opt = DynMuonRoute(
        our_params, lr=lr, momentum=mu, nesterov=True, weight_decay=weight_decay,
        routing_mode="global_schedule", compute_mode="reference",
        adjust_lr_fn="spectral_norm", eps=1e-8, tau_ratio=tau_ratio,
        width_ratio=width_ratio, total_steps=total_steps, track_proxies=False,
    )

    for step_grads in grads:
        for p_ref, p_ours, g in zip(ref_params, our_params, step_grads):
            p_ref.grad = g.clone()
            p_ours.grad = g.clone()
        ref_opt.step()
        our_opt.step()
    return ref_params, our_params


def test_full_step_parity_exact_scalars():
    """With hyperparameters exactly representable in float32 (so scalar
    rounding cannot differ between tensor and Python-float arithmetic), the
    full trajectories are bitwise identical across all three schedule phases.

    tau=0.5, width=0.125, T=8 sweeps p through raw momentum (p >= 0.25 at
    steps 1-4), Newton-Schulz Muon (steps 5-6), and fast_spectral (steps 7-8).
    """
    ref_params, our_params = _run_pair(
        [(16, 16), (8, 32), (32, 8)], steps=8,
        lr=0.5, mu=0.5, weight_decay=0.25,
        tau_ratio=0.5, width_ratio=0.125, total_steps=8,
    )
    for p_ref, p_ours in zip(ref_params, our_params):
        assert torch.equal(p_ref.detach(), p_ours.detach())


def test_full_step_parity_realistic_scalars():
    """With the paper hyperparameters (lr=0.01, mu=0.95, wd=0.01, tau=w=0.04)
    the trajectories agree to floating-point noise (scalar products are
    computed once in float64 and once in float32 tensor arithmetic, which can
    differ by ~1 ulp; any algorithmic mismatch would be orders of magnitude
    larger)."""
    ref_params, our_params = _run_pair(
        [(12, 20), (20, 12)], steps=10,
        lr=0.01, mu=0.95, weight_decay=0.01,
        tau_ratio=0.04, width_ratio=0.04, total_steps=10,
    )
    for p_ref, p_ours in zip(ref_params, our_params):
        ref, ours = p_ref.detach(), p_ours.detach()
        denom = ref.abs().max().clamp(min=1e-12)
        rel = (ours - ref).abs().max() / denom
        assert rel < 1e-4, f"trajectory diverged: max rel err {rel}"


def test_muon_endpoint_parity():
    """fixed_p=0 through the reference compute path equals the reference
    Newton-Schulz Muon update exactly."""
    torch.manual_seed(3)
    init = torch.randn(10, 14)
    g = torch.randn(10, 14)

    w = init.clone().requires_grad_(True)
    opt = DynMuonRoute([w], lr=0.25, momentum=0.5, nesterov=True, weight_decay=0.0,
                       routing_mode="fixed", fixed_p=0.0, compute_mode="reference",
                       adjust_lr_fn="spectral_norm", eps=1e-8, track_proxies=False)
    w.grad = g.clone()
    opt.step()

    M = torch.zeros_like(init)
    U = ref_muon_update_pre_orthogonalize([g.clone()], [M], torch.tensor(0.5), True)
    D = ref_zeropower_via_newtonschulz5(U[0], epsilon=torch.tensor(1e-8))
    adjusted = ref_adjust_lr_spectral_norm(torch.tensor(0.25), init.shape, False)
    expected = init - (D * adjusted)
    assert torch.equal(w.detach(), expected)


def test_state_dict_resume_matches_uninterrupted():
    """Checkpoint resume must continue the p schedule (and trajectories)
    exactly where they stopped — regression test for the step counter not
    being part of the optimizer state."""
    torch.manual_seed(5)
    init = torch.randn(8, 12)
    grads = [torch.randn(8, 12) for _ in range(6)]

    def make(w):
        return DynMuonRoute([w], lr=0.02, momentum=0.95, nesterov=True,
                            weight_decay=0.01, routing_mode="global_schedule",
                            compute_mode="reference", total_steps=6,
                            track_proxies=False)

    # Uninterrupted run.
    w_full = init.clone().requires_grad_(True)
    opt_full = make(w_full)
    for g in grads:
        w_full.grad = g.clone()
        opt_full.step()

    # Interrupted at step 3, resumed via state_dict round-trip.
    w_a = init.clone().requires_grad_(True)
    opt_a = make(w_a)
    for g in grads[:3]:
        w_a.grad = g.clone()
        opt_a.step()
    snapshot = opt_a.state_dict()

    w_b = w_a.detach().clone().requires_grad_(True)
    opt_b = make(w_b)
    opt_b.load_state_dict(snapshot)
    assert opt_b._step_count == 3
    for g in grads[3:]:
        w_b.grad = g.clone()
        opt_b.step()

    assert torch.equal(w_full.detach(), w_b.detach())
