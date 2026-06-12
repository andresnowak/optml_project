"""Plot Newton-Schulz singular-value response curves.

This is a scalar playground for the matrix recurrence: if
X = U diag(sigma) V^T, the NS updates keep U,V and only transform sigma.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


QUINTIC_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

# Experimental slow-push coefficients. Each row has a + b + c = 1, so
# sigma=1 is fixed, but the small-sigma slope is only 1.2 per step.
GENTLE10_NS_COEFFS = (
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
    (1.2, -0.2, 0.0),
)

COEFF_SETS = {
    "reference5": QUINTIC_NS_COEFFS,
    "gentle10": GENTLE10_NS_COEFFS,
}


def cubic_response(sigma: np.ndarray, steps: int) -> np.ndarray:
    y = sigma.copy()
    for _ in range(steps):
        y = 1.5 * y - 0.5 * y**3
    return y


def ns_response(
    sigma: np.ndarray,
    coeffs: tuple[tuple[float, float, float], ...],
    steps: int,
) -> np.ndarray:
    y = sigma.copy()
    for a, b, c in coeffs[:steps]:
        y = a * y + b * y**3 + c * y**5
    return y


def ns_responses_by_step(
    sigma: np.ndarray,
    coeffs: tuple[tuple[float, float, float], ...],
    steps: int,
) -> list[np.ndarray]:
    ys = []
    y = sigma.copy()
    for a, b, c in coeffs[:steps]:
        y = a * y + b * y**3 + c * y**5
        ys.append(y.copy())
    return ys


def damped_polar(sigma: np.ndarray, lam: float) -> np.ndarray:
    return sigma / np.sqrt(sigma**2 + lam**2)


def gated(y: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    gate = sigma**2 / (sigma**2 + tau**2)
    return gate * y


def interp_gate(sigma: np.ndarray, tau: float, mult: float, power: float) -> np.ndarray:
    alpha = mult * tau
    return sigma**power / (sigma**power + alpha**power)


def interp_gated_target(
    sigma: np.ndarray,
    tau: float,
    mult: float,
    power: float,
) -> np.ndarray:
    h = interp_gate(sigma, tau, mult, power)
    return sigma + h * (1.0 - sigma)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coeff-set", choices=COEFF_SETS, default="reference5",
                    help="reference5 is the tuned Muon schedule; gentle10 is experimental")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="ridge scale for damped polar")
    ap.add_argument("--tau", type=float, default=1e-3,
                    help="gate scale applied to NS outputs")
    ap.add_argument("--gate-mult", type=float, default=10.0,
                    help="new interpolation gate midpoint is gate_mult * tau")
    ap.add_argument("--gate-power", type=float, default=6.0,
                    help="sharpness exponent for the new interpolation gate")
    ap.add_argument("--sigma-min", type=float, default=1e-12,
                    help="smallest input singular value shown on the x-axis")
    ap.add_argument("--sigma-max", type=float, default=1.0,
                    help="largest input singular value shown on the x-axis")
    ap.add_argument("--out", type=Path, default=Path("results/ns_singular_map.png"))
    args = ap.parse_args()

    coeffs = COEFF_SETS[args.coeff_set]
    if not 1 <= args.steps <= len(coeffs):
        raise ValueError(
            f"--steps must be in [1, {len(coeffs)}] for --coeff-set={args.coeff_set}"
        )

    sigma = np.logspace(np.log10(args.sigma_min), np.log10(args.sigma_max), 2400)
    cubic = cubic_response(sigma, args.steps)
    ns = ns_response(sigma, coeffs, args.steps)
    soft = damped_polar(sigma, args.lam)
    gated_ns = gated(ns, sigma, args.tau)
    interp_target = interp_gated_target(sigma, args.tau, args.gate_mult, args.gate_power)
    ns_steps = ns_responses_by_step(sigma, coeffs, args.steps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.loglog(sigma, sigma, "k--", lw=1, label="identity")
    ax.loglog(sigma, cubic, label=f"cubic NS, {args.steps} steps")
    ax.loglog(sigma, ns, label=f"{args.coeff_set}, {args.steps} steps")
    ax.loglog(sigma, soft, label=rf"damped polar, $\lambda={args.lam:g}$")
    ax.loglog(sigma, gated_ns, label=rf"gated {args.coeff_set}, $\tau={args.tau:g}$")
    ax.loglog(
        sigma,
        interp_target,
        label=rf"interp-gated {args.coeff_set}, $\alpha={args.gate_mult * args.tau:g}$, $r={args.gate_power:g}$",
    )
    ax.axvline(args.lam, color="tab:green", lw=1, alpha=0.35)
    ax.axvline(args.tau, color="tab:red", lw=1, alpha=0.35)
    ax.axvline(args.gate_mult * args.tau, color="tab:purple", lw=1, alpha=0.35)
    ax.set_xlabel(r"input singular value $\sigma$")
    ax.set_ylabel(r"output scale $f(\sigma)$")
    ax.set_title("Singular-value map")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    powers = [2, 4, 6, 12]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(powers)))
    for power, color in zip(powers, colors):
        target_i = interp_gated_target(sigma, args.tau, args.gate_mult, power)
        ax.loglog(sigma, target_i, color=color, label=rf"$r={power}$")
    ax.loglog(sigma, sigma, "k--", lw=1, label="identity")
    ax.axvline(args.tau, color="tab:red", lw=1, alpha=0.35)
    ax.axvline(args.gate_mult * args.tau, color="tab:purple", lw=1, alpha=0.35)
    ax.set_xlabel(r"input singular value $\sigma$")
    ax.set_ylabel(r"target output scale")
    ax.set_title(rf"Target-to-one gates, $\alpha={args.gate_mult * args.tau:g}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    for name, y in (
        (f"cubic {args.steps}", cubic),
        (f"{args.coeff_set} {args.steps}", ns),
        ("damped polar", soft),
        (f"gated {args.coeff_set}", gated_ns),
        ("interp-gated NS", interp_target),
    ):
        ax.semilogx(sigma, y / sigma, label=name)
    ax.axhline(1.0, color="k", lw=1, alpha=0.4)
    ax.set_xlabel(r"input singular value $\sigma$")
    ax.set_ylabel(r"amplification $f(\sigma)/\sigma$")
    ax.set_title("Tiny-direction amplification")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)

    probes = np.array([1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5])
    print(f"wrote {args.out}")
    print(f"coefficient set: {args.coeff_set}")
    print(
        "sigma      cubic       ns-map      damped      old-gated   interp-target"
    )
    for s, c, q, d, g in zip(
        probes,
        cubic_response(probes, args.steps),
        ns_response(probes, coeffs, args.steps),
        damped_polar(probes, args.lam),
        gated(ns_response(probes, coeffs, args.steps), probes, args.tau),
    ):
        target = interp_gated_target(np.array([s]), args.tau, args.gate_mult, args.gate_power)[0]
        print(
            f"{s:8.1e}  {c:10.3e}  {q:10.3e}  {d:10.3e}  {g:10.3e}  {target:13.3e}"
        )


if __name__ == "__main__":
    main()
