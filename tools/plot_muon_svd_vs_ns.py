"""Plot exact SVD-polar Muon versus finite-step Newton-Schulz Muon.

The repo's Muon baseline applies the quintic scalar map

    p(s) = 2s - 1.5s^3 + 0.5s^5

for ``ns_steps`` iterations after Frobenius normalization. Exact-polar Muon
instead maps every nonzero singular value to one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def muon_quintic_step(sigma: np.ndarray) -> np.ndarray:
    return 2.0 * sigma - 1.5 * sigma**3 + 0.5 * sigma**5


def muon_ns_by_step(sigma: np.ndarray, steps: int) -> list[np.ndarray]:
    ys = []
    y = sigma.copy()
    for _ in range(steps):
        y = muon_quintic_step(y)
        ys.append(y.copy())
    return ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--sigma-min", type=float, default=1e-12)
    ap.add_argument("--sigma-max", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("results/muon_svd_vs_ns_steps.png"))
    args = ap.parse_args()

    sigma = np.logspace(np.log10(args.sigma_min), np.log10(args.sigma_max), 2400)
    ys = muon_ns_by_step(sigma, args.steps)
    show_steps = [s for s in (1, 2, 3, 4, 5, 8, args.steps) if 1 <= s <= args.steps]
    show_steps = sorted(set(show_steps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(show_steps)))

    ax = axes[0]
    ax.loglog(sigma, sigma, "k--", lw=1, label="identity")
    ax.loglog(sigma, np.ones_like(sigma), color="tab:red", lw=2.0, label="exact SVD polar")
    for step, color in zip(show_steps, colors):
        ax.loglog(sigma, ys[step - 1], color=color, label=f"NS {step} steps")
    ax.set_xlabel(r"input singular value $\sigma$")
    ax.set_ylabel(r"output singular value")
    ax.set_title("Muon singular-value response")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogx(sigma, 1.0 / sigma, color="tab:red", lw=2.0, label="exact SVD polar")
    for step, color in zip(show_steps, colors):
        ax.semilogx(sigma, ys[step - 1] / sigma, color=color, label=f"NS {step} steps")
    ax.axhline(1.0, color="k", lw=1, alpha=0.4)
    ax.set_xlabel(r"input singular value $\sigma$")
    ax.set_ylabel(r"amplification $f(\sigma)/\sigma$")
    ax.set_title("Amplification")
    ax.set_ylim(0.8, 1e6)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)

    probes = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    probe_ys = muon_ns_by_step(probes, args.steps)
    print(f"wrote {args.out}")
    print("sigma      NS-5        NS-12       exact-SVD")
    for i, s in enumerate(probes):
        ns5 = probe_ys[min(5, args.steps) - 1][i]
        ns_last = probe_ys[-1][i]
        print(f"{s:8.1e}  {ns5:10.3e}  {ns_last:10.3e}  {1.0:10.3e}")


if __name__ == "__main__":
    main()
