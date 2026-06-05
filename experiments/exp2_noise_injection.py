"""Experiment 2 — spectral anisotropic noise-injection stress test.

Injects ``M += lambda * sigma_1 * z * u_1 v_1ᵀ`` (anisotropic noise aligned with
the top singular direction of the momentum buffer) and compares the Stable-Rank
router against the global time schedule. The router should detect the spike via a
dropping stable rank, route p_{t,l} negative to suppress it, and stay stable,
whereas the global schedule is expected to destabilize.

Usage:
    python experiments/exp2_noise_injection.py --config configs/gpt124m.yaml --model small --max-steps 300 --noise-lambda 3.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynmuon import MemoryLogger, load_config, train  # noqa: E402

OUT_DIR = os.path.join("results", "exp2_noise_injection")


def _scalar(history: dict, key: str):
    if key not in history:
        return None
    return zip(*history[key])  # (steps, values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2_noise.yaml")
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {}
    for mode in ("global_schedule", "stable_rank"):
        cfg = load_config(args.config, {"model": args.model, "max_steps": args.max_steps,
                                        "routing_mode": mode, "noise_lambda": args.noise_lambda,
                                        "wandb": False})
        logger = MemoryLogger()
        print(f"\n=== training routing_mode={mode} (lambda={cfg.get('noise_lambda')}) ===")
        train(cfg, logger=logger)
        runs[mode] = logger.history

    fig, (ax_loss, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, hist in runs.items():
        loss = _scalar(hist, "train/loss")
        if loss:
            steps, vals = list(loss[0]), list(loss[1])
            finite = [v if math.isfinite(v) else float("nan") for v in vals]
            ax_loss.plot(steps, finite, label=mode)
        # mean p across all mlp.c_fc layers as a representative trajectory
        pkeys = [k for k in hist if k.startswith("route/p/") and k.endswith("mlp.c_fc.weight")]
        if pkeys:
            steps = [s for s, _ in hist[pkeys[0]]]
            avg = [sum(hist[k][i][1] for k in pkeys) / len(pkeys) for i in range(len(steps))]
            ax_p.plot(steps, avg, label=mode)

    ax_loss.set_title("training loss under noise injection")
    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss"); ax_loss.legend()
    ax_p.axhline(-0.25, color="red", ls=":", lw=0.8)
    ax_p.set_title("mlp.c_fc routed exponent p"); ax_p.set_xlabel("step")
    ax_p.set_ylabel("p"); ax_p.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "noise_stability.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
