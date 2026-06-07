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
from dynmuon import analysis, build_arm_logger, load_config, train  # noqa: E402

OUT_DIR = os.path.join("results", "exp2_noise_injection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2_noise.yaml")
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float)
    ap.add_argument("--modulate-metric", dest="modulate_metric",
                    choices=["stable_rank", "snr", "alignment"])
    ap.add_argument("--beta", type=float)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", dest="wandb_group", default="exp2_noise")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {}
    for mode in ("global_schedule", "schedule_modulated"):
        cfg = load_config(args.config, {"model": args.model, "max_steps": args.max_steps,
                                        "routing_mode": mode, "noise_lambda": args.noise_lambda,
                                        "modulate_metric": args.modulate_metric, "beta": args.beta})
        print(f"\n=== training routing_mode={mode} (lambda={cfg.get('noise_lambda')}) ===")
        logger, mem, wb = build_arm_logger(cfg, args.wandb, f"exp2_{mode}", args.wandb_group)
        train(cfg, logger=logger)
        if wb:
            wb.finish()
        runs[mode] = mem.history
        analysis.dump_history(mem.history, os.path.join(OUT_DIR, f"history_{mode}.json"))

    fig, (ax_loss, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))
    for mode, hist in runs.items():
        if "train/loss" in hist:
            steps, vals = zip(*hist["train/loss"])
            finite = [v if math.isfinite(v) else float("nan") for v in vals]
            ax_loss.plot(steps, finite, label=mode)
        # mean p across all mlp.c_fc layers as a representative trajectory
        s = analysis.mean_series_by_suffix(hist, "mlp.c_fc.weight")
        if s:
            ax_p.plot(s[0], s[1], label=mode)

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
