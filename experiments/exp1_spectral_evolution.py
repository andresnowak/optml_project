"""Experiment 1 — modded-nanoGPT spectral evolution.

Trains the GPT twice on WikiText-103 — once with the global time schedule, once
with the Stable-Rank router — and plots the per-layer ``p_{t,l}`` trajectories for
``c_attn``, ``c_proj``, ``mlp.c_fc`` and ``mlp.c_proj``.

Scientific question: do Attention layers reject negative p (stay p >= 0) while MLP
layers transition toward p = -0.25?

Usage:
    python experiments/exp1_spectral_evolution.py --config configs/gpt124m.yaml --model small --max-steps 400
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynmuon import analysis, build_arm_logger, load_config, train  # noqa: E402

TARGETS = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
OUT_DIR = os.path.join("results", "exp1_spectral_evolution")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp1_spectral.yaml")
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", dest="wandb_group", default="exp1_spectral")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {}
    for mode in ("global_schedule", "schedule_modulated"):
        cfg = load_config(args.config, {"model": args.model, "max_steps": args.max_steps,
                                        "routing_mode": mode})
        print(f"\n=== training routing_mode={mode} ===")
        logger, mem, wb = build_arm_logger(cfg, args.wandb, f"exp1_{mode}", args.wandb_group)
        train(cfg, logger=logger)
        if wb:
            wb.finish()
        runs[mode] = mem.history
        analysis.dump_history(mem.history, os.path.join(OUT_DIR, f"history_{mode}.json"))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, suffix in zip(axes.flat, TARGETS):
        for mode, hist in runs.items():
            s = analysis.mean_series_by_suffix(hist, suffix)
            if s:
                ax.plot(s[0], s[1], label=mode)
        ax.axhline(0.0, color="grey", ls=":", lw=0.8)
        ax.axhline(-0.25, color="red", ls=":", lw=0.8)
        ax.set_title(suffix)
        ax.set_ylabel("p")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.suptitle("Experiment 1 — per-layer spectral exponent p_{t,l}")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "p_trajectories.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
