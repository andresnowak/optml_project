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
from dynmuon import MemoryLogger, load_config, train  # noqa: E402

TARGETS = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
OUT_DIR = os.path.join("results", "exp1_spectral_evolution")


def _series_for(history: dict, suffix: str):
    """Average p trajectory across all layers whose name ends with ``suffix``."""
    keys = [k for k in history if k.startswith("route/p/") and k.endswith(suffix)]
    if not keys:
        return None
    steps = [s for s, _ in history[keys[0]]]
    n = len(keys)
    avg = []
    for i in range(len(steps)):
        avg.append(sum(history[k][i][1] for k in keys) / n)
    return steps, avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp1_spectral.yaml")
    ap.add_argument("--model", choices=["small", "gpt124m"])
    ap.add_argument("--max-steps", dest="max_steps", type=int)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    runs = {}
    for mode in ("global_schedule", "stable_rank"):
        cfg = load_config(args.config, {"model": args.model, "max_steps": args.max_steps,
                                        "routing_mode": mode, "wandb": False})
        logger = MemoryLogger()
        print(f"\n=== training routing_mode={mode} ===")
        train(cfg, logger=logger)
        runs[mode] = logger.history

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, suffix in zip(axes.flat, TARGETS):
        for mode, hist in runs.items():
            s = _series_for(hist, suffix)
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
