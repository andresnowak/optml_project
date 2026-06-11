"""Classic baselines (spec Part 2.2): AdamW vs Muon (p=0) vs DynMuon (global
schedule) vs DynMuon-Route (stable-rank router).

Trains all four methods on the same data/seed, plots validation-loss curves, and
computes step efficiency = how many fewer steps each method needs to reach a
common target validation loss. The validation target for the project is that the
global DynMuon schedule reaches the target in ~10-26% fewer steps than Muon; this
script reports that number and the router's gain on top.

Usage:
    python experiments/baselines_step_efficiency.py
    python experiments/baselines_step_efficiency.py --train-steps 20000
    python experiments/baselines_step_efficiency.py --target-loss 4.0
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analysis, build_arm_logger, load_config, train  # noqa: E402

METHODS = [
    ("adamw", "configs/adamw.yaml"),
    ("muon", "configs/muon.yaml"),
    ("dynmuon", "configs/dynmuon.yaml"),
    ("dynmuon_route", "configs/route.yaml"),
    ("random_route", "configs/random_spectrum.yaml"),  # sanity check: random routing should be worse than stable-rank
    ("relmuon", "configs/relmuon.yaml"),
    ("inverted_spectrum", "configs/inverted_spectrum.yaml")
]
OUT_DIR = os.path.join("results", "baselines")
VAL_KEY = "val/loss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", dest="train_steps", type=int)
    ap.add_argument("--val-loss-every", dest="val_loss_every", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--target-loss", dest="target_loss", type=float,
                    help="common val-loss target; default = the loss every method reaches")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", dest="wandb_group", default="baselines")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    common = {"train_steps": args.train_steps, "val_loss_every": args.val_loss_every,
              "seed": args.seed}

    runs: dict[str, dict] = {}
    for name, cfg_path in METHODS:
        cfg = load_config(cfg_path, common)
        print(f"\n=== {name} ===")
        logger, mem, wb = build_arm_logger(cfg, args.wandb, name, args.wandb_group)
        train(cfg, logger=logger)
        if wb:
            wb.finish()
        runs[name] = mem.history
        analysis.dump_history(mem.history, os.path.join(OUT_DIR, f"history_{name}.json"))

    # Common target: either user-given, or the worst of the per-method best val
    # losses (so every method is able to reach it).
    best = {n: analysis.min_value(h, VAL_KEY) for n, h in runs.items()}
    reached = {n: v for n, v in best.items() if v is not None}
    target = args.target_loss if args.target_loss is not None else max(reached.values())

    steps = {n: analysis.steps_to_target(h, VAL_KEY, target) for n, h in runs.items()}

    def pct_fewer(a: str, b: str) -> str:
        """% fewer steps for `b` vs `a` to reach the target."""
        if steps.get(a) and steps.get(b):
            return f"{100 * (steps[a] - steps[b]) / steps[a]:+.1f}%"
        return "n/a"

    # -- summary.md ----------------------------------------------------------
    lines = [
        "# Baseline step-efficiency", "",
        f"Common target val loss: **{target:.4f}**", "",
        "| method | best val loss | steps to target |",
        "|--------|---------------|-----------------|",
    ]
    for n, _ in METHODS:
        s = steps[n]
        lines.append(f"| {n} | {best[n]:.4f} | {s if s is not None else 'not reached'} |")
    lines += [
        "", "## Step efficiency (fewer steps to target)",
        f"- DynMuon vs Muon: **{pct_fewer('muon', 'dynmuon')}**  (project target: ~10-26%)",
        f"- DynMuon-Route vs Muon: **{pct_fewer('muon', 'dynmuon_route')}**",
        f"- DynMuon-Route vs DynMuon: **{pct_fewer('dynmuon', 'dynmuon_route')}**",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)

    # -- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, _ in METHODS:
        series = runs[n].get(VAL_KEY)
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker="o", ms=3, label=n)
    ax.axhline(target, color="grey", ls="--", lw=0.8, label=f"target {target:.3f}")
    ax.set_xlabel("step"); ax.set_ylabel("val loss")
    ax.set_title("Baselines — validation loss"); ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "loss_curves.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
