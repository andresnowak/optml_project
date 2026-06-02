"""Gate sensitivity — how does SAV-gating's τ × window grid affect outcome?

Sweeps ``gate_threshold ∈ {0, 0.01, 0.05, 0.1, 0.2}`` against
``gate_window ∈ {5, 10, 20}`` at a fixed (experiment, lr). τ=0 reproduces the
paper-default (always-on SAV). For each cell the script reports
``score_losses`` (min over last 10%) so it's possible to read off the regime
where the gate helps vs. hurts.

Emits ``results/gate_sensitivity/<experiment>/{heatmap.png, results.json, summary.md}``.

Usage:
    uv run python scripts/gate_sensitivity.py --experiment shakespeare --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiments import EXPERIMENTS
from src.logger import MemoryLogger
from src.training import TrainConfig, train
from src.utils import score_losses, select_device


THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2]
WINDOWS = [5, 10, 20]


def _run(cfg: TrainConfig, seed: int) -> list[float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return train(cfg, log_sink=MemoryLogger())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="shakespeare")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"results/gate_sensitivity/{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = TrainConfig(
        experiment_name=args.experiment,
        optimizer_name="specmuon",
        device=select_device(args.device),
        steps=args.steps,
        lr=args.lr,
        log_every=max(1, args.steps // 10),
    )

    grid = np.full((len(THRESHOLDS), len(WINDOWS)), np.nan, dtype=float)
    results: list[dict] = []
    for i, tau in enumerate(THRESHOLDS):
        for j, window in enumerate(WINDOWS):
            cfg = base_cfg.with_(opt_kwargs={
                "sigma_mode": "baseline", "top_k": args.top_k,
                "gate_threshold": float(tau), "gate_window": int(window),
            })
            losses = _run(cfg, args.seed)
            score = score_losses(losses)
            grid[i, j] = score
            results.append({"gate_threshold": float(tau), "gate_window": int(window),
                            "score": score, "final_loss": losses[-1]})
            print(f"τ={tau:>4}  window={window:>2}   score={score:.3e}   final={losses[-1]:.3e}")

    # Heatmap: log10(score) so wide ranges still read.
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    log_grid = np.log10(np.clip(grid, 1e-30, None))
    im = ax.imshow(log_grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(WINDOWS)), [str(w) for w in WINDOWS])
    ax.set_yticks(range(len(THRESHOLDS)), [f"{t:g}" for t in THRESHOLDS])
    ax.set_xlabel("gate_window")
    ax.set_ylabel("gate_threshold τ")
    ax.set_title(f"Gate sensitivity — {args.experiment} @ lr={args.lr:.0e}\nlog₁₀(score)")
    fig.colorbar(im, ax=ax, label="log₁₀(score)")
    for i in range(len(THRESHOLDS)):
        for j in range(len(WINDOWS)):
            ax.text(j, i, f"{grid[i, j]:.1e}", ha="center", va="center",
                    color="w", fontsize=7)
    fig.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'heatmap.png'}")

    with open(out_dir / "results.json", "w") as f:
        json.dump({"experiment": args.experiment, "lr": args.lr, "steps": args.steps,
                   "top_k": args.top_k, "grid": results}, f, indent=2)

    md = [f"# Gate sensitivity — {args.experiment}", "",
          f"lr = {args.lr:.2e}, steps = {args.steps}, top_k = {args.top_k}, seed = {args.seed}", "",
          "score = min loss over last 10% of training; τ=0 ⇒ gate disabled (paper default).", "",
          "| τ \\ window | " + " | ".join(str(w) for w in WINDOWS) + " |",
          "|---|" + "|".join(["---"] * len(WINDOWS)) + "|"]
    for i, tau in enumerate(THRESHOLDS):
        row = [f"{tau:g}"] + [f"{grid[i, j]:.3e}" for j in range(len(WINDOWS))]
        md.append("| " + " | ".join(row) + " |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Saved {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
