"""SAV isolation — does the SAV mechanism help or hurt?

For a fixed (experiment, lr) tuple, run SpecMuon at ``top_k ∈ {0, 1, 6, 32}``
plus the gated variant (``gate_threshold=0.05``), all with paper-faithful
σ-treatment (``sigma_mode="baseline"``). At ``top_k=0`` the SAV branch is
empty and every direction takes the paper-tail (Muon-with-σ) update — this
is the no-SAV ablation isolating the SAV mechanism's contribution.

Emits ``results/sav_isolation/<experiment>/{trajectories.png, summary.md, results.json}``.

Usage:
    uv run python scripts/sav_isolation.py --experiment matrix_factorization
    uv run python scripts/sav_isolation.py --experiment matrix_factorization --lr 1e-1 --steps 500
"""

from __future__ import annotations

import argparse
import json
import os
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


VARIANTS = [
    ("top_k=0 (no-SAV / paper tail)", {"top_k": 0, "gate_threshold": 0.0}),
    ("top_k=1",                       {"top_k": 1, "gate_threshold": 0.0}),
    ("top_k=6 (paper default)",       {"top_k": 6, "gate_threshold": 0.0}),
    ("top_k=32",                      {"top_k": 32, "gate_threshold": 0.0}),
    ("gated (τ=0.05, k=6)",           {"top_k": 6, "gate_threshold": 0.05, "gate_window": 10}),
]


def _run(cfg: TrainConfig, seed: int) -> list[float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return train(cfg, log_sink=MemoryLogger())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="matrix_factorization")
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: results/sav_isolation/<experiment>).")
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"results/sav_isolation/{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = TrainConfig(
        experiment_name=args.experiment,
        optimizer_name="specmuon",
        device=select_device(args.device),
        steps=args.steps,
        lr=args.lr,
        log_every=max(1, args.steps // 10),
    )

    results: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, kwargs in VARIANTS:
        cfg = base_cfg.with_(opt_kwargs={"sigma_mode": "baseline", **kwargs})
        losses = _run(cfg, args.seed)
        results[label] = {
            "opt_kwargs": kwargs,
            "score": score_losses(losses),
            "final_loss": losses[-1],
            "losses": losses,
        }
        ax.plot(losses, label=label, linewidth=1.2)
        print(f"{label:35s}  score={results[label]['score']:.3e}  final={losses[-1]:.3e}")

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"SAV isolation — {args.experiment} @ lr={args.lr:.0e}")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'trajectories.png'}")

    serializable = {k: {"opt_kwargs": v["opt_kwargs"],
                        "score": v["score"], "final_loss": v["final_loss"]}
                    for k, v in results.items()}
    with open(out_dir / "results.json", "w") as f:
        json.dump({"experiment": args.experiment, "lr": args.lr, "steps": args.steps,
                   "variants": serializable}, f, indent=2)

    md = [f"# SAV isolation — {args.experiment}", "",
          f"lr = {args.lr:.2e}, steps = {args.steps}, seed = {args.seed}", "",
          "| variant | score (min over last 10%) | final loss |",
          "|---|---|---|"]
    for label, info in results.items():
        md.append(f"| {label} | {info['score']:.3e} | {info['final_loss']:.3e} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Saved {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
