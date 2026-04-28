from __future__ import annotations

import argparse

import numpy as np
import torch

from optml_project.experiments import EXPERIMENTS
from optml_project.optimizers import OPTIMIZERS
from optml_project.training import train


def _plot(
    losses_by_label: dict[str, list[float]],
    title: str,
    log_scale: bool = False,
    smooth: int = 1,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    # Sort by final loss so the legend reads best → worst
    sorted_items = sorted(losses_by_label.items(), key=lambda kv: kv[1][-1])
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, losses) in enumerate(sorted_items):
        color = colors[i % len(colors)]
        steps = np.arange(1, len(losses) + 1)

        if smooth > 1:
            # show raw trace faintly, overlay smoothed line
            ax.plot(steps, losses, color=color, alpha=0.15, linewidth=0.8)
            kernel = np.ones(smooth) / smooth
            smoothed = np.convolve(losses, kernel, mode="valid")
            x_sm = steps[smooth - 1:]
            ax.plot(x_sm, smoothed, label=label, color=color, linewidth=2)
            x_end, y_end = x_sm[-1], smoothed[-1]
        else:
            ax.plot(steps, losses, label=label, color=color, linewidth=2)
            x_end, y_end = steps[-1], losses[-1]

        ax.annotate(
            f"{y_end:.4f}",
            xy=(x_end, y_end),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=8,
            fontweight="bold",
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)" if log_scale else "Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizer convergence benchmarks.")

    # --- experiment ---
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="linear_regression")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--matrix-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)

    # --- optimizer ---
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # shared
    parser.add_argument("--momentum", type=float, default=None, help="SGD / Muon momentum.")
    # Adam / AdamW
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    # Muon-specific
    parser.add_argument("--ns-steps", type=int, default=None, help="Newton-Schulz iterations (Muon).")

    # --- plotting ---
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-scale", action="store_true", help="Log y-axis on convergence plot.")
    parser.add_argument("--smooth", type=int, default=1, help="Rolling-mean window for plot smoothing.")
    parser.add_argument("--save-plot", type=str, default=None, metavar="PATH")

    # --- modes ---
    parser.add_argument("--compare-all", action="store_true", help="Run all optimizers, plot together.")
    parser.add_argument("--sweep-lr", action="store_true", help="Sweep lr over a log-spaced grid.")
    parser.add_argument("--compare-best-lr", action="store_true", help="Sweep lr per optimizer, plot each at its best lr.")
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=1.0)
    parser.add_argument("--lr-n", type=int, default=8)

    args = parser.parse_args()

    modes = [args.sweep_lr, args.compare_all, args.compare_best_lr]
    if sum(modes) > 1:
        parser.error("--sweep-lr, --compare-all, and --compare-best-lr are mutually exclusive.")
    if args.lr_min >= args.lr_max:
        parser.error("--lr-min must be less than --lr-max.")
    if args.lr_n < 2:
        parser.error("--lr-n must be at least 2.")
    if args.smooth < 1:
        parser.error("--smooth must be at least 1.")
    if args.steps < 1:
        parser.error("--steps must be at least 1.")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    # build optimizer extra kwargs
    opt_kwargs: dict = {}
    if args.momentum is not None:
        opt_kwargs["momentum"] = args.momentum
    if args.ns_steps is not None:
        opt_kwargs["ns_steps"] = args.ns_steps
    if args.beta1 is not None or args.beta2 is not None:
        opt_kwargs["betas"] = (args.beta1 or 0.9, args.beta2 or 0.999)
    if args.eps is not None:
        opt_kwargs["eps"] = args.eps

    common = dict(
        experiment_name=args.experiment,
        device=device,
        steps=args.steps,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        log_every=args.log_every,
        feature_dim=args.feature_dim,
        samples=args.samples,
        matrix_size=args.matrix_size,
        rank=args.rank,
        opt_kwargs=opt_kwargs,
    )

    plot_kwargs = dict(log_scale=args.log_scale, smooth=args.smooth, save_path=args.save_plot)

    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_n)

    if args.sweep_lr:
        losses_by_lr = {}
        for lr in lrs:
            losses_by_lr[f"lr={lr:.2e}"] = train(optimizer_name=args.optimizer, lr=lr, **common)
        _plot(losses_by_lr, f"LR sweep — {args.experiment} / {args.optimizer}", **plot_kwargs)

    elif args.compare_best_lr:
        best_by_opt = {}
        for name in sorted(OPTIMIZERS):
            print(f"\n── sweeping {name} ──")
            best_losses, best_lr = None, None
            for lr in lrs:
                losses = train(optimizer_name=name, lr=lr, **common)
                if best_losses is None or losses[-1] < best_losses[-1]:
                    best_losses, best_lr = losses, lr
            label = f"{name}  (lr={best_lr:.2e})"
            best_by_opt[label] = best_losses
            print(f"   → best lr={best_lr:.2e}  final loss={best_losses[-1]:.6f}")
        _plot(best_by_opt, f"Best-lr comparison — {args.experiment}", **plot_kwargs)

    elif args.compare_all:
        losses_by_opt = {}
        for name in sorted(OPTIMIZERS):
            losses_by_opt[name] = train(optimizer_name=name, lr=args.lr, **common)
        _plot(losses_by_opt, f"Optimizer comparison — {args.experiment}", **plot_kwargs)

    else:
        losses = train(optimizer_name=args.optimizer, lr=args.lr, **common)
        if args.plot:
            _plot({args.optimizer: losses}, f"{args.experiment} / {args.optimizer}", **plot_kwargs)


if __name__ == "__main__":
    main()
