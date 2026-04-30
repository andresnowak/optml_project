from __future__ import annotations

import argparse

import numpy as np
import torch

from optml_project.experiments import EXPERIMENTS
from optml_project.logger import make_logger
from optml_project.optimizers import OPTIMIZERS
from optml_project.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizer convergence benchmarks.")

    exp = parser.add_argument_group("experiment")
    exp.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="linear_regression")
    exp.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    exp.add_argument("--steps", type=int, default=300)
    exp.add_argument("--batch-size", type=int, default=128)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--log-every", type=int, default=25)
    exp.add_argument("--samples", type=int, default=2048, help="Samples for linear regression.")
    exp.add_argument("--feature-dim", type=int, default=32, help="Feature dim for linear regression.")
    exp.add_argument("--matrix-rows", type=int, default=64, help="m: rows of A (matrix factorization).")
    exp.add_argument("--matrix-cols", type=int, default=64, help="n: cols of A (matrix factorization).")
    exp.add_argument("--rank", type=int, default=8, help="k: inner rank (matrix factorization).")

    opt = parser.add_argument_group("optimizer")
    opt.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adamw")
    opt.add_argument("--lr", type=float, default=1e-2)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--momentum", type=float, default=None, help="SGD / Muon.")
    opt.add_argument("--beta1", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--beta2", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--eps", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--ns-steps", type=int, default=None, help="Newton-Schulz iterations (Muon).")
    opt.add_argument("--top-k", type=int, default=None, help="Top-k SAV singular directions (SpecMuon).")
    opt.add_argument("--sav-smooth", type=float, default=None, help="SAV smoothing factor ξ (SpecMuon).")

    log_group = parser.add_argument_group("logging")
    log_group.add_argument("--backend", choices=("matplotlib", "wandb"), default=None,
                           help="Logging backend. Omit for console-only output.")
    log_group.add_argument("--log-grad-svd", action="store_true",
                           help="Log singular values of parameter gradients.")
    log_group.add_argument("--svd-every", type=int, default=None,
                           help="Log SVD every N steps (default: same as --log-every).")
    log_group.add_argument("--svd-top-k", type=int, default=None,
                           help="Only show top-k singular values. Default: all.")
    log_group.add_argument("--log-scale", action="store_true", help="Log y-axis (matplotlib).")
    log_group.add_argument("--smooth", type=int, default=1, help="Rolling-mean window (matplotlib).")
    log_group.add_argument("--save-plot", type=str, default=None, metavar="PATH")
    log_group.add_argument("--wandb-project", type=str, default="optml-bench")

    modes = parser.add_argument_group("modes")
    modes.add_argument("--compare-all", action="store_true", help="Run all optimizers at --lr.")
    modes.add_argument("--sweep-lr", action="store_true", help="Sweep lr over a log-spaced grid.")
    modes.add_argument("--compare-best-lr", action="store_true", help="Sweep lr per optimizer, compare at best lr.")
    modes.add_argument("--lr-min", type=float, default=1e-4)
    modes.add_argument("--lr-max", type=float, default=1.0)
    modes.add_argument("--lr-n", type=int, default=8)

    args = parser.parse_args()

    active_modes = [args.sweep_lr, args.compare_all, args.compare_best_lr]
    if sum(active_modes) > 1:
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

    opt_kwargs: dict = {}
    if args.momentum is not None:
        opt_kwargs["momentum"] = args.momentum
    if args.ns_steps is not None:
        opt_kwargs["ns_steps"] = args.ns_steps
    if args.top_k is not None:
        opt_kwargs["top_k"] = args.top_k
    if args.sav_smooth is not None:
        opt_kwargs["sav_smooth"] = args.sav_smooth
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
        matrix_rows=args.matrix_rows,
        matrix_cols=args.matrix_cols,
        rank=args.rank,
        opt_kwargs=opt_kwargs,
        log_grad_svd=args.log_grad_svd,
        svd_every=args.svd_every,
    )

    logger_kwargs = dict(log_scale=args.log_scale, smooth=args.smooth,
                         save_path=args.save_plot, wandb_project=args.wandb_project,
                         config=vars(args), svd_top_k=args.svd_top_k)
    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_n)

    if args.sweep_lr:
        logger = make_logger(args.backend, f"LR sweep — {args.experiment} / {args.optimizer}", **logger_kwargs)
        for lr in lrs:
            train(optimizer_name=args.optimizer, lr=lr, logger=logger, run_name=f"lr={lr:.2e}", **common)
        if logger:
            logger.finish()

    elif args.compare_best_lr:
        logger = make_logger(args.backend, f"Best-lr comparison — {args.experiment}", **logger_kwargs)
        for name in sorted(OPTIMIZERS):
            print(f"\n── sweeping {name} ──")
            best_losses, best_lr = None, None
            for lr in lrs:
                losses = train(optimizer_name=name, lr=lr, **common)
                if best_losses is None or losses[-1] < best_losses[-1]:
                    best_losses, best_lr = losses, lr
            print(f"   → best lr={best_lr:.2e}  final loss={best_losses[-1]:.6f}")
            train(optimizer_name=name, lr=best_lr, logger=logger,
                  run_name=f"{name} lr={best_lr:.2e}", **common)
        if logger:
            logger.finish()

    elif args.compare_all:
        logger = make_logger(args.backend, f"Optimizer comparison — {args.experiment}", **logger_kwargs)
        for name in sorted(OPTIMIZERS):
            train(optimizer_name=name, lr=args.lr, logger=logger, run_name=name, **common)
        if logger:
            logger.finish()

    else:
        logger = make_logger(args.backend, f"{args.experiment} / {args.optimizer}", **logger_kwargs)
        train(optimizer_name=args.optimizer, lr=args.lr, logger=logger, **common)
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
