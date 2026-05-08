from __future__ import annotations

import argparse

import numpy as np
import torch

from src.experiments import EXPERIMENTS
from src.logger import make_logger
from src.optimizers import OPTIMIZERS
from src.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizer convergence benchmarks.")

    exp = parser.add_argument_group("experiment")
    exp.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="linear_regression")
    exp.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    exp.add_argument("--steps", type=int, default=300)
    exp.add_argument("--batch-size", type=int, default=128)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--log-every", type=int, default=25)
    exp.add_argument("--samples", type=int, default=2048, help="N: number of data points (linear regression).")
    exp.add_argument("--feature-dim", type=int, default=32, help="n: input dimension / cols of W (linear regression).")
    exp.add_argument("--output-dim", type=int, default=16, help="m: output dimension / rows of W (linear regression).")
    exp.add_argument("--matrix-rows", type=int, default=64, help="m: rows of A (matrix factorization).")
    exp.add_argument("--matrix-cols", type=int, default=64, help="n: cols of A (matrix factorization).")
    exp.add_argument("--rank", type=int, default=8, help="k: inner rank (matrix factorization).")
    exp.add_argument("--block-size", type=int, default=128, help="Context length (shakespeare).")
    exp.add_argument("--d-model", type=int, default=128, help="Model dimension (shakespeare).")
    exp.add_argument("--n-heads", type=int, default=4, help="Number of attention heads (shakespeare).")
    exp.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers (shakespeare).")

    opt = parser.add_argument_group("optimizer")
    opt.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="adamw")
    opt.add_argument("--lr", type=float, default=1e-2)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--momentum", type=float, default=None, help="SGD / Muon / Specmuon.")
    opt.add_argument("--beta1", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--beta2", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--eps", type=float, default=None, help="Adam / AdamW.")
    opt.add_argument("--ns-steps", type=int, default=None, help="Newton-Schulz iterations (Muon).")
    opt.add_argument("--top-k", type=int, default=None, help="Top-k SAV singular directions (SpecMuon).")
    opt.add_argument("--sav-smooth", type=float, default=None, help="SAV smoothing factor ξ (SpecMuon).")
    opt.add_argument("--adjust-lr-fn", choices=("shape_scaling",), default=None, help="LR scaling mode (SpecMuon).")

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
    modes.add_argument("--lr-min", type=float, default=1e-4)
    modes.add_argument("--lr-max", type=float, default=1.0)
    modes.add_argument("--lr-n", type=int, default=8)

    args = parser.parse_args()

    active_modes = [args.sweep_lr, args.compare_all]
    if sum(active_modes) > 1:
        parser.error("--sweep-lr and --compare-all are mutually exclusive.")
    if args.lr_min >= args.lr_max:
        parser.error("--lr-min must be less than --lr-max.")
    if args.lr_n < 2:
        parser.error("--lr-n must be at least 2.")
    if args.smooth < 1:
        parser.error("--smooth must be at least 1.")
    if args.steps < 1:
        parser.error("--steps must be at least 1.")

    _specmuon_only = {"--top-k": args.top_k, "--sav-smooth": args.sav_smooth, "--adjust-lr-fn": args.adjust_lr_fn}
    _muon_only = {"--ns-steps": args.ns_steps}
    _muon_like = {"--momentum": args.momentum}

    if not args.compare_all:
        for flag, val in _specmuon_only.items():
            if val is not None and args.optimizer != "specmuon":
                parser.error(f"{flag} can only be used with --optimizer specmuon.")
        for flag, val in _muon_only.items():
            if val is not None and args.optimizer not in ("muon", "specmuon"):
                parser.error(f"{flag} can only be used with --optimizer muon or specmuon.")
        for flag, val in _muon_like.items():
            if val is not None and args.optimizer not in ("muon", "specmuon", "sgd"):
                parser.error(f"{flag} can only be used with --optimizer muon, specmuon, or sgd.")

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
    if args.adjust_lr_fn is not None:
        opt_kwargs["adjust_lr_fn"] = args.adjust_lr_fn
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
        output_dim=args.output_dim,
        samples=args.samples,
        matrix_rows=args.matrix_rows,
        matrix_cols=args.matrix_cols,
        rank=args.rank,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        opt_kwargs=opt_kwargs,
        log_grad_svd=args.log_grad_svd,
        svd_every=args.svd_every,
    )

    logger_kwargs = dict(log_scale=args.log_scale, smooth=args.smooth,
                         save_path=args.save_plot, wandb_project=args.wandb_project,
                         config=vars(args), svd_top_k=args.svd_top_k)
    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_n)

    def reset_seeds() -> None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    def run_paired(optimizer_name: str, lr: float, logger=None, run_name: str | None = None) -> list[float]:
        reset_seeds()
        return train(optimizer_name=optimizer_name, lr=lr, logger=logger, run_name=run_name, **common)

    mode = (
        "sweep_lr" if args.sweep_lr else
        "compare_all" if args.compare_all else
        "single"
    )
    title = {
        "sweep_lr": f"LR sweep — {args.experiment} / {args.optimizer}",
        "compare_all": f"Optimizer comparison — {args.experiment}",
        "single": f"{args.experiment} / {args.optimizer}",
    }[mode]
    logger = make_logger(args.backend, title, **logger_kwargs)

    try:
        if mode == "sweep_lr":
            for lr in lrs:
                run_paired(args.optimizer, lr, logger=logger, run_name=f"lr={lr:.2e}")
        elif mode == "compare_all":
            for name in sorted(OPTIMIZERS):
                run_paired(name, args.lr, logger=logger, run_name=name)
        else:
            run_paired(args.optimizer, args.lr, logger=logger)
    finally:
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
