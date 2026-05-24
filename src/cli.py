from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch

from src.experiments import EXPERIMENTS
from src.logger import make_logger
from src.optimizers import OPTIMIZERS
from src.training import train


SWEEP_SPECS = {
    "lr": {"type": float},
    "weight_decay": {"type": float},
    "momentum": {"type": float, "optimizers": {"sgd", "muon", "specmuon"}},
    "beta1": {"type": float, "optimizers": {"adam", "adamw"}},
    "beta2": {"type": float, "optimizers": {"adam", "adamw"}},
    "eps": {"type": float, "optimizers": {"adam", "adamw", "specmuon"}},
    "ns_steps": {"type": int, "optimizers": {"muon"}},
    "top_k": {"type": int, "optimizers": {"specmuon"}},
    "sav_smooth": {"type": float, "optimizers": {"specmuon"}},
    "adjust_lr_fn": {"type": str, "choices": {"shape_scaling"}, "optimizers": {"specmuon"}},
}


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_sweep_arg(raw: str) -> tuple[str, list[object]]:
    if "=" not in raw:
        raise ValueError(f"invalid sweep {raw!r}; expected PARAM=v1,v2,...")

    name, values_raw = raw.split("=", 1)
    name = name.strip().replace("-", "_")
    spec = SWEEP_SPECS.get(name)
    if spec is None:
        choices = ", ".join(sorted(SWEEP_SPECS))
        raise ValueError(f"unknown sweep parameter {name!r}; choose from {choices}")

    values = [item.strip() for item in values_raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"sweep {name!r} must include at least one value")

    parsed: list[object] = []
    for value in values:
        if "choices" in spec and value not in spec["choices"]:
            choices = ", ".join(sorted(spec["choices"]))
            raise ValueError(f"invalid value {value!r} for sweep {name!r}; choose from {choices}")
        try:
            parsed.append(spec["type"](value))
        except ValueError as exc:
            raise ValueError(f"invalid value {value!r} for sweep {name!r}") from exc
    return name, parsed


def build_wandb_run_name(experiment: str, optimizer: str, lr: float, weight_decay: float, opt_kwargs: dict) -> str:
    parts = [experiment, optimizer, f"lr={lr:.2e}"]
    if weight_decay:
        parts.append(f"wd={weight_decay:.2e}")
    for k, v in sorted(opt_kwargs.items()):
        if k == "betas":
            parts.append(f"b1={v[0]:.3g}_b2={v[1]:.3g}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def format_sweep_value(name: str, value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2e}" if name in {"lr", "weight_decay", "eps"} else f"{value:.4g}"
    return str(value)


def assert_optimizer_supports_params(optimizer_name: str, param_names: set[str]) -> None:
    for name in param_names:
        allowed = SWEEP_SPECS[name].get("optimizers")
        if allowed is not None:
            assert optimizer_name in allowed, (
                f"{name} is only valid for optimizers {sorted(allowed)}, got {optimizer_name}"
            )


def main() -> None:
    load_dotenv()

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
    log_group.add_argument("--log-grad-norms", action="store_true",
                           help="Log per-parameter gradient norms and total gradient norm.")
    log_group.add_argument("--log-weight-norms", action="store_true",
                           help="Log per-parameter weight norms.")
    log_group.add_argument("--svd-every", type=int, default=None,
                           help="Log SVD every N steps (default: same as --log-every).")
    log_group.add_argument("--svd-top-k", type=int, default=None,
                           help="Only show top-k singular values. Default: all.")
    log_group.add_argument("--log-scale", action="store_true", help="Log y-axis (matplotlib).")
    log_group.add_argument("--smooth", type=int, default=1, help="Rolling-mean window (matplotlib).")
    log_group.add_argument("--save-plot", type=str, default=None, metavar="PATH")
    log_group.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "optml-bench"))

    modes = parser.add_argument_group("modes")
    modes.add_argument("--compare-all", action="store_true", help="Run all optimizers at --lr.")
    modes.add_argument("--sweep-lr", action="store_true", help="Sweep lr over a log-spaced grid.")
    modes.add_argument("--sweep", action="append", default=[], metavar="PARAM=V1,V2,...",
                       help="Sweep one or more hyperparameters. Repeat to run a grid sweep.")
    modes.add_argument("--lr-min", type=float, default=1e-4)
    modes.add_argument("--lr-max", type=float, default=1.0)
    modes.add_argument("--lr-n", type=int, default=8)

    args = parser.parse_args()

    active_modes = [args.sweep_lr, bool(args.sweep), args.compare_all]
    if sum(active_modes) > 1:
        parser.error("--sweep-lr, --sweep, and --compare-all are mutually exclusive.")
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
    sweep_params: dict[str, list[object]] = {}

    try:
        for raw in args.sweep:
            name, values = parse_sweep_arg(raw)
            sweep_params[name] = values
    except ValueError as exc:
        parser.error(str(exc))

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
        for name in sweep_params:
            allowed = SWEEP_SPECS[name].get("optimizers")
            if allowed is not None and args.optimizer not in allowed:
                parser.error(f"--sweep {name}=... can only be used with --optimizer {', '.join(sorted(allowed))}.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    common = dict(
        experiment_name=args.experiment,
        device=device,
        steps=args.steps,
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
        log_grad_svd=args.log_grad_svd,
        svd_every=args.svd_every,
        log_grad_norms=args.log_grad_norms,
        log_weight_norms=args.log_weight_norms,
    )

    logger_kwargs = dict(log_scale=args.log_scale, smooth=args.smooth,
                         save_path=args.save_plot, wandb_project=args.wandb_project,
                         config=vars(args), svd_top_k=args.svd_top_k)
    def reset_seeds() -> None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    def build_run_settings(overrides: dict[str, object]) -> tuple[float, float, dict]:
        assert_optimizer_supports_params(args.optimizer, set(overrides))
        lr = float(overrides.get("lr", args.lr))
        weight_decay = float(overrides.get("weight_decay", args.weight_decay))
        beta1 = float(overrides.get("beta1", args.beta1 if args.beta1 is not None else 0.9))
        beta2 = float(overrides.get("beta2", args.beta2 if args.beta2 is not None else 0.999))

        opt_kwargs: dict = {}
        momentum = overrides.get("momentum", args.momentum)
        if momentum is not None:
            opt_kwargs["momentum"] = momentum
        ns_steps = overrides.get("ns_steps", args.ns_steps)
        if ns_steps is not None:
            opt_kwargs["ns_steps"] = ns_steps
        top_k = overrides.get("top_k", args.top_k)
        if top_k is not None:
            opt_kwargs["top_k"] = top_k
        sav_smooth = overrides.get("sav_smooth", args.sav_smooth)
        if sav_smooth is not None:
            opt_kwargs["sav_smooth"] = sav_smooth
        adjust_lr_fn = overrides.get("adjust_lr_fn", args.adjust_lr_fn)
        if adjust_lr_fn is not None:
            opt_kwargs["adjust_lr_fn"] = adjust_lr_fn
        if args.beta1 is not None or args.beta2 is not None or "beta1" in overrides or "beta2" in overrides:
            opt_kwargs["betas"] = (beta1, beta2)
        eps = overrides.get("eps", args.eps)
        if eps is not None:
            opt_kwargs["eps"] = eps
        return lr, weight_decay, opt_kwargs

    def run_paired(
        optimizer_name: str,
        logger=None,
        run_name: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> list[float]:
        override_keys = set((overrides or {}).keys())
        assert_optimizer_supports_params(optimizer_name, override_keys)
        reset_seeds()
        lr, weight_decay, opt_kwargs = build_run_settings(overrides or {})
        if logger is not None:
            wandb_name = build_wandb_run_name(args.experiment, optimizer_name, lr, weight_decay, opt_kwargs)
            run_config = {"experiment": args.experiment, "optimizer": optimizer_name,
                          "lr": lr, "weight_decay": weight_decay, "steps": args.steps,
                          "batch_size": args.batch_size, **opt_kwargs}
            metric_prefix = f"{run_name}/" if run_name else ""
            logger.start_run(wandb_name, run_config, metric_prefix=metric_prefix)
        return train(
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            opt_kwargs=opt_kwargs,
            logger=logger,
            run_name=run_name,
            **common,
        )

    def iter_sweep_overrides() -> list[dict[str, object]]:
        if args.sweep_lr:
            assert not sweep_params, "--sweep-lr and --sweep cannot be combined"
            lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_n)
            return [{"lr": float(lr)} for lr in lrs]
        if not sweep_params:
            return [{}]
        names = list(sweep_params)
        return [dict(zip(names, combo)) for combo in itertools.product(*(sweep_params[name] for name in names))]

    sweep_overrides = iter_sweep_overrides()

    mode = (
        "sweep" if args.sweep_lr or args.sweep else
        "compare_all" if args.compare_all else
        "single"
    )
    assert (mode == "compare_all") == args.compare_all
    assert (mode == "sweep") == bool(args.sweep_lr or args.sweep)
    title = {
        "sweep": (
            f"Grid sweep — {args.experiment} / {args.optimizer}"
            if len(sweep_params) > 1 else
            f"Sweep {next(iter(sweep_params), 'lr')} — {args.experiment} / {args.optimizer}"
        ),
        "compare_all": f"Optimizer comparison — {args.experiment}",
        "single": f"{args.experiment} / {args.optimizer}",
    }[mode]
    logger = make_logger(args.backend, title, **logger_kwargs)

    try:
        if mode == "sweep":
            assert sweep_overrides, "sweep mode requires at least one override set"
            for overrides in sweep_overrides:
                run_name = ", ".join(f"{name}={format_sweep_value(name, value)}" for name, value in overrides.items())
                run_paired(args.optimizer, logger=logger, run_name=run_name, overrides=overrides)
        elif mode == "compare_all":
            assert not args.sweep_lr and not args.sweep
            for name in sorted(OPTIMIZERS):
                run_paired(name, logger=logger, run_name=name)
        else:
            assert not args.compare_all and not args.sweep_lr and not args.sweep
            run_paired(args.optimizer, logger=logger)
    finally:
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
