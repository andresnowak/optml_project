from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch

from src.experiments import EXPERIMENTS
from src.logger import make_logger
from src.optimizers import OPTIMIZER_KWARGS, OPTIMIZERS
from src.training import TrainConfig, train
from src.utils import select_device


# `optimizers` is derived from OPTIMIZER_KWARGS so it stays in sync as new
# optimizer kwargs land. `lr` and `weight_decay` are always sweepable.
def _optimizers_accepting(kwarg: str) -> set[str]:
    return {name for name, kws in OPTIMIZER_KWARGS.items() if kwarg in kws}


SWEEP_SPECS = {
    "lr": {"type": float},
    "weight_decay": {"type": float},
    "momentum": {"type": float, "optimizers": _optimizers_accepting("momentum")},
    "beta1": {"type": float, "optimizers": _optimizers_accepting("betas")},
    "beta2": {"type": float, "optimizers": _optimizers_accepting("betas")},
    "eps": {"type": float, "optimizers": _optimizers_accepting("eps")},
    "ns_steps": {"type": int, "optimizers": _optimizers_accepting("ns_steps")},
    "top_k": {"type": int, "optimizers": _optimizers_accepting("top_k")},
    "sav_smooth": {"type": float, "optimizers": _optimizers_accepting("sav_smooth")},
    "kappa": {"type": float, "optimizers": _optimizers_accepting("kappa")},
    "adjust_lr_fn": {"type": str, "choices": {"shape_scaling"},
                     "optimizers": _optimizers_accepting("adjust_lr_fn")},
    "sigma_mode": {"type": str,
                   "choices": {"baseline", "sqrt", "power", "clip", "truncate", "energy"},
                   "optimizers": _optimizers_accepting("sigma_mode")},
    "sigma_clip": {"type": float, "optimizers": _optimizers_accepting("sigma_clip")},
    "sigma_truncate": {"type": float, "optimizers": _optimizers_accepting("sigma_truncate")},
    "power_beta": {"type": float, "optimizers": _optimizers_accepting("power_beta")},
    "energy_threshold": {"type": float, "optimizers": _optimizers_accepting("energy_threshold")},
    "gate_window": {"type": int, "optimizers": _optimizers_accepting("gate_window")},
    "gate_threshold": {"type": float, "optimizers": _optimizers_accepting("gate_threshold")},
}


# CLI flag → optimizer kwarg name. Drives `_validate_optimizer_flags` so a new
# kwarg just needs an entry here (and in `OPTIMIZER_KWARGS`) — no per-optimizer
# branching to update.
_FLAG_TO_KWARG: dict[str, str] = {
    "--top-k": "top_k",
    "--sav-smooth": "sav_smooth",
    "--kappa": "kappa",
    "--adjust-lr-fn": "adjust_lr_fn",
    "--sigma-mode": "sigma_mode",
    "--sigma-clip": "sigma_clip",
    "--sigma-truncate": "sigma_truncate",
    "--power-beta": "power_beta",
    "--energy-threshold": "energy_threshold",
    "--gate-window": "gate_window",
    "--gate-threshold": "gate_threshold",
    "--ns-steps": "ns_steps",
    "--momentum": "momentum",
    "--beta1": "betas",
    "--beta2": "betas",
    "--eps": "eps",
}


# Args that flow into experiment_kwargs (filtered against the experiment class signature in train()).
EXPERIMENT_FIELDS = frozenset({
    "feature_dim", "output_dim", "samples", "condition_number",
    "matrix_rows", "matrix_cols", "rank",
    "block_size", "d_model", "n_heads", "n_layers",
})


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


def _selected_optimizers(args) -> list[str]:
    """Optimizer subset for compare-all / compare-best-lr."""
    if args.compare_optimizers:
        names = [n.strip() for n in args.compare_optimizers.split(",") if n.strip()]
        unknown = [n for n in names if n not in OPTIMIZERS]
        if unknown:
            raise ValueError(f"--compare-optimizers contains unknown: {unknown}; "
                             f"choose from {sorted(OPTIMIZERS)}")
        return sorted(names)
    return sorted(OPTIMIZERS)


def _run_compare_best_lr(args, run_paired, logger) -> None:
    """Per-optimizer lr sweep (silent), then re-run each at its winning lr.

    Uses `score_losses` (min over the last 10% of training) as the rank,
    breaking ties in favor of earlier-min trajectories within ``compare_rtol``.
    """
    from src.utils import score_losses
    lrs = np.logspace(np.log10(args.lr_min), np.log10(args.lr_max), args.lr_n)
    rtol = args.compare_rtol
    for name in _selected_optimizers(args):
        print(f"── sweeping {name} ──")
        best_losses: list[float] | None = None
        best_lr: float | None = None
        for lr in lrs:
            losses = run_paired(name, logger=None, run_name=None, overrides={"lr": float(lr)})
            curr = score_losses(losses)
            best = score_losses(best_losses) if best_losses is not None else float("inf")
            if (best_losses is None
                    or curr < best
                    or (curr < best * (1 + rtol)
                        and np.argmin(losses) < np.argmin(best_losses))):
                best_losses, best_lr = losses, float(lr)
        assert best_losses is not None and best_lr is not None
        print(f"   → best lr={best_lr:.2e}  final loss={best_losses[-1]:.6f}")
        run_paired(name, logger=logger, run_name=f"{name} lr={best_lr:.2e}",
                   overrides={"lr": best_lr})


def _validate_optimizer_flags(args, parser) -> None:
    """Reject CLI flags that don't apply to the chosen optimizer.

    Data-driven from `OPTIMIZER_KWARGS` and `_FLAG_TO_KWARG`. Skipped in
    `--compare-all` mode where we cycle through optimizers and `build_optimizer`
    filters silently.
    """
    if args.compare_all:
        return
    accepted = OPTIMIZER_KWARGS[args.optimizer]
    for flag, kwarg in _FLAG_TO_KWARG.items():
        val = getattr(args, flag.lstrip("-").replace("-", "_"))
        if val is not None and kwarg not in accepted:
            parser.error(f"{flag} can only be used with optimizers that accept '{kwarg}'.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimizer convergence benchmarks.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. Values become argparse defaults; CLI flags override.")

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
    exp.add_argument("--condition-number", type=float, default=1e4,
                     help="Condition number for ill-conditioned linear regression.")
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
    opt.add_argument("--kappa", type=float, default=None,
                     help="Loss-shift constant κ ≥ 0 for SpecMuon (paper §2.1). Default 0.")
    opt.add_argument("--adjust-lr-fn", choices=("shape_scaling",), default=None,
                     help="Repo-controlled LR scaling mode (Muon / SpecMuon).")
    opt.add_argument("--sigma-mode",
                     choices=("power", "clip", "truncate", "energy", "baseline", "sqrt"),
                     default=None,
                     help="σ-intervention for SpecMuon step size. "
                          "`baseline` and `sqrt` are aliases for power(β=1) and power(β=0.5).")
    opt.add_argument("--sigma-clip", type=float, default=None,
                     help="Step-size cap for --sigma-mode clip.")
    opt.add_argument("--sigma-truncate", type=float, default=None,
                     help="Relative threshold for --sigma-mode truncate.")
    opt.add_argument("--power-beta", type=float, default=None,
                     help="β exponent for --sigma-mode power (β=1 is baseline, β=0.5 is sqrt).")
    opt.add_argument("--energy-threshold", type=float, default=None,
                     help="Cumulative-energy threshold τ for --sigma-mode energy.")
    opt.add_argument("--gate-window", type=int, default=None,
                     help="SAV-gating rolling window size (SpecMuon).")
    opt.add_argument("--gate-threshold", type=float, default=None,
                     help="SAV-gating threshold τ on rolling relative loss drop. "
                          "0 disables gating (paper default, always-on SAV).")
    opt.add_argument("--specmuon-target", choices=("all", "mlp", "attention"),
                     default="all",
                     help="Layer-selectivity ablation: which 2-D matrix weights receive "
                          "the SAV branch. 'all' (default) = paper. 'mlp' = SAV on MLP weights "
                          "only, attention through paper-tail (SpecMuon top_k=0). 'attention' "
                          "= the symmetric variant.")

    log_group = parser.add_argument_group("logging")
    log_group.add_argument("--backend", choices=("matplotlib", "wandb"), default=None,
                           help="Logging backend. Omit for console-only output.")
    log_group.add_argument("--log-grad-svd", action="store_true",
                           help="Log singular values of parameter gradients.")
    log_group.add_argument("--log-sav-r", action="store_true",
                           help="Log SpecMuon SAV r-tracker and last_iota diagnostic.")
    log_group.add_argument("--checkpoint-dir", type=str, default=None,
                           help="Directory to write model+optimizer checkpoints "
                                "(.pt files). Final-step checkpoint always written if set.")
    log_group.add_argument("--checkpoint-every", type=int, default=None,
                           help="Save intermediate checkpoint every N steps "
                                "(in addition to the final one). Requires --checkpoint-dir.")
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
    log_group.add_argument("--wandb-project", type=str,
                           default=os.getenv("WANDB_PROJECT", "mlo-specmuon"),
                           help="WandB project name. Defaults to $WANDB_PROJECT or 'mlo-specmuon'.")
    log_group.add_argument("--wandb-entity", type=str,
                           default=os.getenv("WANDB_ENTITY", "cs-439-project"),
                           help="WandB entity / team. Defaults to $WANDB_ENTITY or 'cs-439-project'.")

    modes = parser.add_argument_group("modes")
    modes.add_argument("--compare-all", action="store_true", help="Run all optimizers at --lr.")
    modes.add_argument("--sweep-lr", action="store_true", help="Sweep lr over a log-spaced grid.")
    modes.add_argument("--sweep", action="append", default=[], metavar="PARAM=V1,V2,...",
                       help="Sweep one or more hyperparameters. Repeat to run a grid sweep.")
    modes.add_argument("--compare-best-lr", action="store_true",
                       help="Per-optimizer lr sweep, then compare each at its winning lr.")
    modes.add_argument("--lr-min", type=float, default=1e-4)
    modes.add_argument("--lr-max", type=float, default=1.0)
    modes.add_argument("--lr-n", type=int, default=8)
    modes.add_argument("--compare-rtol", type=float, default=0.01,
                       help="Relative tolerance for --compare-best-lr tiebreak (prefers earlier-min).")
    modes.add_argument("--compare-optimizers", type=str, default=None,
                       help="Comma-separated subset of optimizers to include in "
                            "--compare-all / --compare-best-lr (default: all).")
    return parser


def main() -> None:
    import sys
    load_dotenv()

    parser = _build_parser()

    # Pre-parse for --config so its values become defaults before the real parse.
    pre, _ = parser.parse_known_args()
    if pre.config is not None:
        from src.utils import load_config
        if not os.path.exists(pre.config):
            parser.error(f"--config file not found: {pre.config}")
        cfg = load_config(pre.config)
        valid = {a.dest for a in parser._actions}
        unknown = set(cfg) - valid
        if unknown:
            parser.error(f"{pre.config}: unknown keys {sorted(unknown)}")
        parser.set_defaults(**cfg)
        print(f"Loaded config from {pre.config}", file=sys.stderr)

    args = parser.parse_args()

    active_modes = [args.sweep_lr, bool(args.sweep), args.compare_all, args.compare_best_lr]
    if sum(active_modes) > 1:
        parser.error("--sweep-lr, --sweep, --compare-all, and --compare-best-lr are mutually exclusive.")
    if args.lr_min >= args.lr_max:
        parser.error("--lr-min must be less than --lr-max.")
    if args.lr_n < 2:
        parser.error("--lr-n must be at least 2.")
    if args.smooth < 1:
        parser.error("--smooth must be at least 1.")
    if args.steps < 1:
        parser.error("--steps must be at least 1.")

    sweep_params: dict[str, list[object]] = {}
    try:
        for raw in args.sweep:
            name, values = parse_sweep_arg(raw)
            sweep_params[name] = values
    except ValueError as exc:
        parser.error(str(exc))

    _validate_optimizer_flags(args, parser)

    if not args.compare_all:
        for name in sweep_params:
            allowed = SWEEP_SPECS[name].get("optimizers")
            if allowed is not None and args.optimizer not in allowed:
                parser.error(f"--sweep {name}=... can only be used with --optimizer {', '.join(sorted(allowed))}.")

    device = select_device(args.device)

    experiment_kwargs = {k: getattr(args, k, None) for k in EXPERIMENT_FIELDS
                         if getattr(args, k, None) is not None}

    logger_kwargs = dict(log_scale=args.log_scale, smooth=args.smooth,
                         save_path=args.save_plot,
                         wandb_project=args.wandb_project,
                         wandb_entity=args.wandb_entity,
                         config=vars(args), svd_top_k=args.svd_top_k)

    def reset_seeds() -> None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # All optimizer kwargs flowing from CLI flags. `betas` is composed
    # separately from --beta1/--beta2. Drives `build_run_settings`.
    _OPT_KWARG_FIELDS = (
        "momentum", "ns_steps", "top_k", "sav_smooth", "kappa",
        "adjust_lr_fn", "sigma_mode", "sigma_clip", "sigma_truncate",
        "power_beta", "energy_threshold", "gate_window", "gate_threshold", "eps",
    )

    def build_run_settings(overrides: dict[str, object]) -> tuple[float, float, dict]:
        assert_optimizer_supports_params(args.optimizer, set(overrides))
        lr = float(overrides.get("lr", args.lr))
        weight_decay = float(overrides.get("weight_decay", args.weight_decay))
        beta1 = float(overrides.get("beta1", args.beta1 if args.beta1 is not None else 0.9))
        beta2 = float(overrides.get("beta2", args.beta2 if args.beta2 is not None else 0.999))

        opt_kwargs: dict = {}
        for key in _OPT_KWARG_FIELDS:
            val = overrides.get(key, getattr(args, key))
            if val is not None:
                opt_kwargs[key] = val
        if args.beta1 is not None or args.beta2 is not None or "beta1" in overrides or "beta2" in overrides:
            opt_kwargs["betas"] = (beta1, beta2)
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
        cfg = TrainConfig(
            experiment_name=args.experiment,
            optimizer_name=optimizer_name,
            device=device,
            steps=args.steps,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=args.batch_size,
            log_every=args.log_every,
            experiment_kwargs=experiment_kwargs,
            opt_kwargs=opt_kwargs,
            log_grad_svd=args.log_grad_svd,
            log_grad_norms=args.log_grad_norms,
            log_weight_norms=args.log_weight_norms,
            log_sav_r=args.log_sav_r,
            svd_every=args.svd_every,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            specmuon_target=args.specmuon_target,
        )
        return train(cfg, log_sink=logger, run_name=run_name)

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
        "compare_best_lr" if args.compare_best_lr else
        "single"
    )
    title = {
        "sweep": (
            f"Grid sweep — {args.experiment} / {args.optimizer}"
            if len(sweep_params) > 1 else
            f"Sweep {next(iter(sweep_params), 'lr')} — {args.experiment} / {args.optimizer}"
        ),
        "compare_all": f"Optimizer comparison — {args.experiment}",
        "compare_best_lr": f"Best-lr comparison — {args.experiment}",
        "single": f"{args.experiment} / {args.optimizer}",
    }[mode]
    logger = make_logger(args.backend, title, **logger_kwargs)

    try:
        if mode == "sweep":
            for overrides in sweep_overrides:
                run_name = ", ".join(f"{name}={format_sweep_value(name, value)}" for name, value in overrides.items())
                run_paired(args.optimizer, logger=logger, run_name=run_name, overrides=overrides)
        elif mode == "compare_all":
            for name in _selected_optimizers(args):
                run_paired(name, logger=logger, run_name=name)
        elif mode == "compare_best_lr":
            _run_compare_best_lr(args, run_paired, logger)
        else:
            run_paired(args.optimizer, logger=logger)
    finally:
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
