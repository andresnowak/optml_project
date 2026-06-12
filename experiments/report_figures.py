"""Generate the report figures from pulled W&B histories.

Every figure reads the local dumps produced by ``experiments/pull_wandb.py``
(``results/wandb/<group>/history_<run>.json`` + ``summary.csv``) so the whole
report is reproducible offline from one pull. Figures are intentionally
minimal: one panel, labeled axes, no styling beyond defaults.

    python experiments/report_figures.py lr_sweep
    python experiments/report_figures.py losses
    python experiments/report_figures.py depth --run route_fill_beta0p15_mlr0p02_20260611_routefill
    python experiments/report_figures.py beta --lr 0p02
    python experiments/report_figures.py proxies
    python experiments/report_figures.py cost
    python experiments/report_figures.py all             # everything available

Outputs land in report/figures/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimizers.dynmuon import logistic_schedule_p  # noqa: E402

WANDB_DIR = os.environ.get("REPORT_WANDB_DIR", os.path.join("results", "wandb"))
OUT_DIR = os.path.join("report", "figures")

METHOD_LABELS = {
    "bowl_adamw": "AdamW",
    "bowl_muon": "Muon",
    "bowl_dynmuon": "DynMuon",
    "bowl_relmuon_log1p": "RelMuon-log1p",
    "bowl_relmuon_rms": "RelMuon-RMS",
}
METHOD_COLORS = {
    "AdamW": "#1f77b4",
    "Muon": "#ff7f0e",
    "Muon-SVD": "#2ca02c",
    "DynMuon": "#d62728",
    "Route-align": "#9467bd",
    "RelMuon-log1p": "#8c564b",
    "RelMuon-RMS": "#7f7f7f",
}
SEED_NOISE = 0.001
MAIN_SWEEPS = (
    ("AdamW", lambda r: r.get("run", "").startswith("bowl_adamw_"), "adam_lr"),
    ("Muon", lambda r: r.get("run", "").startswith("bowl_muon_"), "muon_lr"),
    ("Muon-SVD", lambda r: r.get("run", "").startswith("muon_svd_polar_wd_lr_400m_"), "muon_lr"),
    ("DynMuon", lambda r: r.get("run", "").startswith("bowl_dynmuon_"), "muon_lr"),
    ("Route-align", lambda r: r.get("run", "").startswith("route_align_beta0p15_"), "muon_lr"),
    ("RelMuon-log1p", lambda r: r.get("run", "").startswith("bowl_relmuon_log1p_"), "muon_lr"),
    ("RelMuon-RMS", lambda r: r.get("run", "").startswith("bowl_relmuon_rms_"), "muon_lr"),
)
LAYER_TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
LAYER_LABELS = {
    "attn.q": "Attention Q",
    "attn.k": "Attention K",
    "attn.v": "Attention V",
    "attn.proj": "Attention Output",
    "mlp.fc": "MLP Input",
    "mlp.proj": "MLP Output",
}
LAYER_SHORT_LABELS = {
    "attn.q": "Q",
    "attn.k": "K",
    "attn.v": "V",
    "attn.proj": "Attn out",
    "mlp.fc": "MLP in",
    "mlp.proj": "MLP out",
}
PROXY_DEPTH_RUNS = (
    ("Stable rank\n$\\beta=+.15$", "route_fill_beta0p15_mlr0p02_20260611_clean"),
    ("SNR\n$\\beta=+.15$", "proxy_snr_posbeta_0p02"),
    ("SNR\n$\\beta=-.15$", "proxy_snr_negbeta_0p02"),
    ("EMA-SNR\n$\\beta=-.15$", "proxy_snr_ema_negbeta_0p02"),
    ("Alignment\n$\\beta=+.15$", "route_align_beta0p15_mlr0p02_20260611_clean"),
)
_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -- data access --------------------------------------------------------------

def _load_summaries() -> list[dict]:
    rows = _load_all_summaries()
    # de-duplicate by run name, newest file wins
    seen: dict[str, dict] = {}
    for row in rows:
        seen[row["run"]] = row
    return list(seen.values())


def _load_all_summaries() -> list[dict]:
    rows: list[dict] = []
    paths = sorted(glob.glob(os.path.join(WANDB_DIR, "*", "summary.csv")), key=os.path.getmtime)
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                row["_summary_path"] = path
                rows.append(row)
    return rows


def _find_history(run_name: str) -> dict | None:
    for path in glob.glob(os.path.join(WANDB_DIR, "*", f"history_{run_name}.json")):
        with open(path) as f:
            return json.load(f)
    return None


def _series(history: dict, key: str) -> tuple[list[int], list[float]]:
    pts = history.get(key, [])
    return [int(s) for s, _ in pts], [float(v) for _, v in pts]


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/{name}.pdf")


def _flt(row: dict, key: str) -> float | None:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finished_1526(row: dict) -> bool:
    return row.get("state") == "finished" and _flt(row, "train_steps") == 1526.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate_by_lr(rows: list[dict], pred, lr_key: str) -> list[tuple[float, float, float, int]]:
    by_lr: dict[float, list[float]] = {}
    for row in rows:
        if not _is_finished_1526(row) or not pred(row):
            continue
        lr, val = _flt(row, lr_key), _flt(row, "final_val_loss")
        if lr is None or val is None:
            continue
        by_lr.setdefault(lr, []).append(val)
    pts = []
    for lr, vals in sorted(by_lr.items()):
        mean, std = _mean_std(vals)
        pts.append((lr, mean, std, len(vals)))
    return pts


def _best_index(points: list[tuple[float, float, float, int]]) -> int | None:
    return min(range(len(points)), key=lambda i: points[i][1]) if points else None


def _is_bracketed(points: list[tuple[float, float, float, int]]) -> bool:
    idx = _best_index(points)
    return idx is not None and len(points) >= 3 and 0 < idx < len(points) - 1


def _row_for_run(run: str) -> dict | None:
    for row in _load_summaries():
        if row.get("run") == run:
            return row
    return None


def _value_for_run(run: str, key: str = "final_val_loss") -> float:
    row = _row_for_run(run)
    value = _flt(row or {}, key)
    if value is None:
        raise KeyError(f"{key} for run {run} not found")
    return value


def _steps_to(run: str, target: float) -> int | None:
    hist = _find_history(run)
    if not hist:
        return None
    for step, value in hist.get("val/loss", []):
        if float(value) <= target:
            return int(step)
    return None


# -- figures -------------------------------------------------------------------

def _save_aliases(fig, *names: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in names:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=160)
        print(f"wrote {OUT_DIR}/{name}.pdf")
    plt.close(fig)


def _label_delta_bars(ax, values: list[float], errors: list[float] | None = None, fontsize: int = 7) -> None:
    """Place delta labels outside bars and reserve enough y-axis room."""
    errors = errors or [0.0] * len(values)
    lower, upper = 0.0, 0.0
    for value, err in zip(values, errors):
        if not math.isfinite(value):
            continue
        err = err if math.isfinite(err) else 0.0
        lower = min(lower, value - err)
        upper = max(upper, value + err)
    span = max(upper - lower, 0.01)
    label_offset = 0.035 * span
    axis_pad = 0.12 * span
    ax.set_ylim(lower - axis_pad, upper + axis_pad)
    for i, (value, err) in enumerate(zip(values, errors)):
        if not math.isfinite(value):
            continue
        err = err if math.isfinite(err) else 0.0
        if value >= 0:
            y, va = value + err + label_offset, "bottom"
        else:
            y, va = value - err - label_offset, "top"
        ax.text(i, y, f"{value:+.3f}", ha="center", va=va, fontsize=fontsize, clip_on=False)


def _label_positive_bars(ax, values: list[float], fmt: str, fontsize: int = 7) -> None:
    """Label positive bars with enough top padding."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    upper = max(finite)
    span = max(upper, 1.0)
    label_offset = 0.035 * span
    ax.set_ylim(0.0, upper + 0.16 * span)
    for i, value in enumerate(values):
        if not math.isfinite(value):
            continue
        ax.text(i, value + label_offset, fmt.format(value), ha="center", va="bottom",
                fontsize=fontsize, clip_on=False)


def fig_lr_sweep(args) -> None:
    """Strict LR bowl: only finished sweeps whose best point is bracketed."""
    rows = _load_all_summaries()
    fig, ax = plt.subplots(figsize=(5.2, 3.45))
    plotted = []
    for label, pred, lr_key in MAIN_SWEEPS:
        pts = _aggregate_by_lr(rows, pred, lr_key)
        if not _is_bracketed(pts):
            print(f"skip {label}: best LR is not bracketed or sweep has too few points")
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [p[2] if p[3] > 1 else 0.0 for p in pts]
        line = ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=4, lw=1.8,
                           capsize=2.5, label=label)
        best = _best_index(pts)
        assert best is not None
        ax.plot([xs[best]], [ys[best]], marker="*", ms=12,
                markeredgecolor="black", markeredgewidth=0.4,
                color=line.lines[0].get_color(), zorder=5)
        plotted.append(label)
    ax.set_xscale("log")
    ax.set_xlabel("Matrix Learning Rate")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("Bracketed Learning-Rate Bowls")
    ax.legend(fontsize=7.2, ncol=2, frameon=True)
    ax.grid(alpha=0.3)
    ax.text(
        0.99,
        0.02,
        "star = interior best LR",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    print("plotted strict bowls:", ", ".join(plotted))
    _save_aliases(fig, "lr_sweep", "lr_bowls")


def _filter_from_step(steps: list[int], vals: list[float], start_step: int) -> tuple[list[int], list[float]]:
    kept = [(s, v) for s, v in zip(steps, vals) if s >= start_step and v > 0]
    return [s for s, _ in kept], [v for _, v in kept]


def fig_losses(args) -> None:
    """Train and validation loss in one report figure.

    The first points are dominated by the shared initialization loss. Cropping
    them makes optimizer differences visible without changing the data.
    """
    start_step = getattr(args, "start_step", 125)
    runs = [
        ("bowl_adamw_alr0p0012", "AdamW"),
        ("bowl_muon_mlr0p02", "Muon"),
        ("muon_svd_polar_wd_lr_400m_muon_2e-2_adam_0.002", "Muon-SVD"),
        ("bowl_dynmuon_mlr0p02", "DynMuon"),
        ("route_align_beta0p15_mlr0p02_20260611_clean", "Route-align"),
        ("bowl_relmuon_log1p_mlr0p1", "RelMuon-log1p"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(4.9, 4.15), sharex=True)
    panels = [
        ("train/loss", "Training Loss", axes[0]),
        ("val/loss", "Validation Loss", axes[1]),
    ]
    for metric, title, ax in panels:
        for run, label in runs:
            hist = _find_history(run)
            if hist is None:
                print(f"  (missing history for {run})")
                continue
            steps, vals = _series(hist, metric)
            steps, vals = _filter_from_step(steps, vals, start_step)
            if not steps:
                print(f"  (no {metric} points after step {start_step} for {run})")
                continue
            ax.plot(steps, vals, marker="o" if metric == "val/loss" else None,
                    ms=3, lw=1.8, label=label)
        ax.set_yscale("log")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[-1].set_xlabel("Training Step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=6.4,
        handlelength=1.6,
        columnspacing=0.8,
    )
    fig.tight_layout(h_pad=0.9, rect=(0, 0, 1, 0.88))
    _save(fig, "loss_late")


def fig_depth(args) -> None:
    """Heatmap of mean p_{t,l} - p_t per matrix type and block depth.

    The clock p_t is computed analytically from the run's logistic schedule, so
    no baseline run is needed. Shows where the router actually leans relative
    to having no per-layer personalization at all.
    """
    hist = _find_history(args.run)
    if hist is None:
        raise SystemExit(f"history for {args.run} not found; pull it first")
    total = args.total_steps
    per_type: dict[str, dict[int, float]] = {t: {} for t in LAYER_TYPES}
    for key in hist:
        if not key.startswith("route/p/"):
            continue
        name = key[len("route/p/"):]
        block = _BLOCK_RE.search(name)
        ltype = next((t for t in LAYER_TYPES if name.endswith(t + ".weight")), None)
        if block is None or ltype is None:
            continue
        steps, ps = _series(hist, key)
        deltas = [p - logistic_schedule_p(s, total, -0.25, 1.0, 0.04, 0.04)
                  for s, p in zip(steps, ps)]
        per_type[ltype][int(block.group(1))] = sum(deltas) / len(deltas)

    depths = sorted({d for values in per_type.values() for d in values})
    if not depths:
        raise SystemExit("no routed layer-depth metrics found")
    grid = np.full((len(LAYER_TYPES), len(depths)), np.nan)
    for i, ltype in enumerate(LAYER_TYPES):
        for j, depth in enumerate(depths):
            if depth in per_type[ltype]:
                grid[i, j] = per_type[ltype][depth]

    finite = grid[np.isfinite(grid)]
    limit = max(0.05, float(np.nanmax(np.abs(finite)))) if finite.size else 0.1
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths)
    ax.set_yticks(range(len(LAYER_TYPES)))
    ax.set_yticklabels([LAYER_LABELS[t] for t in LAYER_TYPES])
    ax.set_xlabel("Transformer Block Depth")
    ax.set_title("Where the Router Changes the DynMuon Clock")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"Mean Offset $p_{t,\ell} - p_t$")
    fig.tight_layout()
    _save(fig, "depth_routing")


def _route_offset_grid(run: str, depths: list[int], total_steps: int) -> np.ndarray:
    hist = _find_history(run)
    if hist is None:
        raise SystemExit(f"history for {run} not found; pull it first")
    values: dict[tuple[str, int], list[float]] = {}
    for key in hist:
        if not key.startswith("route/p/"):
            continue
        name = key[len("route/p/"):]
        block = _BLOCK_RE.search(name)
        ltype = next((t for t in LAYER_TYPES if name.endswith(t + ".weight")), None)
        if block is None or ltype is None:
            continue
        steps, ps = _series(hist, key)
        deltas = [
            p - logistic_schedule_p(s, total_steps, -0.25, 1.0, 0.04, 0.04)
            for s, p in zip(steps, ps)
        ]
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        values.setdefault((ltype, int(block.group(1))), []).append(mean_delta)

    grid = np.full((len(LAYER_TYPES), len(depths)), np.nan)
    for i, ltype in enumerate(LAYER_TYPES):
        for j, depth in enumerate(depths):
            cell = values.get((ltype, depth), [])
            if cell:
                grid[i, j] = statistics.mean(cell)
    return grid


def _route_depths(run: str) -> set[int]:
    hist = _find_history(run)
    if hist is None:
        raise SystemExit(f"history for {run} not found; pull it first")
    depths = set()
    for key in hist:
        if not key.startswith("route/p/"):
            continue
        block = _BLOCK_RE.search(key)
        if block is not None:
            depths.add(int(block.group(1)))
    return depths


def fig_proxy_depth(args) -> None:
    """Compact matrix-type-by-depth summary of routed p-offsets for proxy arms."""
    total_steps = getattr(args, "total_steps", 1526)
    all_depths: set[int] = set()
    for _, run in PROXY_DEPTH_RUNS:
        all_depths.update(_route_depths(run))
    depths = sorted(all_depths)
    if not depths:
        raise SystemExit("no routed layer-depth metrics found for proxy summary")

    grids = [(label, _route_offset_grid(run, depths, total_steps)) for label, run in PROXY_DEPTH_RUNS]
    finite = np.concatenate([grid[np.isfinite(grid)] for _, grid in grids])
    limit = max(0.05, float(np.nanmax(np.abs(finite)))) if finite.size else 0.1

    fig, axes = plt.subplots(
        len(grids), 1, figsize=(7.35, 6.25), sharex=True, constrained_layout=True
    )
    last_im = None
    for ax, (label, grid) in zip(axes, grids):
        last_im = ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(LAYER_TYPES)))
        ax.set_yticklabels([LAYER_SHORT_LABELS[t] for t in LAYER_TYPES], fontsize=6.8)
        ax.text(
            -0.10, 0.5, label, transform=ax.transAxes,
            ha="right", va="center", fontsize=7.2
        )
        ax.tick_params(axis="both", length=2)
    axes[-1].set_xticks(range(len(depths)))
    axes[-1].set_xticklabels(depths, fontsize=7)
    axes[-1].set_xlabel("Transformer Block")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    assert last_im is not None
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), pad=0.015, shrink=0.92)
    cbar.set_label(r"Mean Offset $p_{t,\ell} - p_t$")
    fig.suptitle("Proxy Routers Change Different Matrix Types at Different Depths", fontsize=10)
    _save(fig, "proxy_depth_summary")


def fig_beta(args) -> None:
    """Final validation loss vs router gain beta for the completed proxy arms."""
    baseline_runs = [
        "route_lrfix_beta0_mlr0p02_20260611_lrfix",
        "route_lrfix_beta0_mlr0p02_20260611_clean",
    ]
    baseline = statistics.mean([_value_for_run(run) for run in baseline_runs])
    cases = [
        ("Stable rank", {
            -0.15: ["proxy_stable_rank_negbeta_0p02"],
            0.0: baseline_runs,
            0.15: [
                "route_fill_beta0p15_mlr0p02_20260611_clean",
                "route_fill_beta0p15_mlr0p02_20260611_routefill",
                "seed1_route_0p02",
                "seed2_route_0p02",
            ],
            0.30: ["route_fill_beta0p3_mlr0p02_20260611_routefill"],
        }),
        ("Alignment", {
            -0.15: ["proxy_alignment_negbeta_0p02"],
            0.0: baseline_runs,
            0.15: ["route_align_beta0p15_mlr0p02_20260611_clean"],
            0.30: ["route_align_beta0p3_mlr0p02_20260611_clean"],
        }),
        ("SNR", {
            -0.15: ["proxy_snr_negbeta_0p02"],
            0.0: baseline_runs,
            0.15: ["proxy_snr_posbeta_0p02"],
            0.30: ["proxy_snr_beta0p3_0p02"],
        }),
        ("EMA-SNR", {
            -0.15: ["proxy_snr_ema_negbeta_0p02"],
            0.0: baseline_runs,
            0.15: ["proxy_snr_ema_posbeta_0p02"],
            0.30: ["proxy_snr_ema_beta0p3_0p02"],
        }),
    ]
    fig, ax = plt.subplots(figsize=(5.25, 3.25))
    ax.axhspan(-SEED_NOISE, SEED_NOISE, color="0.85", alpha=0.65,
               label=r"seed noise ($\pm0.001$)")
    ax.axhline(0.0, color="0.25", lw=1.0)
    markers = ["o", "s", "^", "D"]
    for (label, by_beta), marker in zip(cases, markers):
        betas = sorted(by_beta)
        means, stds = [], []
        for beta in betas:
            vals = [_value_for_run(run) for run in by_beta[beta]]
            mean, std = _mean_std(vals)
            means.append(mean - baseline)
            stds.append(std)
            if len(vals) > 1:
                ax.scatter(
                    [beta] * len(vals), [v - baseline for v in vals],
                    marker=marker, s=14, alpha=0.25, color="0.2", zorder=2,
                )
        ax.errorbar(betas, means, yerr=stds, marker=marker, lw=1.8,
                    capsize=3, label=label, zorder=3)
    ax.set_xlabel(r"Router gain $\beta$")
    ax.set_ylabel(r"$\Delta$ final validation loss vs $\beta=0$")
    ax.set_title("Proxy Gain Response at the Tuned LR")
    ax.set_xticks([-0.15, 0.0, 0.15, 0.30])
    ax.legend(fontsize=7.6, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "beta_sweep")


def fig_proxies(args) -> None:
    """Proxy comparison as delta from the no-route baseline at LR 0.02."""
    baseline_runs = [
        "route_lrfix_beta0_mlr0p02_20260611_lrfix",
        "route_lrfix_beta0_mlr0p02_20260611_clean",
    ]
    baseline_vals = [_value_for_run(run) for run in baseline_runs]
    baseline = statistics.mean(baseline_vals)
    cases = [
        ("Alignment\n$\\beta=+0.15$", [_value_for_run("route_align_beta0p15_mlr0p02_20260611_clean")]),
        ("SNR\n$\\beta=+0.15$", [_value_for_run("proxy_snr_posbeta_0p02")]),
        ("SNR\n$\\beta=-0.15$", [_value_for_run("proxy_snr_negbeta_0p02")]),
        ("Stable rank\n$\\beta=+0.15$", [
            _value_for_run("route_fill_beta0p15_mlr0p02_20260611_clean"),
            _value_for_run("seed1_route_0p02"),
            _value_for_run("seed2_route_0p02"),
        ]),
        ("EMA SNR\n$\\beta=-0.15$", [_value_for_run("proxy_snr_ema_negbeta_0p02")]),
    ]
    labels = [c[0] for c in cases]
    means, stds = zip(*[_mean_std(vals) for _, vals in cases])
    deltas = [m - baseline for m in means]
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]
    fig, ax = plt.subplots(figsize=(5.1, 3.25))
    ax.axhspan(-SEED_NOISE, SEED_NOISE, color="0.85", alpha=0.65,
               label=r"seed noise ($\pm0.001$)")
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.bar(labels, deltas, yerr=stds, capsize=3, width=0.58, color=colors, alpha=0.82)
    ax.set_ylabel(r"$\Delta$ final validation loss vs no-route")
    ax.set_title("Only Alignment Recovers the Baseline Neighborhood")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    _label_delta_bars(ax, list(deltas), list(stds))
    _save(fig, "proxy_comparison")


def fig_equivalence(args) -> None:
    """Empirical SVD == Newton-Schulz evidence for the DynMuon shaping.

    Left: relative error ||D_ns - D_svd|| / ||D_svd|| of the Gram-identity
    Newton-Schulz path vs the exact SVD across exponents p, for the cubic and
    the reference quintic polar iterations. Right: relative divergence of full
    optimizer trajectories run in the two compute modes for 30 steps. Local
    computation; W&B not needed.
    """
    import torch
    from src.optimizers.dynmuon import DynMuonRoute, shape_exact_ns, shape_exact_svd

    torch.manual_seed(0)
    ps = [x / 20 for x in range(-5, 21)]                 # -0.25 .. 1.0
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.35, 4.75), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    for variant, steps in (("quintic", 5), ("cubic", 30)):
        errs = []
        for p in ps:
            rel = []
            for _ in range(5):
                M = torch.randn(48, 64)
                d_svd = shape_exact_svd(M, p)
                d_ns = shape_exact_ns(M, p, ns_variant=variant, ns_steps=steps)
                rel.append(float(torch.linalg.norm(d_ns - d_svd) / torch.linalg.norm(d_svd)))
            errs.append(sum(rel) / len(rel))
        label = "5-step quintic NS (reference)" if variant == "quintic" else "30-step cubic NS"
        ax1.semilogy(ps, errs, marker=".", lw=1.7, label=label)
    ax1.set_xlabel("Spectral Exponent $p$")
    ax1.set_ylabel("Relative Operator Error")
    ax1.set_title("Single Shaped Update: Newton-Schulz vs Exact SVD")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")
    ax1.text(0.99, 0.08, "lower is closer to exact $U\\Sigma^pV^\\top$",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")

    def run(mode):
        torch.manual_seed(1)
        w = torch.zeros(32, 48, requires_grad=True)
        opt = DynMuonRoute([w], lr=0.02, momentum=0.95, nesterov=True,
                           routing_mode="global_schedule", compute_mode=mode,
                           ns_variant="cubic", ns_steps=30, adjust_lr_fn=None,
                           total_steps=30, track_proxies=False)
        traj = []
        for _ in range(30):
            w.grad = torch.randn(32, 48)
            opt.step()
            traj.append(w.detach().clone())
        return traj

    svd_traj, ns_traj = run("svd"), run("ns")
    div = [float(torch.linalg.norm(a - b) / torch.linalg.norm(a))
           for a, b in zip(svd_traj, ns_traj)]
    ax2.semilogy(range(1, 31), div, marker=".", lw=1.7, color="C2")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("Relative Weight Divergence")
    ax2.set_title("Full Optimizer Trajectory: SVD and NS Stay Numerically Identical")
    ax2.grid(alpha=0.3, which="both")
    _save(fig, "svd_ns_equivalence")


def fig_cost(args) -> None:
    """Two panels: seconds/step per method, and steps to a common val target."""
    rows = _load_summaries()
    # one representative (best final val) run per method label
    chosen: dict[str, dict] = {}
    for row in rows:
        run, val = row["run"], _flt(row, "final_val_loss")
        if val is None:
            continue
        for prefix, label in METHOD_LABELS.items():
            if run.startswith(prefix):
                if label not in chosen or val < _flt(chosen[label], "final_val_loss"):
                    chosen[label] = row
    if not chosen:
        raise SystemExit("no bowl summaries found")
    # common target: the worst best-val among methods (everyone reaches it)
    target = max(_flt(r, "best_val_loss") for r in chosen.values())
    labels, sps, stt = [], [], []
    for label in METHOD_LABELS.values():
        if label not in chosen:
            continue
        row = chosen[label]
        hist = _find_history(row["run"])
        steps_to = None
        if hist:
            for s, v in hist.get("val/loss", []):
                if v <= target:
                    steps_to = s
                    break
        labels.append(label)
        sps.append(_flt(row, "seconds_per_step") or 0.0)
        stt.append(steps_to or 0)
    colors = [METHOD_COLORS.get(label, "0.55") for label in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    ax1.bar(labels, sps, width=0.58, color=colors, edgecolor="0.25", linewidth=0.4, zorder=3)
    ax1.set_title("Per-step Runtime")
    ax1.set_ylabel("Seconds per Step")
    _label_positive_bars(ax1, list(sps), "{:.2f}")
    ax2.bar(labels, stt, width=0.58, color=colors, edgecolor="0.25", linewidth=0.4, zorder=3)
    ax2.set_title("Progress to Shared Target")
    ax2.set_ylabel(f"Steps to Validation Loss {target:.3f}")
    _label_positive_bars(ax2, [float(x) for x in stt], "{:.0f}")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(alpha=0.25, axis="y", zorder=0)
    _save(fig, "cost_comparison")
    print(f"  common target = {target:.4f}")


def fig_route_ablation(args) -> None:
    """Appendix route ablation: proxy, gain, and magnitude decomposition."""
    dyn = statistics.mean([
        _value_for_run("bowl_dynmuon_mlr0p02"),
        _value_for_run("seed1_dynmuon_0p02"),
        _value_for_run("seed2_dynmuon_0p02"),
    ])
    tuned_cases = [
        ("DynMuon\n(no route)", [_value_for_run("bowl_dynmuon_mlr0p02")]),
        ("Route\n$\\beta=0$", [
            _value_for_run("route_lrfix_beta0_mlr0p02_20260611_lrfix"),
            _value_for_run("route_lrfix_beta0_mlr0p02_20260611_clean"),
        ]),
        ("Route-align\n$\\beta=.15$", [_value_for_run("route_align_beta0p15_mlr0p02_20260611_clean")]),
        ("Stable-rank\n$\\beta=.15$", [
            _value_for_run("route_fill_beta0p15_mlr0p02_20260611_clean"),
            _value_for_run("seed1_route_0p02"),
            _value_for_run("seed2_route_0p02"),
        ]),
        ("Decoupled\nbest grid", [_value_for_run("route_lrgrid_decoupled_ref_mlr0p01_20260611_lrgrid")]),
    ]
    high_lr_cases = [
        ("DynMuon\n$\\eta=.2$", [
            _value_for_run("route_ctrl_dynmuon_0p2"),
            _value_for_run("review_route_dynmuon_mlr0p2_seed1"),
        ]),
        ("Route\n$\\eta=.2$", [
            _value_for_run("route_0p2"),
            _value_for_run("review_route_stable_mlr0p2_seed1"),
        ]),
        ("Route $\\beta=.3$\n$\\eta=.2$", [_value_for_run("route_beta0p3_0p2")]),
        ("Decoupled\n$\\eta=.2$", [_value_for_run("route_decoupled_0p2")]),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.3, 3.2), constrained_layout=True)
    for ax, cases, title, base in (
        (ax0, tuned_cases, "Tuned-LR Decomposition", dyn),
        (ax1, high_lr_cases, "High-LR Stress Test", _mean_std(high_lr_cases[0][1])[0]),
    ):
        labels = [c[0] for c in cases]
        means, stds = zip(*[_mean_std(vals) for _, vals in cases])
        deltas = [m - base for m in means]
        colors = ["#2ca02c" if d < -SEED_NOISE else ("0.55" if abs(d) <= SEED_NOISE else "#d62728")
                  for d in deltas]
        ax.axhspan(-SEED_NOISE, SEED_NOISE, color="0.88", alpha=0.75)
        ax.axhline(0.0, color="0.25", lw=1.0)
        ax.bar(labels, deltas, yerr=stds, capsize=3, color=colors, alpha=0.85, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(r"$\Delta$ final validation loss")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3, axis="y")
        _label_delta_bars(ax, list(deltas), list(stds))
    _save(fig, "route_ablation")


def fig_route_robustness(args) -> None:
    """Alignment-only route robustness across the completed LR points."""
    dyn = {
        0.01: ["bowl_dynmuon_mlr0p01"],
        0.02: ["bowl_dynmuon_mlr0p02", "seed1_dynmuon_0p02", "seed2_dynmuon_0p02"],
        0.05: ["bowl_dynmuon_mlr0p05", "review_route_dynmuon_mlr0p05_seed1"],
    }
    route = {
        0.01: ["route_align_beta0p15_mlr0p01_20260611_clean"],
        0.02: ["route_align_beta0p15_mlr0p02_20260611_clean"],
        0.05: ["route_align_beta0p15_mlr0p05_20260611_clean", "review_route_align_mlr0p05_seed1"],
    }

    def series(by_lr: dict[float, list[str]]) -> tuple[list[float], list[float], list[float], list[int]]:
        xs, means, stds, counts = [], [], [], []
        for lr in sorted(by_lr):
            vals = [_value_for_run(run) for run in by_lr[lr]]
            mean, std = _mean_std(vals)
            xs.append(lr)
            means.append(mean)
            stds.append(std)
            counts.append(len(vals))
        return xs, means, stds, counts

    x_dyn, y_dyn, e_dyn, n_dyn = series(dyn)
    x_route, y_route, e_route, n_route = series(route)
    gains = [d - r for d, r in zip(y_dyn, y_route)]
    gain_errs = [
        math.sqrt(de ** 2 + re ** 2) if nd > 1 or nr > 1 else 0.0
        for de, re, nd, nr in zip(e_dyn, e_route, n_dyn, n_route)
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    ax0.errorbar(x_dyn, y_dyn, yerr=e_dyn, marker="o", lw=2.0, capsize=3,
                 label="DynMuon")
    ax0.errorbar(x_route, y_route, yerr=e_route, marker="s", lw=2.0, capsize=3,
                 label=r"Route-align ($\beta=0.15$)")
    ax0.set_xscale("log")
    ax0.set_xticks(x_dyn)
    ax0.set_xticklabels(["0.01", "0.02", "0.05"])
    ax0.xaxis.set_minor_locator(mticker.NullLocator())
    ax0.set_xlabel("Matrix Learning Rate")
    ax0.set_ylabel("Final Validation Loss")
    ax0.set_title("Alignment Router Across Learning Rates")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    ax1.axhline(0.0, color="0.25", lw=1.0)
    ax1.axhspan(-SEED_NOISE, SEED_NOISE, color="0.88", alpha=0.75,
                label=r"seed noise")
    colors = ["#d62728" if g < -SEED_NOISE else ("0.55" if abs(g) <= SEED_NOISE else "#2ca02c")
              for g in gains]
    ax1.bar([str(x).rstrip("0").rstrip(".") for x in x_dyn], gains,
            yerr=gain_errs, capsize=3, color=colors, alpha=0.85, width=0.62)
    ax1.set_xlabel("Matrix Learning Rate")
    ax1.set_ylabel("DynMuon loss - Route-align loss")
    ax1.set_title("Positive Means Routing Helps")
    ax1.grid(alpha=0.3, axis="y")
    _label_delta_bars(ax1, list(gains), list(gain_errs))
    ax1.text(
        0.02, 0.98,
        "Single proxy: alignment.\nError bars show available seeds.",
        transform=ax1.transAxes, ha="left", va="top", fontsize=7, color="0.35",
    )
    _save(fig, "route_robustness")


def fig_relmuon_attention(args) -> None:
    """Appendix RelMuon spatial ablation: full vs attention-only."""
    rows = _load_all_summaries()
    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    specs = [
        ("Full RelMuon-log1p", lambda r: r.get("run", "").startswith("relmuon_full_log1p_"), "C3"),
        ("Attention-only RelMuon", lambda r: r.get("run", "").startswith("relmuon_attention_log1p_"), "C4"),
    ]
    for label, pred, color in specs:
        pts = _aggregate_by_lr(rows, pred, "muon_lr")
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", lw=2.0, color=color, label=label)
        best = _best_index(pts)
        if best is not None:
            ax.plot(xs[best], ys[best], marker="*", ms=12, color=color,
                    markeredgecolor="black", markeredgewidth=0.4)
    muon = statistics.mean([
        _value_for_run("bowl_muon_mlr0p02"),
        _value_for_run("seed1_muon_0p02"),
        _value_for_run("seed2_muon_0p02"),
    ])
    adam = _value_for_run("bowl_adamw_alr0p0012")
    ax.axhline(muon, color="C1", lw=1.2, ls="--", label="Muon best")
    ax.axhline(adam, color="C0", lw=1.2, ls=":", label="AdamW best")
    ax.set_xscale("log")
    ax.set_xlabel("Matrix Learning Rate")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("Attention-Only RelMuon Gives Up Most of the Gain")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    _save(fig, "relmuon_attention_ablation")


def fig_spectrum_controls(args) -> None:
    """Appendix controls: keep magnitude and perturb spectrum values/order."""
    baseline = statistics.mean([
        _value_for_run("bowl_muon_mlr0p02"),
        _value_for_run("seed1_muon_0p02"),
        _value_for_run("seed2_muon_0p02"),
    ])
    cases = [
        ("Muon\n3 seeds", baseline),
        ("Flat exact\npolar", _value_for_run("ctrl_power_0p02")),
        ("Random\nspectrum", _value_for_run("ctrl_random_0p02")),
        ("Kaon", _value_for_run("ctrl_kaon_0p02")),
        ("Inverted\nspectrum", _value_for_run("ctrl_inverted_0p02")),
    ]
    labels = [c[0] for c in cases]
    vals = [c[1] for c in cases]
    deltas = [v - baseline for v in vals]
    colors = ["0.55", "#4c78a8", "#54a24b", "#f58518", "#d62728"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhspan(-SEED_NOISE, SEED_NOISE, color="0.88", alpha=0.75,
               label=r"seed noise ($\pm0.001$)")
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.bar(labels, deltas, color=colors, width=0.62, alpha=0.85)
    ax.set_ylabel(r"$\Delta$ final validation loss vs Muon")
    ax.set_title("Spectrum Values Are Less Important Than Ordering")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    _label_delta_bars(ax, list(deltas))
    _save(fig, "spectrum_controls")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lr_sweep")
    sub.add_parser("bowls")  # backward-compatible alias
    l = sub.add_parser("losses")
    l.add_argument("--start-step", dest="start_step", type=int, default=125)
    d = sub.add_parser("depth")
    d.add_argument("--run", required=True)
    d.add_argument("--total-steps", dest="total_steps", type=int, default=1526)
    pd = sub.add_parser("proxy_depth")
    pd.add_argument("--total-steps", dest="total_steps", type=int, default=1526)
    b = sub.add_parser("beta")
    b.add_argument("--lr", default="0p02", help="LR tag in run names, e.g. 0p02")
    sub.add_parser("proxies")
    sub.add_parser("equivalence")
    sub.add_parser("cost")
    sub.add_parser("route_ablation")
    sub.add_parser("route_robustness")
    sub.add_parser("relmuon_attention")
    sub.add_parser("spectrum_controls")
    sub.add_parser("all")
    args = ap.parse_args()

    if args.cmd == "all":
        fig_lr_sweep(args)
        args.start_step = 125
        fig_losses(args)
        fig_equivalence(args)
        fig_cost(args)
        fig_route_ablation(args)
        fig_route_robustness(args)
        fig_relmuon_attention(args)
        fig_spectrum_controls(args)
        try:
            args.total_steps = 1526
            fig_proxy_depth(args)
        except SystemExit as e:
            print(f"skip proxy depth: {e}")
        for fn, label in ((fig_beta, "beta"), (fig_proxies, "proxies")):
            try:
                args.lr = "0p02"
                fn(args)
            except SystemExit as e:
                print(f"skip {label}: {e}")
        return
    {"lr_sweep": fig_lr_sweep, "bowls": fig_lr_sweep,
     "losses": fig_losses, "depth": fig_depth, "beta": fig_beta,
     "proxies": fig_proxies, "proxy_depth": fig_proxy_depth,
     "equivalence": fig_equivalence, "cost": fig_cost,
     "route_ablation": fig_route_ablation, "route_robustness": fig_route_robustness,
     "relmuon_attention": fig_relmuon_attention,
     "spectrum_controls": fig_spectrum_controls}[args.cmd](args)


if __name__ == "__main__":
    main()
