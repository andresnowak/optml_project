"""Pull W&B LR sweep runs and plot train/validation loss bowls."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

LR_KEYS = ("muon_lr", "adam_lr", "lr")
DEFAULT_METRICS = ("val/loss", "train/loss")


def _split_csv(values: list[str]) -> list[str]:
    out = []
    for value in values:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def _config_value(config: dict, key: str):
    return config.get(key, config.get(key.replace("_", "-")))


def _metric_name(metric: str) -> str:
    return metric.replace("/", "_").replace("-", "_")


def _summary_value(run, metric: str, *, best: bool = False):
    if best:
        keys = (f"{metric}.min", f"{metric}_min", f"best_{metric}", f"best_{_metric_name(metric)}")
    else:
        keys = (metric,)
    for key in keys:
        value = run.summary.get(key)
        if value is not None:
            return float(value)
    return None


def _lr_value(config: dict, x_key: str) -> tuple[str, float]:
    if x_key == "auto":
        optimizer = _config_value(config, "matrix_optimizer")
        keys = ("adam_lr", "lr", "muon_lr") if optimizer == "adamw" else LR_KEYS
    else:
        keys = (x_key,)
    for key in keys:
        value = _config_value(config, key)
        if value is not None:
            return key, float(value)
    raise ValueError(f"missing LR config value; tried {', '.join(keys)}")


def _metric_history(run, metric: str) -> list[tuple[int | None, float]]:
    rows = []
    for row in run.scan_history(keys=["_step", metric], page_size=1000):
        value = row.get(metric)
        if value is not None:
            rows.append((row.get("_step"), float(value)))
    return rows


def _run_row(run, source: str, label: str, metrics: list[str], x_key: str, project: str, selection: str) -> dict:
    lr_key, lr = _lr_value(run.config, x_key)
    row = {
        "source": source,
        "project": project,
        "group": run.group or "",
        "series": label,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "matrix_optimizer": _config_value(run.config, "matrix_optimizer") or "",
        "lr_key": lr_key,
        "lr": lr,
    }

    found = False
    for metric in metrics:
        name = _metric_name(metric)
        final_value = _summary_value(run, metric)
        best_value = _summary_value(run, metric, best=True)
        needs_history = final_value is None or (selection == "best" and best_value is None)
        if needs_history:
            hist = _metric_history(run, metric)
            if not hist:
                row[f"final_{name}_step"] = ""
                row[f"final_{name}"] = ""
                row[f"best_{name}_step"] = ""
                row[f"best_{name}"] = ""
                continue
            found = True
            final_step, final_value = hist[-1]
            best_step, best_value = min(hist, key=lambda item: item[1])
            row[f"final_{name}_step"] = final_step
            row[f"final_{name}"] = final_value
            row[f"best_{name}_step"] = best_step
            row[f"best_{name}"] = best_value
        else:
            found = True
            row[f"final_{name}_step"] = run.summary.get("_step", "")
            row[f"final_{name}"] = final_value
            row[f"best_{name}_step"] = ""
            row[f"best_{name}"] = best_value if best_value is not None else ""
    if not found:
        raise ValueError(f"{run.name} has none of the requested metrics")
    return row


def _write_csv(path: str, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_exclude_lr_ranges(values: list[str]) -> list[tuple[str, float, float]]:
    ranges = []
    for value in _split_csv(values):
        try:
            series, lo, hi = value.split(":", 2)
        except ValueError as exc:
            raise ValueError(f"bad --exclude-lr-range {value!r}; expected SERIES:LOW:HIGH") from exc
        ranges.append((series, float(lo), float(hi)))
    return ranges


def _filter_rows(rows: list[dict], exclude_lr_ranges: list[tuple[str, float, float]]) -> list[dict]:
    if not exclude_lr_ranges:
        return rows
    kept = []
    for row in rows:
        lr = float(row["lr"])
        drop = any(row["series"] == series and lo <= lr <= hi for series, lo, hi in exclude_lr_ranges)
        if not drop:
            kept.append(row)
    return kept


def _plot_metric(ax, rows: list[dict], metric: str, selection: str) -> None:
    y_key = f"{selection}_{_metric_name(metric)}"
    by_series: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get(y_key) != "":
            by_series[row["series"]].append(row)

    for series, series_rows in sorted(by_series.items()):
        series_rows = sorted(series_rows, key=lambda r: r["lr"])
        xs = np.array([math.log10(r["lr"]) for r in series_rows], dtype=float)
        ys = np.array([float(r[y_key]) for r in series_rows], dtype=float)
        (line,) = ax.plot(xs, ys, "o-", label=series)
        color = line.get_color()
        for row, x, y in zip(series_rows, xs, ys):
            ax.annotate(f"{row['lr']:.0e}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

        best_idx = int(np.argmin(ys))
        best_x = xs[best_idx]
        best_y = ys[best_idx]
        ax.axvline(best_x, color=color, linestyle=":", linewidth=1.2, alpha=0.32)
        ax.scatter(
            [best_x],
            [best_y],
            s=190,
            marker="*",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        if len(series_rows) >= 3:
            coeff = np.polyfit(xs, ys, deg=2)
            if coeff[0] > 0:
                grid = np.linspace(xs.min(), xs.max(), 200)
                ax.plot(grid, np.polyval(coeff, grid), "--", alpha=0.45)

    ax.set_title(f"{selection} {metric}")
    ax.set_xlabel("log10(lr)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.text(0.012, 0.985, "star = best point", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.legend()


def _plot(path: str, rows: list[dict], metrics: list[str], selection: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        _plot_metric(ax, rows, metric, selection)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", action="append", default=[],
                    help="W&B group name. Repeat or comma-separate to compare methods.")
    ap.add_argument("--sweep-project", action="append", default=[],
                    help="W&B project containing one LR sweep. Repeat or comma-separate to compare methods.")
    ap.add_argument("--sweep-projects", nargs="+", default=[],
                    help="List of W&B projects, each containing one LR sweep. Values may also be comma-separated.")
    ap.add_argument("--label", action="append",
                    help="Series label for each group/project. Defaults to the group/project name.")
    ap.add_argument("--labels", nargs="+", default=[],
                    help="List of series labels matching --group/--sweep-projects order. Values may also be comma-separated.")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "dynmuon-route"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "cs-439-project"))
    ap.add_argument("--metrics", action="append", default=[],
                    help="Metrics to plot. Default: val/loss,train/loss.")
    ap.add_argument("--x-key", default="auto",
                    help="LR config key, or auto to try adam_lr, muon_lr, lr.")
    ap.add_argument("--state", action="append", default=["finished"],
                    help="W&B run state to include. Repeat or comma-separate; use 'all' to disable filtering.")
    ap.add_argument("--selection", choices=("final", "best"), default="final")
    ap.add_argument("--exclude-lr-range", action="append", default=[],
                    help="Drop rows for a plotted series in inclusive SERIES:LOW:HIGH LR range.")
    ap.add_argument("--out-dir", default="results/lr_bowls")
    args = ap.parse_args()

    groups = _split_csv(args.group)
    sweep_projects = _split_csv(args.sweep_project + args.sweep_projects)
    sources = groups + sweep_projects
    if not sources:
        ap.error("provide at least one --group, --sweep-project, or --sweep-projects")
    labels = _split_csv((args.label or []) + args.labels)
    if labels and len(labels) != len(sources):
        ap.error("--label/--labels count must match --group/--sweep-projects count")
    if not labels:
        labels = sources
    metrics = _split_csv(args.metrics) or list(DEFAULT_METRICS)
    states = set(_split_csv(args.state))
    try:
        exclude_lr_ranges = _parse_exclude_lr_ranges(args.exclude_lr_range)
    except ValueError as exc:
        ap.error(str(exc))

    import wandb

    rows = []
    label_iter = iter(labels)
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    for group in groups:
        label = next(label_iter)
        runs = list(wandb.Api().runs(path, filters={"group": group}))
        if not runs:
            print(f"warning: no W&B runs found for group {group!r} in {path}")
            continue
        for run in runs:
            if "all" not in states and run.state not in states:
                print(f"skipping {group}: {run.name} has state {run.state!r}")
                continue
            try:
                rows.append(_run_row(run, group, label, metrics, args.x_key, args.project, args.selection))
            except ValueError as exc:
                print(f"skipping {group}: {exc}")
    for project in sweep_projects:
        label = next(label_iter)
        project_path = f"{args.entity}/{project}" if args.entity else project
        runs = list(wandb.Api().runs(project_path))
        if not runs:
            print(f"warning: no W&B runs found in project {project_path}")
            continue
        for run in runs:
            if "all" not in states and run.state not in states:
                print(f"skipping {project}: {run.name} has state {run.state!r}")
                continue
            try:
                rows.append(_run_row(run, project, label, metrics, args.x_key, project, args.selection))
            except ValueError as exc:
                print(f"skipping {project}: {exc}")
    if not rows:
        raise RuntimeError("no runs had an LR config and one of the requested metrics")
    rows = _filter_rows(rows, exclude_lr_ranges)
    if not rows:
        raise RuntimeError("all rows were filtered out")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = "_vs_".join(sources)
    csv_path = os.path.join(args.out_dir, f"{stem}.csv")
    png_path = os.path.join(args.out_dir, f"{stem}_{args.selection}.png")
    rows = sorted(rows, key=lambda r: (r["series"], r["lr"]))
    _write_csv(csv_path, rows)
    _plot(png_path, rows, metrics, args.selection, stem)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    for metric in metrics:
        y_key = f"{args.selection}_{_metric_name(metric)}"
        for series in sorted({r["series"] for r in rows}):
            series_rows = [r for r in rows if r["series"] == series and r.get(y_key) != ""]
            if not series_rows:
                continue
            best = min(series_rows, key=lambda r: float(r[y_key]))
            print(f"best {args.selection} {metric} [{series}]: lr={best['lr']:.6g}, loss={float(best[y_key]):.6f}")


if __name__ == "__main__":
    main()
