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
TIME_KEY = "time/train_seconds"


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


def _metric_history(run, metric: str) -> list[tuple[int, float]]:
    rows = []
    for row in run.scan_history(keys=["_step", metric], page_size=1000):
        value = row.get(metric)
        step = row.get("_step")
        if value is not None and step is not None:
            rows.append((int(step), float(value)))
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


def _first_to_target(history: list[tuple[int, float]], target: float) -> tuple[int, float] | None:
    for step, value in sorted(history):
        if value <= target:
            return step, value
    return None


def _time_at_step(time_history: list[tuple[int, float]], step: int) -> float | None:
    if not time_history:
        return None
    points = sorted(time_history)
    for time_step, value in points:
        if time_step == step:
            return value
    for (lo_step, lo_time), (hi_step, hi_time) in zip(points, points[1:]):
        if lo_step <= step <= hi_step and hi_step != lo_step:
            frac = (step - lo_step) / (hi_step - lo_step)
            return lo_time + frac * (hi_time - lo_time)
    return None


def _target_for_metric(rows: list[dict], metric: str, selection: str, target_series: str) -> float | None:
    y_key = f"{selection}_{_metric_name(metric)}"
    candidates = [float(row[y_key]) for row in rows if row["series"] == target_series and row.get(y_key) != ""]
    if not candidates:
        return None
    return min(candidates)


def _target_rows(
    rows: list[dict],
    runs_by_key: dict[tuple[str, str], object],
    metrics: list[str],
    selection: str,
    target_series: str,
) -> list[dict]:
    targets = {
        metric: _target_for_metric(rows, metric, selection, target_series)
        for metric in metrics
    }
    out = []
    history_cache: dict[tuple[tuple[str, str], str], list[tuple[int, float]]] = {}

    def hist(run_key: tuple[str, str], metric: str) -> list[tuple[int, float]]:
        cache_key = (run_key, metric)
        if cache_key not in history_cache:
            history_cache[cache_key] = _metric_history(runs_by_key[run_key], metric)
        return history_cache[cache_key]

    for row in rows:
        run_key = (row["project"], row["run_id"])
        if run_key not in runs_by_key:
            continue
        time_history = hist(run_key, TIME_KEY)
        for metric in metrics:
            target = targets[metric]
            if target is None:
                continue
            crossing = _first_to_target(hist(run_key, metric), target)
            step_to_target = crossing[0] if crossing else None
            value_at_target = crossing[1] if crossing else None
            time_to_target = _time_at_step(time_history, step_to_target) if step_to_target is not None else None
            out.append({
                "metric": metric,
                "target_series": target_series,
                "target_loss": target,
                "series": row["series"],
                "project": row["project"],
                "source": row["source"],
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "lr": row["lr"],
                "reached": crossing is not None,
                "step_to_target": step_to_target if step_to_target is not None else "",
                "time_to_target": time_to_target if time_to_target is not None else "",
                "value_at_target": value_at_target if value_at_target is not None else "",
            })
    return out


def _best_target_rows(target_rows: list[dict], field: str, metrics: list[str], series_order: list[str]) -> list[dict]:
    best = []
    for metric in metrics:
        for series in series_order:
            candidates = [
                row for row in target_rows
                if row["metric"] == metric and row["series"] == series and row[field] != ""
            ]
            if candidates:
                best.append(min(candidates, key=lambda row: float(row[field])))
    return best


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


def _plot_target_bars(
    path: str,
    target_rows: list[dict],
    metrics: list[str],
    series_order: list[str],
    field: str,
    ylabel: str,
    title: str,
) -> None:
    best_rows = _best_target_rows(target_rows, field, metrics, series_order)
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in best_rows:
        by_metric[row["metric"]].append(row)

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        metric_rows = by_metric.get(metric, [])
        names = [row["series"] for row in metric_rows]
        values = [float(row[field]) for row in metric_rows]
        bars = ax.bar(names, values)
        target = metric_rows[0]["target_loss"] if metric_rows else None
        target_series = metric_rows[0]["target_series"] if metric_rows else ""
        ax.set_title(f"{metric} to {target_series} target")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        if target is not None:
            ax.text(
                0.012,
                0.985,
                f"target loss = {float(target):.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
        for bar, row, value in zip(bars, metric_rows, values):
            label = f"{value:.1f}" if field == "time_to_target" else f"{int(value)}"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )
            ax.annotate(
                f"lr={float(row['lr']):.0e}",
                (bar.get_x() + bar.get_width() / 2, 0),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                color="white",
            )
        missing = [series for series in series_order if series not in names]
        if missing:
            ax.text(
                0.988,
                0.985,
                "not reached: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )

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
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "dynmuon-route-sweeps"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "cs-439-project"))
    ap.add_argument("--metrics", action="append", default=[],
                    help="Metrics to plot. Default: val/loss,train/loss.")
    ap.add_argument("--x-key", default="auto",
                    help="LR config key, or auto to try adam_lr, muon_lr, lr.")
    ap.add_argument("--state", action="append", default=["finished"],
                    help="W&B run state to include. Repeat or comma-separate; use 'all' to disable filtering.")
    ap.add_argument("--selection", choices=("final", "best"), default="final")
    ap.add_argument("--target-series", default="AdamW",
                    help="Series whose best plotted loss defines the threshold for time/step target plots.")
    ap.add_argument("--no-target-plots", action="store_true",
                    help="Only write the LR bowl CSV/PNG; skip Adam-target time/step plots.")
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
    runs_by_key = {}
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
                runs_by_key[(args.project, run.id)] = run
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
                runs_by_key[(project, run.id)] = run
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
    if not args.no_target_plots:
        target_rows = _target_rows(rows, runs_by_key, metrics, args.selection, args.target_series)
        if target_rows:
            target_csv_path = os.path.join(args.out_dir, f"{stem}_{args.target_series}_targets.csv")
            target_time_path = os.path.join(args.out_dir, f"{stem}_{args.target_series}_target_time.png")
            target_steps_path = os.path.join(args.out_dir, f"{stem}_{args.target_series}_target_steps.png")
            series_order = [label for label in labels if label in {row["series"] for row in rows}]
            _write_csv(target_csv_path, target_rows)
            _plot_target_bars(
                target_time_path,
                target_rows,
                metrics,
                series_order,
                "time_to_target",
                "training time (s)",
                f"Fastest wall-clock time to {args.target_series} loss",
            )
            _plot_target_bars(
                target_steps_path,
                target_rows,
                metrics,
                series_order,
                "step_to_target",
                "training steps",
                f"Fastest steps to {args.target_series} loss",
            )
            print(f"wrote {target_csv_path}")
            print(f"wrote {target_time_path}")
            print(f"wrote {target_steps_path}")
        else:
            print(f"warning: no target plots written; no rows found for target series {args.target_series!r}")
    for metric in metrics:
        y_key = f"{args.selection}_{_metric_name(metric)}"
        for series in sorted({r["series"] for r in rows}):
            series_rows = [r for r in rows if r["series"] == series and r.get(y_key) != ""]
            if not series_rows:
                continue
            best = min(series_rows, key=lambda r: float(r[y_key]))
            print(f"best {args.selection} {metric} [{series}]: lr={best['lr']:.6g}, loss={float(best[y_key]):.6f}")
        if not args.no_target_plots:
            metric_target_rows = [r for r in target_rows if r["metric"] == metric] if target_rows else []
            for row in _best_target_rows(metric_target_rows, "step_to_target", metrics, labels):
                print(
                    f"fastest steps to {args.target_series} {metric} [{row['series']}]: "
                    f"lr={float(row['lr']):.6g}, step={int(row['step_to_target'])}"
                )
            for row in _best_target_rows(metric_target_rows, "time_to_target", metrics, labels):
                print(
                    f"fastest time to {args.target_series} {metric} [{row['series']}]: "
                    f"lr={float(row['lr']):.6g}, time={float(row['time_to_target']):.2f}s"
                )


if __name__ == "__main__":
    main()
