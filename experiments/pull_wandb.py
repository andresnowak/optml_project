"""Pull run histories from a remote W&B project into local history JSONs.

The dumps use the MemoryLogger history format ``{metric: [(step, value), ...]}``
so every existing analysis tool works on remote runs unchanged:

    python experiments/pull_wandb.py --group route_arms          # one sweep group
    python experiments/pull_wandb.py --name bowl_muon_mlr0p02    # one run
    python experiments/pull_wandb.py --group bowl_dynmuon --keys "val/loss,train/loss,lr"
    python experiments/cost_table.py --results-dir results/wandb/route_arms
    python experiments/depth_routing.py --routed results/wandb/route_arms/history_route_0p2.json ...

Also writes ``summary.csv`` per group (final/best val loss, wall-clock,
seconds/step, key config) for quick comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analysis  # noqa: E402

DEFAULT_ENTITY = "cs-439-project"
DEFAULT_PROJECT = "dynmuon-route-sweeps"
# Metric prefixes worth dumping by default (full route/* is large but needed
# for depth_routing; val/train/lr/time are what cost_table consumes).
DEFAULT_KEY_PREFIXES = ("val/", "train/", "lr", "time/", "tokens/", "route/", "weight_update/", "weight_svd/")
CONFIG_COLUMNS = (
    "matrix_optimizer", "routing_mode", "compute_mode", "magnitude", "spectrum",
    "muon_lr", "adam_lr", "embed_lr", "weight_decay", "beta", "lean_norm",
    "lean_max", "modulate_metric", "dynamic_ref", "ref_decay", "train_steps", "seed",
)


def _project_from_url(url: str | None) -> tuple[str | None, str | None]:
    """Extract ``entity/project`` from a W&B workspace/run URL."""
    if not url:
        return None, None
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def _config_value(config: dict, key: str):
    """Read a top-level config value, falling back to the active route block.

    W&B stores the full nested YAML config. For defaults like
    ``route.schedule_modulated.beta`` there is no top-level ``beta`` unless the
    CLI overrode it, so summaries must resolve the effective nested value.
    """
    if key in config:
        return config.get(key)
    routing_mode = config.get("routing_mode")
    route_cfg = config.get("route", {})
    mode_cfg = route_cfg.get(routing_mode, {}) if isinstance(route_cfg, dict) else {}
    if key == "modulate_metric":
        return mode_cfg.get("metric")
    return mode_cfg.get(key) if isinstance(mode_cfg, dict) else None


def pull_wandb_logs(
    *,
    url: str | None = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    group: str | None = None,
    name: str | None = None,
    key_prefixes: tuple[str, ...] = DEFAULT_KEY_PREFIXES,
    out_dir: str | None = None,
    include_running: bool = False,
) -> list[dict]:
    """Download matching remote W&B runs into local analysis-ready files.

    This is the programmatic entry point. Histories use the repository's
    ``MemoryLogger`` JSON shape so existing scripts can consume remote logs as
    if they were produced locally.
    """
    url_entity, url_project = _project_from_url(url)
    return pull_runs(
        entity=url_entity or entity,
        project=url_project or project,
        group=group,
        name=name,
        key_prefixes=key_prefixes,
        out_dir=out_dir,
        include_running=include_running,
    )


def pull_runs(
    *,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    group: str | None = None,
    name: str | None = None,
    key_prefixes: tuple[str, ...] = DEFAULT_KEY_PREFIXES,
    out_dir: str | None = None,
    include_running: bool = False,
) -> list[dict]:
    """Download matching runs and dump per-run history JSONs + a summary CSV.

    ``group``/``name`` accept fnmatch patterns (``bowl_*``). Returns the list
    of summary rows (one per run). Histories are written to
    ``results/wandb/<group or 'runs'>/history_<run_name>.json``.
    """
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    out_base = out_dir or os.path.join("results", "wandb", group or name or "runs")
    os.makedirs(out_base, exist_ok=True)

    for run in api.runs(f"{entity}/{project}", order="-created_at"):
        run_group = run.config.get("wandb_group") or run.group or ""
        if group and not fnmatch.fnmatch(run_group, group):
            continue
        if name and not fnmatch.fnmatch(run.name, name):
            continue
        if run.state == "running" and not include_running:
            print(f"skip (running): {run.name}")
            continue

        history: dict[str, list[tuple[int, float]]] = {}
        for row in run.scan_history(page_size=2000):
            step = row.get("_step")
            if step is None:
                continue
            for key, value in row.items():
                if key.startswith("_") or not isinstance(value, (int, float)):
                    continue
                if not any(key.startswith(p) for p in key_prefixes):
                    continue
                history.setdefault(key, []).append((int(step), float(value)))
        for series in history.values():
            series.sort(key=lambda sv: sv[0])

        path = os.path.join(out_base, f"history_{run.name}.json")
        analysis.dump_history(history, path)

        summary = dict(run.summary)
        seconds = summary.get("time/train_seconds")
        steps = run.config.get("train_steps")
        row = {
            "run_id": run.id,
            "run": run.name,
            "run_url": run.url,
            "group": run_group,
            "state": run.state,
            "final_val_loss": analysis.final_value(history, "val/loss"),
            "best_val_loss": analysis.min_value(history, "val/loss"),
            "train_seconds": seconds,
            "seconds_per_step": (seconds / steps) if seconds and steps else None,
            **{k: _config_value(run.config, k) for k in CONFIG_COLUMNS},
        }
        rows.append(row)
        print(f"pulled {run.name}: {len(history)} metrics -> {path}")

    if rows:
        csv_path = os.path.join(out_base, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["group"], r["run"])))
        print(f"wrote {csv_path} ({len(rows)} runs)")
    else:
        print("no runs matched")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--url", help="W&B workspace/run URL; overrides --entity/--project when present")
    ap.add_argument("--group", help="fnmatch pattern on the W&B group, e.g. bowl_*")
    ap.add_argument("--name", help="fnmatch pattern on the run name")
    ap.add_argument("--keys", help="comma-separated metric prefixes (default: val/train/lr/time/route/...)")
    ap.add_argument("--out-dir", dest="out_dir")
    ap.add_argument("--include-running", action="store_true")
    args = ap.parse_args()
    prefixes = tuple(k.strip() for k in args.keys.split(",")) if args.keys else DEFAULT_KEY_PREFIXES
    pull_wandb_logs(url=args.url, entity=args.entity, project=args.project,
                    group=args.group, name=args.name, key_prefixes=prefixes,
                    out_dir=args.out_dir, include_running=args.include_running)


if __name__ == "__main__":
    main()
