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

For reviewer-facing reproduction without W&B access, this script can also pack
the local ``results/wandb`` cache into a compressed JSONL bundle:

    python experiments/pull_wandb.py --bundle-only \
        --bundle-out experiments/report_wandb_bundle.jsonl.gz
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import analysis  # noqa: E402

DEFAULT_ENTITY = "cs-439-project"
DEFAULT_PROJECT = "dynmuon-route-sweeps"
DEFAULT_BUNDLE = os.path.join("experiments", "report_wandb_bundle.jsonl.gz")
# Metric prefixes worth dumping by default (full route/* is large but needed
# for depth_routing; val/train/lr/time are what cost_table consumes).
DEFAULT_KEY_PREFIXES = ("val/", "train/", "lr", "time/", "tokens/", "route/", "weight_update/", "weight_svd/")
CONFIG_COLUMNS = (
    "matrix_optimizer", "routing_mode", "compute_mode", "magnitude", "spectrum",
    "orthogonalize", "muon_lr", "adam_lr", "embed_lr", "weight_decay", "beta", "lean_norm",
    "lean_max", "modulate_metric", "dynamic_ref", "ref_decay", "train_steps", "seed",
)


def _safe_bundle_path(rel_path: str) -> str:
    """Reject paths that could escape the restored cache directory."""
    norm = os.path.normpath(rel_path)
    if norm.startswith("..") or os.path.isabs(norm):
        raise ValueError(f"unsafe bundle path: {rel_path}")
    return norm


def _iter_cache_files(source_dir: str, group_patterns: tuple[str, ...] | None):
    """Yield report cache files as ``(relative_path, absolute_path)``."""
    for root, dirs, files in os.walk(source_dir):
        rel_dir = os.path.relpath(root, source_dir)
        top = "" if rel_dir == "." else rel_dir.split(os.sep)[0]
        if rel_dir == "." and group_patterns:
            dirs[:] = [d for d in dirs if any(fnmatch.fnmatch(d, p) for p in group_patterns)]
        elif group_patterns and not any(fnmatch.fnmatch(top, p) for p in group_patterns):
            dirs[:] = []
            continue
        for filename in sorted(files):
            if filename != "summary.csv" and not (filename.startswith("history_") and filename.endswith(".json")):
                continue
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, source_dir)
            yield rel_path, abs_path


def bundle_cache(
    *,
    source_dir: str = os.path.join("results", "wandb"),
    bundle_out: str = DEFAULT_BUNDLE,
    group_patterns: tuple[str, ...] | None = None,
    history_key_prefixes: tuple[str, ...] | None = None,
) -> int:
    """Pack local W&B CSV/JSON cache files into a compressed JSONL bundle.

    The bundle is streamable: one manifest line followed by one JSON object per
    file. It is intentionally a repository-native data artifact, not a W&B API
    dependency.
    """
    os.makedirs(os.path.dirname(bundle_out) or ".", exist_ok=True)
    count = 0
    with gzip.open(bundle_out, "wt", encoding="utf-8") as f:
        manifest = {
            "type": "manifest",
            "schema": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": source_dir,
            "group_patterns": list(group_patterns or []),
            "history_key_prefixes": list(history_key_prefixes or []),
        }
        f.write(json.dumps(manifest, separators=(",", ":")) + "\n")
        for rel_path, abs_path in _iter_cache_files(source_dir, group_patterns):
            with open(abs_path, encoding="utf-8") as src:
                text = src.read()
            if history_key_prefixes and os.path.basename(abs_path).startswith("history_"):
                history = json.loads(text)
                history = {
                    k: v for k, v in history.items()
                    if any(k.startswith(prefix) for prefix in history_key_prefixes)
                }
                text = json.dumps(history, separators=(",", ":"))
            record = {
                "type": "file",
                "path": rel_path.replace(os.sep, "/"),
                "text": text,
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    print(f"wrote bundle {bundle_out} ({count} files)", flush=True)
    return count


def restore_bundle(
    *,
    bundle_path: str = DEFAULT_BUNDLE,
    out_dir: str = os.path.join("results", "wandb"),
    clean: bool = True,
) -> int:
    """Restore a compressed JSONL bundle into a local W&B-style cache."""
    if clean and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if line_no == 1 and record.get("type") == "manifest":
                continue
            if record.get("type") != "file":
                raise ValueError(f"unexpected bundle record on line {line_no}")
            rel_path = _safe_bundle_path(record["path"])
            path = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as out:
                out.write(record["text"])
            count += 1
    print(f"restored {count} files from {bundle_path} -> {out_dir}", flush=True)
    return count


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
    ap.add_argument("--bundle-out", default=None,
                    help=f"write a compressed offline bundle after pulling/cache scan (default path: {DEFAULT_BUNDLE})")
    ap.add_argument("--bundle-only", action="store_true",
                    help="do not contact W&B; bundle the existing local cache")
    ap.add_argument("--bundle-source", default=os.path.join("results", "wandb"),
                    help="local cache directory to bundle")
    ap.add_argument("--bundle-groups",
                    help="comma-separated fnmatch patterns for cache group dirs to include")
    ap.add_argument("--bundle-history-keys",
                    help="comma-separated metric prefixes to keep inside bundled history JSONs")
    ap.add_argument("--restore-bundle",
                    help="restore this compressed bundle into --restore-out-dir and exit")
    ap.add_argument("--restore-out-dir", default=os.path.join("results", "wandb"))
    args = ap.parse_args()
    if args.restore_bundle:
        restore_bundle(bundle_path=args.restore_bundle, out_dir=args.restore_out_dir, clean=True)
        return
    bundle_groups = tuple(g.strip() for g in args.bundle_groups.split(",")) if args.bundle_groups else None
    bundle_history_keys = (
        tuple(k.strip() for k in args.bundle_history_keys.split(","))
        if args.bundle_history_keys else None
    )
    if args.bundle_only:
        bundle_cache(source_dir=args.bundle_source,
                     bundle_out=args.bundle_out or DEFAULT_BUNDLE,
                     group_patterns=bundle_groups,
                     history_key_prefixes=bundle_history_keys)
        return
    prefixes = tuple(k.strip() for k in args.keys.split(",")) if args.keys else DEFAULT_KEY_PREFIXES
    pull_wandb_logs(url=args.url, entity=args.entity, project=args.project,
                    group=args.group, name=args.name, key_prefixes=prefixes,
                    out_dir=args.out_dir, include_running=args.include_running)
    if args.bundle_out:
        bundle_cache(source_dir=args.bundle_source,
                     bundle_out=args.bundle_out,
                     group_patterns=bundle_groups,
                     history_key_prefixes=bundle_history_keys)


if __name__ == "__main__":
    main()
