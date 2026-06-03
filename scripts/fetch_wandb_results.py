"""Pull WandB runs into local artifacts for analysis.

Default project/entity match the rest of the codebase (`mlo-specmuon` /
`cs-439-project`). Use `--filter` to grab a subset by run-name regex
(e.g. `--filter 'shakespeare.*top_k=(0|32)'` for the §4.2 comparison).

Outputs land under ``results/wandb_export/<timestamp>/``:

  runs/<safe-name>/history.csv     per-step metrics (loss, lr, ι, ...)
  runs/<safe-name>/summary.json    final-step values + best-loss summary
  runs/<safe-name>/config.json     hyperparameters
  summary.md                       human-readable table; cat to paste in chat
  loss_comparison.png              overlay of loss curves (log-y if requested)

Usage:
    uv run python scripts/fetch_wandb_results.py
    uv run python scripts/fetch_wandb_results.py --filter "selsav|shakespeare-long"
    uv run python scripts/fetch_wandb_results.py --metric loss --log-y
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import wandb


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", name)[:120]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="cs-439-project")
    p.add_argument("--project", default="mlo-specmuon")
    p.add_argument("--filter", default=None,
                   help="Regex filter on run NAME (not ID). All runs if omitted.")
    p.add_argument("--tags", nargs="*", default=None,
                   help="Optional: only runs that carry ALL of these tags.")
    p.add_argument("--metric", default="loss",
                   help="Metric key to overlay in the comparison plot (default: loss).")
    p.add_argument("--log-y", action="store_true",
                   help="Log-y axis for the comparison plot (use for loss).")
    p.add_argument("--max-runs", type=int, default=50,
                   help="Cap total runs pulled (default 50) so a runaway regex "
                        "doesn't download a thousand runs.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory; default results/wandb_export/<stamp>/.")
    args = p.parse_args()

    api = wandb.Api()
    runs_iter = api.runs(f"{args.entity}/{args.project}",
                         filters={"tags": {"$all": args.tags}} if args.tags else None)
    pattern = re.compile(args.filter) if args.filter else None

    selected = []
    for run in runs_iter:
        if pattern and not pattern.search(run.name):
            continue
        selected.append(run)
        if len(selected) >= args.max_runs:
            break

    if not selected:
        print(f"No runs matched filter={args.filter!r} tags={args.tags!r} in "
              f"{args.entity}/{args.project}.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir or f"results/wandb_export/{time.strftime('%Y%m%d-%H%M%S')}")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Per-run dumps + collect (name, df) for the comparison plot.
    rows: list[dict] = []
    series: list[tuple[str, list[int], list[float]]] = []
    for run in selected:
        safe = _safe(run.name)
        rdir = runs_dir / safe
        rdir.mkdir(parents=True, exist_ok=True)

        # history() can fail on very large runs without pandas — fall back to
        # scan_history() which is paginated and dict-based.
        hist_records = list(run.scan_history(keys=["_step", args.metric, "loss", "lr"],
                                              page_size=10_000))
        if hist_records:
            # Write CSV by hand to avoid the pandas dep on the cluster (it's
            # in the uv env locally; cluster invocation goes through the
            # entry script which pip-installs only wandb/pyyaml/matplotlib).
            keys = sorted({k for r in hist_records for k in r.keys()})
            with (rdir / "history.csv").open("w") as f:
                f.write(",".join(keys) + "\n")
                for r in hist_records:
                    f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

        (rdir / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, default=str))
        (rdir / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str))

        metric_vals = [r[args.metric] for r in hist_records
                       if args.metric in r and r[args.metric] is not None]
        steps = [r["_step"] for r in hist_records
                 if args.metric in r and r[args.metric] is not None]
        final = metric_vals[-1] if metric_vals else None
        best = min(metric_vals) if metric_vals else None
        rows.append({
            "name": run.name,
            "id": run.id,
            "state": run.state,
            f"final_{args.metric}": final,
            f"best_{args.metric}": best,
            "steps_logged": len(metric_vals),
        })
        if steps and metric_vals:
            series.append((run.name, steps, metric_vals))

    # Markdown summary — designed to be cat'd into chat.
    md = [f"# WandB results — {args.project}",
          "",
          f"Filter: `{args.filter or '<all>'}`  |  tags: `{args.tags or '<any>'}`  |  N = {len(selected)}",
          "",
          f"| run | state | final `{args.metric}` | best `{args.metric}` | steps |",
          "|---|---|---|---|---|"]
    rows.sort(key=lambda r: (r[f"best_{args.metric}"] if r[f"best_{args.metric}"] is not None else float("inf")))
    for r in rows:
        fb = f"{r[f'best_{args.metric}']:.4g}" if r[f"best_{args.metric}"] is not None else "—"
        ff = f"{r[f'final_{args.metric}']:.4g}" if r[f"final_{args.metric}"] is not None else "—"
        md.append(f"| {r['name']} | {r['state']} | {ff} | {fb} | {r['steps_logged']} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    # Comparison plot — single panel, one curve per run.
    if series:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for name, steps, vals in series:
            ax.plot(steps, vals, label=name, linewidth=1.0, alpha=0.85)
        if args.log_y:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(args.metric + (" (log)" if args.log_y else ""))
        ax.set_title(f"{args.project} — {args.metric}")
        ax.legend(fontsize=7, loc="best")
        fig.savefig(out_dir / "loss_comparison.png", dpi=150, bbox_inches="tight")

    print(f"Wrote {out_dir}/")
    print(f"  - summary.md           ({len(rows)} runs)")
    print(f"  - loss_comparison.png  ({len(series)} curves)")
    print(f"  - runs/<name>/         per-run history.csv, summary.json, config.json")
    print()
    print("Paste `summary.md` into chat and attach loss_comparison.png to share with me.")


if __name__ == "__main__":
    main()
