"""Reproduce every result, table, and figure in the report from W&B logs.

    python run.py                 # pull all sweep groups + regenerate all figures
    python run.py --skip-pull     # figures only, from the local results/wandb dumps
    python run.py --train         # additionally retrain every arm (cluster; days of GPU)

Pipeline (all steps are idempotent):
  1. Pull the run histories of every report sweep group from
     wandb.ai/cs-439-project/dynmuon-route-sweeps into results/wandb/
     (experiments/pull_wandb.py).
  2. Regenerate the report figures into report/figures/
     (experiments/report_figures.py): LR bowls, loss curves, depth-resolved
     routing, beta sweep, proxy comparison, SVD-vs-Newton-Schulz evidence,
     and the cost comparison.
  3. Print the summary tables used in the report.

Training itself runs on the cluster via scripts/sweeps.sh (see README); the
exact submission commands for every report run are:

    scripts/sweeps.sh bowls && scripts/sweeps.sh route 0.02 && scripts/sweeps.sh final 0.02
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

PY = sys.executable

REPORT_GROUPS = (
    "bowl_dynmuon", "bowl_muon", "bowl_adamw", "bowl_relmuon",
    "route_arms", "route_lrfix*", "route_lrgrid*", "route_fill*",
    "route_alignment*", "route_proxies", "spectrum_controls", "seed_replicates",
)
# Loss-curve panel: best-LR baselines vs the routed variants.
CURVE_RUNS = ("bowl_muon_mlr0p02,bowl_dynmuon_mlr0p02,"
              "route_lrfix_beta0_mlr0p02_20260611_lrfix,"
              "route_lrfix_decoupled_ref_mlr0p02_20260611_lrfix")
CURVE_LABELS = "Muon,DynMuon,Route (beta=0),Route decoupled"
DEPTH_RUN = "route_fill_beta0p15_mlr0p02_20260611_routefill"


def sh(*cmd: str, check: bool = True) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check).returncode


def step_pull() -> None:
    for group in REPORT_GROUPS:
        sh(PY, "experiments/pull_wandb.py", "--group", group, check=False)


def step_figures() -> None:
    fig = lambda *a: sh(PY, "experiments/report_figures.py", *a, check=False)
    fig("bowls")
    fig("svd_ns")
    fig("cost")
    fig("curves", "--runs", CURVE_RUNS, "--labels", CURVE_LABELS)
    fig("depth", "--run", DEPTH_RUN)
    fig("beta", "--lr", "0p02")
    fig("proxies")


def step_tables() -> None:
    rows = []
    for path in glob.glob(os.path.join("results", "wandb", "*", "summary.csv")):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    seen = {}
    for r in rows:
        seen[r["run"]] = r
    rows = sorted(seen.values(), key=lambda r: (r.get("group") or "", r["run"]))
    print(f"\n{'run':44s} {'group':20s} {'final_val':>9s} {'s/step':>7s}")
    for r in rows:
        fv = r.get("final_val_loss") or ""
        sps = r.get("seconds_per_step") or ""
        fv = f"{float(fv):.4f}" if fv else "  -"
        sps = f"{float(sps):.2f}" if sps else "  -"
        print(f"{r['run'][:44]:44s} {(r.get('group') or '')[:20]:20s} {fv:>9s} {sps:>7s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pull", action="store_true", help="reuse local results/wandb dumps")
    ap.add_argument("--train", action="store_true",
                    help="print the cluster submission commands instead of assuming runs exist")
    args = ap.parse_args()
    if args.train:
        print(__doc__)
        return
    if not args.skip_pull:
        step_pull()
    step_figures()
    step_tables()


if __name__ == "__main__":
    main()
