"""Reproduce every result, table, and figure in the report from bundled logs.

    python run.py                 # offline: restore bundled logs + regenerate figures
    python run.py --refresh-wandb # maintainers only: pull W&B + rewrite bundle
    python run.py --skip-restore  # figures only, from REPORT_WANDB_DIR/results/wandb
    python run.py --train         # print cluster training commands

Pipeline (all steps are idempotent):
  1. Restore experiments/report_wandb_bundle.jsonl.gz into a clean local cache.
     Reviewers do not need W&B credentials.
  2. Regenerate the report figures into report/figures/
     (experiments/report_figures.py): LR bowls, loss curves, depth-resolved
     routing, beta sweep, proxy comparison, SVD-vs-Newton-Schulz evidence,
     exponent/magnitude diagnostics, and the cost comparison.
  3. Print the summary tables used in the report.

Maintainers can refresh the bundle from W&B with --refresh-wandb. Training itself
runs on the cluster via scripts/sweeps.sh (see README); the exact submission
commands for every report run are:

    scripts/sweeps.sh bowls && scripts/sweeps.sh route 0.02 && scripts/sweeps.sh final 0.02
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

from experiments.pull_wandb import DEFAULT_BUNDLE, bundle_cache, restore_bundle

PY = sys.executable
DEFAULT_CACHE_DIR = os.path.join("results", "report_wandb_cache")

REPORT_GROUPS = (
    "bowl_dynmuon", "bowl_muon", "bowl_adamw", "bowl_relmuon",
    "route_arms", "route_lrfix*", "route_lrgrid*", "route_fill*",
    "route_alignment*", "route_proxies", "spectrum_controls", "seed_replicates",
    "relmuon_compare", "review_lean_route", "review_lean_noise",
    "review_lean_cost", "review_lean_relmuon",
)
BUNDLE_GROUPS = (*REPORT_GROUPS, "muon_svd_vs_ns")
REPORT_HISTORY_PREFIXES = ("val/", "train/", "route/p/")
# Loss-curve panel: the same selected representatives used in the LR bowl.
CURVE_RUNS = (
    "bowl_adamw_alr0p0012,bowl_muon_mlr0p02,"
    "muon_svd_polar_wd_lr_400m_muon_2e-2_adam_0.002,bowl_dynmuon_mlr0p02,"
    "route_align_beta0p15_mlr0p02_20260611_clean,bowl_relmuon_log1p_mlr0p1"
)
CURVE_LABELS = "AdamW,Muon,Muon-SVD,DynMuon,Route-align,RelMuon-log1p"


def sh(*cmd: str, check: bool = True, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=env).returncode


def step_pull() -> None:
    for group in REPORT_GROUPS:
        sh(PY, "experiments/pull_wandb.py", "--group", group, check=False)
    sh(
        PY,
        "experiments/pull_wandb.py",
        "--project",
        "muon-svd-vs-ns",
        "--out-dir",
        os.path.join("results", "wandb", "muon_svd_vs_ns"),
        check=False,
    )


def step_figures(wandb_dir: str) -> None:
    env = {**os.environ, "REPORT_WANDB_DIR": wandb_dir}
    fig = lambda *a: sh(PY, "experiments/report_figures.py", *a, check=False, env=env)
    fig("bowls")
    fig("svd_ns")
    fig("equivalence")
    fig("cost")
    fig("curves", "--runs", CURVE_RUNS, "--labels", CURVE_LABELS)
    fig("losses")
    fig("proxy_depth")
    fig("beta", "--lr", "0p02")
    fig("proxies")
    fig("route_robustness")
    fig("route_ablation")
    fig("relmuon_attention")
    fig("spectrum_controls")


def step_tables(wandb_dir: str) -> None:
    rows = []
    for path in glob.glob(os.path.join(wandb_dir, "*", "summary.csv")):
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
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE,
                    help="compressed offline W&B-log bundle to restore")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                    help="local cache directory restored from --bundle")
    ap.add_argument("--refresh-wandb", action="store_true",
                    help="maintainers only: pull W&B logs and rewrite --bundle")
    ap.add_argument("--skip-restore", "--skip-pull", action="store_true", dest="skip_restore",
                    help="reuse REPORT_WANDB_DIR or results/wandb instead of restoring --bundle")
    ap.add_argument("--train", action="store_true",
                    help="print the cluster submission commands instead of assuming runs exist")
    args = ap.parse_args()
    if args.train:
        print(__doc__)
        return
    if args.refresh_wandb:
        step_pull()
        bundle_cache(
            source_dir=os.path.join("results", "wandb"),
            bundle_out=args.bundle,
            group_patterns=BUNDLE_GROUPS,
            history_key_prefixes=REPORT_HISTORY_PREFIXES,
        )
    if args.skip_restore:
        wandb_dir = os.environ.get("REPORT_WANDB_DIR", os.path.join("results", "wandb"))
    else:
        if not os.path.exists(args.bundle):
            raise SystemExit(
                f"offline bundle not found: {args.bundle}\n"
                "Maintainers can create it with: python run.py --refresh-wandb"
            )
        restore_bundle(bundle_path=args.bundle, out_dir=args.cache_dir, clean=True)
        wandb_dir = args.cache_dir
    step_figures(wandb_dir)
    step_tables(wandb_dir)


if __name__ == "__main__":
    main()
