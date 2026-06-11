"""Generate the report figures from pulled W&B histories.

Every figure reads the local dumps produced by ``experiments/pull_wandb.py``
(``results/wandb/<group>/history_<run>.json`` + ``summary.csv``) so the whole
report is reproducible offline from one pull. Figures are intentionally
minimal: one panel, labeled axes, no styling beyond defaults.

    python experiments/report_figures.py bowls
    python experiments/report_figures.py curves --runs bowl_muon_mlr0p02,bowl_dynmuon_mlr0p02
    python experiments/report_figures.py depth --run route_fill_beta0p15_mlr0p02_20260611_routefill
    python experiments/report_figures.py beta --lr 0p02
    python experiments/report_figures.py proxies
    python experiments/report_figures.py svd_ns          # local computation, no W&B needed
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
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optimizers.dynmuon import logistic_schedule_p  # noqa: E402

WANDB_DIR = os.path.join("results", "wandb")
OUT_DIR = os.path.join("report", "figures")

METHOD_LABELS = {
    "bowl_adamw": "AdamW",
    "bowl_muon": "Muon",
    "bowl_dynmuon": "DynMuon",
    "bowl_relmuon_log1p": "RelMuon-log1p",
    "bowl_relmuon_rms": "RelMuon-RMS",
}
LAYER_TYPES = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


# -- data access --------------------------------------------------------------

def _load_summaries() -> list[dict]:
    rows: list[dict] = []
    for path in glob.glob(os.path.join(WANDB_DIR, "*", "summary.csv")):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    # de-duplicate by run name, newest file wins
    seen: dict[str, dict] = {}
    for row in rows:
        seen[row["run"]] = row
    return list(seen.values())


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


# -- figures -------------------------------------------------------------------

def fig_bowls(args) -> None:
    """Final validation loss vs matrix LR, one line per method (log-x)."""
    rows = _load_summaries()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for group_prefix, label in METHOD_LABELS.items():
        pts = []
        for row in rows:
            run = row["run"]
            if not run.startswith(group_prefix.replace("bowl_relmuon_", "bowl_relmuon_")):
                continue
            if group_prefix in ("bowl_relmuon_log1p", "bowl_relmuon_rms"):
                if not run.startswith(group_prefix):
                    continue
            elif not run.startswith(group_prefix + "_"):
                continue
            lr = _flt(row, "adam_lr" if group_prefix == "bowl_adamw" else "muon_lr")
            val = _flt(row, "final_val_loss")
            if lr and val:
                pts.append((lr, val))
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        line, = ax.plot(xs, ys, marker="o", ms=4, label=label)
        best = min(pts, key=lambda p: p[1])
        ax.plot([best[0]], [best[1]], marker="*", ms=13, color=line.get_color(), zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("matrix learning rate")
    ax.set_ylabel("final validation loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "lr_bowls")


def fig_curves(args) -> None:
    """Validation-loss curves for a comma-separated list of runs."""
    runs = [r.strip() for r in args.runs.split(",")]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for run in runs:
        hist = _find_history(run)
        if hist is None:
            print(f"  (missing history for {run})")
            continue
        steps, vals = _series(hist, "val/loss")
        label = args.labels.split(",")[runs.index(run)] if args.labels else run
        ax.plot(steps, vals, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    if args.ymax:
        ax.set_ylim(top=args.ymax)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, args.out or "loss_curves")


def fig_depth(args) -> None:
    """Mean over time of p_{t,l} - p_t per matrix type vs block depth.

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
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for ltype in LAYER_TYPES:
        if not per_type[ltype]:
            continue
        depths = sorted(per_type[ltype])
        ax.plot(depths, [per_type[ltype][d] for d in depths], marker="o", ms=3, label=ltype)
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("block depth")
    ax.set_ylabel(r"mean$_t\,(p_{t,\ell} - p_t)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    _save(fig, "depth_routing")


def fig_beta(args) -> None:
    """Final validation loss vs router gain beta (at one fixed LR)."""
    rows = _load_summaries()
    pts = []
    for row in rows:
        run = row["run"]
        if f"mlr{args.lr}" not in run and f"_{args.lr}" not in run:
            continue
        if row.get("routing_mode") != "schedule_modulated" or row.get("magnitude") == "polar_fro":
            continue
        beta, val = _flt(row, "beta"), _flt(row, "final_val_loss")
        if beta is not None and val:
            pts.append((beta, val, run))
    if not pts:
        raise SystemExit(f"no route runs found at lr tag {args.lr}")
    pts.sort()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
    ax.set_xlabel(r"router gain $\beta$ (p-units per cross-layer std)")
    ax.set_ylabel("final validation loss")
    ax.grid(alpha=0.3)
    _save(fig, "beta_sweep")
    for beta, val, run in pts:
        print(f"  beta={beta:g}: {val:.4f}  ({run})")


def fig_proxies(args) -> None:
    """Final validation loss per routing proxy (stable_rank vs alignment...)."""
    rows = _load_summaries()
    by_proxy: dict[str, list[float]] = {}
    for row in rows:
        if row.get("routing_mode") != "schedule_modulated":
            continue
        val = _flt(row, "final_val_loss")
        metric = row.get("modulate_metric") or "stable_rank"
        if val:
            by_proxy.setdefault(metric, []).append(val)
    if not by_proxy:
        raise SystemExit("no routed runs with proxies found")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    names = sorted(by_proxy)
    best = [min(by_proxy[n]) for n in names]
    ax.bar(names, best, width=0.5)
    ax.set_ylabel("best final validation loss")
    ax.set_ylim(bottom=min(best) - 0.05)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "proxy_comparison")


def fig_svd_ns(args) -> None:
    """Empirical SVD == Newton-Schulz evidence for the DynMuon shaping.

    Left: relative error ||D_ns - D_svd|| / ||D_svd|| vs exponent p for the
    cubic and quintic polar iterations on random matrices. Right: divergence
    of full optimizer trajectories (ns vs svd compute modes) over 30 steps.
    Local computation; W&B not needed.
    """
    import torch
    from src.optimizers.dynmuon import DynMuonRoute, shape_exact_ns, shape_exact_svd

    torch.manual_seed(0)
    ps = [x / 20 for x in range(-5, 21)]                 # -0.25 .. 1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for variant, steps in (("quintic", 5), ("cubic", 30)):
        errs = []
        for p in ps:
            rel = []
            for _ in range(5):
                M = torch.randn(48, 64, dtype=torch.float64).float()
                d_svd = shape_exact_svd(M, p)
                d_ns = shape_exact_ns(M, p, ns_variant=variant, ns_steps=steps)
                rel.append(float(torch.linalg.norm(d_ns - d_svd) / torch.linalg.norm(d_svd)))
            errs.append(sum(rel) / len(rel))
        ax1.semilogy(ps, errs, marker=".", label=f"{variant} NS")
    ax1.set_xlabel("spectral exponent $p$")
    ax1.set_ylabel(r"$\|D_{NS}-D_{SVD}\|_F / \|D_{SVD}\|_F$")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

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

    torch.manual_seed(2)
    svd_traj, ns_traj = run("svd"), run("ns")
    div = [float(torch.linalg.norm(a - b) / torch.linalg.norm(a))
           for a, b in zip(svd_traj, ns_traj)]
    ax2.semilogy(range(1, 31), div, marker=".")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("relative weight divergence")
    ax2.grid(alpha=0.3)
    _save(fig, "svd_vs_ns")


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
    labels, spс, stt = [], [], []
    for label, row in chosen.items():
        hist = _find_history(row["run"])
        steps_to = None
        if hist:
            for s, v in hist.get("val/loss", []):
                if v <= target:
                    steps_to = s
                    break
        labels.append(label)
        spс.append(_flt(row, "seconds_per_step") or 0.0)
        stt.append(steps_to or 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax1.bar(labels, spс, width=0.5)
    ax1.set_ylabel("seconds / step")
    ax1.tick_params(axis="x", rotation=25)
    ax2.bar(labels, stt, width=0.5)
    ax2.set_ylabel(f"steps to val loss {target:.3f}")
    ax2.tick_params(axis="x", rotation=25)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, axis="y")
    _save(fig, "cost_comparison")
    print(f"  common target = {target:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bowls")
    c = sub.add_parser("curves")
    c.add_argument("--runs", required=True)
    c.add_argument("--labels")
    c.add_argument("--ymax", type=float)
    c.add_argument("--out")
    d = sub.add_parser("depth")
    d.add_argument("--run", required=True)
    d.add_argument("--total-steps", dest="total_steps", type=int, default=1526)
    b = sub.add_parser("beta")
    b.add_argument("--lr", default="0p02", help="LR tag in run names, e.g. 0p02")
    sub.add_parser("proxies")
    sub.add_parser("svd_ns")
    sub.add_parser("cost")
    sub.add_parser("all")
    args = ap.parse_args()

    if args.cmd == "all":
        fig_bowls(args)
        fig_svd_ns(args)
        fig_cost(args)
        for fn, label in ((fig_beta, "beta"), (fig_proxies, "proxies")):
            try:
                args.lr = "0p02"
                fn(args)
            except SystemExit as e:
                print(f"skip {label}: {e}")
        return
    {"bowls": fig_bowls, "curves": fig_curves, "depth": fig_depth, "beta": fig_beta,
     "proxies": fig_proxies, "svd_ns": fig_svd_ns, "cost": fig_cost}[args.cmd](args)


if __name__ == "__main__":
    main()
