"""Per-optimizer cost comparison.

Reads the ``history_*.json`` dumps produced by the experiment scripts (each holds
``time/train_seconds``, ``val/loss`` and ``train/loss`` trajectories) and reports,
per method: best/final validation loss, total wall-clock training time, mean
per-step time, and both the number of steps **and** the wall-clock time to reach a
common target loss. This separates the two cost axes the project cares about — who
finishes in fewer *steps* vs. who is cheaper per *step* (Muon-family steps run a
Newton-Schulz / SVD that AdamW does not).

Usage:
    # scan a directory of history_<method>.json
    python experiments/cost_table.py --results-dir results/baselines
    # or name them explicitly, and pin the target
    python experiments/cost_table.py \
        --history muon=results/baselines/history_muon.json \
        --history route=results/baselines/history_dynmuon_route.json \
        --target-loss 4.0 --reference muon
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

VAL_KEY = "val/loss"
TRAIN_KEY = "train/loss"
TIME_KEY = "time/train_seconds"


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _series(history: dict, key: str) -> list[tuple[int, float]]:
    return [(int(s), float(v)) for s, v in history.get(key, [])]


def _final(history: dict, key: str) -> float | None:
    s = _series(history, key)
    return s[-1][1] if s else None


def _best(history: dict, key: str) -> float | None:
    s = _series(history, key)
    return min(v for _, v in s) if s else None


def _max_step(history: dict) -> int:
    steps = [s for key in (VAL_KEY, TRAIN_KEY) for s, _ in _series(history, key)]
    return max(steps) if steps else 0


def _steps_to_target(history: dict, target: float) -> int | None:
    for step, value in _series(history, VAL_KEY):
        if value <= target:
            return step
    return None


def _time_to_target(history: dict, target: float) -> float | None:
    """Wall-clock time at the first validation step that reaches the target.

    ``time/train_seconds`` and ``val/loss`` are logged together at validation
    steps, so we look the time up at the same step.
    """
    step = _steps_to_target(history, target)
    if step is None:
        return None
    times = dict(_series(history, TIME_KEY))
    return times.get(step)


def _discover(results_dir: str) -> dict[str, str]:
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "history_*.json"))):
        name = os.path.basename(path)[len("history_"):-len(".json")]
        out[name] = path
    return out


def _parse_history_args(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--history expects name=path, got {pair!r}")
        name, path = pair.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="scan this dir for history_<method>.json")
    ap.add_argument("--history", action="append", default=[],
                    help="explicit name=path (repeatable); overrides --results-dir order")
    ap.add_argument("--target-loss", type=float,
                    help="common val-loss target; default = worst of the per-method best losses")
    ap.add_argument("--reference", help="method to compute %-fewer-steps / %-less-time against")
    ap.add_argument("--out-dir", help="where to write cost.md/cost.png (default: --results-dir)")
    args = ap.parse_args()

    methods = _parse_history_args(args.history)
    if args.results_dir:
        methods = {**_discover(args.results_dir), **methods}
    if not methods:
        raise SystemExit("no histories: pass --results-dir and/or --history name=path")

    hist = {name: _load(path) for name, path in methods.items()}
    best = {n: _best(h, VAL_KEY) for n, h in hist.items()}
    reached = [v for v in best.values() if v is not None]
    if not reached:
        raise SystemExit(f"no '{VAL_KEY}' series found in any history")
    target = args.target_loss if args.target_loss is not None else max(reached)

    # -- per-method metrics --------------------------------------------------
    table = {}
    for n, h in hist.items():
        steps = _max_step(h)
        total_t = _final(h, TIME_KEY)
        ms = (total_t / steps * 1000.0) if (total_t and steps) else None
        table[n] = {
            "final_val": _final(h, VAL_KEY),
            "best_val": best[n],
            "steps": steps,
            "total_time": total_t,
            "ms_per_step": ms,
            "steps_to_target": _steps_to_target(h, target),
            "time_to_target": _time_to_target(h, target),
        }

    ref = args.reference if args.reference in table else None

    def cell(v, fmt):
        return fmt.format(v) if v is not None else "—"

    def rel(field: str, m: str) -> str:
        """% fewer (steps/time) for method m vs the reference (lower is better)."""
        if not ref or m == ref:
            return ""
        a, b = table[ref].get(field), table[m].get(field)
        if a and b:
            return f" ({100 * (a - b) / a:+.1f}%)"
        return ""

    # -- markdown table ------------------------------------------------------
    lines = [
        "# Optimizer cost comparison", "",
        f"Common target val loss: **{target:.4f}**"
        + (f"  ·  reference: **{ref}**" if ref else ""), "",
        "| method | best val | final val | steps | total time (s) | ms/step | steps→target | time→target (s) |",
        "|--------|----------|-----------|-------|----------------|---------|--------------|-----------------|",
    ]
    for n in methods:
        t = table[n]
        lines.append(
            f"| {n} "
            f"| {cell(t['best_val'], '{:.4f}')} "
            f"| {cell(t['final_val'], '{:.4f}')} "
            f"| {t['steps']} "
            f"| {cell(t['total_time'], '{:.1f}')} "
            f"| {cell(t['ms_per_step'], '{:.1f}')} "
            f"| {cell(t['steps_to_target'], '{:d}')}{rel('steps_to_target', n)} "
            f"| {cell(t['time_to_target'], '{:.1f}')}{rel('time_to_target', n)} |"
        )
    md = "\n".join(lines)

    out_dir = args.out_dir or args.results_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "cost.md")
    with open(md_path, "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\nwrote {md_path}")

    # -- bar chart: total time and steps-to-target side by side --------------
    names = list(methods)
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_t.bar(names, [table[n]["total_time"] or 0 for n in names], color="tab:blue")
    ax_t.set_title("total wall-clock training time"); ax_t.set_ylabel("seconds")
    ax_t.tick_params(axis="x", rotation=20)
    ax_s.bar(names, [table[n]["steps_to_target"] or 0 for n in names], color="tab:orange")
    ax_s.set_title(f"steps to val loss {target:.3f}"); ax_s.set_ylabel("steps")
    ax_s.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    png_path = os.path.join(out_dir, "cost.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
