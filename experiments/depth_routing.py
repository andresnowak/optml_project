"""Depth-resolved view of per-layer routing-

DynMuon-Route sets a per-layer exponent ``p_{t,l} = clip(p_t + beta*(g_l - mean_l g))``
around the shared global clock ``p_t``. This script answers: *how far, and in which
direction, does each layer's routed exponent depart from the no-personalization
baseline ``p_t``?* — by plotting layer depth on the x-axis against the
time-averaged deviation ``mean_t (p_{t,l} - p_t)`` on the y-axis, one series per
matrix type (attn q/k/v/proj, mlp fc/proj).

The clock ``p_t`` is the exponent a layer would receive with no routing, i.e. the
``global_schedule`` run (every layer identical). Pass that run as ``--baseline``
for an exact reference; otherwise the per-step cross-layer mean of the routed run
is used as a fallback (≈ p_t when deviations are mean-zero, before clipping).

Usage:
    python experiments/depth_routing.py \
        --routed   results/exp1_spectral_evolution/history_schedule_modulated.json \
        --baseline results/exp1_spectral_evolution/history_global_schedule.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

P_PREFIX = "route/p/"
# Compact, ordered matrix-type labels (depth plot draws one series per type).
LAYER_TYPES = [
    ("attn.q.weight", "attn.q"),
    ("attn.k.weight", "attn.k"),
    ("attn.v.weight", "attn.v"),
    ("attn.proj.weight", "attn.proj"),
    ("mlp.fc.weight", "mlp.fc"),
    ("mlp.proj.weight", "mlp.proj"),
    # legacy (old fused naming) so stale dumps still parse
    ("attn.c_attn.weight", "attn.qkv"),
    ("attn.c_proj.weight", "attn.proj"),
    ("mlp.c_fc.weight", "mlp.fc"),
    ("mlp.c_proj.weight", "mlp.proj"),
]
_DEPTH_RE = re.compile(r"(?:blocks|h|layers)\.(\d+)\.")


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _layer_type(name: str) -> str | None:
    for suffix, label in LAYER_TYPES:
        if name.endswith(suffix):
            return label
    return None


def _depth(name: str) -> int | None:
    m = _DEPTH_RE.search(name)
    return int(m.group(1)) if m else None


def _p_layers(history: dict) -> dict[str, dict[int, float]]:
    """{param_name: {step: p}} for every finite route/p/* series."""
    out: dict[str, dict[int, float]] = {}
    for key, series in history.items():
        if not key.startswith(P_PREFIX):
            continue
        name = key[len(P_PREFIX):]
        steps = {int(s): float(v) for s, v in series if math.isfinite(v)}
        if steps:
            out[name] = steps
    return out


def _clock_by_step(baseline: dict) -> dict[int, float]:
    """p_t per step = cross-layer mean of the global-schedule run (all equal)."""
    layers = _p_layers(baseline)
    per_step: dict[int, list[float]] = defaultdict(list)
    for steps in layers.values():
        for s, v in steps.items():
            per_step[s].append(v)
    return {s: sum(vs) / len(vs) for s, vs in per_step.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routed", required=True, help="history JSON of the routed run")
    ap.add_argument("--baseline", help="history JSON of the global_schedule run (exact p_t). "
                                       "Omit to use the routed run's cross-layer mean.")
    ap.add_argument("--out", help="output PNG (default: alongside --routed as depth_routing.png)")
    args = ap.parse_args()

    routed = _p_layers(_load(args.routed))
    if not routed:
        raise SystemExit(f"no '{P_PREFIX}*' series in {args.routed} — is this a routed run?")

    if args.baseline:
        clock = _clock_by_step(_load(args.baseline))
        clock_label = "global_schedule p_t"
    else:
        clock = _clock_by_step(_load(args.routed))  # cross-layer mean of the routed run
        clock_label = "cross-layer mean (fallback p_t)"

    # Per-layer time-averaged deviation from the clock, on shared steps only.
    rows = []  # (depth, layer_type, mean_dev, mean_abs_dev, n)
    for name, steps in routed.items():
        depth, ltype = _depth(name), _layer_type(name)
        if depth is None or ltype is None:
            continue
        devs = [p - clock[s] for s, p in steps.items() if s in clock]
        if not devs:
            continue
        mean_dev = sum(devs) / len(devs)
        mean_abs = sum(abs(d) for d in devs) / len(devs)
        rows.append((depth, ltype, mean_dev, mean_abs, len(devs)))

    if not rows:
        raise SystemExit("no layers shared steps with the clock; check the two runs match")

    rows.sort(key=lambda r: (r[1], r[0]))
    by_type: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for depth, ltype, mean_dev, _abs, _n in rows:
        by_type[ltype].append((depth, mean_dev))

    # -- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for ltype in sorted(by_type):
        pts = sorted(by_type[ltype])
        xs = [d for d, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, "o-", ms=4, label=ltype)
    ax.axhline(0.0, color="grey", ls="--", lw=0.9)
    ax.set_xlabel("layer depth (block index)")
    ax.set_ylabel(r"time-averaged $\,\overline{p_{t,\ell}-p_t}\,$")
    ax.set_title(f"Per-layer routing departure from the shared clock\n(clock = {clock_label})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(args.routed) or ".", "depth_routing.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # -- numeric companion + console summary --------------------------------
    json_out = os.path.splitext(out)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump([
            {"depth": d, "layer_type": lt, "mean_dev": md, "mean_abs_dev": ma, "n_steps": n}
            for d, lt, md, ma, n in rows
        ], f, indent=2)
    print(f"saved {json_out}")

    overall_abs = sum(r[3] for r in rows) / len(rows)
    print(f"\nmean |p_{{t,l}} - p_t| across all layers: {overall_abs:.4f}")
    print(f"{'layer_type':12s} {'mean_dev':>10s} {'mean|dev|':>10s}")
    agg: dict[str, list[float]] = defaultdict(list)
    aggabs: dict[str, list[float]] = defaultdict(list)
    for _d, lt, md, ma, _n in rows:
        agg[lt].append(md)
        aggabs[lt].append(ma)
    for lt in sorted(agg):
        print(f"{lt:12s} {sum(agg[lt])/len(agg[lt]):>10.4f} {sum(aggabs[lt])/len(aggabs[lt]):>10.4f}")


if __name__ == "__main__":
    main()
