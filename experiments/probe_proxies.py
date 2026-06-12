"""Probe the routing proxies before committing GPU to a long run.

Runs a short training trajectory (default: the global-schedule baseline) and
reports, per layer type and over time, the distribution of the three routing
proxies — stable rank (sr), SNR (gamma), alignment (alpha). Use it to:
  * set `ref` for schedule_modulated (≈ the median stable rank), and
  * set `mu`/`omega` for the logistic modes (mu ≈ median, omega ≈ (p90-p10)/4),
  * check whether a proxy has dynamic range (early vs late) and separates layers.

Usage:
    python experiments/probe_proxies.py --config configs/small.yaml --steps 150
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import MemoryLogger, load_config, train  # noqa: E402

METRICS = ["sr", "gamma", "alpha"]


def _layer_type(name: str) -> str:
    return "attn" if ".attn." in name else "mlp" if ".mlp." in name else "other"


def _collect(history: dict, metric: str):
    """{layer_type: {"all": [...], "early": [...], "late": [...]}} for route/<metric>/*."""
    keys = [k for k in history if k.startswith(f"route/{metric}/")]
    out: dict[str, dict[str, list]] = {}
    for k in keys:
        series = [v for _, v in history[k] if np.isfinite(v)]
        if not series:
            continue
        lt = _layer_type(k)
        cut = max(1, len(series) // 5)
        d = out.setdefault(lt, {"all": [], "early": [], "late": []})
        d["all"] += series
        d["early"] += series[:cut]
        d["late"] += series[-cut:]
    return out


def _stats(v):
    a = np.array(v, dtype=float)
    return dict(n=len(a), min=a.min(), p10=np.percentile(a, 10), median=np.median(a),
               p90=np.percentile(a, 90), max=a.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--routing-mode", dest="routing_mode", default="global_schedule")
    args = ap.parse_args()

    cfg = load_config(args.config, {"train_steps": args.steps, "routing_mode": args.routing_mode,
                                    "val_loss_every": 0,
                                    "log_every": 1})
    logger = MemoryLogger()
    print(f"probing proxies over {args.steps} steps (routing_mode={args.routing_mode})...\n")
    train(cfg, logger=logger)

    for metric in METRICS:
        col = _collect(logger.history, metric)
        if not col:
            continue
        print(f"=== {metric} ===")
        print(f"  {'layer':7s} {'n':>6s} {'min':>8s} {'p10':>8s} {'median':>8s} {'p90':>8s} {'max':>8s}  {'early→late median':>18s}")
        for lt in ("attn", "mlp", "other"):
            if lt not in col:
                continue
            s = _stats(col[lt]["all"])
            em, lm = np.median(col[lt]["early"]), np.median(col[lt]["late"])
            print(f"  {lt:7s} {s['n']:6d} {s['min']:8.3f} {s['p10']:8.3f} {s['median']:8.3f} "
                  f"{s['p90']:8.3f} {s['max']:8.3f}  {em:8.3f} → {lm:.3f}")
        print()

    # Suggested calibration from stable rank.
    sr = _collect(logger.history, "sr")
    if sr:
        allv = [x for lt in sr.values() for x in lt["all"]]
        med = float(np.median(allv))
        spread = float((np.percentile(allv, 90) - np.percentile(allv, 10)) / 4) or 0.5
        print("--- suggested calibration (stable rank) ---")
        print(f"  schedule_modulated: route.schedule_modulated.{{attn,mlp,default}}.ref = {med:.2f}")
        print(f"                      beta ~ {max(0.05, round(0.5 / max(spread,1e-3), 2))}  "
              f"(so a p10→p90 swing moves p by ~0.5)")
        for lt in ("attn", "mlp"):
            if lt in sr:
                m = float(np.median(sr[lt]["all"]))
                print(f"  stable_rank logistic: route.stable_rank.{lt} = {{mu: {m:.2f}, omega: {spread:.2f}}}")


if __name__ == "__main__":
    main()
