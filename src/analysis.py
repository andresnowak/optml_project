"""Helpers for turning logged metric histories into experiment artifacts.

A ``history`` is the ``MemoryLogger.history`` dict: ``{metric_name: [(step, value), ...]}``.
These functions are shared by the experiment scripts so they characterize runs
numerically (JSON dumps, steps-to-target) rather than only visually.
"""

from __future__ import annotations

import json
import os


def dump_history(history: dict, path: str) -> None:
    """Write a metric history to JSON (raw trajectories for offline analysis)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v for k, v in history.items()}, f, indent=2)


def mean_series_by_suffix(history: dict, suffix: str, prefix: str = "route/p/"):
    """Average the per-layer series whose key is ``prefix...suffix``."""
    keys = [k for k in history if k.startswith(prefix) and k.endswith(suffix)]
    if not keys:
        return None
    steps = [s for s, _ in history[keys[0]]]
    values = [sum(history[k][i][1] for k in keys) / len(keys) for i in range(len(steps))]
    return steps, values


def steps_to_target(history: dict, key: str, target: float) -> int | None:
    """First logged step at which ``key`` first drops to/below ``target`` (e.g.
    val loss reaching a threshold). None if never reached."""
    for step, value in history.get(key, []):
        if value <= target:
            return step
    return None


def final_value(history: dict, key: str) -> float | None:
    series = history.get(key)
    return series[-1][1] if series else None


def min_value(history: dict, key: str) -> float | None:
    series = history.get(key)
    return min(v for _, v in series) if series else None
