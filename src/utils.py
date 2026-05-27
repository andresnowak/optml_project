"""Helpers shared by the CLI, scripts, and tests."""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import yaml

from src.experiments import EXPERIMENTS


def select_device(spec: str = "auto") -> torch.device:
    """Pick a torch device. ``spec="auto"`` prefers cuda → mps → cpu."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str, _seen: tuple[str, ...] = ()) -> dict:
    """Load a YAML config, recursively resolving ``extends:``. Cycles raise."""
    abs_path = os.path.abspath(path)
    if abs_path in _seen:
        chain = " -> ".join(_seen + (abs_path,))
        raise ValueError(f"config inheritance cycle: {chain}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(cfg).__name__}")
    cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
    base_path = cfg.pop("extends", None)
    if base_path is not None:
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base = load_config(base_path, _seen + (abs_path,))
        return {**base, **cfg}
    return cfg


def filtered_experiment_kwargs(experiment_name: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop kwargs the experiment class doesn't accept (and any None values)."""
    cls = EXPERIMENTS[experiment_name]
    accepted = set(inspect.signature(cls).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted and v is not None}


def score_losses(losses: list[float]) -> float:
    """Min loss over the last 10% of training; +inf for diverged runs."""
    if not losses:
        return float("inf")
    arr = np.asarray(losses, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < max(1, len(arr) // 2):
        return float("inf")
    n_last = max(1, len(finite) // 10)
    return float(finite[-n_last:].min())


def steps_to_target(losses: list[float], target: float) -> int | None:
    """Index (1-based) of the first step at which loss <= target; None if never."""
    for i, loss in enumerate(losses):
        if np.isfinite(loss) and loss <= target:
            return i + 1
    return None
