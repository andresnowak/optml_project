"""YAML configuration loading.

All run customization lives in ``configs/*.yaml``. A config may declare
``extends: <other.yaml>`` (resolved relative to the config's own directory) to
inherit and override a base config; nested mappings (e.g. ``route``) are merged
recursively. CLI overrides are applied last.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config, resolving ``extends`` and applying CLI overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent)
        cfg = _deep_merge(load_config(parent_path), cfg)
    if overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
