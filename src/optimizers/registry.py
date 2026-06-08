"""Optimizer construction from config."""

from __future__ import annotations

import torch
from torch import nn

from .dynmuon import DynMuonRoute
from .muon import Muon
from .param_groups import split_gpt_params


def _adamw_aux_groups(split, cfg: dict) -> list[dict]:
    params = split.embed + split.scalar
    if not params:
        return []
    lr = cfg["adam_lr"]
    return [{
        "params": params,
        "name": "aux",
        "lr": lr,
        "initial_lr": lr,
        "weight_decay": cfg.get("scalar_weight_decay", 0.0),
    }]


def build_optimizers(model: nn.Module, cfg: dict):
    matrix_optimizer = cfg.get("matrix_optimizer", "dynmuon")
    if matrix_optimizer == "adamw":
        split = split_gpt_params(model, routed=False)
        matrix_params = sorted(split.matrix.get("matrix", []), key=lambda p: p.size(), reverse=True)
        param_groups = []
        if matrix_params:
            lr = cfg["adam_lr"]
            param_groups.append({
                "params": matrix_params,
                "name": "matrix",
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": cfg.get("weight_decay", 0.1),
            })
        param_groups.extend(_adamw_aux_groups(split, cfg))
        return None, torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    if matrix_optimizer == "muon":
        split = split_gpt_params(model, routed=False)
        matrix_params = sorted(split.matrix.get("matrix", []), key=lambda p: p.size(), reverse=True)
        muon = Muon(
            matrix_params,
            lr=cfg["muon_lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            mu=cfg.get("momentum", 0.95),
            nesterov=cfg.get("nesterov", True),
            ns_steps=cfg.get("ns_steps", 12),
            adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        )
        aux_groups = _adamw_aux_groups(split, cfg)
        adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
        return muon, adamw

    routed = cfg.get("routing_mode") == "schedule_modulated"
    split = split_gpt_params(model, routed=routed)
    routing_mode = cfg["routing_mode"]
    route_mode = cfg.get("route", {}).get(routing_mode, {})
    default_lt = route_mode.get("default", {})

    def _lt(lt: str, key: str, dflt: float) -> float:
        return route_mode.get(lt, default_lt).get(key, dflt)

    param_groups = []
    for group_name, params in split.matrix.items():
        lookup = group_name if routed else "default"
        param_groups.append({
            "params": params,
            "name": group_name,
            "mu": _lt(lookup, "mu", 0.0),
            "omega": _lt(lookup, "omega", 1.0),
            "ref": _lt(lookup, "ref", 0.0),
        })

    needs_schedule = routing_mode in ("global_schedule", "schedule_modulated")
    dynmuon = DynMuonRoute(
        param_groups,
        lr=cfg["muon_lr"],
        momentum=cfg.get("momentum", 0.95),
        nesterov=cfg.get("nesterov", True),
        routing_mode=routing_mode,
        compute_mode=cfg["compute_mode"],
        ns_variant=cfg.get("ns_variant", "quintic"),
        ns_steps=cfg.get("ns_steps", 5),
        adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        beta=cfg.get("beta", route_mode.get("beta", 0.1)),
        dynamic_ref=cfg.get("dynamic_ref", route_mode.get("dynamic_ref", False)),
        ref_decay=cfg.get("ref_decay", route_mode.get("ref_decay", 0.9)),
        modulate_metric=cfg.get("modulate_metric", route_mode.get("metric", "stable_rank")),
        fixed_p=cfg.get("fixed_p", 0.0),
        tau_ratio=cfg.get("tau_ratio", 0.04),
        width_ratio=cfg.get("width_ratio", 0.04),
        total_steps=cfg["train_steps"] if needs_schedule else None,
    ) if param_groups else None
    aux_groups = _adamw_aux_groups(split, cfg)
    adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
    return dynmuon, adamw
