"""Optimizer construction from config."""

from __future__ import annotations

import torch
from torch import nn

from .dynmuon import DynMuonRoute
from .gated_muon import GatedMuon
from .kaon import Kaon
from .muon import Muon
from .param_groups import split_gpt_params
from .relmuon import RelMuon


def _adamw_aux_groups(split, cfg: dict) -> list[dict]:
    groups = []
    if split.embed:
        # The tied embedding/head carries the logit scale; it usually wants a
        # higher LR than biases/gains (embed_lr defaults to adam_lr).
        lr = cfg.get("embed_lr", cfg["adam_lr"])
        groups.append({
            "params": split.embed,
            "name": "embed",
            "lr": lr,
            "initial_lr": lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })
    if split.scalar:
        lr = cfg["adam_lr"]
        groups.append({
            "params": split.scalar,
            "name": "aux",
            "lr": lr,
            "initial_lr": lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })
    return groups


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
            orthogonalize=cfg.get("orthogonalize", "ns"),
            adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        )
        aux_groups = _adamw_aux_groups(split, cfg)
        adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
        return muon, adamw

    if matrix_optimizer == "gated_muon":
        split = split_gpt_params(model, routed=False)
        matrix_params = sorted(split.matrix.get("matrix", []), key=lambda p: p.size(), reverse=True)
        gated_muon = GatedMuon(
            matrix_params,
            lr=cfg["muon_lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            mu=cfg.get("momentum", 0.95),
            nesterov=cfg.get("nesterov", True),
            ns_steps=cfg.get("ns_steps", 5),
            gate_tau=cfg.get("gate_tau", 1e-5),
            adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        )
        aux_groups = _adamw_aux_groups(split, cfg)
        adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
        return gated_muon, adamw

    if matrix_optimizer == "kaon":
        split = split_gpt_params(model, routed=False)
        matrix_params = sorted(split.matrix.get("matrix", []), key=lambda p: p.size(), reverse=True)
        kaon = Kaon(
            matrix_params,
            lr=cfg["muon_lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            mu=cfg.get("momentum", 0.95),
            nesterov=cfg.get("nesterov", True),
            adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
            chaos_steps=cfg.get("kaon_steps", 5),
            chaos_lambda=cfg.get("kaon_lambda", 4.1),
            output_scale=cfg.get("kaon_output_scale", 1.175),
            eps=cfg.get("kaon_eps", 1e-7),
        )
        aux_groups = _adamw_aux_groups(split, cfg)
        adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
        return kaon, adamw

    if matrix_optimizer == "relmuon":
        split = split_gpt_params(model, routed=False)
        matrix_params = sorted(split.matrix.get("matrix", []), key=lambda p: p.size(), reverse=True)
        relmuon = RelMuon(
            matrix_params,
            lr=cfg["muon_lr"],
            weight_decay=cfg.get("weight_decay", 0.0),
            mu=cfg.get("momentum", 0.95),
            nesterov=cfg.get("nesterov", True),
            eps=cfg.get("relmuon_eps", 1e-8),
            adjust_lr_fn=cfg.get("adjust_lr_fn", None),
            scale_mode=cfg.get("relmuon_scale_mode", "log1p"),
            scale_cap=cfg.get("relmuon_scale_cap"),
        )
        aux_groups = _adamw_aux_groups(split, cfg)
        adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
        return relmuon, adamw

    routing_mode = cfg["routing_mode"]
    route_mode = cfg.get("route", {}).get(routing_mode, {})
    beta = cfg.get("beta", route_mode.get("beta", 0.1))
    routed = routing_mode == "schedule_modulated" and float(beta) != 0.0
    split = split_gpt_params(model, routed=routed)
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
        weight_decay=cfg.get("weight_decay", 0.0),
        routing_mode=routing_mode,
        spectrum_mode=cfg.get("spectrum_mode", "power"),
        compute_mode=cfg.get("compute_mode", "reference"),
        ns_variant=cfg.get("ns_variant", "quintic"),
        ns_steps=cfg.get("ns_steps", 5),
        eps=cfg.get("dynmuon_eps", 1e-8),
        adjust_lr_fn=cfg.get("adjust_lr_fn", "spectral_norm"),
        beta=beta,
        dynamic_ref=cfg.get("dynamic_ref", route_mode.get("dynamic_ref", False)),
        ref_decay=cfg.get("ref_decay", route_mode.get("ref_decay", 0.9)),
        lean_norm=cfg.get("lean_norm", route_mode.get("lean_norm", "raw")),
        lean_max=cfg.get("lean_max", route_mode.get("lean_max")),
        modulate_metric=cfg.get("modulate_metric", route_mode.get("metric", "stable_rank")),
        fixed_p=cfg.get("fixed_p", 0.0),
        tau_ratio=cfg.get("tau_ratio", 0.04),
        width_ratio=cfg.get("width_ratio", 0.04),
        total_steps=cfg["train_steps"] if needs_schedule else None,
        magnitude=cfg.get("magnitude", "none"),
        spectrum=cfg.get("spectrum", "power"),
        spectrum_seed=cfg.get("seed", 0),
        track_proxies=cfg.get("track_proxies", True),
        snr_ema_decay=cfg.get("snr_ema_decay", 0.95),
    ) if param_groups else None
    aux_groups = _adamw_aux_groups(split, cfg)
    adamw = torch.optim.AdamW(aux_groups, betas=(0.9, 0.95)) if aux_groups else None
    return dynmuon, adamw
