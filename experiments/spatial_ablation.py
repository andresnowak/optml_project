"""Run a spatial optimizer ablation.

This entry point keeps the selected matrix optimizer only on attention matrices
and moves every other trainable parameter to AdamW. It is intended for the
RelMuon attention-only efficiency test:

    scripts/run_job.sh spatial --config configs/relmuon_attention.yaml ...
"""

from __future__ import annotations

import torch
import src.trainer
from src.cli import main
from src.optimizers.param_groups import layer_type


def _is_attention_matrix(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim >= 2 and layer_type(name) == "attn"


def _sorted_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return sorted(params, key=lambda p: p.size(), reverse=True)


def spatial_ablation_builder(model, cfg):
    # Build the native optimizer first so its config-specific defaults and state
    # conventions stay exactly the same as in the full-matrix experiment.
    dynmuon, adamw = src.trainer.build_optimizers_original(model, cfg)

    print("\n[SPATIAL ABLATION] matrix optimizer on attention matrices only")

    if dynmuon is None:
        raise ValueError(
            "Spatial ablation expects a matrix optimizer config, e.g. "
            "configs/relmuon_attention.yaml"
        )

    muon_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if _is_attention_matrix(name, param):
            muon_params.append(param)
            print(f" Routed attention to matrix optimizer: {name}")
        else:
            print(f" Routed fallback to AdamW:             {name}")

    muon_param_ids = {id(p) for p in muon_params}
    for group in dynmuon.param_groups:
        group["params"] = _sorted_params([p for p in group["params"] if id(p) in muon_param_ids])
    dynmuon.param_groups[:] = [group for group in dynmuon.param_groups if group["params"]]
    if not dynmuon.param_groups:
        raise ValueError("Spatial ablation found no attention matrix parameters")

    embed_params = []
    mlp_params = []
    scalar_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_attention_matrix(name, param):
            continue

        if name == "embed.weight":
            embed_params.append(param)
        elif param.ndim >= 2:
            mlp_params.append(param)
        else:
            scalar_params.append(param)

    adamw_groups = []
    if embed_params:
        embed_lr = cfg.get("embed_lr", cfg.get("adam_lr", 6e-4))
        adamw_groups.append({
            "params": _sorted_params(embed_params),
            "name": "embed",
            "lr": embed_lr,
            "initial_lr": embed_lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })
    if mlp_params:
        adam_lr = cfg.get("adam_lr", 6e-4)
        adamw_groups.append({
            "params": _sorted_params(mlp_params),
            "name": "mlp",
            "lr": adam_lr,
            "initial_lr": adam_lr,
            "weight_decay": cfg.get("weight_decay", 0.1),
        })
    if scalar_params:
        adam_lr = cfg.get("adam_lr", 6e-4)
        adamw_groups.append({
            "params": _sorted_params(scalar_params),
            "name": "aux",
            "lr": adam_lr,
            "initial_lr": adam_lr,
            "weight_decay": cfg.get("scalar_weight_decay", 0.0),
        })

    adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95)) if adamw_groups else None

    return dynmuon, adamw


if __name__ == "__main__":
    src.trainer.build_optimizers_original = src.trainer.build_optimizers
    src.trainer.build_optimizers = spatial_ablation_builder
    main()
