"""Parameter grouping for optimizer recipes."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass
class ParamSplit:
    embed: list[nn.Parameter]
    scalar: list[nn.Parameter]
    matrix: dict[str, list[nn.Parameter]]


def layer_type(name: str) -> str:
    if ".attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def split_gpt_params(model: nn.Module, *, routed: bool) -> ParamSplit:
    """Split GPT params into AdamW fallback and matrix-optimizer groups."""
    seen: set[int] = set()
    assigned: set[int] = set()
    embed: list[nn.Parameter] = []
    scalar: list[nn.Parameter] = []
    matrix: dict[str, list[nn.Parameter]] = {"matrix": []} if not routed else {"attn": [], "mlp": [], "other": []}

    named = list(model.named_parameters())
    for name, p in named:
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_embedding = name == "embed.weight"
        if is_embedding:
            embed.append(p)
        elif p.ndim < 2:
            scalar.append(p)
        elif routed:
            matrix[layer_type(name)].append(p)
        else:
            matrix["matrix"].append(p)
        assigned.add(id(p))

    expected = {id(p) for _, p in named if p.requires_grad}
    if assigned != expected:
        missing = expected - assigned
        extra = assigned - expected
        raise RuntimeError(f"parameter split mismatch: missing={len(missing)} extra={len(extra)}")

    return ParamSplit(embed=embed, scalar=scalar, matrix={k: v for k, v in matrix.items() if v})
