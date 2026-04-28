from __future__ import annotations

from torch.optim import Adam, AdamW, SGD
from torch.optim import Muon as _TorchMuon


class Muon(_TorchMuon):
    """Our Muon subclass — adjust_lr shape-scaling functions will be added here."""
    pass


class SpecMuon(SGD):
    """Placeholder — update rule TBD."""

    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)


OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW,
    "muon": Muon,
    "sgd": SGD,
    "specmuon": SpecMuon,
}


def build_optimizer(name, params, lr, weight_decay, **kwargs):
    """Build an optimizer, forwarding only the kwargs each one understands."""
    if name == "muon":
        kw = {k: kwargs[k] for k in ("momentum", "ns_steps") if kwargs.get(k) is not None}
        return Muon(params, lr=lr, **kw)
    if name == "sgd":
        kw = {k: kwargs[k] for k in ("momentum",) if kwargs.get(k) is not None}
        return SGD(params, lr=lr, weight_decay=weight_decay, **kw)
    if name in ("adam", "adamw"):
        kw = {}
        if kwargs.get("betas") is not None:
            kw["betas"] = kwargs["betas"]
        if kwargs.get("eps") is not None:
            kw["eps"] = kwargs["eps"]
        return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay, **kw)
    return OPTIMIZERS[name](params, lr=lr, weight_decay=weight_decay)
