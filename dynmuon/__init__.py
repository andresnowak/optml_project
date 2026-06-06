"""DynMuon-Route: dynamic layer-wise spectral-exponent routing for Muon."""

from . import analysis
from .config import load_config, pick_device
from .models import GPT, GPTConfig
from .optimizer import DynMuonRoute, logistic_route, newton_schulz
from .trainer import (
    MemoryLogger, TeeLogger, WandbLogger, build_arm_logger, build_optimizers, train,
)

__all__ = [
    "DynMuonRoute", "logistic_route", "newton_schulz",
    "GPT", "GPTConfig",
    "load_config", "pick_device",
    "train", "build_optimizers",
    "MemoryLogger", "WandbLogger", "TeeLogger", "build_arm_logger",
    "analysis",
]
