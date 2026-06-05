"""WikiText-103 batch loading (nanoGPT-style uint16 memmap)."""

from __future__ import annotations

import os

import numpy as np
import torch


def load_bin(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_wikitext.py` first."
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)
