"""Cached uint16 token data loading."""

from __future__ import annotations

import os

import numpy as np
import torch


def load_bin(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `uv run python data/prepare_wikitext.py` first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_token_batch(
    data: np.ndarray,
    batch_tokens: int,
    sequence_length: int,
    device: torch.device,
    *,
    offset: int | None = None,
):
    """Return a token batch shaped as sequences.

    ``batch_tokens`` is the derived token count for a training or validation chunk.
    It must be divisible by ``sequence_length``.
    """
    if batch_tokens % sequence_length != 0:
        raise ValueError("batch_tokens must be divisible by sequence_length")
    max_start = len(data) - batch_tokens - 1
    if max_start <= 0:
        raise ValueError(f"dataset has {len(data):,} tokens, too short for {batch_tokens:,}-token batch")
    start = int(torch.randint(max_start, (1,)).item()) if offset is None else offset % max_start
    buf = torch.from_numpy(data[start:start + batch_tokens + 1].astype(np.int64))
    x = buf[:-1].view(-1, sequence_length)
    y = buf[1:].view(-1, sequence_length)
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def iter_microbatches(x: torch.Tensor, y: torch.Tensor, mbs: int):
    if len(x) % mbs != 0:
        raise ValueError(f"{len(x)} sequences is not divisible by mbs={mbs}")
    for i in range(0, len(x), mbs):
        yield x[i:i + mbs], y[i:i + mbs]


def validation_offsets(val_tokens: int, batch_tokens: int) -> list[int]:
    chunks = max(1, (val_tokens + batch_tokens - 1) // batch_tokens)
    return [i * batch_tokens for i in range(chunks)]
