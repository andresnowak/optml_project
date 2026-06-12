"""Cached uint16 token data loading."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

LLMC_MAGIC = 20240520
LLMC_VERSION = 1
LLMC_HEADER_BYTES = 256 * 4


@dataclass(frozen=True)
class TokenData:
    arrays: tuple[np.ndarray, ...]
    paths: tuple[str, ...]

    def __len__(self) -> int:
        return sum(len(a) for a in self.arrays)

    def span(self, start: int, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.uint16)
        if start < 0 or start + length > len(self):
            raise ValueError(f"span [{start}, {start + length}) is outside {len(self):,} tokens")

        out = np.empty(length, dtype=np.uint16)
        written = 0
        pos = start
        for arr in self.arrays:
            if pos >= len(arr):
                pos -= len(arr)
                continue
            take = min(len(arr) - pos, length - written)
            out[written:written + take] = arr[pos:pos + take]
            written += take
            if written == length:
                break
            pos = 0
        return out

    def random_span(self, length: int) -> np.ndarray:
        capacities = [max(0, len(arr) - length) for arr in self.arrays]
        total = sum(capacities)
        if total <= 0:
            raise ValueError(f"dataset has {len(self):,} tokens, too short for {length:,}-token span")
        pick = int(torch.randint(total, (1,)).item())
        for arr, cap in zip(self.arrays, capacities):
            if pick >= cap:
                pick -= cap
                continue
            start = int(torch.randint(cap, (1,)).item())
            return arr[start:start + length]
        raise RuntimeError("unreachable random span selection")


def _load_one_bin(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) == 256 and int(header[0]) == LLMC_MAGIC:
        if int(header[1]) != LLMC_VERSION:
            raise ValueError(f"{path} has unsupported llm.c data version {int(header[1])}")
        return np.memmap(path, dtype=np.uint16, mode="r", offset=LLMC_HEADER_BYTES, shape=(int(header[2]),))
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_bin(path: str) -> TokenData:
    paths = sorted(glob.glob(path))
    if not paths and os.path.exists(path):
        paths = [path]
    if not paths:
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python data/prepare_wikitext.py` or "
            "`uv run python data/prepare_fineweb.py 500M` first."
        )
    return TokenData(tuple(_load_one_bin(p) for p in paths), tuple(paths))


def get_token_batch(
    data: TokenData,
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
    span_tokens = batch_tokens + 1
    if offset is None:
        buf_np = data.random_span(span_tokens)
    else:
        max_start = len(data) - span_tokens
        if max_start <= 0:
            raise ValueError(f"dataset has {len(data):,} tokens, too short for {batch_tokens:,}-token batch")
        buf_np = data.span(offset % max_start, span_tokens)
    buf = torch.from_numpy(buf_np.astype(np.int64))
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
