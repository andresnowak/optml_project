"""Download GPT-2-tokenized FineWeb shards, modded-nanogpt style."""

from __future__ import annotations

import argparse
import math
import os

from huggingface_hub import hf_hub_download

OUT_DIR = os.path.join("data", "fineweb10B")
REPO_ID = "kjj0/fineweb10B-gpt2"
TOKENS_PER_SHARD = 100_000_000
MAX_TRAIN_SHARDS = 103


def parse_tokens(text: str) -> int:
    s = text.strip().lower().replace("_", "")
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def download(filename: str) -> None:
    local_path = os.path.join(OUT_DIR, filename)
    if os.path.exists(local_path):
        print(f"exists {local_path}")
        return
    hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", help="training tokens to cache, e.g. 500M, 0.5B, 1000000000")
    args = ap.parse_args()

    tokens = parse_tokens(args.tokens)
    num_chunks = math.ceil(tokens / TOKENS_PER_SHARD)
    if not 1 <= num_chunks <= MAX_TRAIN_SHARDS:
        raise SystemExit(f"requested {num_chunks} chunks, expected 1..{MAX_TRAIN_SHARDS}")

    print(f"downloading {num_chunks} FineWeb train chunks ({num_chunks * TOKENS_PER_SHARD:,} tokens)")
    download("fineweb_val_000000.bin")
    for i in range(1, num_chunks + 1):
        download(f"fineweb_train_{i:06d}.bin")


if __name__ == "__main__":
    main()
