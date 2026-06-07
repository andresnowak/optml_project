"""Cache WikiText-103 as GPT-2 BPE uint16 token files."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join("data", "wikitext103")
DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TOKENIZER = "gpt2"
SPLITS = (("train", "train.bin"), ("validation", "val.bin"))


def _write_split(ds, enc, split: str, path: str, force: bool) -> int:
    if os.path.exists(path) and not force:
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"{split}: {len(arr):,} cached tokens -> {path}")
        return len(arr)
    text = "\n".join(t for t in ds[split]["text"] if t)
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    print(f"{split}: {len(arr):,} tokens -> {path}")
    return len(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild existing token files")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding(TOKENIZER)
    ds = load_dataset(DATASET_REPO, DATASET_CONFIG)

    counts = {}
    for split, out_name in SPLITS:
        counts[split] = _write_split(ds, enc, split, os.path.join(OUT_DIR, out_name), args.force)

    metadata = {
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "tokenizer": TOKENIZER,
        "train_tokens": counts["train"],
        "val_tokens": counts["validation"],
        "files": {"train": "train.bin", "validation": "val.bin"},
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
