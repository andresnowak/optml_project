"""Download WikiText-103 and tokenize to GPT-2 BPE uint16 .bin files.

Produces ``data/wikitext103/{train,val}.bin`` (nanoGPT-style memmap arrays) that
the trainer reads. Validation uses the wikitext-103 validation split.

Usage:  python scripts/prepare_wikitext.py
"""

from __future__ import annotations

import os

import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join("data", "wikitext103")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    # Namespaced repo id: recent huggingface_hub rejects the bare "wikitext"
    # canonical name. Salesforce/wikitext is the current home of the dataset.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    for split, out_name in (("train", "train.bin"), ("validation", "val.bin")):
        text = "\n".join(t for t in ds[split]["text"] if t)
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(OUT_DIR, out_name)
        arr.tofile(path)
        print(f"{split}: {len(arr):,} tokens -> {path}")


if __name__ == "__main__":
    main()
