#!/usr/bin/env python3
"""Compute mDeBERTa query embeddings for exact RACER router reproduction.

The RACER paper specifies mDeBERTa-v3-base but not a pooling rule. This script
saves both first-token (CLS-style) and attention-mask mean pooled embeddings so
the pooling choice can be reported as an ablation instead of guessed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_prompt(value) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x) for x in value)
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-pkl", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    inp = Path(args.input_pkl)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    df = pd.read_pickle(inp).reset_index(drop=True)
    indices = np.arange(len(df), dtype=np.int64)[args.shard_index::args.num_shards]
    prompts = [normalize_prompt(df.loc[int(i), "prompt"]) for i in indices]

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True)
    model = model.cuda().eval()

    cls_parts, mean_parts = [], []
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            texts = prompts[start:start + args.batch_size]
            enc = tok(texts, padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt")
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            cls = torch.nn.functional.normalize(hidden[:, 0, :], dim=1)
            mean = torch.nn.functional.normalize(
                (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1), dim=1)
            cls_parts.append(cls.float().cpu().numpy().astype(np.float16))
            mean_parts.append(mean.float().cpu().numpy().astype(np.float16))
            if start % (args.batch_size * 20) == 0:
                print(f"shard={args.shard_index} {start}/{len(prompts)}", flush=True)

    np.save(out / "indices.npy", indices)
    np.save(out / "cls.npy", np.concatenate(cls_parts, axis=0))
    np.save(out / "mean.npy", np.concatenate(mean_parts, axis=0))
    meta = {
        "input": str(inp),
        "input_sha256": sha256_file(inp),
        "model_path": args.model_path,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": int(len(indices)),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "torch": torch.__version__,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print("DONE", json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
