"""Reproduce RouterDC's RouterBench cost-aware routing trend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS
from components.routerdc_baseline import RouterDCBaseline, RouterDCConfig
from scripts.run_learned_baselines import sha256_file

FAMILIES = ("gsm8k", "hellaswag", "mbpp", "mmlu", "winograde", "arc_challenge")


def repeat_frame(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot repeat an empty RouterBench frame")
    indices = np.arange(count) % len(frame)
    return frame.iloc[indices].reset_index(drop=True)


def load_inputs(data_dir: Path, cost_rate: float) -> dict[str, Any]:
    prompts = []
    adjusted = []
    clusters = []
    validation_prompts = []
    validation_adjusted = []
    source_hashes = {}
    cost_columns = [f"{name}|total_cost" for name in ROUTERBENCH_MODEL_COLUMNS]
    for family_index, family in enumerate(FAMILIES):
        path = data_dir / "train" / f"{family}_train.csv"
        source_hashes[str(path.relative_to(data_dir))] = sha256_file(path)
        frame = pd.read_csv(path).dropna(subset=["prompt", *ROUTERBENCH_MODEL_COLUMNS, *cost_columns])
        sampled = repeat_frame(frame, 1000)
        raw = sampled.loc[:, ROUTERBENCH_MODEL_COLUMNS].to_numpy(dtype=float)
        costs = sampled.loc[:, cost_columns].to_numpy(dtype=float)
        values = raw - cost_rate * costs
        prompts.extend(sampled["prompt"].astype(str).tolist())
        adjusted.append(values)
        cluster_column = sampled["cluster_id"] if "cluster_id" in sampled else family_index
        clusters.extend(np.asarray(cluster_column, dtype=int).tolist())
        validation_prompts.extend(frame["prompt"].astype(str).tolist())
        validation_adjusted.append(
            frame.loc[:, ROUTERBENCH_MODEL_COLUMNS].to_numpy(dtype=float)
            - cost_rate * frame.loc[:, cost_columns].to_numpy(dtype=float)
        )
        test_path = data_dir / "test" / f"{family}_test.csv"
        source_hashes[str(test_path.relative_to(data_dir))] = sha256_file(test_path)
    return {
        "prompts": prompts,
        "adjusted": np.concatenate(adjusted).astype(np.float32),
        "clusters": np.asarray(clusters, dtype=np.int64),
        "validation_prompts": validation_prompts,
        "validation_adjusted": np.concatenate(validation_adjusted).astype(np.float32),
        "source_hashes": source_hashes,
    }


def evaluate(model: RouterDCBaseline, data_dir: Path) -> dict[str, Any]:
    cost_columns = [f"{name}|total_cost" for name in ROUTERBENCH_MODEL_COLUMNS]
    tasks = {}
    for family in FAMILIES:
        frame = pd.read_csv(data_dir / "test" / f"{family}_test.csv").dropna(
            subset=["prompt", *ROUTERBENCH_MODEL_COLUMNS, *cost_columns]
        )
        probabilities = model.predict_proba(frame["prompt"].astype(str).tolist())
        chosen = probabilities.argmax(axis=1)
        quality = frame.loc[:, ROUTERBENCH_MODEL_COLUMNS].to_numpy(dtype=float)
        costs = frame.loc[:, cost_columns].to_numpy(dtype=float)
        rows = np.arange(len(frame))
        tasks[family] = {
            "query_count": len(frame),
            "selected_quality": float(quality[rows, chosen].mean()),
            "selected_cost": float(costs[rows, chosen].mean()),
        }
    return {
        "tasks": tasks,
        "macro_selected_quality": float(np.mean([value["selected_quality"] for value in tasks.values()])),
        "macro_selected_cost": float(np.mean([value["selected_cost"] for value in tasks.values()])),
    }


def run(args: argparse.Namespace) -> Path:
    destination = Path(args.out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    data = load_inputs(Path(args.data_dir), args.cost_rate)
    config = RouterDCConfig(
        seed=7,
        top_k=4,
        bottom_k=4,
        training_steps=1000,
        evaluation_steps=100,
        cluster_loss_weight=1.0,
        micro_batch_size=32,
    )
    model = RouterDCBaseline(
        args.backbone_path, ROUTERBENCH_MODEL_COLUMNS, config=config, device=args.device
    )
    model.fit(
        data["prompts"], data["adjusted"], data["clusters"],
        validation_prompts=data["validation_prompts"],
        validation_quality=data["validation_adjusted"],
    )
    result = {
        "protocol": "routerdc_official_routerbench_cost_trend_v1",
        "identity": {
            "cost_rate": args.cost_rate,
            "seed": 7,
            "host": socket.gethostname(),
            "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "run_stamp": args.run_stamp,
        },
        "official_launcher_settings": {
            "top_k": 4, "bottom_k": 4, "training_steps": 1000,
            "evaluation_steps": 100, "learning_rate": 5e-5,
            "cluster_loss_weight": 1, "cost_rates": [0, 1],
        },
        "source_hashes": data["source_hashes"],
        "evaluation": evaluate(model, Path(args.data_dir)),
        "best_step": model.best_step,
        "best_validation_quality": model.best_validation_quality,
        "official_code_reused": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        model.save(temporary / "checkpoint.pt")
        (temporary / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        files = {name: sha256_file(temporary / name) for name in ("checkpoint.pt", "result.json")}
        (temporary / "manifest.json").write_text(json.dumps({
            "protocol": result["protocol"], "immutable": True,
            "identity": result["identity"], "files": files,
            "checkpoint_sha256": files["checkpoint.pt"],
        }, indent=2, sort_keys=True) + "\n")
        (temporary / "DONE").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--cost-rate", required=True, type=float, choices=(0.0, 1.0))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--out-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
