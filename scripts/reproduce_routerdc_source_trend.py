"""Reproduce RouterDC's published lambda-one versus lambda-zero trend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from components.routerdc_baseline import RouterDCBaseline, RouterDCConfig
from scripts.run_learned_baselines import sha256_file

FAMILIES = ("gsm8k", "humaneval", "arc_challenge", "mmlu", "cmmlu")
TRAIN_FILES = {
    family: f"{family}_train.json" if family != "gsm8k" else "gsm8k-train.json"
    for family in FAMILIES
}
TEST_FILES = {
    "gsm8k": "gsm8k-test.json",
    "humaneval": "humaneval_test.json",
    "arc_challenge": "arc_challenge_test.json",
    "mmlu": "mmlu_test.json",
    "cmmlu": "cmmlu_test.json",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"invalid RouterDC data file {path}")
    return value


def repeat(values: Sequence[Any], count: int) -> list[Any]:
    if not values:
        raise ValueError("cannot repeat empty data")
    return [values[index % len(values)] for index in range(count)]


def evaluate(model: RouterDCBaseline, data_dir: Path) -> dict[str, Any]:
    by_family = {}
    all_values = []
    for family in FAMILIES:
        rows = load_rows(data_dir / "test" / TEST_FILES[family])
        prompts = [str(row["question"]) for row in rows]
        quality = np.asarray([list(row["scores"].values()) for row in rows], dtype=float)
        probabilities = model.predict_proba(prompts)
        selected = probabilities.argmax(axis=1)
        values = quality[np.arange(len(rows)), selected]
        by_family[family] = {
            "selected_quality": float(values.mean()),
            "query_count": len(rows),
        }
        all_values.extend(values.tolist())
    macro = float(np.mean([record["selected_quality"] for record in by_family.values()]))
    return {
        "families": by_family,
        "macro_selected_quality": macro,
        "pooled_query_selected_quality": float(np.mean(all_values)),
    }


def run(arguments: argparse.Namespace) -> Path:
    destination = Path(arguments.out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    data_dir = Path(arguments.data_dir)
    rows_by_family = {
        family: load_rows(data_dir / "train" / TRAIN_FILES[family])
        for family in FAMILIES
    }
    first = rows_by_family[FAMILIES[0]][0]
    model_ids = tuple(str(value) for value in first["scores"])
    prompts = []
    quality = []
    clusters = []
    validation_prompts = []
    validation_quality = []
    source_hashes = {}
    for family_index, family in enumerate(FAMILIES):
        path = data_dir / "train" / TRAIN_FILES[family]
        source_hashes[str(path.relative_to(data_dir))] = sha256_file(path)
        source = repeat(rows_by_family[family], 1000)
        for row in source:
            if tuple(str(value) for value in row["scores"]) != model_ids:
                raise ValueError(f"model order differs in {family}")
            prompts.append(str(row["question"]))
            quality.append(list(row["scores"].values()))
            clusters.append(int(row.get("cluster_id", family_index)))
        validation_prompts.extend(str(row["question"]) for row in rows_by_family[family])
        validation_quality.extend(list(row["scores"].values()) for row in rows_by_family[family])
        test_path = data_dir / "test" / TEST_FILES[family]
        source_hashes[str(test_path.relative_to(data_dir))] = sha256_file(test_path)
    config = RouterDCConfig(
        seed=arguments.seed,
        cluster_loss_weight=arguments.lambda_weight,
        training_steps=arguments.training_steps,
        evaluation_steps=arguments.evaluation_steps,
        max_length=arguments.max_length,
    )
    model = RouterDCBaseline(
        arguments.backbone_path, model_ids, config=config, device=arguments.device
    )
    model.fit(
        prompts,
        np.asarray(quality, dtype=np.float32),
        clusters,
        validation_prompts=validation_prompts,
        validation_quality=np.asarray(validation_quality, dtype=np.float32),
    )
    result = {
        "protocol": "routerdc_source_lambda_trend_v1",
        "identity": {
            "lambda_weight": arguments.lambda_weight,
            "seed": arguments.seed,
            "host": socket.gethostname(),
            "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "run_stamp": arguments.run_stamp,
        },
        "paper_claim": "lambda one has higher average testing accuracy than lambda zero",
        "model_ids": list(model_ids),
        "source_hashes": source_hashes,
        "configuration": model.metadata()["config"],
        "validation_selected_quality": model.best_validation_quality,
        "best_step": model.best_step,
        "evaluation": evaluate(model, data_dir),
        "official_code_reused": False,
        "official_data_used_ephemerally": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        model.save(temporary / "checkpoint.pt")
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = {name: sha256_file(temporary / name) for name in ("checkpoint.pt", "result.json")}
        manifest = {
            "protocol": result["protocol"], "immutable": True,
            "identity": result["identity"], "files": files,
            "checkpoint_sha256": files["checkpoint.pt"],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "DONE").touch()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--lambda-weight", required=True, type=float, choices=(0.0, 1.0))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--evaluation-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
