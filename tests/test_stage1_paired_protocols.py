from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_paired_main_suite as paired


def _task() -> SimpleNamespace:
    rng = np.random.default_rng(20260920)
    labels = np.zeros((100, 4), dtype=np.int8)
    for row in range(100):
        labels[row, row % 4] = 1
        labels[row, (row + 1) % 4] = 1
    return SimpleNamespace(
        task="mmlu",
        query_ids=tuple(f"q{index}" for index in range(100)),
        embeddings=rng.normal(size=(100, 5)),
        correctness=labels,
        model_ids=tuple(f"m{index}" for index in range(4)),
        source_hashes={"data": "a", "embeddings": "b"},
        feature_audit={"degenerate": False},
    )


def _knn(train_x, train_y, query_x, k):
    probabilities = np.tile(np.asarray([0.7, 0.6, 0.5, 0.4]), (len(query_x), 1))
    support = np.zeros((len(query_x), 1, train_y.shape[1]), dtype=np.int8)
    return probabilities, support


def test_run_payload_records_routerbench_protocol() -> None:
    payload = paired.run_paired_suite(
        _task(),
        pool_size=4,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        protocol="routerbench",
        knn_function=_knn,
    )
    assert payload["protocol"] == "routerbench"
    assert payload["benchmark"] == "mmlu"


def test_unknown_protocol_and_missing_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="protocol must"):
        paired.run_paired_suite(
            _task(), pool_size=4, pool_seed=7001, redraw_start=0,
            redraw_end=1, knn_k=8, protocol="unknown", knn_function=_knn,
        )
    with pytest.raises(ValueError, match="requires leaderboard"):
        paired._load_task("routereval", "ifeval")
    with pytest.raises(ValueError, match="requires its pickle"):
        paired._load_task("routerbench", "mmlu")


def test_checkpoint_loader_refuses_wrong_target_family_before_model_load(tmp_path: Path) -> None:
    model_ids = ["m0", "m1"]
    for seed in paired.TRAINING_SEEDS:
        directory = tmp_path / f"seed{seed}"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(b"not loaded because family is wrong")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata = {
            "identity": {
                "source": "routereval",
                "target_family": "gpqa",
                "excluded_family": "gpqa",
                "method": "flexrouter",
                "pool_size": 2,
                "pool_seed": 7001,
                "training_seed": seed,
            },
            "model_pool": {
                "model_ids": model_ids,
                "model_ids_sha256": paired.hash_json(model_ids),
            },
        }
        (directory / "training_metadata.json").write_text(json.dumps(metadata))
        manifest = {
            "checkpoint_sha256": checkpoint_hash,
            "files": {"checkpoint.pt": checkpoint_hash},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing registered checkpoint seeds"):
        paired.load_checkpoint_ensemble(
            tmp_path,
            "flexrouter",
            pool_size=2,
            pool_seed=7001,
            model_ids=model_ids,
            expected_source="routereval",
            expected_target_family="bbh",
        )
