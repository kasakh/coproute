from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from components import stage1_task_adapters as adapters
from scripts.run_learned_baselines import LearnedTaskData, ROUTEREVAL_KEYS


def _family_task(count: int = 2) -> LearnedTaskData:
    return LearnedTaskData(
        task="bbh",
        query_ids=tuple(f"family:{index}" for index in range(count)),
        embeddings=np.arange(count * 3, dtype=float).reshape(count, 3),
        correctness=np.asarray([[1, 0], [0, 1]], dtype=np.int8)[:count],
        model_ids=("m0", "m1"),
        prompts=tuple(f"prompt {index}" for index in range(count)),
        feature_audit={"degenerate": False},
        source_hashes={"leaderboard": "a", "router_dataset": "b"},
    )


def _leaderboard(path: Path, selected: np.ndarray) -> None:
    data = {
        key: {"correctness": np.empty((0, 2), dtype=float)}
        for key in ROUTEREVAL_KEYS["bbh"]
    }
    data["bbh_boolean_expressions"] = {"correctness": selected.astype(float)}
    with path.open("wb") as handle:
        pickle.dump({"data": data, "model": np.asarray(["m0", "m1"])}, handle)


def test_routereval_registered_slice_reproduces_direct_labels(tmp_path: Path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.pkl"
    selected = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    _leaderboard(leaderboard, selected)
    monkeypatch.setattr(adapters, "load_routereval_task", lambda *args, **kwargs: _family_task())
    result = adapters.load_routereval_registered_task(
        leaderboard, tmp_path, "bbh_boolean_expressions"
    )
    assert result.task == "bbh_boolean_expressions"
    assert np.array_equal(result.correctness, selected)
    assert result.query_ids == (
        "routereval:bbh_boolean_expressions:0",
        "routereval:bbh_boolean_expressions:1",
    )
    assert result.feature_audit["alignment"] == "family_original_index_exact_then_registered_slice"


def test_routereval_registered_slice_fails_on_alignment_mismatch(tmp_path: Path, monkeypatch) -> None:
    leaderboard = tmp_path / "leaderboard.pkl"
    _leaderboard(leaderboard, np.asarray([[1, 0], [0, 1]], dtype=np.int8))
    wrong = _family_task()
    wrong = LearnedTaskData(**{**wrong.__dict__, "correctness": 1 - wrong.correctness})
    monkeypatch.setattr(adapters, "load_routereval_task", lambda *args, **kwargs: wrong)
    with pytest.raises(RuntimeError, match="does not reproduce direct labels"):
        adapters.load_routereval_registered_task(
            leaderboard, tmp_path, "bbh_boolean_expressions"
        )


def test_routerbench_registered_task_filters_exact_family(monkeypatch) -> None:
    frame = pd.DataFrame({
        "task": ["mmlu", "hellaswag", "mmlu"],
        "prompt": ["first", "second", "third"],
    })
    data = SimpleNamespace(
        frame=frame,
        query_ids=("q0", "q1", "q2"),
        model_columns=tuple(adapters.ROUTERBENCH_MODEL_COLUMNS),
        source_sha256="source",
    )
    labels = SimpleNamespace(standard=np.asarray([
        [1] + [0] * 10,
        [0, 1] + [0] * 9,
        [0, 0, 1] + [0] * 8,
    ], dtype=np.int8))
    embeddings = np.arange(12, dtype=float).reshape(3, 4)
    audit = {"source_sha256": "embed", "manifest_sha256": "manifest"}
    monkeypatch.setattr(adapters, "load_routerbench", lambda *args, **kwargs: data)
    monkeypatch.setattr(adapters, "derive_coverage_labels", lambda *args, **kwargs: labels)
    monkeypatch.setattr(adapters, "load_aligned_routerbench_embeddings", lambda *args, **kwargs: (embeddings, audit))
    result = adapters.load_routerbench_registered_task("data.pkl", "embeddings.npz", "mmlu")
    assert result.query_ids == ("q0", "q2")
    assert result.correctness.shape == (2, 11)
    assert result.embeddings.shape == (2, 4)
    assert result.feature_audit["alignment"] == "exact_query_id_set_and_reorder_then_task_slice"


def test_checkpoint_target_family_mapping_is_strict() -> None:
    assert adapters.checkpoint_target_family("routereval", "bbh_navigate") == "bbh"
    assert adapters.checkpoint_target_family("routereval", "ifeval") is None
    assert adapters.checkpoint_target_family("routerbench", "mmlu") == "mmlu"
    with pytest.raises(ValueError, match="unregistered"):
        adapters.checkpoint_target_family("routerbench", "gsm8k")
