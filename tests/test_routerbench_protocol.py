"""Tests for the RouterBench correctness loader."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routerbench_protocol import (
    BENCHMARK_FAMILIES,
    MAX_DUPLICATE_RATE,
    ROUTERBENCH_MODEL_COLUMNS,
    RouterBenchDegenerateFeatures,
    load_correctness_only,
    subsample_pool,
)


def _make_fake_pickle(tmp_path: Path, rows: list[dict]) -> Path:
    frame = pd.DataFrame(rows)
    path = tmp_path / "routerbench_test.pkl"
    with path.open("wb") as handle:
        pickle.dump(frame, handle)
    return path


def _default_row(prompt: str, eval_name: str, correctness: dict | None = None) -> dict:
    row: dict = {"prompt": prompt, "eval_name": eval_name}
    for name in ROUTERBENCH_MODEL_COLUMNS:
        row[name] = 1.0 if (correctness or {}).get(name, False) else 0.0
    return row


def test_loader_returns_expected_shape(tmp_path: Path) -> None:
    rows = [
        _default_row(f"prompt-{idx}", "mbpp", {ROUTERBENCH_MODEL_COLUMNS[idx % 11]: True})
        for idx in range(30)
    ]
    path = _make_fake_pickle(tmp_path, rows)
    result = load_correctness_only(pickle_path=path, benchmark="mbpp")
    assert result.correctness.shape == (30, 11)
    assert result.model_names == ROUTERBENCH_MODEL_COLUMNS
    assert result.feature_audit["family_key"] == "mbpp"
    assert result.feature_audit["duplicate_rate"] == 0.0


def test_family_aggregation_for_mmlu(tmp_path: Path) -> None:
    rows = [
        _default_row(f"mmlu-anatomy-{idx}", "mmlu-anatomy") for idx in range(10)
    ] + [
        _default_row(f"mmlu-astronomy-{idx}", "mmlu-astronomy") for idx in range(15)
    ] + [
        _default_row(f"unrelated-{idx}", "hellaswag") for idx in range(20)
    ]
    path = _make_fake_pickle(tmp_path, rows)
    result = load_correctness_only(pickle_path=path, benchmark="mmlu")
    assert result.correctness.shape == (25, 11)
    breakdown = result.feature_audit["eval_name_breakdown"]
    assert breakdown == {"mmlu-anatomy": 10, "mmlu-astronomy": 15}


def test_duplicate_guard_raises(tmp_path: Path) -> None:
    single = _default_row("shared-prompt", "grade-school-math")
    rows = [dict(single) for _ in range(50)]
    path = _make_fake_pickle(tmp_path, rows)
    with pytest.raises(RouterBenchDegenerateFeatures) as info:
        load_correctness_only(pickle_path=path, benchmark="gsm8k")
    assert "duplicate rate" in str(info.value)


def test_duplicate_guard_allows_below_threshold(tmp_path: Path) -> None:
    unique_rows = [
        _default_row(f"prompt-{idx}", "winogrande") for idx in range(95)
    ]
    duplicated_rows = [
        _default_row("shared", "winogrande") for _ in range(5)
    ]
    path = _make_fake_pickle(tmp_path, unique_rows + duplicated_rows)
    result = load_correctness_only(pickle_path=path, benchmark="winogrande")
    assert result.correctness.shape == (100, 11)
    assert result.feature_audit["duplicate_rate"] == pytest.approx(0.04)
    assert result.feature_audit["duplicate_rate"] < MAX_DUPLICATE_RATE


def test_unknown_benchmark_raises(tmp_path: Path) -> None:
    rows = [_default_row("prompt", "mbpp") for _ in range(5)]
    path = _make_fake_pickle(tmp_path, rows)
    with pytest.raises(KeyError):
        load_correctness_only(pickle_path=path, benchmark="not-a-family")


def test_empty_family_raises(tmp_path: Path) -> None:
    rows = [_default_row("prompt", "hellaswag") for _ in range(5)]
    path = _make_fake_pickle(tmp_path, rows)
    with pytest.raises(KeyError):
        load_correctness_only(pickle_path=path, benchmark="mbpp")


def test_threshold_binarizes(tmp_path: Path) -> None:
    row = _default_row("prompt", "hellaswag")
    for idx, name in enumerate(ROUTERBENCH_MODEL_COLUMNS):
        row[name] = 0.3 + idx * 0.06  # values from 0.3 to 0.9
    path = _make_fake_pickle(tmp_path, [row])
    result = load_correctness_only(pickle_path=path, benchmark="hellaswag", threshold=0.5)
    # models with value >= 0.5 are correct
    expected = np.array([[int(0.3 + idx * 0.06 >= 0.5) for idx in range(11)]])
    assert np.array_equal(result.correctness, expected)


def test_nan_treated_as_incorrect(tmp_path: Path) -> None:
    row = _default_row("prompt", "arc-challenge")
    row[ROUTERBENCH_MODEL_COLUMNS[0]] = float("nan")
    row[ROUTERBENCH_MODEL_COLUMNS[1]] = 1.0
    path = _make_fake_pickle(tmp_path, [row])
    result = load_correctness_only(pickle_path=path, benchmark="arc-challenge")
    assert result.correctness[0, 0] == 0
    assert result.correctness[0, 1] == 1


def test_subsample_pool_identity_at_full(tmp_path: Path) -> None:
    ordering = subsample_pool(11, 11, seed=42)
    assert np.array_equal(ordering, np.arange(11))


def test_subsample_pool_reproducible(tmp_path: Path) -> None:
    a = subsample_pool(11, 7, seed=123)
    b = subsample_pool(11, 7, seed=123)
    assert np.array_equal(a, b)


def test_subsample_pool_rejects_oversize(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        subsample_pool(11, 20, seed=0)


def test_feature_audit_records_hash_and_metadata(tmp_path: Path) -> None:
    rows = [
        _default_row(f"prompt-{idx}", "grade-school-math") for idx in range(40)
    ]
    path = _make_fake_pickle(tmp_path, rows)
    result = load_correctness_only(pickle_path=path, benchmark="gsm8k")
    assert result.feature_audit["protocol"] == "routerbench"
    assert result.feature_audit["threshold"] == 0.5
    assert result.feature_audit["features_present"] is True
    assert isinstance(result.source_sha256, str)
    assert len(result.source_sha256) == 64
