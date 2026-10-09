"""Tests for the RouterEval loader, its feature guard and pool subsampling."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.routereval_protocol import (  # noqa: E402
    RouterEvalDegenerateFeatures,
    audit_features,
    headroom_census,
    load_benchmark,
    subsample_pool,
)

N_TRAIN, N_VAL, N_TEST = 6, 2, 3
N_ROWS = N_TRAIN + N_VAL + N_TEST
N_MODELS = 8


def _write_fixtures(tmp_path: Path, *, prompts: list[str] | None = None) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    correctness = (rng.random((N_ROWS, N_MODELS)) > 0.4).astype(float)
    leaderboard = {
        "model": np.asarray([f"model-{i}" for i in range(N_MODELS)]),
        "data": {"toy": {"correctness": correctness}},
    }
    if prompts is None:
        prompts = [f"question {i}" for i in range(N_ROWS)]
    router_dataset = {
        "split_index": {
            "train_indices": np.arange(N_TRAIN),
            "val_indices": np.arange(N_TRAIN, N_TRAIN + N_VAL),
            "test_indices": np.arange(N_TRAIN + N_VAL, N_ROWS),
        },
        "embedding": {
            "train_embed": rng.random((N_TRAIN, 4)),
            "val_embed": rng.random((N_VAL, 4)),
            "test_embed": rng.random((N_TEST, 4)),
        },
        "prompt": {
            "train_prompt": np.asarray(prompts[:N_TRAIN], dtype=object),
            "val_prompt": np.asarray(prompts[N_TRAIN : N_TRAIN + N_VAL], dtype=object),
            "test_prompt": np.asarray(prompts[N_TRAIN + N_VAL :], dtype=object),
        },
        "easy": {},
        "hard": {},
    }
    leaderboard_path = tmp_path / "leaderboard.pkl"
    router_path = tmp_path / "toy_router_dataset.pkl"
    leaderboard_path.write_bytes(pickle.dumps(leaderboard))
    router_path.write_bytes(pickle.dumps(router_dataset))
    return leaderboard_path, router_path


def test_loads_and_audits_a_clean_benchmark(tmp_path: Path) -> None:
    leaderboard_path, router_path = _write_fixtures(tmp_path)
    data = load_benchmark(
        leaderboard_path=leaderboard_path,
        router_dataset_path=router_path,
        benchmark="toy",
    )
    assert data.n_queries == N_ROWS
    assert data.n_models == N_MODELS
    assert data.feature_audit["degenerate"] is False
    assert data.feature_audit["duplicate_rate"] == 0.0
    assert len(data.source_sha256) == 64


def test_degenerate_prompts_are_refused(tmp_path: Path) -> None:
    """The RouterBench GSM8K failure mode must be fatal, not a warning."""

    leaderboard_path, router_path = _write_fixtures(
        tmp_path, prompts=["identical"] * N_ROWS
    )
    with pytest.raises(RouterEvalDegenerateFeatures, match="collapse"):
        load_benchmark(
            leaderboard_path=leaderboard_path,
            router_dataset_path=router_path,
            benchmark="toy",
        )


def test_high_duplicate_rate_is_refused(tmp_path: Path) -> None:
    prompts = ["a"] * (N_ROWS - 1) + ["b"]
    leaderboard_path, router_path = _write_fixtures(tmp_path, prompts=prompts)
    with pytest.raises(RouterEvalDegenerateFeatures, match="duplicate prompt rate"):
        load_benchmark(
            leaderboard_path=leaderboard_path,
            router_dataset_path=router_path,
            benchmark="toy",
        )


def test_splits_are_disjoint_and_cover_every_row(tmp_path: Path) -> None:
    leaderboard_path, router_path = _write_fixtures(tmp_path)
    data = load_benchmark(
        leaderboard_path=leaderboard_path,
        router_dataset_path=router_path,
        benchmark="toy",
    )
    train = set(data.split_indices["train"].tolist())
    val = set(data.split_indices["val"].tolist())
    test = set(data.split_indices["test"].tolist())
    assert not train & val and not train & test and not val & test
    assert train | val | test == set(range(N_ROWS))


def test_pool_subsample_is_reproducible_and_bounded() -> None:
    first = subsample_pool(100, 10, seed=42)
    second = subsample_pool(100, 10, seed=42)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, subsample_pool(100, 10, seed=43))
    assert first.min() >= 0 and first.max() < 100
    assert len(set(first.tolist())) == 10
    with pytest.raises(ValueError, match="exceeds available models"):
        subsample_pool(5, 10, seed=0)


def test_headroom_census_reports_all_miss_per_pool_size(tmp_path: Path) -> None:
    leaderboard_path, router_path = _write_fixtures(tmp_path)
    data = load_benchmark(
        leaderboard_path=leaderboard_path,
        router_dataset_path=router_path,
        benchmark="toy",
    )
    census = headroom_census(data, [2, 4, N_MODELS], seed=1)
    assert [r["pool_size"] for r in census["records"]] == [2, 4, N_MODELS]
    rates = [r["complete_pool_all_miss_rate"] for r in census["records"]]
    # A larger pool can only cover more, so all-miss must be non-increasing.
    assert rates == sorted(rates, reverse=True)
    assert census["n_models_available"] == N_MODELS


def test_audit_features_counts_correctly() -> None:
    assert audit_features(["a", "a", "b"])["unique_prompts"] == 2
    assert audit_features(["a"] * 4)["degenerate"] is True
    assert audit_features([])["duplicate_rate"] == 0.0
