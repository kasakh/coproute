"""Integration tests for the ISSUE-77 paired-suite arm."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_paired_main_suite as paired


def _task(rows: int = 100, models: int = 6) -> SimpleNamespace:
    rng = np.random.default_rng(20260920)
    correctness = np.zeros((rows, models), dtype=np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
        correctness[row, (row + 2) % models] = 1
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=rng.normal(size=(rows, 5)),
        correctness=correctness,
        model_ids=tuple(f"model-{column}" for column in range(models)),
        source_hashes={"leaderboard": "a" * 64, "router_dataset": "b" * 64},
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
    )


def _fake_knn(train_x, train_y, query_x, k):
    del train_x, k
    n_queries = len(query_x)
    n_models = train_y.shape[1]
    base = np.linspace(0.20, 0.80, n_models)
    variation = (np.arange(n_queries)[:, None] % 5 - 2) * 0.01
    probabilities = np.clip(base[None, :] + variation, 1e-4, 1.0 - 1e-4)
    support = np.broadcast_to(train_y[:1], (n_queries, 1, n_models)).copy()
    return probabilities, support


def _smoke_payload() -> dict:
    return paired.run_paired_suite(
        _task(),
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        include_chain_discount=True,
        knn_function=_fake_knn,
    )


def test_registered_chain_arm_tunes_on_validation_and_records_endpoint_evidence() -> None:
    payload = _smoke_payload()
    assert payload["design"] == "ISSUE_77_chain_discounted_stopping"
    assert payload["include_chain_discount"] is True
    assert payload["chain_discount_min_survivor_grid"] == list(
        paired.CHAIN_DISCOUNT_MIN_SURVIVOR_GRID
    )
    assert paired.CHAIN_DISCOUNT_METHOD in payload["method_identities"]

    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        result = redraw["methods"][paired.CHAIN_DISCOUNT_METHOD]
        assert result["selection_source"] == "validation_only"
        assert result["final_calibration_count"] == 1
        assert result["endpoint_reference_method"] == "fixed_tally_factorized"
        assert result["endpoint_result_identity_verified"] is True
        trajectory = result["validation_trajectory"]
        assert [row["minimum_survivor_count"] for row in trajectory] == list(
            paired.CHAIN_DISCOUNT_MIN_SURVIVOR_GRID
        )
        assert trajectory[0]["discount_fully_off"] is True
        assert result["chosen_hyper"]["minimum_survivor_count"] in (
            paired.CHAIN_DISCOUNT_MIN_SURVIVOR_GRID
        )
        profile = result["chain_discount_profile"]
        assert len(profile["order"]) == payload["pool_size"]
        assert len(profile["raw_discount"]) == payload["pool_size"] + 1
        assert profile["discount_fully_off"] == result["discount_fully_off"]


def test_registered_arm_presence_check_fails_loudly_for_missing_rows() -> None:
    payload = _smoke_payload()
    broken = copy.deepcopy(payload["cells"])
    del broken[0]["redraws"][0]["methods"][paired.CHAIN_DISCOUNT_METHOD]
    with pytest.raises(RuntimeError, match="registered arm presence mismatch"):
        paired.assert_registered_arm_presence(broken, payload["method_identities"])


def test_chain_arm_is_opt_in_and_default_method_set_is_unchanged() -> None:
    payload = paired.run_paired_suite(
        _task(),
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        knn_function=_fake_knn,
    )
    assert payload["method_identities"] == sorted(paired.NONLEARNED_METHODS)
    assert payload["include_chain_discount"] is False
    assert payload["chain_discount_min_survivor_grid"] is None
