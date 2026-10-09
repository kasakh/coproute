"""Focused invariants for the ISSUE-72 degraded-router runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_issue72_degraded_router as issue72


def _split() -> dict[str, np.ndarray]:
    return {
        "design": np.asarray([0, 1, 2]),
        "router": np.asarray([3, 4]),
        "final_calibration": np.asarray([5, 6]),
        "validation": np.asarray([7, 8]),
        "test": np.asarray([9, 10]),
    }


def _task(rows: int = 500, models: int = 11) -> SimpleNamespace:
    generator = np.random.default_rng(20260920)
    correctness = np.zeros((rows, models), dtype=np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
        correctness[row, (row + 3) % models] = 1
        correctness[row, (row + 7) % models] = 1
    return SimpleNamespace(
        task="mmlu",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=generator.normal(size=(rows, 7)),
        correctness=correctness,
        model_ids=tuple(f"model-{index}" for index in range(models)),
        source_hashes={"routerbench": "a" * 64, "embeddings": "b" * 64},
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
    )


def _fake_knn(train_x, train_y, query_x, k):
    del train_x, k
    n_rows = len(query_x)
    n_models = train_y.shape[1]
    marginal = np.asarray(train_y, dtype=float).mean(axis=0)
    base = 0.15 + 0.70 * marginal
    row_effect = (np.arange(n_rows)[:, None] % 9 - 4) * 0.006
    model_effect = np.linspace(-0.025, 0.025, n_models)[None, :]
    probabilities = np.clip(base[None, :] + row_effect + model_effect, 0.01, 0.99)
    return probabilities, np.empty((n_rows, 0, n_models))


def test_temperature_fit_uses_validation_rows_only() -> None:
    probabilities = np.asarray(
        [
            [0.10, 0.90],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.40, 0.60],
            [0.55, 0.45],
            [0.65, 0.35],
        ]
    )
    labels = np.asarray(
        [[0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.int8
    )
    validation_rows = np.asarray([2, 3])
    first = issue72.fit_validation_temperature(probabilities, labels, validation_rows)

    changed_probabilities = probabilities.copy()
    changed_probabilities[[0, 1, 4, 5]] = 1.0 - changed_probabilities[[0, 1, 4, 5]]
    changed_labels = labels.copy()
    changed_labels[[0, 1, 4, 5]] = 1 - changed_labels[[0, 1, 4, 5]]
    second = issue72.fit_validation_temperature(
        changed_probabilities, changed_labels, validation_rows
    )

    assert first == second
    assert first["fit_roles"] == ["validation"]
    assert first["held_out_rows_used"] is False
    assert first["fit_row_count"] == 2


def test_sigma_zero_is_exact_raw_identity() -> None:
    generator = np.random.default_rng(4)
    primary = generator.uniform(0.01, 0.99, size=(20, 4))
    matched = generator.uniform(0.01, 0.99, size=(20, 4))
    labels = (generator.uniform(size=(20, 4)) < 0.5).astype(np.int8)
    variants = issue72.build_probability_variants(
        primary, matched, labels, np.arange(5, 12), redraw=3
    )
    indexed = {record["variant"]: record for record in variants}
    np.testing.assert_array_equal(
        indexed[issue72.SIGMA_ZERO_VARIANT]["primary_probabilities"], primary
    )
    np.testing.assert_array_equal(
        indexed[issue72.SIGMA_ZERO_VARIANT]["matched_control_probabilities"], matched
    )
    assert tuple(indexed) == issue72.VARIANT_IDS


def test_logit_noise_is_deterministic_and_seed_is_effective() -> None:
    probabilities = np.asarray([[0.2, 0.5, 0.8], [0.35, 0.65, 0.9]])
    first = issue72.deterministic_logit_noise(
        probabilities, 0.20, seed=991
    )
    second = issue72.deterministic_logit_noise(
        probabilities, 0.20, seed=991
    )
    different = issue72.deterministic_logit_noise(
        probabilities, 0.20, seed=992
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_systematic_over_and_underconfidence_are_monotone() -> None:
    probabilities = np.asarray([[0.05, 0.20, 0.49, 0.50, 0.51, 0.80, 0.95]])
    over = issue72.scale_logits(
        probabilities, issue72.OVERCONFIDENCE_LOGIT_SCALE
    )
    under = issue72.scale_logits(
        probabilities, issue72.UNDERCONFIDENCE_LOGIT_SCALE
    )
    original_distance = np.abs(probabilities - 0.5)
    assert np.all(np.abs(over - 0.5) >= original_distance - 1e-15)
    assert np.all(np.abs(under - 0.5) <= original_distance + 1e-15)
    assert over[0, 3] == pytest.approx(0.5)
    assert under[0, 3] == pytest.approx(0.5)
    assert np.all(np.diff(over) >= 0.0)
    assert np.all(np.diff(under) >= 0.0)


def test_matched_budget_rows_are_exact_design_router_union() -> None:
    split = _split()
    embeddings = np.arange(44, dtype=float).reshape(11, 4)
    labels = np.tile(np.asarray([[1, 0, 1]], dtype=np.int8), (11, 1))
    query_ids = [f"q-{index}" for index in range(11)]
    observed: list[np.ndarray] = []

    def capturing_knn(train_x, train_y, query_x, k):
        del train_x, k
        observed.append(np.array(train_y, copy=True))
        probabilities = np.broadcast_to(
            np.clip(train_y.mean(axis=0), 0.01, 0.99),
            (len(query_x), train_y.shape[1]),
        ).copy()
        return probabilities, np.empty((len(query_x), 0, train_y.shape[1]))

    _, _, provenance = issue72.fit_probability_streams(
        embeddings,
        labels,
        split,
        query_ids,
        knn_k=2,
        knn_function=capturing_knn,
    )
    np.testing.assert_array_equal(observed[0], labels[split["router"]])
    expected_rows = np.concatenate([split["design"], split["router"]])
    np.testing.assert_array_equal(observed[1], labels[expected_rows])
    matched = provenance["data_budget_matched_marginal_controls"]
    assert matched["fit_rows"] == expected_rows.tolist()
    assert matched["fit_query_ids"] == [query_ids[index] for index in expected_rows]
    assert matched["exact_union_verified"] is True
    assert matched["used_by_methods"] == [
        "exact_independence_matched_budget", "racer_g_matched_budget"
    ]
    assert matched["excluded_from_methods"] == [
        issue72.D1_METHOD,
        issue72.D2_METHOD,
        "exact_independence",
        "racer_g",
    ]


def test_decomposition_arithmetic_is_exact() -> None:
    methods = {
        "racer_g": {"certified": True, "test_calls": 7.5},
        "exact_independence": {"certified": True, "test_calls": 6.0},
        "racer_g_matched_budget": {"certified": True, "test_calls": 7.0},
        "exact_independence_matched_budget": {"certified": True, "test_calls": 5.5},
        issue72.D1_METHOD: {"certified": True, "test_calls": 4.25},
        issue72.D2_METHOD: {"certified": True, "test_calls": 4.75},
    }
    result = issue72._decomposition(methods)
    assert result["sizing_gain"]["calls"] == 1.5
    assert result["ordering_gain"]["calls"] == 1.75
    assert result["d2_increment"]["calls"] == 0.5
    assert result["matched_budget_sizing_gain"]["calls"] == 1.5
    assert result["exact_independence_budget_effect"]["calls"] == 0.5
    assert result["racer_g_budget_effect"]["calls"] == 0.5

    methods["racer_g"]["certified"] = False
    assert issue72._decomposition(methods)["sizing_gain"]["calls"] is None


def test_event_arithmetic_is_exact_and_invalid_vectors_fail() -> None:
    base = {
        "certified": True,
        "test_query_ids": ["a", "b", "c"],
        "test_calls_by_query": [1, 2, 3],
        "test_miss_by_query": [0, 1, 0],
        "test_calls": 2.0,
        "standard_all_miss_risk": 1.0 / 3.0,
        "selected_model_ids_by_query": [["m0"], ["m0", "m1"], ["m0", "m1", "m2"]],
    }
    result = issue72._compact_result(base)
    assert "selected_model_ids_by_query" not in result
    assert result["test_query_count"] == 3
    assert result["test_call_sum"] == 6
    assert result["test_miss_count"] == 1

    broken = dict(base)
    broken["test_calls"] = 9.0
    with pytest.raises(RuntimeError, match="call vector arithmetic"):
        issue72._compact_result(broken)


def test_one_redraw_keeps_splits_pool_arms_and_exact_endpoint() -> None:
    payload = issue72.run_issue72(
        _task(),
        protocol="routerbench",
        host="local",
        stamp="20260920T000000Z",
        redraw_start=0,
        count=1,
        knn_function=_fake_knn,
    )
    assert payload["target_grid"] == list(issue72.ALPHA_GRID)
    assert payload["variant_identities"] == list(issue72.VARIANT_IDS)
    assert payload["method_identities"] == list(issue72.REGISTERED_METHODS)
    assert payload["preserve_prefixes"] is True
    assert payload["chain_discount_grid"] == [0, 5, 20, 50]
    assert "ISSUE-63" in payload["scope_statement"]
    assert "not rerun" in payload["scope_statement"]
    assert payload["redraw_count"] == 1

    split_hashes = set()
    pool_ids = set()
    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        split_hashes.add(redraw["split_ids_sha256"])
        pool_ids.add(redraw["pool_id"])
        raw = redraw["variants"][0]
        sigma_zero = next(
            row for row in redraw["variants"]
            if row["variant"] == issue72.SIGMA_ZERO_VARIANT
        )
        assert raw["probability_hashes"] == sigma_zero["probability_hashes"]
        assert raw["router_metrics"] == sigma_zero["router_metrics"]
        assert raw["score_identity"] == sigma_zero["score_identity"]
        assert raw["methods"] == sigma_zero["methods"]
        for variant in redraw["variants"]:
            assert set(variant["methods"]) == set(issue72.REGISTERED_METHODS)
            assert variant["methods"][issue72.D1_METHOD]["probability_stream"] == "primary_router"
            assert variant["methods"][issue72.D2_METHOD]["probability_stream"] == "primary_router"
            assert variant["methods"]["racer_g"]["probability_stream"] == "primary_router"
            assert variant["methods"]["exact_independence"]["probability_stream"] == "primary_router"
            assert variant["methods"]["racer_g_matched_budget"]["probability_stream"] == "data_budget_matched_marginal_controls"
            assert variant["methods"]["exact_independence_matched_budget"]["probability_stream"] == "data_budget_matched_marginal_controls"
            d2 = variant["methods"][issue72.D2_METHOD]
            assert d2["endpoint_reference_method"] == issue72.D1_METHOD
            assert d2["endpoint_score_identity_verified"] is True
            assert d2["endpoint_result_identity_verified"] is True
            assert d2["chain_discount_profile"]["minimum_survivor_count"] in issue72.CHAIN_GRID
            assert variant["score_identity"]["primary_d1_score_sha256"] == variant["score_identity"]["d2_endpoint_score_sha256"]
            assert variant["decomposition"] == issue72._decomposition(variant["methods"])
            for result in variant["methods"].values():
                if result["certified"]:
                    assert result["test_call_sum"] == sum(result["test_calls_by_query"])
                    assert result["test_miss_count"] == sum(result["test_miss_by_query"])
                    assert result["standard_all_miss_risk"] == pytest.approx(
                        np.mean(result["test_miss_by_query"])
                    )
    assert len(split_hashes) == 1
    assert len(pool_ids) == 1
    assert len(payload["paired_degradation_bootstrap"]) == (
        len(issue72.ALPHA_GRID)
        * (len(issue72.VARIANT_IDS) - 1)
        * len(issue72.REGISTERED_METHODS)
    )
    assert payload["aggregates"] == issue72.recompute_aggregates(payload["cells"])
    assert payload["router_metric_aggregates"] == issue72.recompute_router_metric_aggregates(
        payload["cells"]
    )
    assert len(payload["paired_router_metric_degradation_bootstrap"]) == (
        (len(issue72.VARIANT_IDS) - 1) * 2 * 2 * 3
    )


def test_bootstrap_is_deterministic() -> None:
    values = [1.0, -0.5, 0.25, 2.0, -1.0]
    first = issue72.deterministic_paired_bootstrap(
        values, seed=123, repetitions=10_000
    )
    second = issue72.deterministic_paired_bootstrap(
        values, seed=123, repetitions=10_000
    )
    assert first == second
    assert first["bootstrap_repetitions"] == 10_000
    assert first["difference_direction"] == "variant_minus_noise_sigma_0"


def test_output_collision_is_atomic_and_refused(tmp_path: Path) -> None:
    destination = tmp_path / "issue72_local_20260920T000000Z"
    payload = {
        "host": "local",
        "stamp": "20260920T000000Z",
        "protocol": "routerbench",
        "benchmark": "mmlu",
        "redraw_start": 0,
        "redraw_end": 1,
    }
    published = issue72.publish_output(destination, payload)
    names = {path.name for path in published.iterdir()}
    assert any(name.endswith("_results.json") for name in names)
    assert any(name.endswith("_manifest.json") for name in names)
    assert any(name.endswith("_DONE") for name in names)
    before = {path.name: path.read_bytes() for path in published.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        issue72.publish_output(destination, payload)
    after = {path.name: path.read_bytes() for path in published.iterdir()}
    assert before == after
    assert not list(tmp_path.glob("*.tmp.*"))
