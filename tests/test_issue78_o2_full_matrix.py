from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.chain_discount import chain_discounted_scores, fit_chain_discount
from components.racer_wrapper import fixed_order_factorized_scores
from scripts import run_issue78_o2_full_matrix as issue78


def _task(rows: int = 240, models: int = 6) -> SimpleNamespace:
    generator = np.random.default_rng(20260920)
    correctness = (generator.random((rows, models)) < 0.42).astype(np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
    embeddings = generator.normal(size=(rows, 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{column}" for column in range(models)),
        prompts=tuple(f"prompt-{row}" for row in range(rows)),
        source_hashes={"leaderboard": "a" * 64, "router_dataset": "b" * 64},
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
    )


def test_issue78_registry_has_fourteen_paired_d1_d2_families() -> None:
    assert len(issue78.FIXED_METHODS) == 14
    assert len(issue78.D1_METHODS) == 14
    assert len(issue78.D2_METHODS) == 14
    assert len(issue78.REGISTERED_METHODS) == 28
    assert len(set(issue78.REGISTERED_METHODS)) == 28
    for d1, d2 in zip(issue78.D1_METHODS, issue78.D2_METHODS):
        assert d2 == d1.replace("_factorized", "_chain_discounted")


def test_every_family_discount_off_is_byte_exact_after_shared_smoothing() -> None:
    task = _task(rows=90)
    labels = task.correctness
    probabilities = np.clip(labels * 0.55 + 0.20, 0.01, 0.99)
    fitted = issue78._fit_d1_candidates(
        probabilities,
        labels[:30],
        task.model_ids,
        base_seed=7001,
        split_seed=7014,
    )
    for method, candidates in fitted.items():
        for candidate in candidates:
            profile = fit_chain_discount(labels[:30], candidate["order"])
            raw = chain_discounted_scores(
                probabilities,
                profile,
                minimum_survivor_count=0,
            )
            endpoint = issue78._smooth_fixed(
                raw,
                candidate["order"],
                base_seed=7001,
                split_seed=7014,
            )
            assert np.array_equal(endpoint, candidate["scores"]), method


def test_issue78_one_redraw_emits_all_targets_methods_and_endpoints() -> None:
    payload = issue78.run_issue78(
        _task(),
        protocol="routereval",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        host_id="test_host",
        run_stamp="20260920T000000Z",
        worker_count=3,
    )
    assert payload["schema_version"] == issue78.SCHEMA_VERSION
    assert payload["design"] == "ISSUE_78_full_matrix_O2_D1_D2"
    assert payload["preserve_prefixes"] is True
    assert payload["worker_count"] == 3
    assert payload["method_identities"] == list(issue78.REGISTERED_METHODS)
    assert len(payload["cells"]) == 7
    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        assert set(redraw["methods"]) == set(issue78.REGISTERED_METHODS)
        for d1, d2 in zip(issue78.D1_METHODS, issue78.D2_METHODS):
            product = redraw["methods"][d1]
            chain = redraw["methods"][d2]
            assert product["selection_source"] == "validation_only"
            assert product["final_calibration_count"] == 1
            assert chain["selection_source"] == "validation_only"
            assert chain["final_calibration_count"] == 1
            assert chain["endpoint_reference_method"] == d1
            assert chain["endpoint_score_identity_verified"] is True
            assert chain["endpoint_result_identity_verified"] is True
            assert chain["chosen_hyper"]["minimum_survivor_count"] in issue78.CHAIN_GRID


def test_issue78_rejects_empty_range_before_execution() -> None:
    with pytest.raises(ValueError, match="redraw range"):
        issue78.run_issue78(
            _task(),
            protocol="routereval",
            pool_size=6,
            pool_seed=7001,
            redraw_start=4,
            redraw_end=1,
            knn_k=8,
            host_id="test_host",
            run_stamp="20260920T000000Z",
            worker_count=1,
        )


def test_fallback_scan_matches_brute_force() -> None:
    task = _task(rows=40, models=5)
    generator = np.random.default_rng(7)
    scores = generator.uniform(size=task.correctness.shape)
    rows = np.arange(len(task.query_ids))
    fast = issue78._fallback_standard(scores, task.correctness, rows)
    observed = np.unique(scores)
    brute = min(
        (
            float(
                np.mean(
                    ~np.any(
                        (scores <= threshold) & task.correctness.astype(bool), axis=1
                    )
                )
            ),
            float(np.mean(np.sum(scores <= threshold, axis=1))),
            float(threshold),
        )
        for threshold in [float(np.nextafter(observed[0], -np.inf)), *map(float, observed)]
    )
    assert fast == brute


def test_issue78_stage1_analysis_pair_registry_has_only_new_rows() -> None:
    from scripts.postprocess_issue78_o2_full_matrix import comparison_pairs

    pairs = comparison_pairs()
    assert len(pairs) == 25
    assert len(set(pairs)) == 25
    assert ("fixed_design_marginal_factorized", "fixed_tally_factorized") not in pairs
    assert ("fixed_tally_chain_discounted", "fixed_tally_factorized") not in pairs
    assert (
        "fixed_error_correlation_factorized",
        "fixed_tally_factorized",
    ) in pairs
    assert (
        "fixed_error_correlation_chain_discounted",
        "fixed_error_correlation_factorized",
    ) in pairs
