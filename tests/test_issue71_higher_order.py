from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.higher_order_selectors import (
    discrete_vine_tree_order,
    tetrachoric_diagnostics,
    tetrachoric_normal_score_order,
    vine_diagnostics,
)
from scripts import run_issue71_higher_order as issue71


def _labels(rows: int = 90, models: int = 8) -> np.ndarray:
    generator = np.random.default_rng(71)
    latent = generator.normal(size=(rows, 3))
    loadings = generator.normal(size=(3, models))
    probability = 1.0 / (1.0 + np.exp(-(latent @ loadings)))
    labels = (generator.random((rows, models)) < probability).astype(np.int8)
    for row in range(rows):
        labels[row, row % models] = 1
    return labels


def _task(rows: int = 240, models: int = 6) -> SimpleNamespace:
    labels = _labels(rows, models)
    generator = np.random.default_rng(7100)
    embeddings = generator.normal(size=(rows, 9))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=labels,
        model_ids=tuple(f"model-{column}" for column in range(models)),
        prompts=tuple(f"prompt-{row}" for row in range(rows)),
        source_hashes={"leaderboard": "a" * 64, "router_dataset": "b" * 64},
        feature_audit={"degenerate": False},
    )


def test_higher_order_orders_are_deterministic_full_permutations() -> None:
    labels = _labels()
    tetra_first = tetrachoric_normal_score_order(labels, 8, ridge=0.1, seed=7)
    tetra_second = tetrachoric_normal_score_order(labels, 8, ridge=0.1, seed=7)
    vine_first = discrete_vine_tree_order(labels, 8, laplace=1.0, seed=7)
    vine_second = discrete_vine_tree_order(labels, 8, laplace=1.0, seed=7)
    assert tetra_first == tetra_second
    assert vine_first == vine_second
    assert set(tetra_first) == set(range(8))
    assert set(vine_first) == set(range(8))


def test_tetrachoric_diagnostics_record_psd_and_sample_size() -> None:
    diagnostics = tetrachoric_diagnostics(_labels(), ridge=0.1)
    assert diagnostics["estimator"] == "probit_conditional_normal_score_tetrachoric"
    assert diagnostics["psd_strategy"] == "low_rank_svd_plus_positive_diagonal"
    assert diagnostics["nearest_psd_projection_used"] is False
    assert diagnostics["n_design_rows"] == 90
    assert diagnostics["n_models"] == 8
    assert diagnostics["minimum_idiosyncratic_variance"] > 0.0


def test_vine_diagnostics_refuse_unidentified_higher_levels() -> None:
    diagnostics = vine_diagnostics(_labels(), laplace=1.0)
    assert diagnostics["estimator"] == "discrete_bernoulli_chow_liu_first_tree_vine"
    assert diagnostics["tree_edges"] == 7
    assert diagnostics["higher_vine_levels_fit"] == 0
    assert "not uniquely identifiable" in diagnostics["identifiability_note"]


def test_issue71_registry_and_one_redraw_endpoints() -> None:
    assert len(issue71.D1_METHODS) == 2
    assert len(issue71.D2_METHODS) == 2
    assert len(issue71.REGISTERED_METHODS) == 4
    payload = issue71.run_issue71(
        _task(),
        protocol="routereval",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        host_id="test_host",
        run_stamp="20260920T000000Z",
        worker_count=2,
    )
    assert payload["method_identities"] == list(issue71.REGISTERED_METHODS)
    assert payload["preserve_prefixes"] is True
    assert len(payload["cells"]) == 7
    for cell in payload["cells"]:
        methods = cell["redraws"][0]["methods"]
        assert set(methods) == set(issue71.REGISTERED_METHODS)
        for d1, d2 in zip(issue71.D1_METHODS, issue71.D2_METHODS):
            assert methods[d1]["final_calibration_count"] == 1
            assert methods[d2]["final_calibration_count"] == 1
            assert methods[d2]["endpoint_reference_method"] == d1
            assert methods[d2]["endpoint_score_identity_verified"] is True
            assert methods[d2]["endpoint_result_identity_verified"] is True
            assert methods[d2]["chosen_hyper"]["minimum_survivor_count"] in issue71.CHAIN_GRID


def test_issue71_rejects_empty_or_reversed_redraw_range() -> None:
    try:
        issue71.run_issue71(
            _task(),
            protocol="routereval",
            pool_size=6,
            pool_seed=7001,
            redraw_start=2,
            redraw_end=2,
            knn_k=8,
            host_id="test_host",
            run_stamp="20260920T000000Z",
            worker_count=1,
        )
    except ValueError as error:
        assert "redraw range" in str(error)
    else:
        raise AssertionError("empty redraw range was accepted")
