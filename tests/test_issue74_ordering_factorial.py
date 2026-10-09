from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.racer_protocol import smooth_nonconformity_scores
from components.racer_wrapper import (
    fixed_design_marginal_factorized_scores,
    fixed_order_factorized_scores,
)
from components.scale_selectors import build_order
from scripts import run_issue74_ordering_factorial as issue74


def _data(rows: int = 180, models: int = 6):
    generator = np.random.default_rng(20260920)
    labels = (generator.random((rows, models)) < 0.45).astype(np.int8)
    for row in range(rows):
        labels[row, row % models] = 1
    probabilities = np.clip(
        labels * 0.55 + generator.uniform(0.05, 0.45, size=labels.shape),
        0.01,
        0.99,
    )
    return labels, probabilities


def test_registry_has_every_fixed_order_and_router_control() -> None:
    assert len(issue74.FIXED_METHODS) == 14
    assert len(issue74.REGISTERED_METHODS) == 15
    assert issue74.FIXED_METHODS["fixed_tally_factorized"] == "empirical_tally"
    assert issue74.FIXED_METHODS["fixed_design_marginal_factorized"] == "marginal"
    assert "fixed_seeded_random_factorized" in issue74.FIXED_METHODS
    assert "fixed_accuracy_weighted_random_factorized" in issue74.FIXED_METHODS
    assert issue74.REGISTERED_METHODS[-1] == "exact_independence"


def test_product_candidates_are_complete_nested_prefixes() -> None:
    labels, probabilities = _data()
    fitted = issue74.build_score_candidates(
        probabilities,
        labels[:45],
        [f"model-{index}" for index in range(labels.shape[1])],
        protocol="routereval",
        base_seed=7001,
        redraw=2,
        preserve_score_prefixes=True,
    )
    assert set(fitted) == set(issue74.REGISTERED_METHODS)
    for method, candidates in fitted.items():
        assert candidates, method
        for candidate in candidates:
            if candidate["order"] is None:
                continue
            order = np.asarray(candidate["order"], dtype=int)
            assert sorted(order.tolist()) == list(range(labels.shape[1]))
            ordered = np.take_along_axis(candidate["scores"], np.broadcast_to(order, probabilities.shape), axis=1)
            assert np.all(np.diff(ordered, axis=1) >= -1e-15), method


def test_legacy_tally_and_design_marginal_scores_reproduce_issue50_endpoints() -> None:
    labels, probabilities = _data()
    design = labels[:45]
    model_ids = [f"model-{index}" for index in range(labels.shape[1])]
    fitted = issue74.build_score_candidates(
        probabilities,
        design,
        model_ids,
        protocol="routereval",
        base_seed=7001,
        redraw=0,
        preserve_score_prefixes=False,
    )
    split_seed = 7001
    tally_order = build_order(
        "empirical_tally", design, model_names=model_ids,
        max_depth=labels.shape[1], seed=7001, hyper=0.0,
    )
    tally_expected = smooth_nonconformity_scores(
        fixed_order_factorized_scores(probabilities, tally_order),
        seed=7001, split_seed=split_seed, preserve_prefixes=False,
    )
    marginal_expected = smooth_nonconformity_scores(
        fixed_design_marginal_factorized_scores(probabilities, design),
        seed=7001, split_seed=split_seed, preserve_prefixes=False,
    )
    np.testing.assert_array_equal(
        fitted["fixed_tally_factorized"][0]["scores"], tally_expected
    )
    np.testing.assert_array_equal(
        fitted["fixed_design_marginal_factorized"][0]["scores"], marginal_expected
    )


def test_one_redraw_writes_every_registered_method_for_every_target() -> None:
    labels, _ = _data(rows=240, models=6)
    generator = np.random.default_rng(91)
    embeddings = generator.normal(size=(240, 10))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    task = SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"q-{index}" for index in range(240)),
        embeddings=embeddings,
        correctness=labels,
        model_ids=tuple(f"model-{index}" for index in range(6)),
        source_hashes={"labels": "a" * 64},
        feature_audit={"features_present": True},
    )
    payload = issue74.run_factorial(
        task,
        protocol="routereval",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        preserve_score_prefixes=True,
    )
    assert payload["preserve_score_prefixes"] is True
    assert payload["registered_methods"] == list(issue74.REGISTERED_METHODS)
    assert len(payload["cells"]) == 7
    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        assert set(redraw["methods"]) == set(issue74.REGISTERED_METHODS)
        for record in redraw["methods"].values():
            assert record["selection_source"] == "validation_only"
            assert record["final_calibration_count"] == 1


def test_corrected_smoothing_preserves_fixed_order_through_tied_plateau() -> None:
    probabilities = np.zeros((5, 6), dtype=float)
    labels = np.asarray([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ], dtype=np.int8)
    fitted = issue74.build_score_candidates(
        probabilities,
        labels,
        [f"model-{index}" for index in range(6)],
        protocol="routereval",
        base_seed=7001,
        redraw=3,
        preserve_score_prefixes=True,
    )
    for method, candidates in fitted.items():
        for candidate in candidates:
            if candidate["order"] is None:
                continue
            order = np.asarray(candidate["order"], dtype=int)
            ordered = candidate["scores"][:, order]
            assert np.all(np.diff(ordered, axis=1) >= 0.0), method
            assert np.all(np.diff(ordered, axis=1) > 0.0), method


def test_sorted_fallback_scan_matches_brute_force() -> None:
    labels, probabilities = _data(rows=35, models=6)
    scores = probabilities.copy()
    fast = issue74._fallback_standard(scores, labels)
    observed = np.unique(scores)
    brute = min(
        (
            float(np.mean(~np.any((scores <= threshold) & labels.astype(bool), axis=1))),
            float(np.mean(np.sum(scores <= threshold, axis=1))),
            float(threshold),
        )
        for threshold in [float(np.nextafter(observed[0], -np.inf)), *map(float, observed)]
    )
    assert fast == brute
