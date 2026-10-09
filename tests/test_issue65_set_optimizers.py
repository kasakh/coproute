from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.set_optimizers import (
    beam_min_sum_order,
    exact_min_sum_order,
    local_swap_min_sum_order,
    min_sum_objective,
    prefix_miss_counts,
)


def _labels(rows: int = 40, models: int = 6) -> np.ndarray:
    generator = np.random.default_rng(65)
    labels = (generator.random((rows, models)) < 0.42).astype(np.int8)
    labels[np.arange(rows), np.arange(rows) % models] = 1
    return labels


def test_exact_dynamic_program_matches_brute_force_permutations() -> None:
    labels = _labels(models=6)
    result = exact_min_sum_order(labels)
    brute_order = min(
        itertools.permutations(range(6)),
        key=lambda order: (min_sum_objective(labels, order), order),
    )
    assert result.order == brute_order
    assert result.objective == min_sum_objective(labels, brute_order)
    assert result.objective <= result.greedy_objective


def test_exact_depth_subsets_match_brute_force() -> None:
    labels = _labels(models=6)
    result = exact_min_sum_order(labels)
    failures = 1 - labels
    for depth in range(1, 7):
        best = min(
            int(np.sum(np.all(failures[:, subset] == 1, axis=1)))
            for subset in itertools.combinations(range(6), depth)
        )
        assert result.exact_depth_misses[depth - 1] == best
        chosen = result.exact_depth_subsets[depth - 1]
        observed = int(np.sum(np.all(failures[:, chosen] == 1, axis=1)))
        assert observed == best


def test_full_beam_recovers_exact_order_on_small_pool() -> None:
    labels = _labels(models=6)
    exact = exact_min_sum_order(labels)
    beam, diagnostics = beam_min_sum_order(labels, beam_width=1 << 6)
    assert min_sum_objective(labels, beam) == exact.objective
    assert diagnostics["objective"] == exact.objective


def test_local_swap_never_worsens_greedy_objective() -> None:
    labels = _labels(rows=80, models=10)
    result, diagnostics = local_swap_min_sum_order(
        labels, maximum_positions=10, passes=3
    )
    assert set(result) == set(range(10))
    assert diagnostics["objective"] <= diagnostics["initial_objective"]
    assert diagnostics["objective_improvement"] >= 0
    assert sum(prefix_miss_counts(labels, result)) == diagnostics["objective"]


def test_exact_refuses_pool_above_cap() -> None:
    with pytest.raises(ValueError, match="capped"):
        exact_min_sum_order(_labels(rows=30, models=6), maximum_models=5)


def test_optimizers_are_deterministic() -> None:
    labels = _labels(rows=70, models=9)
    assert beam_min_sum_order(labels, beam_width=8)[0] == beam_min_sum_order(labels, beam_width=8)[0]
    assert local_swap_min_sum_order(labels, maximum_positions=9)[0] == local_swap_min_sum_order(labels, maximum_positions=9)[0]


def _task(rows: int = 240, models: int = 6):
    from types import SimpleNamespace
    generator = np.random.default_rng(6500)
    labels = _labels(rows, models)
    embeddings = generator.normal(size=(rows, 8))
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


def test_issue65_runner_emits_exact_and_greedy_reachable_approximations() -> None:
    from scripts import run_issue65_set_optimizers as issue65

    payload = issue65.run_issue65(
        _task(),
        protocol="routerbench",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        host_id="test_host",
        run_stamp="20260920T000000Z",
        worker_count=2,
    )
    assert payload["include_exact"] is True
    assert set(payload["method_identities"]) == {issue65.EXACT, issue65.BEAM, issue65.SWAP}
    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        assert redraw["optimization_audit"]["beam_greedy_endpoint_verified"] is True
        assert redraw["optimization_audit"]["swap_greedy_endpoint_verified"] is True
        assert redraw["optimization_audit"]["exact"]["nested_objective_gap"] >= 0
        for method, record in redraw["methods"].items():
            assert record["final_calibration_count"] == 1
            if method in {issue65.BEAM, issue65.SWAP}:
                assert record["greedy_endpoint_score_identity_verified"] is True


def test_popcount_does_not_require_new_python_int_method() -> None:
    import inspect
    from components import set_optimizers

    assert set_optimizers._popcount((1 << 130) | (1 << 7) | 1) == 3
    assert ".bit_count(" not in inspect.getsource(set_optimizers)
