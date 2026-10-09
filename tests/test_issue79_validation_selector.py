from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_issue79_validation_selector as issue79


def _task(rows: int = 240, models: int = 6) -> SimpleNamespace:
    generator = np.random.default_rng(79)
    correctness = (generator.random((rows, models)) < 0.38).astype(np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
    embeddings = generator.normal(size=(rows, 9))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{column}" for column in range(models)),
        prompts=tuple(f"prompt-{row}" for row in range(rows)),
        source_hashes={"leaderboard": "a" * 64, "router_dataset": "b" * 64},
        feature_audit={"degenerate": False},
    )


def test_validation_partition_is_complete_disjoint_and_deterministic() -> None:
    rows = np.array([9, 2, 8, 1, 7, 0, 6])
    first = issue79.split_validation_rows(rows)
    second = issue79.split_validation_rows(rows)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(first[0]) | set(first[1]) == set(rows)
    assert not (set(first[0]) & set(first[1]))


def test_discovery_loss_uses_first_correct_rank_and_pool_plus_one() -> None:
    labels = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [1, 1, 0],
    ])
    observed = issue79.discovery_losses(labels, np.arange(3), (2, 1, 0))
    assert observed.tolist() == [2.0, 4.0, 2.0]


def test_family_tuning_collapses_duplicate_orders_before_bias_penalty() -> None:
    labels = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])
    scores = np.zeros_like(labels, dtype=float)
    candidates = [
        {"grid_index": 0, "hyper": 0.0, "order": (0, 1, 2), "scores": scores},
        {"grid_index": 1, "hyper": 0.5, "order": (0, 1, 2), "scores": scores},
        {"grid_index": 2, "hyper": 1.0, "order": (2, 1, 0), "scores": scores},
    ]
    _, audit = issue79.tune_family(candidates, labels, np.arange(4))
    assert audit["nominal_candidate_count"] == 3
    assert audit["effective_candidate_count"] == 2
    assert len(audit["trajectory"]) == 2
    assert np.isclose(audit["bias_multiplier"], np.sqrt(2.0 * np.log(2.0)))


def test_issue79_one_redraw_is_target_free_and_final_calibrates_once() -> None:
    payload = issue79.run_issue79(
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
    assert payload["family_menu_size"] == 12
    assert np.isclose(payload["family_bias_multiplier"], np.sqrt(2.0 * np.log(12.0)))
    assert payload["random_control_methods"]
    redraw = payload["redraws"][0]
    assert redraw["target_free_selection_verified"] is True
    assert redraw["validation_partition"]["complete_and_disjoint_verified"] is True
    selected = redraw["selected_method"]
    observed_sources = set()
    assert len(redraw["alpha_results"]) == len(issue79.ALPHA_GRID)
    for alpha_result in redraw["alpha_results"]:
        methods = alpha_result["methods"]
        assert set(methods) == set(issue79.OUTPUT_METHODS)
        chosen = methods["validation_selected_factorized"]
        observed_sources.add(chosen["source_order_method"])
        assert chosen["final_calibration_count"] == 1
        assert methods[issue79.BASELINE_METHOD]["final_calibration_count"] == 1
        assert methods[issue79.ERROR_CORRELATION_METHOD]["final_calibration_count"] == 1
    assert observed_sources == {selected}


def test_selection_helpers_never_read_final_calibration_or_test_rows() -> None:
    task = _task(rows=120, models=5)
    design = task.correctness[:35]
    probabilities = np.clip(task.correctness * 0.6 + 0.15, 0.01, 0.99)
    fitted = issue79._fit_d1_candidates(
        probabilities,
        design,
        task.model_ids,
        base_seed=7001,
        split_seed=7014,
    )
    tune_rows = np.arange(35, 50)
    select_rows = np.arange(50, 65)
    def choose(labels: np.ndarray) -> str:
        tuned = {}
        for method, candidates in fitted.items():
            candidate, audit = issue79.tune_family(candidates, labels, tune_rows)
            tuned[method] = {"candidate": candidate, "tuning": audit}
        selected, _ = issue79.select_family(tuned, labels, select_rows)
        return selected
    original = choose(task.correctness)
    changed = task.correctness.copy()
    changed[65:] = 1 - changed[65:]
    assert choose(changed) == original
