"""Focused synthetic tests for the ISSUE-50 canonical paired runner."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.paired_orders import per_query_orders_factorized_scores  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402


def _task(rows: int = 100, models: int = 6) -> SimpleNamespace:
    rng = np.random.default_rng(20260919)
    embeddings = rng.normal(size=(rows, 5))
    correctness = np.zeros((rows, models), dtype=np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
        correctness[row, (row + 2) % models] = 1
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"q-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{index}" for index in range(models)),
        source_hashes={"leaderboard": "a" * 64, "router_dataset": "b" * 64},
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
    )


def _fake_knn(train_x, train_y, query_x, k):
    del train_x, k
    n_queries = len(query_x)
    n_models = train_y.shape[1]
    base = np.linspace(0.15, 0.85, n_models)
    variation = (np.arange(n_queries)[:, None] % 5 - 2) * 0.01
    probabilities = np.clip(base[None, :] + variation, 1e-4, 1.0 - 1e-4)
    support = np.broadcast_to(train_y[:1], (n_queries, 1, n_models)).copy()
    return probabilities, support


def test_per_query_orders_factorized_scores_are_exact_nested_prefixes() -> None:
    probabilities = np.asarray([[0.8, 0.5, 0.2], [0.25, 0.75, 0.5]])
    orders = np.asarray([[0, 2, 1], [1, 0, 2]])
    scores = per_query_orders_factorized_scores(probabilities, orders)
    np.testing.assert_allclose(scores[0], [0.0, 0.84, 0.8])
    np.testing.assert_allclose(scores[1], [0.75, 0.0, 0.8125])
    ordered = np.take_along_axis(scores, orders, axis=1)
    assert np.all(np.diff(ordered, axis=1) >= 0.0)
    with pytest.raises(ValueError, match="full permutation"):
        per_query_orders_factorized_scores(probabilities, [[0, 0, 1], [0, 1, 2]])


def test_nonlearned_runner_uses_registered_splits_and_emits_raw_vectors() -> None:
    payload = paired.run_paired_suite(
        _task(),
        pool_size=6,
        pool_seed=7001,
        redraw_start=2,
        redraw_end=4,
        knn_k=8,
        knn_function=_fake_knn,
    )
    assert payload["schema_version"] == "paired_main_suite_v2"
    assert payload["redraw_range"] == {
        "start_inclusive": 2,
        "end_exclusive": 4,
        "count": 2,
    }
    assert payload["target_grid"] == list(paired.ALPHA_GRID)
    assert payload["method_identities"] == sorted(paired.NONLEARNED_METHODS)
    assert payload["checkpoint_identities"] == {}

    for cell in payload["cells"]:
        assert [record["redraw_id"] for record in cell["redraws"]] == [2, 3]
        for redraw in cell["redraws"]:
            assert redraw["split_seed"] == 7001 + 13 * redraw["redraw_id"]
            split_ids = redraw["split_ids"]
            assert {name: len(ids) for name, ids in split_ids.items()} == {
                "design": 30,
                "router": 20,
                "final_calibration": 20,
                "validation": 15,
                "test": 15,
            }
            assert len({query for ids in split_ids.values() for query in ids}) == 100
            for method in redraw["methods"].values():
                assert method["endpoint"]["reaches_full_pool"] is True
                assert method["endpoint"]["model_ids"] == payload["model_ids"]
                assert method["test_query_ids"] == split_ids["test"]
                if method["certified"]:
                    assert len(method["test_calls_by_query"]) == 15
                    assert len(method["test_miss_by_query"]) == 15
                    assert method["test_calls"] == pytest.approx(
                        np.mean(method["test_calls_by_query"])
                    )
                    assert method["standard_all_miss_risk"] == pytest.approx(
                        np.mean(method["test_miss_by_query"])
                    )
                else:
                    assert method["test_calls_by_query"] is None
                    assert method["test_miss_by_query"] is None
    assert payload["aggregates"] == paired.recompute_aggregates(payload["cells"])


def test_optional_e3_selects_only_from_nonlearned_validation_menu() -> None:
    payload = paired.run_paired_suite(
        _task(),
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        include_e3=True,
        knn_function=_fake_knn,
    )
    assert "e3_selected_nonlearned" in payload["method_identities"]
    for cell in payload["cells"]:
        record = cell["redraws"][0]["methods"]["e3_selected_nonlearned"]
        assert record["selection_source"] == "validation_only"
        assert record["selected_method"] in paired.E3_MENU
        assert {row["method"] for row in record["validation_trajectory"]} == set(
            paired.E3_MENU
        )


def test_atomic_json_refuses_overwrite_and_is_canonical(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    value = {"z": [2, 1], "a": {"b": True}}
    paired.atomic_write_json(destination, value)
    assert destination.read_bytes() == paired.canonical_json_bytes(value)
    assert json.loads(destination.read_text(encoding="utf-8")) == value
    assert not list(tmp_path.glob("*.tmp.*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired.atomic_write_json(destination, value)


def test_full_learned_order_helpers_use_complete_pool_and_registered_seed_order() -> None:
    torch = pytest.importorskip("torch")

    class FakePath:
        order = (2, 0)

    class FakeFlex:
        n_models = 4

        def predict_quality(self, embeddings):
            del embeddings
            return torch.tensor([[0.8, 0.7, 0.9, 0.7]], dtype=torch.float64)

        def select_from_qualities(self, qualities, **kwargs):
            del qualities
            assert kwargs == {"max_size": 4, "relative_threshold": 0.0, "top_k0": None}
            return FakePath()

    orders, lengths = paired.full_flexrouter_orders(FakeFlex(), np.ones((1, 3)))
    assert orders.tolist() == [[2, 0, 1, 3]]
    assert lengths == [2]

    class FakeBaRP:
        n_models = 4

        def rank_models(self, embeddings, preference, *, top_k):
            assert len(embeddings) == 2
            assert preference == [1.0, 0.0]
            assert top_k == 4
            return torch.tensor([[1, 0, 3, 2], [2, 3, 0, 1]])

    assert paired.full_barp_orders(FakeBaRP(), np.ones((2, 3))).tolist() == [
        [1, 0, 3, 2],
        [2, 3, 0, 1],
    ]
