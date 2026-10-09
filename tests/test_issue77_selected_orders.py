from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_issue74_ordering_factorial as issue74
from scripts import run_issue77_selected_orders as issue77


def _task() -> SimpleNamespace:
    generator = np.random.default_rng(20260920)
    rows, models = 240, 6
    correctness = (generator.random((rows, models)) < 0.4).astype(np.int8)
    for row in range(rows):
        correctness[row, row % models] = 1
    embeddings = generator.normal(size=(rows, 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"q-{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{index}" for index in range(models)),
        source_hashes={"labels": "a" * 64},
        feature_audit={"features_present": True},
    )


def _reference(task: SimpleNamespace, payload: dict) -> dict:
    methods = issue77.DEFAULT_METHODS
    redraws = {"0": {}}
    for cell in payload["cells"]:
        alpha = str(cell["alpha"])
        redraws["0"][alpha] = {}
        records = cell["redraws"][0]["methods"]
        for method in methods:
            record = records[method]
            redraws["0"][alpha][method] = {
                "chosen_hyper": record["chosen_hyper"],
                "chosen_order": record["chosen_order"],
                "result_signature": issue77.result_signature(record),
            }
    return {
        "manifest_sha256": "b" * 64,
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_size": 6,
        "model_ids_sha256": payload["model_ids_sha256"],
        "query_ids_sha256": payload["query_ids_sha256"],
        "selected_methods": list(methods),
        "redraws": redraws,
    }


def test_selected_order_cross_reproduces_product_and_emits_discounted_pairs() -> None:
    task = _task()
    factorial = issue74.run_factorial(
        task,
        protocol="routereval",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        preserve_score_prefixes=True,
    )
    crossed = issue77.run_cross(
        task,
        _reference(task, factorial),
        methods=issue77.DEFAULT_METHODS,
        protocol="routereval",
        pool_size=6,
        pool_seed=7001,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
    )
    assert crossed["chain_discount_grid"] == [0, 5, 20, 50]
    expected = {
        name
        for method in issue77.DEFAULT_METHODS
        for name in (method, method.replace("_factorized", "_chain_discounted"))
    }
    assert set(crossed["registered_methods"]) == expected
    for cell in crossed["cells"]:
        methods = cell["redraws"][0]["methods"]
        assert set(methods) == expected
        for base in issue77.DEFAULT_METHODS:
            assert methods[base]["endpoint_reference_manifest_verified"] is True
            chain = methods[base.replace("_factorized", "_chain_discounted")]
            assert chain["endpoint_score_identity_verified"] is True
            assert chain["endpoint_result_identity_verified"] is True
            assert chain["chosen_hyper"]["minimum_survivor_count"] in issue77.CHAIN_GRID
            assert chain["final_calibration_count"] == 1


def test_selected_order_cross_rejects_empty_or_reversed_redraw_range() -> None:
    task = _task()
    reference = {
        "protocol": "routereval",
        "benchmark": "ifeval",
        "pool_size": 6,
        "selected_methods": list(issue77.DEFAULT_METHODS),
    }
    import pytest
    with pytest.raises(ValueError, match="redraw range"):
        issue77.run_cross(
            task,
            reference,
            methods=issue77.DEFAULT_METHODS,
            protocol="routereval",
            pool_size=6,
            pool_seed=7001,
            redraw_start=4,
            redraw_end=1,
            knn_k=8,
        )
