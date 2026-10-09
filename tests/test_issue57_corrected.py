from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.scale_selectors import ARMS, HYPERPARAMETER_GRIDS
from scripts import design_bprime_run as issue57
from scripts import merge_issue57_shards as issue57_merge


def _matrix(seed: int, rows: int = 80, models: int = 8) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return (generator.random((rows, models)) < 0.55).astype(np.int8)


def test_issue57_registry_contains_all_orders_and_random_controls() -> None:
    assert len(ARMS) == 14
    assert set(ARMS) == set(HYPERPARAMETER_GRIDS)
    assert "seeded_random" in ARMS
    assert "accuracy_weighted_random" in ARMS
    assert set(issue57.DEPENDENCE_ARMS) == set(ARMS) - {"marginal"}


def test_tuning_uses_validation_then_one_final_calibration_and_emits_curve() -> None:
    result = issue57._tune_one_arm(
        arm="double_fault",
        design=_matrix(1),
        val=_matrix(2),
        cal=_matrix(3),
        test=_matrix(4),
        names=[f"model-{index}" for index in range(8)],
        max_depth=8,
        seed=17,
        alpha=0.20,
    )
    assert result["final_calibration_count"] == 1
    assert len(result["trajectory"]) == len(HYPERPARAMETER_GRIDS["double_fault"])
    assert len(result["chosen_order_indices"]) == 8
    assert len(result["fixed_depth_test_curve"]) == 8
    assert [row["depth"] for row in result["fixed_depth_test_curve"]] == list(range(1, 9))
    risks = [row["standard_all_miss_risk"] for row in result["fixed_depth_test_curve"]]
    assert np.all(np.diff(risks) <= 1e-15)


def test_arm_presence_fails_loudly_on_missing_registered_arm() -> None:
    curve = [{"depth": 1, "standard_all_miss_risk": 0.1}]
    tuned = {arm: {"fixed_depth_test_curve": curve} for arm in issue57.DEPENDENCE_ARMS}
    seed_results = [{
        "seed": 42,
        "per_alpha": {
            "0.05": {
                "marginal": {"fixed_depth_test_curve": curve},
                "tuned": tuned,
            }
        },
    }]
    issue57._assert_arm_presence(seed_results, [0.05])
    tuned.pop("seeded_random")
    with pytest.raises(RuntimeError, match="missing=.*seeded_random"):
        issue57._assert_arm_presence(seed_results, [0.05])


def _fake_payload(seed: int) -> dict:
    curve = [{"depth": 1, "standard_all_miss_risk": 0.1}]
    marginal = {
        "test_calls": 1,
        "test_risk": 0.1,
        "fixed_depth_test_curve": curve,
    }
    tuned = {
        arm: {
            "hyper_name": "none",
            "grid_size": 1,
            "effective_k": 1,
            "chosen_hyper": None,
            "test_calls": 1,
            "test_risk": 0.1,
            "fixed_depth_test_curve": curve,
        }
        for arm in issue57.DEPENDENCE_ARMS
    }
    return {
        "schema_version": "issue57_corrected_e1_v1",
        "design": "B_prime_per_method_tuning",
        "issue": "ISSUE-57",
        "registered_arms": list(ARMS),
        "protocol": "routerbench",
        "benchmark": "mmlu",
        "pool_size": 4,
        "design_budget": 10,
        "alphas": [0.05],
        "split_ratios": [0.4, 0.3, 0.15, 0.15],
        "max_depth": 4,
        "null_shuffle": False,
        "hyperparameter_grids": {key: list(value) for key, value in HYPERPARAMETER_GRIDS.items()},
        "feature_audit": {},
        "source_sha256": "a" * 64,
        "machine": "test",
        "run_started_utc": f"2026-09-20T00:00:0{seed}Z",
        "run_finished_utc": f"2026-09-20T00:00:1{seed}Z",
        "seeds": [seed],
        "seed_results": [{
            "seed": seed,
            "per_alpha": {"0.05": {"marginal": marginal, "tuned": tuned}},
        }],
        "aggregate": {},
    }


def test_issue57_merger_combines_disjoint_seeds_and_refuses_duplicates(tmp_path: Path) -> None:
    paths = []
    for seed in (42, 43):
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(_fake_payload(seed)), encoding="utf-8")
        paths.append(path)
    merged = issue57_merge.merge(paths)
    assert merged["seeds"] == [42, 43]
    assert merged["aggregate"]["0.05"]["methods"]["seeded_random"]["certification_count"] == 2
    with pytest.raises(ValueError, match="duplicate seeds"):
        issue57_merge.merge([paths[0], paths[0]])
