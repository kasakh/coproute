"""Focused tests for the synthetic conditional-independence null."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.synthetic_conditional_independence import (  # noqa: E402
    FROZEN_CONFIG,
    METHODS,
    SCHEMA_VERSION,
    correlation_matrix,
    off_diagonal_values,
    run_experiment,
    run_seed,
    simulate_split,
    true_marginals,
    write_outputs,
)


def small_config(seed_count: int = 3):
    return replace(
        FROZEN_CONFIG,
        seed_count=seed_count,
        design_size=240,
        validation_size=180,
        final_calibration_size=240,
        test_size=500,
    )


def test_frozen_configuration_has_at_least_one_hundred_seeds():
    FROZEN_CONFIG.validate()
    assert FROZEN_CONFIG.seed_count >= 100
    assert set(FROZEN_CONFIG.split_sizes()) == {
        "design",
        "validation",
        "final_calibration",
        "test",
    }


def test_true_marginals_are_bounded_heterogeneous_and_difficulty_dependent():
    difficulty = np.array([-2.0, 0.0, 2.0])
    probabilities = true_marginals(difficulty, FROZEN_CONFIG)
    assert probabilities.shape == (3, FROZEN_CONFIG.n_models)
    assert np.all(probabilities >= FROZEN_CONFIG.probability_floor)
    assert np.all(probabilities <= 1.0 - FROZEN_CONFIG.probability_floor)
    assert len(np.unique(np.round(probabilities[1], 8))) == FROZEN_CONFIG.n_models
    assert np.all(probabilities[0] > probabilities[2])


def test_residual_correlation_centers_near_zero_under_the_null():
    split = simulate_split(FROZEN_CONFIG, 9_001, "design", size=50_000)
    raw = off_diagonal_values(correlation_matrix(split.correctness))
    residual = off_diagonal_values(
        correlation_matrix(split.correctness.astype(float) - split.probabilities)
    )
    assert float(raw.mean()) > 0.05
    assert abs(float(residual.mean())) < 0.01
    assert float(np.abs(residual).mean()) < 0.015


def test_exact_factorization_has_no_worse_population_risk_at_matched_depths():
    result = run_seed(small_config(seed_count=1), seed_index=17)
    fixed = result["fixed_depth"]
    exact = {
        row["depth"]: row
        for row in fixed
        if row["method"] == "exact_factorized"
    }
    assert set(exact) == set(FROZEN_CONFIG.fixed_depths)
    for row in fixed:
        assert row["calls"] == row["depth"]
        if row["method"] == "exact_factorized":
            assert abs(row["population_risk_excess_vs_exact"]) < 1e-12
        else:
            assert row["population_risk_excess_vs_exact"] >= -1e-12
            expected_false_win = (
                row["empirical_standard_loss_risk"]
                < exact[row["depth"]]["empirical_standard_loss_risk"]
            )
            assert row["false_dependence_win"] is expected_false_win


def test_deterministic_json_csv_and_output_schema(tmp_path: Path):
    config = small_config(seed_count=3)
    first_payload = run_experiment(config)
    second_payload = run_experiment(config)
    assert first_payload == second_payload

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_json, first_csv = write_outputs(first_dir, first_payload)
    second_json, second_csv = write_outputs(second_dir, second_payload)
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_csv.read_bytes() == second_csv.read_bytes()

    parsed = json.loads(first_json.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["config"]["seed_count"] == 3
    assert len(parsed["seed_results"]) == 3
    assert {row["method"] for row in parsed["seed_results"][0]["fixed_depth"]} == set(METHODS)
    assert {
        row["method"]
        for row in parsed["seed_results"][0]["calibrated_standard_loss"]
    } == set(METHODS)

    with first_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert set(rows[0]) == {
        "section",
        "method",
        "depth",
        "metric",
        "n",
        "mean",
        "standard_error",
        "ci95_low",
        "ci95_high",
    }
    metrics = {row["metric"] for row in rows}
    assert "pooled_raw_off_diagonal_correlation" in metrics
    assert "residual_off_diagonal_correlation" in metrics
    assert "false_dependence_win_frequency" in metrics
    assert "test_empirical_standard_loss_risk" in metrics

    with pytest.raises(FileExistsError):
        write_outputs(first_dir, first_payload)
