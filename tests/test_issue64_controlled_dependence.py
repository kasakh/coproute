"""Focused scientific and execution tests for the ISSUE-64 runner."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_issue64_controlled_dependence import (  # noqa: E402
    DESIGN_BUDGETS,
    METHODS,
    POOL_SIZES,
    REGIMES,
    ExperimentConfig,
    SyntheticSplit,
    atomic_write_no_overwrite,
    correlation_diagnostics,
    deterministic_paired_bootstrap,
    exact_all_miss_probability,
    fit_candidate,
    full_grid,
    run_experiment,
    run_seed,
    score_candidate,
    simulate_split,
    true_marginals,
)


def config(regime: str, **changes) -> ExperimentConfig:
    base = ExperimentConfig(
        regime=regime,
        pool_size=8,
        design_budget=50,
        seed_count=1,
        validation_size=80,
        final_calibration_size=240,
        test_size=160,
        alphas=(0.10,),
        latent_draws=48,
        diagnostic_draws=20_000,
        bootstrap_repetitions=100,
    )
    return replace(base, **changes)


def test_pool_and_budget_form_an_independent_cartesian_grid():
    grid = full_grid()
    assert len(grid) == len(REGIMES) * len(POOL_SIZES) * len(DESIGN_BUDGETS)
    assert len(set(grid)) == len(grid)
    for regime in REGIMES:
        assert {
            (pool_size, budget)
            for cell_regime, pool_size, budget in grid
            if cell_regime == regime
        } == set((pool_size, budget) for pool_size in POOL_SIZES for budget in DESIGN_BUDGETS)


def test_exact_known_dependence_signatures():
    p = np.full((4, 8), 0.5)
    independent = exact_all_miss_probability(
        "conditional_independence_heterogeneous_difficulty", p, (0, 1, 2)
    )
    complementary = exact_all_miss_probability(
        "complementary_specialists_negative_residual", p, (0, 1)
    )
    redundant = exact_all_miss_probability(
        "redundant_families_positive_residual", p, (0, 1, 2)
    )
    higher = exact_all_miss_probability(
        "higher_order_pairwise_invisible", p, (0, 1, 2)
    )
    assert np.all(independent == 0.125)
    assert np.all(complementary == 0.0)
    assert np.all(redundant == 0.5)
    assert np.all(higher == 0.25)


def test_complementary_and_redundant_residual_signatures():
    complementary = simulate_split(
        config("complementary_specialists_negative_residual"),
        7,
        "ground_truth",
        size=40_000,
    )
    redundant = simulate_split(
        config("redundant_families_positive_residual"),
        7,
        "ground_truth",
        size=40_000,
    )
    assert correlation_diagnostics(complementary)["first_pair_residual_correlation"] < -0.20
    assert correlation_diagnostics(redundant)["first_pair_residual_correlation"] > 0.20


def test_higher_order_regime_is_pairwise_invisible_but_not_jointly_independent():
    split = simulate_split(
        config("higher_order_pairwise_invisible"),
        19,
        "ground_truth",
        size=60_000,
    )
    diagnostics = correlation_diagnostics(split)
    assert abs(diagnostics["first_pair_residual_correlation"]) < 0.02
    assert abs(diagnostics["first_triple_all_miss_rate"] - 0.25) < 0.01
    assert diagnostics["first_triple_independence_product"] == pytest.approx(0.125)


def test_mixed_regime_has_pooled_correlation_but_near_zero_residual_correlation():
    split = simulate_split(
        config("mixed_easy_hard_pooled_correlation_no_residual"),
        23,
        "ground_truth",
        size=60_000,
    )
    diagnostics = correlation_diagnostics(split)
    assert diagnostics["pooled_failure_correlation_mean"] > 0.15
    assert abs(diagnostics["standardized_residual_correlation_mean"]) < 0.015
    assert diagnostics["standardized_residual_absolute_correlation_mean"] < 0.02


def test_simulation_is_deterministic_and_streams_are_disjoint():
    active = config("conditional_independence_heterogeneous_difficulty")
    first = simulate_split(active, 3, "design")
    second = simulate_split(active, 3, "design")
    validation = simulate_split(active, 3, "validation")
    assert np.array_equal(first.difficulty, second.difficulty)
    assert np.array_equal(first.probabilities, second.probabilities)
    assert np.array_equal(first.correctness, second.correctness)
    assert set(first.query_id).isdisjoint(validation.query_id)
    assert not np.array_equal(first.correctness, validation.correctness[: len(first.correctness)])


def test_true_marginals_are_strictly_valid_for_every_regime_and_pool():
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    for regime in REGIMES:
        for pool_size in POOL_SIZES:
            probabilities = true_marginals(regime, x, pool_size)
            assert probabilities.shape == (len(x), pool_size)
            assert np.all(np.isfinite(probabilities))
            assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_heldout_labels_cannot_change_fitted_or_scored_candidate():
    active = config("conditional_independence_heterogeneous_difficulty")
    design = simulate_split(active, 5, "design")
    validation = simulate_split(active, 5, "validation")
    candidate = fit_candidate(
        "pairwise_count", 0.5, design, seed=991, latent_draws=32
    )
    original = score_candidate(candidate, validation, split_seed=771)
    changed = SyntheticSplit(
        validation.regime,
        validation.query_id,
        validation.difficulty,
        validation.probabilities,
        1 - validation.correctness,
    )
    changed_scores = score_candidate(candidate, changed, split_seed=771)
    assert np.array_equal(original, changed_scores)


def test_run_seed_contains_all_methods_and_valid_event_arithmetic():
    result = run_seed(config("higher_order_pairwise_invisible"), 2)
    assert set(result["alphas"]) == {"0.1"}
    methods = result["alphas"]["0.1"]["methods"]
    assert set(methods) == set(METHODS)
    certified_count = 0
    for record in methods.values():
        assert record["final_calibration_count"] == 1
        assert record["preserve_prefixes"] is True
        if not record["certified"]:
            continue
        certified_count += 1
        assert record["event_count"] == round(record["risk"] * record["trial_count"])
        assert record["call_sum"] >= record["trial_count"]
        packed = bytes.fromhex(record["event_vector"]["packed_hex"])
        unpacked = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8), bitorder="little"
        )[: record["event_vector"]["length"]]
        assert int(unpacked.sum()) == record["event_count"]
        for metric in (
            "prefix_order_kendall_similarity",
            "prefix_top_k_jaccard",
        ):
            assert 0.0 <= record[metric] <= 1.0
        if record["selected_model_jaccard_resample"] is not None:
            assert 0.0 <= record["selected_model_jaccard_resample"] <= 1.0
    assert certified_count > 0


def test_bootstrap_is_deterministic_and_records_protocol():
    values = [-1.0, -0.5, 0.25, 0.75]
    first = deterministic_paired_bootstrap(values, seed=17, repetitions=500)
    second = deterministic_paired_bootstrap(values, seed=17, repetitions=500)
    assert first == second
    assert first["bootstrap_repetitions"] == 500
    assert first["bootstrap_seed"] == 17
    assert first["paired_seed_count"] == len(values)
    assert first["mean_difference"] == pytest.approx(np.mean(values))
    assert first["ci95"][0] <= first["mean_difference"] <= first["ci95"][1]


def test_atomic_output_refuses_collision(tmp_path: Path):
    path = tmp_path / "shard.json"
    atomic_write_no_overwrite(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError):
        atomic_write_no_overwrite(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_two_alpha_payload_is_canonical_json_and_has_full_bootstrap_metrics():
    active = config(
        "mixed_easy_hard_pooled_correlation_no_residual",
        validation_size=100,
        final_calibration_size=300,
        test_size=150,
        alphas=(0.10, 0.05),
        latent_draws=32,
        diagnostic_draws=2_000,
    )
    payload = run_experiment(active, host="local", stamp="test")
    json.dumps(payload, allow_nan=False)
    assert {row["alpha"] for row in payload["summary"]} == {0.10, 0.05}
    contrast = next(
        row
        for row in payload["mechanism_contrasts_and_paired_bootstrap"]
        if row.get("candidate") == "empirical_tally" and row.get("alpha") == 0.10
    )
    assert contrast["calls"]["bootstrap_repetitions"] == 100
    assert contrast["prefix_joint_miss_mae"]["bootstrap_repetitions"] == 100
    assert contrast["order_kendall_similarity"]["bootstrap_repetitions"] == 100
