"""Tests for corrected DACR conditional dependence estimators."""

from pathlib import Path
import inspect
import sys

import numpy as np
import pytest


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
if str(CORRECTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CORRECTED_ROOT))

from components.dependence import (  # noqa: E402
    DESIGN_SEED,
    ClusteredResidualFit,
    CrossFittedResidualEstimator,
    FrozenGaussianSampler,
    FrozenStudentTSampler,
    IndependenceModel,
    PooledTetrachoricEstimator,
    deterministic_fold_ids,
    evaluate_heldout_all_miss,
    fit_cross_fitted_residual_correlation,
    fit_pooled_tetrachoric,
    nearest_correlation,
    predict_all_miss_by_subset,
)


def _assert_correlation(matrix: np.ndarray) -> None:
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=0.0)
    assert float(np.min(np.linalg.eigvalsh(matrix))) >= -1e-8


def _mean_absolute_off_diagonal(matrix: np.ndarray) -> float:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return float(np.mean(np.abs(matrix[indices])))


def test_independence_model_uses_router_marginals_exactly() -> None:
    model = IndependenceModel(3)

    assert model.method == "independence"
    np.testing.assert_array_equal(model.correlation, np.eye(3))
    assert model.all_miss_probability([0.2, 0.5, 0.8], []) == 1.0
    assert model.all_miss_probability([0.2, 0.5, 0.8], [0, 2]) == pytest.approx(
        0.8 * 0.2
    )


def test_pooled_tetrachoric_is_explicitly_labeled_and_psd() -> None:
    rng = np.random.default_rng(11)
    shared = rng.normal(size=2500)
    labels = np.column_stack(
        [
            shared + rng.normal(scale=0.8, size=shared.size) > -0.1,
            shared + rng.normal(scale=0.8, size=shared.size) > 0.2,
            shared + rng.normal(scale=1.0, size=shared.size) > -0.3,
        ]
    ).astype(int)

    fit = fit_pooled_tetrachoric(labels)

    assert fit.method == "pooled_tetrachoric"
    assert fit.label == "pooled_tetrachoric"
    assert fit.n_design_queries == labels.shape[0]
    _assert_correlation(fit.correlation)
    assert np.mean(fit.correlation[np.triu_indices(3, 1)]) > 0.35
    assert np.isfinite(fit.projection.frobenius_adjustment)


def test_nearest_correlation_reports_adjustment_and_eigenvalues() -> None:
    invalid = np.array(
        [
            [1.0, 1.20, -0.95],
            [1.20, 1.0, 0.85],
            [-0.95, 0.85, 1.0],
        ]
    )

    projected, diagnostics = nearest_correlation(invalid)

    _assert_correlation(projected)
    assert diagnostics.original_min_eigenvalue < 0.0
    assert diagnostics.projected_min_eigenvalue >= -1e-8
    assert diagnostics.frobenius_adjustment > 0.0
    assert diagnostics.iterations > 0
    assert isinstance(diagnostics.converged, bool)


def test_deterministic_design_seed_and_balanced_folds() -> None:
    first = deterministic_fold_ids(103, 5)
    repeated = deterministic_fold_ids(103, 5, design_seed=DESIGN_SEED)
    other = deterministic_fold_ids(103, 5, design_seed=DESIGN_SEED + 1)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
    counts = np.bincount(first)
    assert int(np.max(counts) - np.min(counts)) <= 1
    assert not first.flags.writeable


def test_cross_fitted_estimator_has_no_heldout_label_input() -> None:
    fit_parameters = inspect.signature(CrossFittedResidualEstimator.fit).parameters
    pooled_parameters = inspect.signature(PooledTetrachoricEstimator.fit).parameters
    prohibited = {"heldout", "test", "calibration", "evaluation"}

    assert not any(
        token in parameter.lower()
        for parameter in fit_parameters
        for token in prohibited
    )
    assert not any(
        token in parameter.lower()
        for parameter in pooled_parameters
        for token in prohibited
    )
    assert "cross_fitted_router_probabilities" in fit_parameters
    assert "design_fold_ids" in fit_parameters
    assert "heldout_correctness" in inspect.signature(
        evaluate_heldout_all_miss
    ).parameters


def test_cross_fitted_estimator_requires_cross_fold_provenance() -> None:
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    probabilities = np.full(labels.shape, 0.5)

    with pytest.raises(ValueError, match="at least two folds"):
        CrossFittedResidualEstimator(min_cluster_size=2).fit(
            labels, probabilities, np.zeros(labels.shape[0], dtype=int)
        )


def test_residualization_removes_spurious_common_difficulty_correlation() -> None:
    rng = np.random.default_rng(20260903)
    n_queries = 8000
    difficulty = rng.normal(size=n_queries)
    intercepts = np.array([0.55, 0.25, -0.05, -0.30])
    slopes = np.array([1.35, 1.15, 1.45, 1.25])
    logits = intercepts[None, :] + difficulty[:, None] * slopes[None, :]
    router_probabilities = 1.0 / (1.0 + np.exp(-logits))
    correctness = (rng.random(router_probabilities.shape) < router_probabilities).astype(
        int
    )
    folds = deterministic_fold_ids(n_queries, 8, design_seed=77)

    pooled = fit_pooled_tetrachoric(correctness)
    residual = fit_cross_fitted_residual_correlation(
        correctness, router_probabilities, folds
    )

    pooled_strength = _mean_absolute_off_diagonal(pooled.correlation)
    residual_strength = _mean_absolute_off_diagonal(residual.correlation)
    assert pooled_strength > 0.15
    assert residual_strength < 0.05
    assert residual_strength < 0.25 * pooled_strength
    _assert_correlation(residual.correlation)


def test_query_clustered_residual_matrices_capture_cluster_specific_dependence() -> None:
    rng = np.random.default_rng(991)
    n_per_cluster = 1200
    clusters = np.repeat(["shared", "independent"], n_per_cluster)
    probabilities = np.full((2 * n_per_cluster, 3), 0.62)

    shared_base = rng.random(n_per_cluster) < 0.62
    shared_labels = np.column_stack(
        [
            shared_base,
            np.where(rng.random(n_per_cluster) < 0.82, shared_base, rng.random(n_per_cluster) < 0.62),
            rng.random(n_per_cluster) < 0.62,
        ]
    )
    independent_labels = rng.random((n_per_cluster, 3)) < 0.62
    correctness = np.vstack([shared_labels, independent_labels]).astype(int)
    folds = deterministic_fold_ids(correctness.shape[0], 6, design_seed=18)

    fit = CrossFittedResidualEstimator(min_cluster_size=100).fit(
        correctness,
        probabilities,
        folds,
        query_clusters=clusters,
    )

    assert isinstance(fit, ClusteredResidualFit)
    assert fit.method == "query_clustered_cross_fitted_residual_correlation"
    assert set(fit.by_cluster) == {"shared", "independent"}
    for cluster_fit in fit.by_cluster.values():
        _assert_correlation(cluster_fit.correlation)
        assert cluster_fit.method == fit.method
    assert fit.matrix_for("shared")[0, 1] > 0.55
    assert abs(fit.matrix_for("independent")[0, 1]) < 0.08
    np.testing.assert_array_equal(fit.matrix_for("unseen"), fit.pooled.correlation)


def test_frozen_gaussian_mapping_is_deterministic_and_query_order_invariant() -> None:
    correlation, _ = nearest_correlation(
        [[1.0, 0.35, -0.10], [0.35, 1.0, 0.20], [-0.10, 0.20, 1.0]]
    )
    first = FrozenGaussianSampler(3, n_draws=4096, design_seed=41)
    repeated = FrozenGaussianSampler(3, n_draws=4096, design_seed=41)
    other = FrozenGaussianSampler(3, n_draws=4096, design_seed=42)
    probabilities = [0.25, 0.55, 0.80]

    initial = first.failure_sample(probabilities, correlation)
    first.failure_sample([0.9, 0.2, 0.4], np.eye(3))
    after_other_query = first.failure_sample(probabilities, correlation)

    np.testing.assert_array_equal(initial, after_other_query)
    np.testing.assert_array_equal(
        initial, repeated.failure_sample(probabilities, correlation)
    )
    assert not np.array_equal(first.design_latent_bank, other.design_latent_bank)
    assert not initial.flags.writeable
    assert np.isfinite(first.all_miss_probability(probabilities, correlation, [0, 1]))


def test_frozen_student_t_mapping_is_deterministic_and_finite() -> None:
    correlation, _ = nearest_correlation(
        [[1.0, 0.55, 0.15], [0.55, 1.0, 0.30], [0.15, 0.30, 1.0]]
    )
    first = FrozenStudentTSampler(
        3, degrees_of_freedom=3.5, n_draws=4096, design_seed=73
    )
    repeated = FrozenStudentTSampler(
        3, degrees_of_freedom=3.5, n_draws=4096, design_seed=73
    )
    probabilities = [0.35, 0.60, 0.75]

    sample = first.failure_sample(probabilities, correlation)
    np.testing.assert_array_equal(
        sample, repeated.failure_sample(probabilities, correlation)
    )
    probability = first.all_miss_probability(probabilities, correlation, [0, 1, 2])
    assert 0.0 <= probability <= 1.0
    assert np.all(np.isfinite(first.design_normal_bank))
    assert np.all(np.isfinite(first.design_scale_bank))
    assert not sample.flags.writeable


def test_all_miss_prediction_and_calibration_are_reported_by_subset() -> None:
    router_probabilities = np.array(
        [
            [0.8, 0.4, 0.7],
            [0.2, 0.5, 0.9],
            [0.6, 0.3, 0.4],
            [0.7, 0.8, 0.5],
            [0.1, 0.4, 0.2],
            [0.9, 0.6, 0.7],
        ]
    )
    heldout_correctness = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )
    subsets = [(), (0,), (0, 1), (0, 1, 2)]

    predictions = predict_all_miss_by_subset(
        router_probabilities, subsets, IndependenceModel(3)
    )
    metrics = evaluate_heldout_all_miss(
        heldout_correctness, predictions, n_bins=4
    )
    unsorted_metrics = evaluate_heldout_all_miss(
        heldout_correctness, {(1, 0): predictions[(0, 1)]}, n_bins=4
    )

    np.testing.assert_allclose(predictions[()], np.ones(6))
    np.testing.assert_allclose(
        predictions[(0, 1)],
        (1.0 - router_probabilities[:, 0])
        * (1.0 - router_probabilities[:, 1]),
    )
    assert set(metrics) == set(subsets)
    assert set(unsorted_metrics) == {(0, 1)}
    for subset, summary in metrics.items():
        assert summary.subset == subset
        assert summary.n_queries == heldout_correctness.shape[0]
        assert 0.0 <= summary.observed_rate <= 1.0
        assert 0.0 <= summary.mean_prediction <= 1.0
        assert summary.expected_calibration_error >= 0.0
        assert np.isfinite(summary.brier_score)
        assert np.isfinite(summary.log_loss)


def test_clustered_prediction_uses_frozen_sampler_and_cluster_fallback() -> None:
    rng = np.random.default_rng(9)
    labels = (rng.random((400, 3)) < 0.6).astype(int)
    probabilities = np.full(labels.shape, 0.6)
    folds = deterministic_fold_ids(400, 4, design_seed=5)
    clusters = np.repeat(["a", "b"], 200)
    fit = CrossFittedResidualEstimator(min_cluster_size=50).fit(
        labels, probabilities, folds, query_clusters=clusters
    )
    heldout_probabilities = np.array(
        [[0.8, 0.7, 0.4], [0.4, 0.5, 0.6], [0.9, 0.3, 0.8]]
    )
    heldout_clusters = np.array(["a", "b", "unseen"])
    sampler = FrozenGaussianSampler(3, n_draws=2048, design_seed=61)

    first = predict_all_miss_by_subset(
        heldout_probabilities,
        [(0, 1), (0, 1, 2)],
        fit,
        sampler=sampler,
        query_clusters=heldout_clusters,
    )
    repeated = predict_all_miss_by_subset(
        heldout_probabilities,
        [(0, 1), (0, 1, 2)],
        fit,
        sampler=sampler,
        query_clusters=heldout_clusters,
    )

    for subset in first:
        np.testing.assert_array_equal(first[subset], repeated[subset])
        assert np.all(np.isfinite(first[subset]))
        assert np.all((first[subset] >= 0.0) & (first[subset] <= 1.0))
        assert not first[subset].flags.writeable


def test_binary_copula_scope_is_stated_without_uniqueness_claim() -> None:
    import components.dependence as dependence

    module_text = Path(dependence.__file__).read_text().lower()
    assert "bernoulli subcopula" in module_text
    assert "does not claim" in module_text
    assert "unique bernoulli copula" in module_text
