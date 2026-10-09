"""Conditional dependence estimators for corrected DACR.

Correctness matrices use rows for queries and columns for models. A value of
one means that a model answered a query correctly. Router probabilities are
probabilities of correctness for the same rows and models.

Binary margins identify only the Bernoulli subcopula values on their support
grid. The latent Gaussian and Student t constructions below are explicit
working dependence models. This module does not claim that either construction
is a unique Bernoulli copula.

Fitting APIs accept design data only. Held-out outcomes are accepted only by
``evaluate_heldout_all_miss`` after the dependence estimate and frozen sampler
have produced predictions. This separation makes accidental calibration or test
label leakage visible at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


DESIGN_SEED = 314159
_EPSILON = 1e-8


def _as_binary_matrix(values: Any, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix.astype(float))):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return matrix.astype(float, copy=False)


def _as_probability_matrix(values: Any, shape: Tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return matrix


def _as_cluster_key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _normalize_subsets(
    subsets: Iterable[Iterable[int]], n_models: int
) -> Tuple[Tuple[int, ...], ...]:
    normalized = []
    seen = set()
    for subset in subsets:
        member = tuple(sorted(int(index) for index in subset))
        if len(member) != len(set(member)):
            raise ValueError("subsets must not contain duplicate model indices")
        if any(index < 0 or index >= n_models for index in member):
            raise IndexError("subset contains an out-of-range model index")
        if member in seen:
            raise ValueError("subsets must be unique")
        normalized.append(member)
        seen.add(member)
    if not normalized:
        raise ValueError("at least one subset is required")
    return tuple(normalized)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical diagnostics for a nearest-correlation projection."""

    frobenius_adjustment: float
    original_min_eigenvalue: float
    projected_min_eigenvalue: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class DependenceFit:
    """One design-split dependence estimate."""

    method: str
    correlation: np.ndarray
    projection: ProjectionDiagnostics
    n_design_queries: int

    def __post_init__(self) -> None:
        matrix = np.asarray(self.correlation, dtype=float).copy()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("correlation must be square")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("correlation must be finite")
        matrix.setflags(write=False)
        object.__setattr__(self, "correlation", matrix)

    @property
    def label(self) -> str:
        return self.method


@dataclass(frozen=True)
class ClusteredResidualFit:
    """Pooled and query-clustered cross-fitted residual correlations."""

    method: str
    pooled: DependenceFit
    by_cluster: Mapping[Any, DependenceFit]
    min_cluster_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_cluster", MappingProxyType(dict(self.by_cluster)))

    @property
    def n_models(self) -> int:
        return int(self.pooled.correlation.shape[0])

    def matrix_for(self, query_cluster: Any) -> np.ndarray:
        key = _as_cluster_key(query_cluster)
        fit = self.by_cluster.get(key, self.pooled)
        return fit.correlation


@dataclass(frozen=True)
class AllMissCalibration:
    """Held-out prediction and calibration summary for one model subset."""

    subset: Tuple[int, ...]
    n_queries: int
    observed_rate: float
    mean_prediction: float
    calibration_gap: float
    expected_calibration_error: float
    brier_score: float
    log_loss: float


def nearest_correlation(
    matrix: Any,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, ProjectionDiagnostics]:
    """Project a finite symmetric matrix to the nearest correlation matrix.

    The implementation uses Higham's alternating projections, followed by an
    eigenvalue clip and diagonal rescaling. Diagnostics compare the projected
    matrix with the symmetrized, unit-diagonal input.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square array")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
        raise ValueError("max_iterations must be a positive integer")

    target = (values + values.T) / 2.0
    np.fill_diagonal(target, 1.0)
    original_min = float(np.min(np.linalg.eigvalsh(target)))

    current = target.copy()
    correction = np.zeros_like(current)
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        previous = current.copy()
        residual = current - correction
        eigenvalues, eigenvectors = np.linalg.eigh((residual + residual.T) / 2.0)
        positive = np.maximum(eigenvalues, 0.0)
        projected_psd = (eigenvectors * positive) @ eigenvectors.T
        correction = projected_psd - residual
        current = (projected_psd + projected_psd.T) / 2.0
        np.fill_diagonal(current, 1.0)
        if np.linalg.norm(current - previous, ord="fro") <= tolerance:
            converged = True
            break

    eigenvalues, eigenvectors = np.linalg.eigh((current + current.T) / 2.0)
    positive = np.maximum(eigenvalues, 0.0)
    current = (eigenvectors * positive) @ eigenvectors.T
    scales = np.sqrt(np.maximum(np.diag(current), _EPSILON))
    current = current / np.outer(scales, scales)
    current = np.clip((current + current.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(current, 1.0)

    projected_min = float(np.min(np.linalg.eigvalsh(current)))
    diagnostics = ProjectionDiagnostics(
        frobenius_adjustment=float(np.linalg.norm(current - target, ord="fro")),
        original_min_eigenvalue=original_min,
        projected_min_eigenvalue=projected_min,
        iterations=iterations,
        converged=converged,
    )
    current.setflags(write=False)
    return current, diagnostics


def deterministic_fold_ids(
    n_queries: int,
    n_folds: int,
    *,
    design_seed: int = DESIGN_SEED,
) -> np.ndarray:
    """Create deterministic balanced design-fold assignments."""

    if isinstance(n_queries, bool) or int(n_queries) < 2:
        raise ValueError("n_queries must be at least two")
    if isinstance(n_folds, bool) or int(n_folds) < 2 or int(n_folds) > int(n_queries):
        raise ValueError("n_folds must be between two and n_queries")
    if isinstance(design_seed, bool) or int(design_seed) < 0:
        raise ValueError("design_seed must be a non-negative integer")
    rng = np.random.default_rng(np.random.SeedSequence(int(design_seed)))
    permutation = rng.permutation(int(n_queries))
    fold_ids = np.empty(int(n_queries), dtype=int)
    fold_ids[permutation] = np.arange(int(n_queries)) % int(n_folds)
    fold_ids.setflags(write=False)
    return fold_ids


@dataclass(frozen=True)
class IndependenceModel:
    """Conditional independence working model using router marginals."""

    n_models: int
    method: str = "independence"

    def __post_init__(self) -> None:
        if isinstance(self.n_models, bool) or int(self.n_models) <= 0:
            raise ValueError("n_models must be a positive integer")
        object.__setattr__(self, "n_models", int(self.n_models))

    @property
    def correlation(self) -> np.ndarray:
        matrix = np.eye(self.n_models)
        matrix.setflags(write=False)
        return matrix

    def all_miss_probability(
        self, router_correctness_probabilities: Any, subset: Iterable[int]
    ) -> float:
        probabilities = np.asarray(router_correctness_probabilities, dtype=float)
        if probabilities.shape != (self.n_models,):
            raise ValueError("router probabilities must have one entry per model")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("router probabilities must be finite and lie in [0, 1]")
        member = _normalize_subsets([subset], self.n_models)[0]
        if not member:
            return 1.0
        return float(np.prod(1.0 - probabilities[list(member)]))


# Fixed quadrature nodes make tetrachoric fitting deterministic.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(48)
_NORMAL = NormalDist()


def _bivariate_normal_cdf(first: float, second: float, correlation: float) -> float:
    if correlation == 0.0:
        return float(_NORMAL.cdf(first) * _NORMAL.cdf(second))
    lower, upper = (0.0, correlation) if correlation > 0.0 else (correlation, 0.0)
    transformed = 0.5 * (upper - lower) * _LEGENDRE_NODES + 0.5 * (upper + lower)
    denominator = 1.0 - transformed * transformed
    exponent = -(
        first * first - 2.0 * first * second * transformed + second * second
    ) / (2.0 * denominator)
    integrand = np.exp(exponent) / (2.0 * np.pi * np.sqrt(denominator))
    integral = 0.5 * (upper - lower) * float(np.dot(_LEGENDRE_WEIGHTS, integrand))
    if correlation < 0.0:
        integral = -integral
    return float(np.clip(_NORMAL.cdf(first) * _NORMAL.cdf(second) + integral, 0.0, 1.0))


def _tetrachoric_pair(left: np.ndarray, right: np.ndarray) -> float:
    left_probability = float(np.mean(left))
    right_probability = float(np.mean(right))
    if (
        left_probability <= _EPSILON
        or left_probability >= 1.0 - _EPSILON
        or right_probability <= _EPSILON
        or right_probability >= 1.0 - _EPSILON
    ):
        return 0.0
    threshold_left = _NORMAL.inv_cdf(left_probability)
    threshold_right = _NORMAL.inv_cdf(right_probability)
    target = float(np.mean(left * right))
    low, high = -0.999, 0.999
    low_value = _bivariate_normal_cdf(threshold_left, threshold_right, low)
    high_value = _bivariate_normal_cdf(threshold_left, threshold_right, high)
    if target <= low_value:
        return low
    if target >= high_value:
        return high
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        value = _bivariate_normal_cdf(threshold_left, threshold_right, midpoint)
        if value < target:
            low = midpoint
        else:
            high = midpoint
    return float(0.5 * (low + high))


class PooledTetrachoricEstimator:
    """Fit the explicitly pooled, unconditional tetrachoric baseline."""

    method = "pooled_tetrachoric"

    def fit(self, design_correctness: Any) -> DependenceFit:
        labels = _as_binary_matrix(design_correctness, "design_correctness")
        n_queries, n_models = labels.shape
        matrix = np.eye(n_models)
        for left in range(n_models):
            for right in range(left + 1, n_models):
                value = _tetrachoric_pair(labels[:, left], labels[:, right])
                matrix[left, right] = matrix[right, left] = value
        correlation, diagnostics = nearest_correlation(matrix)
        return DependenceFit(self.method, correlation, diagnostics, n_queries)


def fit_pooled_tetrachoric(design_correctness: Any) -> DependenceFit:
    """Convenience wrapper for the pooled tetrachoric baseline."""

    return PooledTetrachoricEstimator().fit(design_correctness)


def _residual_correlation(
    correctness: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    variance = np.maximum(probabilities * (1.0 - probabilities), epsilon)
    residuals = (correctness - probabilities) / np.sqrt(variance)
    n_models = correctness.shape[1]
    matrix = np.eye(n_models)
    centered = residuals - np.mean(residuals, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    for left in range(n_models):
        for right in range(left + 1, n_models):
            denominator = norms[left] * norms[right]
            value = (
                0.0
                if denominator <= epsilon
                else float(np.dot(centered[:, left], centered[:, right]) / denominator)
            )
            matrix[left, right] = matrix[right, left] = float(np.clip(value, -1.0, 1.0))
    return matrix


class CrossFittedResidualEstimator:
    """Estimate conditional dependence from cross-fitted router residuals.

    ``cross_fitted_router_probabilities`` must be produced without training on
    the corresponding row's correctness labels. ``design_fold_ids`` records
    that provenance and must contain at least two folds. The estimator never
    accepts calibration or test outcomes.
    """

    method = "cross_fitted_residual_correlation"
    clustered_method = "query_clustered_cross_fitted_residual_correlation"

    def __init__(self, *, epsilon: float = 1e-6, min_cluster_size: int = 20) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if isinstance(min_cluster_size, bool) or int(min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be at least two")
        self.epsilon = float(epsilon)
        self.min_cluster_size = int(min_cluster_size)

    def fit(
        self,
        design_correctness: Any,
        cross_fitted_router_probabilities: Any,
        design_fold_ids: Any,
        *,
        query_clusters: Optional[Any] = None,
    ) -> Union[DependenceFit, ClusteredResidualFit]:
        labels = _as_binary_matrix(design_correctness, "design_correctness")
        probabilities = _as_probability_matrix(
            cross_fitted_router_probabilities,
            labels.shape,
            "cross_fitted_router_probabilities",
        )
        folds = np.asarray(design_fold_ids)
        if folds.ndim != 1 or folds.shape[0] != labels.shape[0]:
            raise ValueError("design_fold_ids must have one entry per design query")
        if len(np.unique(folds)) < 2:
            raise ValueError("design_fold_ids must contain at least two folds")

        raw = _residual_correlation(labels, probabilities, self.epsilon)
        pooled_matrix, pooled_diagnostics = nearest_correlation(raw)
        pooled = DependenceFit(
            self.method,
            pooled_matrix,
            pooled_diagnostics,
            labels.shape[0],
        )
        if query_clusters is None:
            return pooled

        clusters = np.asarray(query_clusters)
        if clusters.ndim != 1 or clusters.shape[0] != labels.shape[0]:
            raise ValueError("query_clusters must have one entry per design query")
        by_cluster: Dict[Any, DependenceFit] = {}
        for raw_key in np.unique(clusters):
            mask = clusters == raw_key
            if int(np.sum(mask)) < self.min_cluster_size:
                continue
            cluster_raw = _residual_correlation(
                labels[mask], probabilities[mask], self.epsilon
            )
            cluster_matrix, diagnostics = nearest_correlation(cluster_raw)
            key = _as_cluster_key(raw_key)
            by_cluster[key] = DependenceFit(
                self.clustered_method,
                cluster_matrix,
                diagnostics,
                int(np.sum(mask)),
            )
        return ClusteredResidualFit(
            self.clustered_method,
            pooled,
            by_cluster,
            self.min_cluster_size,
        )


def fit_cross_fitted_residual_correlation(
    design_correctness: Any,
    cross_fitted_router_probabilities: Any,
    design_fold_ids: Any,
    *,
    query_clusters: Optional[Any] = None,
    epsilon: float = 1e-6,
    min_cluster_size: int = 20,
) -> Union[DependenceFit, ClusteredResidualFit]:
    """Convenience wrapper for cross-fitted residual dependence fitting."""

    estimator = CrossFittedResidualEstimator(
        epsilon=epsilon, min_cluster_size=min_cluster_size
    )
    return estimator.fit(
        design_correctness,
        cross_fitted_router_probabilities,
        design_fold_ids,
        query_clusters=query_clusters,
    )


def _correlation_root(correlation: Any, n_models: int) -> np.ndarray:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape != (n_models, n_models):
        raise ValueError("correlation must have shape (n_models, n_models)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("correlation must have a unit diagonal")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) < -1e-8:
        raise ValueError("correlation must be positive semidefinite")
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


class FrozenGaussianSampler:
    """Gaussian latent sampler frozen once by a design seed."""

    model = "gaussian"

    def __init__(
        self,
        n_models: int,
        *,
        n_draws: int = 8192,
        design_seed: int = DESIGN_SEED,
    ) -> None:
        if isinstance(n_models, bool) or int(n_models) <= 0:
            raise ValueError("n_models must be a positive integer")
        if isinstance(n_draws, bool) or int(n_draws) <= 0:
            raise ValueError("n_draws must be a positive integer")
        if isinstance(design_seed, bool) or int(design_seed) < 0:
            raise ValueError("design_seed must be a non-negative integer")
        self.n_models = int(n_models)
        self.n_draws = int(n_draws)
        self.design_seed = int(design_seed)
        rng = np.random.default_rng(np.random.SeedSequence(self.design_seed))
        bank = rng.standard_normal((self.n_draws, self.n_models))
        bank.setflags(write=False)
        self._design_latent_bank = bank

    @property
    def design_latent_bank(self) -> np.ndarray:
        return self._design_latent_bank.copy()

    def failure_sample(
        self, router_correctness_probabilities: Any, correlation: Any
    ) -> np.ndarray:
        probabilities = np.asarray(router_correctness_probabilities, dtype=float)
        if probabilities.shape != (self.n_models,):
            raise ValueError("router probabilities must have one entry per model")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("router probabilities must be finite and lie in [0, 1]")
        root = _correlation_root(correlation, self.n_models)
        latent = self._design_latent_bank @ root.T
        failure_probabilities = 1.0 - probabilities
        thresholds = np.array(
            [
                float("-inf")
                if probability <= 0.0
                else float("inf")
                if probability >= 1.0
                else _NORMAL.inv_cdf(float(probability))
                for probability in failure_probabilities
            ]
        )
        failures = latent <= thresholds[None, :]
        failures.setflags(write=False)
        return failures

    def all_miss_probability(
        self,
        router_correctness_probabilities: Any,
        correlation: Any,
        subset: Iterable[int],
    ) -> float:
        member = _normalize_subsets([subset], self.n_models)[0]
        if not member:
            return 1.0
        failures = self.failure_sample(router_correctness_probabilities, correlation)
        return float(np.mean(np.all(failures[:, list(member)], axis=1)))


class FrozenStudentTSampler:
    """Student t latent sampler frozen once by a design seed.

    Frozen correlated t draws are converted to empirical marginal ranks. This
    keeps the Monte Carlo mapping deterministic and gives uniform finite-bank
    margins without requiring an external Student t inverse-CDF dependency.
    """

    model = "student_t"

    def __init__(
        self,
        n_models: int,
        *,
        degrees_of_freedom: float = 4.0,
        n_draws: int = 8192,
        design_seed: int = DESIGN_SEED,
    ) -> None:
        if isinstance(n_models, bool) or int(n_models) <= 0:
            raise ValueError("n_models must be a positive integer")
        if isinstance(n_draws, bool) or int(n_draws) <= 0:
            raise ValueError("n_draws must be a positive integer")
        if not np.isfinite(degrees_of_freedom) or float(degrees_of_freedom) <= 0.0:
            raise ValueError("degrees_of_freedom must be positive and finite")
        if isinstance(design_seed, bool) or int(design_seed) < 0:
            raise ValueError("design_seed must be a non-negative integer")
        self.n_models = int(n_models)
        self.n_draws = int(n_draws)
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.design_seed = int(design_seed)
        rng = np.random.default_rng(np.random.SeedSequence(self.design_seed))
        normals = rng.standard_normal((self.n_draws, self.n_models))
        scales = rng.chisquare(self.degrees_of_freedom, size=self.n_draws)
        normals.setflags(write=False)
        scales.setflags(write=False)
        self._design_normal_bank = normals
        self._design_scale_bank = scales

    @property
    def design_normal_bank(self) -> np.ndarray:
        return self._design_normal_bank.copy()

    @property
    def design_scale_bank(self) -> np.ndarray:
        return self._design_scale_bank.copy()

    def failure_sample(
        self, router_correctness_probabilities: Any, correlation: Any
    ) -> np.ndarray:
        probabilities = np.asarray(router_correctness_probabilities, dtype=float)
        if probabilities.shape != (self.n_models,):
            raise ValueError("router probabilities must have one entry per model")
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("router probabilities must be finite and lie in [0, 1]")
        root = _correlation_root(correlation, self.n_models)
        latent = (self._design_normal_bank @ root.T) / np.sqrt(
            self._design_scale_bank[:, None] / self.degrees_of_freedom
        )
        uniforms = np.empty_like(latent)
        for column in range(self.n_models):
            order = np.argsort(latent[:, column], kind="mergesort")
            ranks = np.empty(self.n_draws, dtype=int)
            ranks[order] = np.arange(self.n_draws)
            uniforms[:, column] = (ranks + 0.5) / self.n_draws
        failures = uniforms <= (1.0 - probabilities)[None, :]
        failures.setflags(write=False)
        return failures

    def all_miss_probability(
        self,
        router_correctness_probabilities: Any,
        correlation: Any,
        subset: Iterable[int],
    ) -> float:
        member = _normalize_subsets([subset], self.n_models)[0]
        if not member:
            return 1.0
        failures = self.failure_sample(router_correctness_probabilities, correlation)
        return float(np.mean(np.all(failures[:, list(member)], axis=1)))


DependenceInput = Union[IndependenceModel, DependenceFit, ClusteredResidualFit]
FrozenSampler = Union[FrozenGaussianSampler, FrozenStudentTSampler]


def predict_all_miss_by_subset(
    router_correctness_probabilities: Any,
    subsets: Iterable[Iterable[int]],
    dependence: DependenceInput,
    *,
    sampler: Optional[FrozenSampler] = None,
    query_clusters: Optional[Any] = None,
) -> Mapping[Tuple[int, ...], np.ndarray]:
    """Predict held-out all-miss probabilities without using held-out outcomes."""

    probabilities = np.asarray(router_correctness_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("router probabilities must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("router probabilities must be finite and lie in [0, 1]")
    n_queries, n_models = probabilities.shape
    members = _normalize_subsets(subsets, n_models)

    if isinstance(dependence, IndependenceModel):
        if dependence.n_models != n_models:
            raise ValueError("dependence model count does not match router probabilities")
        result = {}
        for member in members:
            prediction = (
                np.ones(n_queries)
                if not member
                else np.prod(1.0 - probabilities[:, list(member)], axis=1)
            )
            prediction.setflags(write=False)
            result[member] = prediction
        return MappingProxyType(result)

    if isinstance(dependence, DependenceFit):
        if dependence.correlation.shape != (n_models, n_models):
            raise ValueError("dependence model count does not match router probabilities")
        correlations = [dependence.correlation] * n_queries
    elif isinstance(dependence, ClusteredResidualFit):
        if dependence.n_models != n_models:
            raise ValueError("dependence model count does not match router probabilities")
        if query_clusters is None:
            raise ValueError("query_clusters are required for clustered dependence")
        clusters = np.asarray(query_clusters)
        if clusters.ndim != 1 or clusters.shape[0] != n_queries:
            raise ValueError("query_clusters must have one entry per query")
        correlations = [dependence.matrix_for(cluster) for cluster in clusters]
    else:
        raise TypeError("unsupported dependence model")

    active_sampler: FrozenSampler = (
        FrozenGaussianSampler(n_models, design_seed=DESIGN_SEED)
        if sampler is None
        else sampler
    )
    if active_sampler.n_models != n_models:
        raise ValueError("sampler model count does not match router probabilities")
    output = {member: np.empty(n_queries, dtype=float) for member in members}
    for row in range(n_queries):
        failures = active_sampler.failure_sample(probabilities[row], correlations[row])
        for member in members:
            output[member][row] = (
                1.0
                if not member
                else float(np.mean(np.all(failures[:, list(member)], axis=1)))
            )
    for prediction in output.values():
        prediction.setflags(write=False)
    return MappingProxyType(output)


def evaluate_heldout_all_miss(
    heldout_correctness: Any,
    predictions_by_subset: Mapping[Iterable[int], Any],
    *,
    n_bins: int = 10,
) -> Mapping[Tuple[int, ...], AllMissCalibration]:
    """Evaluate held-out all-miss prediction and calibration by subset.

    This is the only API in this module that accepts held-out correctness.
    Dependence fitting and prediction must already be complete.
    """

    correctness = _as_binary_matrix(heldout_correctness, "heldout_correctness")
    if isinstance(n_bins, bool) or int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer")
    n_queries, n_models = correctness.shape
    normalized_predictions: Dict[Tuple[int, ...], Any] = {}
    for raw_subset, prediction in predictions_by_subset.items():
        member = _normalize_subsets([raw_subset], n_models)[0]
        if member in normalized_predictions:
            raise ValueError("predictions must contain unique normalized subsets")
        normalized_predictions[member] = prediction
    members = tuple(normalized_predictions)
    if not members:
        raise ValueError("at least one subset prediction is required")
    output: Dict[Tuple[int, ...], AllMissCalibration] = {}
    boundaries = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for member in members:
        predicted = np.asarray(normalized_predictions[member], dtype=float)
        if predicted.shape != (n_queries,):
            raise ValueError("each subset prediction must have one value per held-out query")
        if not np.all(np.isfinite(predicted)) or np.any(
            (predicted < 0.0) | (predicted > 1.0)
        ):
            raise ValueError("predictions must be finite and lie in [0, 1]")
        observed = (
            np.ones(n_queries, dtype=float)
            if not member
            else np.all(correctness[:, list(member)] == 0.0, axis=1).astype(float)
        )
        clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
        bin_ids = np.minimum(np.digitize(predicted, boundaries[1:-1]), int(n_bins) - 1)
        ece = 0.0
        for bin_index in range(int(n_bins)):
            mask = bin_ids == bin_index
            if np.any(mask):
                ece += float(np.mean(mask)) * abs(
                    float(np.mean(observed[mask]) - np.mean(predicted[mask]))
                )
        observed_rate = float(np.mean(observed))
        mean_prediction = float(np.mean(predicted))
        output[member] = AllMissCalibration(
            subset=member,
            n_queries=n_queries,
            observed_rate=observed_rate,
            mean_prediction=mean_prediction,
            calibration_gap=mean_prediction - observed_rate,
            expected_calibration_error=ece,
            brier_score=float(np.mean((predicted - observed) ** 2)),
            log_loss=float(
                -np.mean(observed * np.log(clipped) + (1.0 - observed) * np.log(1.0 - clipped))
            ),
        )
    return MappingProxyType(output)


__all__ = [
    "DESIGN_SEED",
    "AllMissCalibration",
    "ClusteredResidualFit",
    "CrossFittedResidualEstimator",
    "DependenceFit",
    "FrozenGaussianSampler",
    "FrozenStudentTSampler",
    "IndependenceModel",
    "PooledTetrachoricEstimator",
    "ProjectionDiagnostics",
    "deterministic_fold_ids",
    "evaluate_heldout_all_miss",
    "fit_cross_fitted_residual_correlation",
    "fit_pooled_tetrachoric",
    "nearest_correlation",
    "predict_all_miss_by_subset",
]
