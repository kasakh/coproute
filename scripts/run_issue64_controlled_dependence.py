#!/usr/bin/env python3
"""Run one sharded ISSUE-64 controlled synthetic dependence cell.

The runner has five data-generating regimes with known conditional all-miss
probabilities. One process owns one regime, pool-size, design-budget, and seed
range shard. Candidate hyperparameters use validation only. The chosen
candidate is then calibrated once on an independent final-calibration stream
and evaluated once on an independent test stream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from components.dependence import (  # noqa: E402
    FrozenGaussianSampler,
    FrozenStudentTSampler,
    nearest_correlation,
)
from components.racer_protocol import (  # noqa: E402
    select_standard_crc_threshold,
    smooth_nonconformity_scores,
)
from components.racer_wrapper import (  # noqa: E402
    CORRELATION_HEURISTIC,
    FACTORIZED_INDEPENDENCE,
    PAIRWISE_HEURISTIC,
    QUERY_LOCAL_EMPIRICAL_COVERAGE,
    fixed_order_factorized_scores,
    sequential_scores,
)
from components.scale_selectors import build_order  # noqa: E402

SCHEMA_VERSION = "issue64_controlled_dependence_v1"
BASE_SEED = 64_202_609
REGIMES = (
    "conditional_independence_heterogeneous_difficulty",
    "complementary_specialists_negative_residual",
    "redundant_families_positive_residual",
    "higher_order_pairwise_invisible",
    "mixed_easy_hard_pooled_correlation_no_residual",
)
POOL_SIZES = (8, 16, 32)
DESIGN_BUDGETS = (50, 100, 250, 500, 1000)
TARGET_ALPHAS = (0.10, 0.05)
METHODS = (
    "exact_independence",
    "empirical_tally",
    "pairwise_count",
    "error_correlation",
    "latent_normal",
    "heavy_tailed",
    "query_local_empirical_coverage",
    "oracle_marginal_exact_independence",
)
DEPENDENCE_AWARE_METHODS = METHODS[1:-1]
SPLIT_IDS = {
    "design": 11,
    "design_resample": 17,
    "validation": 23,
    "final_calibration": 37,
    "test": 53,
    "ground_truth": 71,
}
BOOTSTRAP_REPETITIONS = 10_000
PRIMARY_ALPHA = 0.10
SECONDARY_ALPHA = 0.05
EPSILON = 1e-12

HYPER_GRIDS: Mapping[str, tuple[Any, ...]] = {
    "exact_independence": (None,),
    "empirical_tally": (0.0, 0.5, 1.0),
    "pairwise_count": (0.0, 0.5, 1.0),
    "error_correlation": (0.90, 0.99, 0.999),
    "latent_normal": (0.0, 0.10, 0.20),
    "heavy_tailed": (3.0, 4.0, 8.0),
    "query_local_empirical_coverage": (10, 20, 40),
    "oracle_marginal_exact_independence": (None,),
}


@dataclass(frozen=True)
class ExperimentConfig:
    regime: str
    pool_size: int
    design_budget: int
    seed_start: int = 0
    seed_count: int = 20
    validation_size: int = 600
    final_calibration_size: int = 1000
    test_size: int = 2000
    alphas: tuple[float, ...] = TARGET_ALPHAS
    latent_draws: int = 512
    diagnostic_draws: int = 100_000
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS
    base_seed: int = BASE_SEED

    def validate(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unknown regime {self.regime!r}")
        if self.pool_size not in POOL_SIZES:
            raise ValueError(f"pool_size must be one of {POOL_SIZES}")
        if self.design_budget not in DESIGN_BUDGETS:
            raise ValueError(f"design_budget must be one of {DESIGN_BUDGETS}")
        for name in (
            "seed_count",
            "validation_size",
            "final_calibration_size",
            "test_size",
            "latent_draws",
            "diagnostic_draws",
            "bootstrap_repetitions",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed_start < 0:
            raise ValueError("seed_start must be non-negative")
        if len(set(self.alphas)) != len(self.alphas) or not self.alphas:
            raise ValueError("alphas must be non-empty and unique")
        if any(not 0.0 < float(alpha) < 1.0 for alpha in self.alphas):
            raise ValueError("alphas must lie strictly between zero and one")


@dataclass(frozen=True)
class SyntheticSplit:
    regime: str
    query_id: np.ndarray
    difficulty: np.ndarray
    probabilities: np.ndarray
    correctness: np.ndarray


@dataclass(frozen=True)
class MarginalFit:
    coefficients: np.ndarray
    probability_floor: float = 0.02

    def predict(self, difficulty: np.ndarray) -> np.ndarray:
        features = _features(difficulty)
        result = _sigmoid(features @ self.coefficients)
        return _validate_probabilities(
            np.clip(result, self.probability_floor, 1.0 - self.probability_floor)
        )


@dataclass(frozen=True)
class FittedCandidate:
    regime: str
    method: str
    hyper: Any
    design_x: np.ndarray
    design_y: np.ndarray
    marginal_fit: MarginalFit
    fixed_order: tuple[int, ...] | None
    raw_correlation: np.ndarray
    fit_seed: int
    latent_draws: int


def full_grid() -> tuple[tuple[str, int, int], ...]:
    """Return the declared independent Cartesian grid."""

    return tuple(
        (regime, pool_size, budget)
        for regime in REGIMES
        for pool_size in POOL_SIZES
        for budget in DESIGN_BUDGETS
    )


def _seed(base_seed: int, seed_index: int, stream: str, extra: int = 0) -> np.random.SeedSequence:
    if stream not in SPLIT_IDS:
        raise ValueError(f"unknown stream {stream!r}")
    return np.random.SeedSequence(
        [int(base_seed), int(seed_index), int(SPLIT_IDS[stream]), int(extra)]
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    positive = array >= 0.0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _features(difficulty: np.ndarray) -> np.ndarray:
    x = np.asarray(difficulty, dtype=float).reshape(-1)
    return np.column_stack((np.ones(len(x)), x, x * x - 1.0))


def _validate_probabilities(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("probabilities must be a non-empty matrix")
    if not np.all(np.isfinite(array)) or np.any((array <= 0.0) | (array >= 1.0)):
        raise ValueError("probabilities must be finite and strictly inside zero and one")
    return array


def true_marginals(regime: str, difficulty: np.ndarray, pool_size: int) -> np.ndarray:
    """Return exact conditional correctness marginals for one regime."""

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    x = np.asarray(difficulty, dtype=float).reshape(-1, 1)
    model = np.arange(pool_size, dtype=float).reshape(1, -1)
    if regime == "higher_order_pairwise_invisible":
        return np.full((len(x), pool_size), 0.5, dtype=float)
    if regime == "mixed_easy_hard_pooled_correlation_no_residual":
        easy = (x[:, 0] < 0.0).astype(float).reshape(-1, 1)
        offsets = 0.06 * np.cos((model + 1.0) * 1.7)
        return _validate_probabilities(np.clip(0.30 + 0.55 * easy + offsets, 0.08, 0.94))
    if regime == "complementary_specialists_negative_residual":
        signs = np.where((model.astype(int) % 2) == 0, 1.0, -1.0)
        intercept = 0.15 + 0.18 * np.cos(model + 0.4)
        logits = intercept + signs * 1.25 * x + 0.05 * np.sin(model + 1.0) * (x * x - 1.0)
        return _validate_probabilities(np.clip(_sigmoid(logits), 0.08, 0.88))
    intercept = 1.25 - 1.0 * model / max(pool_size - 1, 1)
    loading = 0.65 + 0.55 * ((model.astype(int) % 5) / 4.0)
    curvature = 0.10 * np.sin(model + 0.7)
    logits = intercept - loading * x + curvature * (x * x - 1.0)
    return _validate_probabilities(np.clip(_sigmoid(logits), 0.05, 0.95))


def _sample_correctness(
    regime: str,
    probabilities: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    n_queries, n_models = probabilities.shape
    if regime in {
        "conditional_independence_heterogeneous_difficulty",
        "mixed_easy_hard_pooled_correlation_no_residual",
    }:
        return (generator.random(probabilities.shape) < probabilities).astype(np.int8)
    output = np.zeros((n_queries, n_models), dtype=np.int8)
    if regime == "complementary_specialists_negative_residual":
        for left in range(0, n_models, 2):
            uniform = generator.random(n_queries)
            output[:, left] = uniform < probabilities[:, left]
            if left + 1 < n_models:
                output[:, left + 1] = uniform > 1.0 - probabilities[:, left + 1]
        return output
    if regime == "redundant_families_positive_residual":
        for start in range(0, n_models, 4):
            uniform = generator.random(n_queries)
            stop = min(start + 4, n_models)
            output[:, start:stop] = uniform[:, None] < probabilities[:, start:stop]
        return output
    if regime == "higher_order_pairwise_invisible":
        for start in range(0, n_models, 3):
            width = min(3, n_models - start)
            if width == 3:
                first = generator.integers(0, 2, size=n_queries, dtype=np.int8)
                second = generator.integers(0, 2, size=n_queries, dtype=np.int8)
                output[:, start] = first
                output[:, start + 1] = second
                output[:, start + 2] = np.bitwise_xor(first, second)
            else:
                output[:, start : start + width] = generator.integers(
                    0, 2, size=(n_queries, width), dtype=np.int8
                )
        return output
    raise ValueError(f"unknown regime {regime!r}")


def simulate_split(
    config: ExperimentConfig,
    seed_index: int,
    split_name: str,
    *,
    size: int | None = None,
) -> SyntheticSplit:
    """Generate one deterministic stream independent of every other role."""

    config.validate()
    sizes = {
        "design": config.design_budget,
        "design_resample": config.design_budget,
        "validation": config.validation_size,
        "final_calibration": config.final_calibration_size,
        "test": config.test_size,
        "ground_truth": config.diagnostic_draws,
    }
    if split_name not in sizes:
        raise ValueError(f"unknown split {split_name!r}")
    n = sizes[split_name] if size is None else int(size)
    if n < 1:
        raise ValueError("split size must be positive")
    generator = np.random.default_rng(_seed(config.base_seed, seed_index, split_name))
    if config.regime == "mixed_easy_hard_pooled_correlation_no_residual":
        difficulty = np.where(generator.random(n) < 0.5, -1.0, 1.0)
    else:
        difficulty = generator.normal(size=n)
    probabilities = true_marginals(config.regime, difficulty, config.pool_size)
    correctness = _sample_correctness(config.regime, probabilities, generator)
    if not np.all((correctness == 0) | (correctness == 1)):
        raise RuntimeError("simulator emitted non-binary correctness")
    role = SPLIT_IDS[split_name]
    query_id = np.array(
        [f"s{seed_index}:r{role}:q{index}" for index in range(n)], dtype=object
    )
    return SyntheticSplit(
        config.regime, query_id, difficulty, probabilities, correctness
    )


def exact_all_miss_probability(
    regime: str,
    probabilities: np.ndarray,
    subset: Sequence[int],
) -> np.ndarray:
    """Return the analytic conditional all-miss probability for each query."""

    p = _validate_probabilities(probabilities)
    member = tuple(sorted(int(index) for index in subset))
    if len(member) != len(set(member)):
        raise ValueError("subset contains duplicate indices")
    if any(index < 0 or index >= p.shape[1] for index in member):
        raise IndexError("subset contains an invalid model index")
    if not member:
        return np.ones(p.shape[0], dtype=float)
    if regime in {
        "conditional_independence_heterogeneous_difficulty",
        "mixed_easy_hard_pooled_correlation_no_residual",
    }:
        return np.prod(1.0 - p[:, member], axis=1)
    result = np.ones(p.shape[0], dtype=float)
    groups: dict[int, list[int]] = {}
    width = 2 if regime == "complementary_specialists_negative_residual" else 4
    if regime == "higher_order_pairwise_invisible":
        width = 3
    for index in member:
        groups.setdefault(index // width, []).append(index)
    for indices in groups.values():
        if regime == "complementary_specialists_negative_residual":
            if len(indices) == 1:
                factor = 1.0 - p[:, indices[0]]
            else:
                left, right = indices[:2]
                overlap = np.maximum(p[:, left] + p[:, right] - 1.0, 0.0)
                factor = 1.0 - p[:, left] - p[:, right] + overlap
        elif regime == "redundant_families_positive_residual":
            factor = 1.0 - np.max(p[:, indices], axis=1)
        elif regime == "higher_order_pairwise_invisible":
            count = len(indices)
            factor = np.full(p.shape[0], 0.5 if count == 1 else 0.25)
        else:
            raise ValueError(f"unknown regime {regime!r}")
        result *= np.clip(factor, 0.0, 1.0)
    if np.any((result < -EPSILON) | (result > 1.0 + EPSILON)):
        raise RuntimeError("analytic all-miss probability left the unit interval")
    return np.clip(result, 0.0, 1.0)


def oracle_orders(regime: str, probabilities: np.ndarray) -> np.ndarray:
    """Build the exact population-risk greedy order for every query."""

    p = _validate_probabilities(probabilities)
    n_queries, n_models = p.shape
    result = np.empty((n_queries, n_models), dtype=int)
    for row in range(n_queries):
        selected: list[int] = []
        remaining = set(range(n_models))
        for depth in range(n_models):
            best = min(
                remaining,
                key=lambda index: (
                    float(
                        exact_all_miss_probability(
                            regime, p[row : row + 1], selected + [index]
                        )[0]
                    ),
                    index,
                ),
            )
            result[row, depth] = best
            selected.append(best)
            remaining.remove(best)
    return result


def fit_marginals(design: SyntheticSplit, *, ridge: float = 1.0) -> MarginalFit:
    """Fit deterministic design-only logistic marginals by ridge IRLS."""

    x = _features(design.difficulty)
    y = np.asarray(design.correctness, dtype=float)
    coefficients = np.zeros((x.shape[1], y.shape[1]), dtype=float)
    penalty = np.eye(x.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    for model in range(y.shape[1]):
        beta = np.zeros(x.shape[1], dtype=float)
        beta[0] = math.log((float(y[:, model].mean()) + 0.5 / len(y)) / (1.0 - float(y[:, model].mean()) + 0.5 / len(y)))
        for _ in range(30):
            fitted = np.clip(_sigmoid(x @ beta), 1e-5, 1.0 - 1e-5)
            weight = fitted * (1.0 - fitted)
            hessian = x.T @ (weight[:, None] * x) + penalty
            gradient = x.T @ (y[:, model] - fitted) - penalty @ beta
            step = np.linalg.solve(hessian + 1e-8 * np.eye(x.shape[1]), gradient)
            beta += step
            if float(np.max(np.abs(step))) < 1e-8:
                break
        coefficients[:, model] = beta
    return MarginalFit(coefficients)


def _raw_failure_correlation(labels: np.ndarray) -> np.ndarray:
    failure = 1.0 - np.asarray(labels, dtype=float)
    centered = failure - failure.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=0))
    denominator = np.outer(norm, norm)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((failure.shape[1], failure.shape[1]), dtype=float),
        where=denominator > EPSILON,
    )
    np.fill_diagonal(correlation, 1.0)
    projected, _ = nearest_correlation(correlation)
    return projected


def fit_candidate(
    method: str,
    hyper: Any,
    design: SyntheticSplit,
    *,
    seed: int,
    latent_draws: int,
) -> FittedCandidate:
    """Fit one candidate using design labels only."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    marginal_fit = fit_marginals(design)
    fixed_order = None
    model_names = [f"model_{index:03d}" for index in range(design.correctness.shape[1])]
    if method == "empirical_tally":
        fixed_order = build_order(
            "empirical_tally",
            design.correctness,
            model_names=model_names,
            max_depth=len(model_names),
            seed=seed,
            hyper=hyper,
        )
    elif method == "latent_normal":
        fixed_order = build_order(
            "gaussian_copula",
            design.correctness,
            model_names=model_names,
            max_depth=len(model_names),
            n_draws=latent_draws,
            seed=seed,
            hyper=hyper,
        )
    elif method == "heavy_tailed":
        fixed_order = build_order(
            "student_t_copula",
            design.correctness,
            model_names=model_names,
            max_depth=len(model_names),
            n_draws=latent_draws,
            seed=seed,
            hyper=hyper,
        )
    if fixed_order is not None and tuple(sorted(fixed_order)) != tuple(range(len(model_names))):
        raise RuntimeError(f"{method} failed to return a full permutation")
    return FittedCandidate(
        regime=design.regime,
        method=method,
        hyper=hyper,
        design_x=np.asarray(design.difficulty, dtype=float).copy(),
        design_y=np.asarray(design.correctness, dtype=np.int8).copy(),
        marginal_fit=marginal_fit,
        fixed_order=fixed_order,
        raw_correlation=_raw_failure_correlation(design.correctness),
        fit_seed=int(seed),
        latent_draws=int(latent_draws),
    )


def _nearest_labels(candidate: FittedCandidate, difficulty: np.ndarray, k: int) -> np.ndarray:
    query = np.asarray(difficulty, dtype=float).reshape(-1)
    distance = np.abs(query[:, None] - candidate.design_x.reshape(1, -1))
    count = min(max(1, int(k)), len(candidate.design_x))
    indices = np.argpartition(distance, count - 1, axis=1)[:, :count]
    row = np.arange(len(query))[:, None]
    local_distance = distance[row, indices]
    stable = np.argsort(local_distance + indices * EPSILON, axis=1, kind="stable")
    indices = indices[row, stable]
    return candidate.design_y[indices]


def score_candidate(candidate: FittedCandidate, split: SyntheticSplit, *, split_seed: int) -> np.ndarray:
    """Score one held-out split without reading its labels."""

    method = candidate.method
    p_hat = candidate.marginal_fit.predict(split.difficulty)
    if method == "oracle_marginal_exact_independence":
        raw = sequential_scores(split.probabilities, dependence=FACTORIZED_INDEPENDENCE)
    elif method == "exact_independence":
        raw = sequential_scores(p_hat, dependence=FACTORIZED_INDEPENDENCE)
    elif method in {"empirical_tally", "latent_normal", "heavy_tailed"}:
        assert candidate.fixed_order is not None
        raw = fixed_order_factorized_scores(p_hat, candidate.fixed_order)
    elif method == "pairwise_count":
        raw = sequential_scores(
            p_hat,
            dependence=PAIRWISE_HEURISTIC,
            design=candidate.design_y,
            hyper={"laplace_smoothing": float(candidate.hyper)},
        )
    elif method == "error_correlation":
        raw = sequential_scores(
            p_hat,
            dependence=CORRELATION_HEURISTIC,
            design=candidate.design_y,
            hyper={"clip_magnitude": float(candidate.hyper)},
        )
    elif method == "query_local_empirical_coverage":
        neighbors = _nearest_labels(candidate, split.difficulty, int(candidate.hyper))
        raw = sequential_scores(
            p_hat,
            dependence=QUERY_LOCAL_EMPIRICAL_COVERAGE,
            neighbor_correctness=neighbors,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    order = np.argsort(raw, axis=1, kind="stable")
    ordered = np.take_along_axis(raw, order, axis=1)
    smoothed_ordered = smooth_nonconformity_scores(
        ordered,
        seed=candidate.fit_seed,
        split_seed=int(split_seed),
        preserve_prefixes=True,
    )
    scores = np.empty_like(smoothed_ordered)
    np.put_along_axis(scores, order, smoothed_ordered, axis=1)
    if np.any(np.diff(np.take_along_axis(scores, order, axis=1), axis=1) < -EPSILON):
        raise RuntimeError("prefix-preserving smoothing inverted an order")
    return scores


def _validation_parts(split: SyntheticSplit) -> tuple[np.ndarray, np.ndarray]:
    midpoint = len(split.query_id) // 2
    if midpoint < 2 or len(split.query_id) - midpoint < 2:
        raise ValueError("validation needs at least four rows")
    return np.arange(midpoint), np.arange(midpoint, len(split.query_id))


def _candidate_validation_record(
    candidate: FittedCandidate,
    validation: SyntheticSplit,
    alpha: float,
    *,
    split_seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    scores = score_candidate(candidate, validation, split_seed=split_seed)
    temporary_calibration, assessment = _validation_parts(validation)
    calibration = select_standard_crc_threshold(
        scores[temporary_calibration],
        validation.correctness[temporary_calibration],
        alpha,
    )
    if calibration.feasible:
        mask = scores[assessment] <= calibration.threshold
        events = ~np.any(mask & validation.correctness[assessment].astype(bool), axis=1)
        calls = mask.sum(axis=1)
        record = {
            "validation_certified": True,
            "validation_calls": float(calls.mean()),
            "validation_risk": float(events.mean()),
            "validation_event_count": int(events.sum()),
            "validation_trial_count": int(len(events)),
            "temporary_crc_bound": float(calibration.crc_bound),
            "non_certification_reason": None,
        }
    else:
        record = {
            "validation_certified": False,
            "validation_calls": None,
            "validation_risk": 1.0,
            "validation_event_count": int(len(assessment)),
            "validation_trial_count": int(len(assessment)),
            "temporary_crc_bound": float(calibration.crc_bound),
            "non_certification_reason": calibration.non_certification_reason,
        }
    return record, scores


def select_candidate(
    method: str,
    design: SyntheticSplit,
    validation: SyntheticSplit,
    alpha: float,
    *,
    seed: int,
    latent_draws: int,
) -> tuple[FittedCandidate, list[dict[str, Any]], bool]:
    """Select a fitted candidate on validation before final calibration."""

    trajectory: list[dict[str, Any]] = []
    candidates: list[FittedCandidate] = []
    for grid_index, hyper in enumerate(HYPER_GRIDS[method]):
        candidate = fit_candidate(
            method, hyper, design, seed=seed + grid_index, latent_draws=latent_draws
        )
        record, _ = _candidate_validation_record(
            candidate,
            validation,
            alpha,
            split_seed=seed + 101 + grid_index,
        )
        record.update({"grid_index": grid_index, "hyper": hyper})
        trajectory.append(record)
        candidates.append(candidate)
    certified = [row for row in trajectory if row["validation_certified"]]
    pool = certified if certified else trajectory
    if certified:
        chosen_row = min(
            pool,
            key=lambda row: (
                row["validation_calls"], row["validation_risk"], row["grid_index"]
            ),
        )
    else:
        chosen_row = min(
            pool,
            key=lambda row: (
                row["temporary_crc_bound"], row["grid_index"]
            ),
        )
    return candidates[int(chosen_row["grid_index"])], trajectory, not bool(certified)


def _orders_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=float), axis=1, kind="stable")
    expected = np.arange(scores.shape[1])
    if any(not np.array_equal(np.sort(row), expected) for row in order):
        raise RuntimeError("score order is not a full permutation")
    return order


def _packed_events(events: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(events, dtype=np.uint8).reshape(-1)
    packed = np.packbits(vector, bitorder="little").tobytes()
    return {
        "encoding": "numpy_packbits_little_hex",
        "length": int(len(vector)),
        "packed_hex": packed.hex(),
        "sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
    }


def _kendall_similarity(left: np.ndarray, right: np.ndarray) -> float:
    n = len(left)
    if n < 2:
        return 1.0
    position_left = np.empty(n, dtype=int)
    position_right = np.empty(n, dtype=int)
    position_left[left] = np.arange(n)
    position_right[right] = np.arange(n)
    concordant = 0
    total = n * (n - 1) // 2
    for first in range(n):
        for second in range(first + 1, n):
            concordant += int(
                (position_left[first] - position_left[second])
                * (position_right[first] - position_right[second])
                > 0
            )
    return concordant / total


def _set_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(int(value) for value in np.flatnonzero(left))
    right_set = set(int(value) for value in np.flatnonzero(right))
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def _correlation_root(correlation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(correlation, dtype=float))
    if float(values.min()) < -1e-8:
        raise ValueError("correlation must be positive semidefinite")
    return (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T


def _rank_uniforms(latent: np.ndarray) -> np.ndarray:
    values = np.asarray(latent, dtype=float)
    uniforms = np.empty_like(values)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        ranks = np.empty(values.shape[0], dtype=int)
        ranks[order] = np.arange(values.shape[0])
        uniforms[:, column] = (ranks + 0.5) / values.shape[0]
    return uniforms


def _latent_uniform_bank(candidate: FittedCandidate) -> np.ndarray:
    """Reuse repository frozen samplers and expose one rank-uniform draw bank."""

    n_models = candidate.design_y.shape[1]
    root = _correlation_root(candidate.raw_correlation)
    if candidate.method == "latent_normal":
        sampler = FrozenGaussianSampler(
            n_models,
            n_draws=candidate.latent_draws,
            design_seed=candidate.fit_seed,
        )
        latent = sampler.design_latent_bank @ root.T
    elif candidate.method == "heavy_tailed":
        sampler = FrozenStudentTSampler(
            n_models,
            degrees_of_freedom=float(candidate.hyper),
            n_draws=candidate.latent_draws,
            design_seed=candidate.fit_seed,
        )
        latent = (sampler.design_normal_bank @ root.T) / np.sqrt(
            sampler.design_scale_bank[:, None] / float(candidate.hyper)
        )
    else:
        raise ValueError("latent uniform banks require a latent method")
    return _rank_uniforms(latent)


def _prefix_predictions(
    candidate: FittedCandidate,
    split: SyntheticSplit,
    orders: np.ndarray,
    max_depth: int,
) -> np.ndarray:
    """Predict all requested prefix risks without repeated latent resampling."""

    method = candidate.method
    p_hat = candidate.marginal_fit.predict(split.difficulty)
    if method == "oracle_marginal_exact_independence":
        ordered_miss = np.take_along_axis(1.0 - split.probabilities, orders, axis=1)
        return np.cumprod(ordered_miss[:, :max_depth], axis=1)
    if method in {"exact_independence", "pairwise_count", "error_correlation"}:
        ordered_miss = np.take_along_axis(1.0 - p_hat, orders, axis=1)
        return np.cumprod(ordered_miss[:, :max_depth], axis=1)
    if method == "empirical_tally":
        assert candidate.fixed_order is not None
        values = []
        for depth in range(1, max_depth + 1):
            member = candidate.fixed_order[:depth]
            values.append(float(np.mean(np.all(candidate.design_y[:, member] == 0, axis=1))))
        return np.broadcast_to(np.asarray(values), (len(orders), max_depth)).copy()
    if method in {"latent_normal", "heavy_tailed"}:
        uniforms = _latent_uniform_bank(candidate)
        predictions = np.empty((len(orders), max_depth), dtype=float)
        for row in range(len(orders)):
            failures = uniforms <= (1.0 - p_hat[row])[None, :]
            ordered = failures[:, orders[row, :max_depth]]
            predictions[row] = np.logical_and.accumulate(ordered, axis=1).mean(axis=0)
        return predictions
    if method == "query_local_empirical_coverage":
        neighbors = _nearest_labels(candidate, split.difficulty, int(candidate.hyper))
        predictions = np.empty((len(orders), max_depth), dtype=float)
        for row in range(len(orders)):
            ordered = neighbors[row][:, orders[row, :max_depth]] == 0
            predictions[row] = np.logical_and.accumulate(ordered, axis=1).mean(axis=0)
        return predictions
    raise ValueError(f"unknown method {method!r}")


def evaluate_candidate(
    candidate: FittedCandidate,
    final_calibration: SyntheticSplit,
    test: SyntheticSplit,
    alpha: float,
    *,
    split_seed: int,
    resample_mask: np.ndarray | None = None,
    oracle_order: np.ndarray | None = None,
    include_diagnostics: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run one final calibration and test evaluation."""

    calibration_scores = score_candidate(candidate, final_calibration, split_seed=split_seed)
    test_scores = score_candidate(candidate, test, split_seed=split_seed + 1)
    calibration = select_standard_crc_threshold(
        calibration_scores, final_calibration.correctness, alpha
    )
    full_pool_events = np.all(test.correctness == 0, axis=1)
    full_pool_floor = float(full_pool_events.mean())
    output: dict[str, Any] = {
        "certified": bool(calibration.feasible),
        "final_calibration_count": 1,
        "final_calibration_query_count": int(len(final_calibration.query_id)),
        "final_calibration_query_ids_sha256": _hash_json(final_calibration.query_id.tolist()),
        "test_query_count": int(len(test.query_id)),
        "test_query_ids_sha256": _hash_json(test.query_id.tolist()),
        "complete_pool_floor": full_pool_floor,
        "complete_pool_event_count": int(full_pool_events.sum()),
        "complete_pool_trial_count": int(len(full_pool_events)),
        "non_certification_reason": calibration.non_certification_reason,
        "crc_bound": float(calibration.crc_bound),
    }
    if not calibration.feasible:
        output.update(
            threshold=None,
            calls=None,
            call_sum=None,
            event_count=None,
            trial_count=int(len(test.query_id)),
            risk=None,
            event_vector=None,
            selected_sets_sha256=None,
            prefix_joint_miss_mae=None,
            prefix_joint_miss_rmse=None,
            prefix_order_kendall_similarity=None,
            prefix_top_k_jaccard=None,
            selected_model_jaccard_resample=None,
        )
        return output, np.zeros_like(test.correctness, dtype=bool), test_scores
    mask = test_scores <= calibration.threshold
    if np.any(mask.sum(axis=1) < 1):
        raise RuntimeError("certified prefix family emitted an empty selected set")
    events = ~np.any(mask & test.correctness.astype(bool), axis=1)
    calls = mask.sum(axis=1)
    if int(events.sum()) != int(round(float(events.mean()) * len(events))):
        raise RuntimeError("all-miss event arithmetic failed")
    selected_hash = hashlib.sha256(
        np.packbits(mask, bitorder="little").tobytes()
    ).hexdigest()
    output.update(
        threshold=float(calibration.threshold),
        calls=float(calls.mean()),
        call_sum=int(calls.sum()),
        event_count=int(events.sum()),
        trial_count=int(len(events)),
        risk=float(events.mean()),
        event_vector=_packed_events(events),
        selected_sets_sha256=selected_hash,
    )
    if not include_diagnostics:
        output.update(
            prefix_joint_miss_mae=None,
            prefix_joint_miss_rmse=None,
            prefix_order_kendall_similarity=None,
            prefix_top_k_jaccard=None,
            selected_model_jaccard_resample=None,
        )
        return output, mask, test_scores
    orders = _orders_from_scores(test_scores)
    oracle = (
        oracle_orders(configured_regime(candidate), test.probabilities)
        if oracle_order is None
        else np.asarray(oracle_order, dtype=int)
    )
    if oracle.shape != orders.shape:
        raise ValueError("oracle order must align with test score orders")
    kendall = np.mean(
        [_kendall_similarity(orders[row], oracle[row]) for row in range(len(orders))]
    )
    top_k = min(3, test.correctness.shape[1])
    top_jaccard = np.mean(
        [
            len(set(orders[row, :top_k]) & set(oracle[row, :top_k]))
            / len(set(orders[row, :top_k]) | set(oracle[row, :top_k]))
            for row in range(len(orders))
        ]
    )
    max_depth = min(6, test.correctness.shape[1])
    predicted = _prefix_predictions(candidate, test, orders, max_depth)
    truth = np.empty_like(predicted)
    for row in range(len(orders)):
        for depth in range(1, max_depth + 1):
            truth[row, depth - 1] = exact_all_miss_probability(
                configured_regime(candidate),
                test.probabilities[row : row + 1],
                orders[row, :depth],
            )[0]
    difference = predicted - truth
    output.update(
        prefix_joint_miss_mae=float(np.mean(np.abs(difference))),
        prefix_joint_miss_rmse=float(np.sqrt(np.mean(difference * difference))),
        prefix_order_kendall_similarity=float(kendall),
        prefix_top_k_jaccard=float(top_jaccard),
        selected_model_jaccard_resample=(
            None
            if resample_mask is None
            else float(
                np.mean(
                    [
                        _set_jaccard(left, right)
                        for left, right in zip(mask, resample_mask)
                    ]
                )
            )
        ),
    )
    for key in (
        "prefix_order_kendall_similarity",
        "prefix_top_k_jaccard",
        "selected_model_jaccard_resample",
    ):
        value = output[key]
        if value is not None and not 0.0 <= value <= 1.0:
            raise RuntimeError(f"stability metric {key} left the unit interval")
    return output, mask, test_scores


def configured_regime(candidate: FittedCandidate) -> str:
    return candidate.regime


def _hash_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def run_seed(config: ExperimentConfig, seed_index: int) -> dict[str, Any]:
    """Run all methods and both targets for one paired redraw."""

    config.validate()
    design = simulate_split(config, seed_index, "design")
    design_resample = simulate_split(config, seed_index, "design_resample")
    validation = simulate_split(config, seed_index, "validation")
    final_calibration = simulate_split(config, seed_index, "final_calibration")
    test = simulate_split(config, seed_index, "test")
    test_oracle_order = oracle_orders(config.regime, test.probabilities)
    split_hashes = {
        name: _hash_json(split.query_id.tolist())
        for name, split in (
            ("design", design),
            ("design_resample", design_resample),
            ("validation", validation),
            ("final_calibration", final_calibration),
            ("test", test),
        )
    }
    if len(set(split_hashes.values())) != len(split_hashes):
        raise RuntimeError("split identity collision")
    alpha_records: dict[str, Any] = {}
    for alpha_index, alpha in enumerate(config.alphas):
        methods: dict[str, Any] = {}
        for method_index, method in enumerate(METHODS):
            method_seed = config.base_seed + seed_index * 10_000 + alpha_index * 1000 + method_index * 100
            chosen, trajectory, fallback = select_candidate(
                method,
                design,
                validation,
                alpha,
                seed=method_seed,
                latent_draws=config.latent_draws,
            )
            paired, paired_trajectory, paired_fallback = select_candidate(
                method,
                design_resample,
                validation,
                alpha,
                seed=method_seed + 50_000,
                latent_draws=config.latent_draws,
            )
            paired_result, paired_mask, _ = evaluate_candidate(
                paired,
                final_calibration,
                test,
                alpha,
                split_seed=method_seed + 70_000,
                include_diagnostics=False,
            )
            result, _, _ = evaluate_candidate(
                chosen,
                final_calibration,
                test,
                alpha,
                split_seed=method_seed + 80_000,
                resample_mask=paired_mask if paired_result["certified"] else None,
                oracle_order=test_oracle_order,
            )
            result.update(
                chosen_hyper=chosen.hyper,
                selection_source="validation_only",
                selection_protocol="temporary_validation_calibration_then_validation_assessment_then_freeze_then_one_final_calibration_then_test",
                selection_fallback_used=bool(fallback),
                validation_trajectory=trajectory,
                paired_resample_chosen_hyper=paired.hyper,
                paired_resample_selection_fallback_used=bool(paired_fallback),
                paired_resample_validation_trajectory_sha256=_hash_json(paired_trajectory),
                preserve_prefixes=True,
            )
            methods[method] = result
        if set(methods) != set(METHODS):
            raise RuntimeError("method registry mismatch")
        alpha_records[format(alpha, ".12g")] = {"alpha": float(alpha), "methods": methods}
    return {
        "seed_index": int(seed_index),
        "stream_seeds": {
            name: [config.base_seed, seed_index, split_id]
            for name, split_id in SPLIT_IDS.items()
        },
        "split_query_ids_sha256": split_hashes,
        "alphas": alpha_records,
    }


def correlation_diagnostics(split: SyntheticSplit) -> dict[str, float]:
    raw = _raw_failure_correlation(split.correctness)
    variance = np.sqrt(np.maximum(split.probabilities * (1.0 - split.probabilities), 1e-8))
    residual = (split.correctness - split.probabilities) / variance
    residual = np.corrcoef(residual, rowvar=False)
    residual = np.nan_to_num(residual, nan=0.0)
    mask = ~np.eye(raw.shape[0], dtype=bool)
    triple = (0, 1, 2)
    triple_truth = float(exact_all_miss_probability(
        "higher_order_pairwise_invisible" if split.probabilities.shape[1] >= 3 else "conditional_independence_heterogeneous_difficulty",
        np.full((1, split.probabilities.shape[1]), 0.5),
        triple if split.probabilities.shape[1] >= 3 else (0,),
    )[0])
    return {
        "pooled_failure_correlation_mean": float(raw[mask].mean()),
        "pooled_failure_absolute_correlation_mean": float(np.abs(raw[mask]).mean()),
        "standardized_residual_correlation_mean": float(residual[mask].mean()),
        "standardized_residual_absolute_correlation_mean": float(np.abs(residual[mask]).mean()),
        "first_pair_residual_correlation": float(residual[0, 1]),
        "first_triple_all_miss_rate": float(np.mean(np.all(split.correctness[:, :3] == 0, axis=1))),
        "first_triple_independence_product": float(np.mean(np.prod(1.0 - split.probabilities[:, :3], axis=1))),
        "pairwise_invisible_reference_triple_truth": triple_truth,
    }


def ground_truth_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Record equations, parameters, analytic signatures, and high-MC checks."""

    split = simulate_split(config, config.seed_start, "ground_truth")
    equations = {
        "conditional_independence_heterogeneous_difficulty": "Y_j | X independently follows Bernoulli(sigmoid(a_j - b_j X + c_j (X^2 - 1))).",
        "complementary_specialists_negative_residual": "Within pair (2g,2g+1), Y_2g=1[U<p_2g(X)] and Y_2g+1=1[U>1-p_2g+1(X)]. P(both correct|X)=max(p_a+p_b-1,0).",
        "redundant_families_positive_residual": "Within each four-model family, Y_j=1[U_family<p_j(X)]. P(all selected miss|X)=1-max_j p_j(X).",
        "higher_order_pairwise_invisible": "Each triple is (A,B,A xor B) for independent fair bits A and B. Every pair is independent, while P(all three miss)=1/4 instead of 1/8.",
        "mixed_easy_hard_pooled_correlation_no_residual": "X is observed easy or hard with equal probability. Conditional on X, Y_j are independent Bernoulli with p_j(easy) near 0.85 and p_j(hard) near 0.30.",
    }
    analytic = {
        "conditional_independence_heterogeneous_difficulty": {"conditional_residual_pair_sign": "zero"},
        "complementary_specialists_negative_residual": {"within_pair_residual_sign": "negative"},
        "redundant_families_positive_residual": {"within_family_residual_sign": "positive"},
        "higher_order_pairwise_invisible": {
            "pairwise_correlation": 0.0,
            "triple_all_miss": 0.25,
            "independence_triple_all_miss": 0.125,
            "higher_order_excess": 0.125,
        },
        "mixed_easy_hard_pooled_correlation_no_residual": {
            "pooled_pair_sign": "positive",
            "conditional_residual_pair_sign": "zero",
        },
    }
    regime_parameters = {
        "conditional_independence_heterogeneous_difficulty": {
            "difficulty": "standard_normal",
            "intercept": "1.25 - model_index/(pool_size-1)",
            "difficulty_loading": "0.65 + 0.55*(model_index mod 5)/4",
            "curvature": "0.10*sin(model_index+0.7)",
            "probability_clip": [0.05, 0.95],
        },
        "complementary_specialists_negative_residual": {
            "difficulty": "standard_normal",
            "intercept": "0.15 + 0.18*cos(model_index+0.4)",
            "difficulty_loading": "alternating +1.25 and -1.25",
            "curvature": "0.05*sin(model_index+1)",
            "pair_width": 2,
            "probability_clip": [0.08, 0.88],
        },
        "redundant_families_positive_residual": {
            "difficulty": "standard_normal",
            "marginal_formula": "same heterogeneous logistic formula as conditional independence",
            "family_width": 4,
            "shared_uniform_per_family": True,
        },
        "higher_order_pairwise_invisible": {
            "difficulty": "standard_normal_unused",
            "block_width": 3,
            "bit_probabilities": [0.5, 0.5],
            "block_support": ["000", "011", "101", "110"],
        },
        "mixed_easy_hard_pooled_correlation_no_residual": {
            "difficulty_support": [-1.0, 1.0],
            "easy_probability": 0.5,
            "correctness_probability": "0.30 + 0.55*easy + 0.06*cos(1.7*(model_index+1))",
            "probability_clip": [0.08, 0.94],
        },
    }
    return {
        "regime": config.regime,
        "equation": equations[config.regime],
        "parameters": regime_parameters[config.regime],
        "analytic_signature": analytic[config.regime],
        "diagnostic_draws": int(config.diagnostic_draws),
        "diagnostic_seed": [config.base_seed, config.seed_start, SPLIT_IDS["ground_truth"]],
        "high_monte_carlo_diagnostics": correlation_diagnostics(split),
    }


def deterministic_paired_bootstrap(
    values: Sequence[float],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any] | None:
    """Return a deterministic paired bootstrap interval over seed differences."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite vector")
    if repetitions < 2:
        raise ValueError("bootstrap repetitions must be at least two")
    if not len(array):
        return None
    generator = np.random.default_rng(int(seed))
    means = array[
        generator.integers(0, len(array), size=(int(repetitions), len(array)))
    ].mean(axis=1)
    return {
        "paired_seed_count": int(len(array)),
        "mean_difference": float(array.mean()),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "difference_direction": "candidate_minus_exact_independence",
    }


def summarize(config: ExperimentConfig, seed_results: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(config.alphas):
        alpha_key = format(alpha, ".12g")
        for method_index, method in enumerate(METHODS):
            rows = [result["alphas"][alpha_key]["methods"][method] for result in seed_results]
            certified = [row for row in rows if row["certified"]]
            summaries.append({
                "alpha": float(alpha),
                "method": method,
                "seed_count": len(rows),
                "certified_seed_count": len(certified),
                "certification_rate": len(certified) / len(rows),
                "mean_calls": None if not certified else float(np.mean([row["calls"] for row in certified])),
                "risk": None if not certified else float(sum(row["event_count"] for row in certified) / sum(row["trial_count"] for row in certified)),
                "mean_prefix_joint_miss_mae": None if not certified else float(np.mean([row["prefix_joint_miss_mae"] for row in certified])),
                "mean_order_kendall_similarity": None if not certified else float(np.mean([row["prefix_order_kendall_similarity"] for row in certified])),
                "mean_top_k_jaccard": None if not certified else float(np.mean([row["prefix_top_k_jaccard"] for row in certified])),
                "mean_selected_model_jaccard_resample": None if not certified else float(np.mean([row["selected_model_jaccard_resample"] for row in certified if row["selected_model_jaccard_resample"] is not None])) if any(row["selected_model_jaccard_resample"] is not None for row in certified) else None,
            })
            if method == "exact_independence":
                continue
            paired = [
                (
                    row["alphas"][alpha_key]["methods"][method],
                    row["alphas"][alpha_key]["methods"]["exact_independence"],
                )
                for row in seed_results
            ]
            paired = [
                (left, right)
                for left, right in paired
                if left["certified"] and right["certified"]
            ]
            metric_fields = {
                "calls": "calls",
                "risk": "risk",
                "prefix_joint_miss_mae": "prefix_joint_miss_mae",
                "order_kendall_similarity": "prefix_order_kendall_similarity",
                "top_k_jaccard": "prefix_top_k_jaccard",
                "selected_model_jaccard_resample": "selected_model_jaccard_resample",
            }
            seed = config.base_seed + alpha_index * 10_000 + method_index * 100
            bootstrap_metrics: dict[str, Any] = {}
            for metric_offset, (output_name, field) in enumerate(metric_fields.items()):
                differences = [
                    float(left[field]) - float(right[field])
                    for left, right in paired
                    if left[field] is not None and right[field] is not None
                ]
                bootstrap_metrics[output_name] = deterministic_paired_bootstrap(
                    differences,
                    seed=seed + metric_offset,
                    repetitions=config.bootstrap_repetitions,
                )
            contrasts.append({
                "alpha": float(alpha),
                "candidate": method,
                "reference": "exact_independence",
                "jointly_certified_seed_count": len(paired),
                **bootstrap_metrics,
            })
    mechanisms: list[dict[str, Any]] = []
    for method in DEPENDENCE_AWARE_METHODS:
        qualifying = []
        for contrast in contrasts:
            if contrast["candidate"] != method or contrast["calls"] is None:
                continue
            if contrast["calls"]["mean_difference"] < 0.0 and contrast["risk"]["mean_difference"] <= 0.0:
                qualifying.append(contrast)
        if qualifying:
            first = min(qualifying, key=lambda row: list(config.alphas).index(row["alpha"]))
            mechanisms.append({
                "candidate": method,
                "reference": "exact_independence",
                "first_declared_alpha": first["alpha"],
                "call_difference": first["calls"]["mean_difference"],
                "risk_difference": first["risk"]["mean_difference"],
                "criterion": "lower paired mean calls with no higher paired mean standard all-miss risk",
            })
    return summaries, mechanisms + contrasts


def validate_payload(payload: Mapping[str, Any]) -> None:
    config = payload["config"]
    expected_seeds = int(config["seed_count"])
    seed_results = payload.get("seed_results", [])
    if len(seed_results) != expected_seeds:
        raise RuntimeError("missing seed cells")
    alpha_keys = {format(float(value), ".12g") for value in config["alphas"]}
    certification: dict[tuple[str, str], list[bool]] = {
        (alpha, method): [] for alpha in alpha_keys for method in METHODS
    }
    structural: dict[str, list[bool]] = {alpha: [] for alpha in alpha_keys}
    for seed in seed_results:
        if set(seed["alphas"]) != alpha_keys:
            raise RuntimeError("missing alpha cell")
        for alpha_key, cell in seed["alphas"].items():
            if set(cell["methods"]) != set(METHODS):
                raise RuntimeError("missing method cell")
            alpha = float(alpha_key)
            structural[alpha_key].append(
                all(row["complete_pool_floor"] > alpha for row in cell["methods"].values())
            )
            for method, row in cell["methods"].items():
                certification[(alpha_key, method)].append(bool(row["certified"]))
                if row["certified"]:
                    if row["event_count"] != int(round(row["risk"] * row["trial_count"])):
                        raise RuntimeError("event count and risk disagree")
                    if row["call_sum"] < row["trial_count"] or row["call_sum"] > row["trial_count"] * int(config["pool_size"]):
                        raise RuntimeError("call sum is outside valid endpoints")
    for alpha_key in alpha_keys:
        outside_structural_infeasibility = not all(structural[alpha_key])
        if outside_structural_infeasibility:
            for method in METHODS:
                if not any(certification[(alpha_key, method)]):
                    raise RuntimeError(
                        f"zero certification outside structural infeasibility for {method} at alpha {alpha_key}"
                    )


def run_experiment(config: ExperimentConfig, *, host: str, stamp: str) -> dict[str, Any]:
    config.validate()
    seed_results = [
        run_seed(config, seed_index)
        for seed_index in range(config.seed_start, config.seed_start + config.seed_count)
    ]
    summary, contrasts = summarize(config, seed_results)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": 64,
        "host": host,
        "stamp": stamp,
        "config": {
            **config.__dict__,
            "alphas": list(config.alphas),
        },
        "full_manifest": {
            "regimes": list(REGIMES),
            "pool_sizes": list(POOL_SIZES),
            "design_label_budgets": list(DESIGN_BUDGETS),
            "independent_grid": [list(cell) for cell in full_grid()],
            "target_alphas": list(TARGET_ALPHAS),
            "primary_alpha": PRIMARY_ALPHA,
            "secondary_alpha": SECONDARY_ALPHA,
            "seed_count_per_full_cell": 20,
            "one_cell_shard_per_process": True,
            "preserve_prefixes": True,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "methods": list(METHODS),
            "hyperparameter_grids": {name: list(values) for name, values in HYPER_GRIDS.items()},
        },
        "ground_truth": ground_truth_manifest(config),
        "seed_results": seed_results,
        "summary": summary,
        "mechanism_contrasts_and_paired_bootstrap": contrasts,
    }
    validate_payload(payload)
    return payload


def _safe_slug(value: str, name: str) -> str:
    text = str(value)
    if not text or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"{name} must use letters, digits, dot, underscore, or hyphen")
    return text


def output_path(out: Path, config: ExperimentConfig, *, host: str, stamp: str) -> Path:
    filename = (
        f"issue64_{host}_{stamp}_{config.regime}_p{config.pool_size}_"
        f"b{config.design_budget}_s{config.seed_start}n{config.seed_count}.json"
    )
    return Path(out) / filename if Path(out).suffix == "" else Path(out)


def atomic_write_no_overwrite(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish canonical JSON atomically and refuse target collisions."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {target}")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite immutable result {target}") from error
        directory = os.open(target.parent, getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--regime", required=True, choices=REGIMES)
    parser.add_argument("--pool-size", required=True, type=int, choices=POOL_SIZES)
    parser.add_argument("--design-budget", required=True, type=int, choices=DESIGN_BUDGETS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    host = _safe_slug(arguments.host, "host")
    stamp = _safe_slug(arguments.stamp, "stamp")
    config = ExperimentConfig(
        regime=arguments.regime,
        pool_size=arguments.pool_size,
        design_budget=arguments.design_budget,
        seed_start=arguments.seed_start,
        seed_count=arguments.seed_count,
    )
    destination = output_path(arguments.out, config, host=host, stamp=stamp)
    if destination.exists():
        raise SystemExit(f"refusing to overwrite immutable result {destination}")
    payload = run_experiment(config, host=host, stamp=stamp)
    atomic_write_no_overwrite(destination, payload)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
