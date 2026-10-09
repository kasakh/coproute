#!/usr/bin/env python3
"""Reproducible conditional-independence null experiment.

The data-generating process has one observed query difficulty variable X and
heterogeneous model-specific correctness probabilities p_j(X). Model outcomes
are sampled independently conditional on X. Marginalizing over X creates raw
correlation even though the conditional residual correlation is zero.

Three nested set-ordering rules are compared:

* exact_factorized uses the true conditional marginals and no dependence term.
* pooled_raw_correlation adds a design-estimated raw-correlation penalty.
* residual_correlation adds the same penalty using design residuals Y - p(X).

The correlation penalty is selected on validation data. Final calibration then
chooses the smallest depth satisfying the finite-sample corrected standard
all-miss loss target. Test data is used only for reporting. Fixed-depth results
compare all methods at exactly matched call counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "synthetic_conditional_independence.v1"
METHODS = (
    "exact_factorized",
    "pooled_raw_correlation",
    "residual_correlation",
)
SPLIT_IDS = {
    "design": 11,
    "validation": 23,
    "final_calibration": 37,
    "test": 53,
}
SUMMARY_COLUMNS = (
    "section",
    "method",
    "depth",
    "metric",
    "n",
    "mean",
    "standard_error",
    "ci95_low",
    "ci95_high",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings needed to reproduce the v1 experiment."""

    base_seed: int = 24_051_986
    seed_count: int = 128
    n_models: int = 8
    design_size: int = 1_500
    validation_size: int = 1_000
    final_calibration_size: int = 1_500
    test_size: int = 4_000
    target_alpha: float = 0.10
    fixed_depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    penalty_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    probability_floor: float = 0.03
    model_intercepts: tuple[float, ...] = (1.55, 1.30, 1.10, 0.90, 0.72, 0.50, 0.28, 0.05)
    difficulty_loadings: tuple[float, ...] = (0.72, 0.90, 1.08, 1.28, 1.48, 1.12, 0.86, 0.64)
    curvature: tuple[float, ...] = (-0.10, 0.06, 0.12, -0.05, 0.09, -0.12, 0.04, 0.10)

    def validate(self) -> None:
        if self.seed_count < 1:
            raise ValueError("seed_count must be positive")
        if self.n_models < 2:
            raise ValueError("n_models must be at least two")
        if not 0.0 < self.target_alpha < 1.0:
            raise ValueError("target_alpha must lie strictly between zero and one")
        if not 0.0 < self.probability_floor < 0.5:
            raise ValueError("probability_floor must lie strictly between zero and 0.5")
        if any(size < 2 for size in self.split_sizes().values()):
            raise ValueError("every split must have at least two queries")
        if tuple(sorted(set(self.fixed_depths))) != self.fixed_depths:
            raise ValueError("fixed_depths must be unique and increasing")
        if not self.fixed_depths or self.fixed_depths[-1] > self.n_models:
            raise ValueError("fixed_depths must be nonempty and at most n_models")
        if not self.penalty_grid or any(value <= 0.0 for value in self.penalty_grid):
            raise ValueError("penalty_grid must contain positive values")
        for name, values in (
            ("model_intercepts", self.model_intercepts),
            ("difficulty_loadings", self.difficulty_loadings),
            ("curvature", self.curvature),
        ):
            if len(values) != self.n_models:
                raise ValueError(f"{name} must contain n_models values")

    def split_sizes(self) -> dict[str, int]:
        return {
            "design": self.design_size,
            "validation": self.validation_size,
            "final_calibration": self.final_calibration_size,
            "test": self.test_size,
        }


FROZEN_CONFIG = ExperimentConfig()


@dataclass(frozen=True)
class SyntheticSplit:
    difficulty: np.ndarray
    probabilities: np.ndarray
    correctness: np.ndarray


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def true_marginals(difficulty: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Return heterogeneous true p_j(X) for every query and model."""
    x = np.asarray(difficulty, dtype=float).reshape(-1, 1)
    intercepts = np.asarray(config.model_intercepts, dtype=float).reshape(1, -1)
    loadings = np.asarray(config.difficulty_loadings, dtype=float).reshape(1, -1)
    curvature = np.asarray(config.curvature, dtype=float).reshape(1, -1)
    logits = intercepts - x * loadings + curvature * (x * x - 1.0)
    probabilities = _sigmoid(logits)
    floor = config.probability_floor
    return np.clip(probabilities, floor, 1.0 - floor)


def simulate_split(
    config: ExperimentConfig,
    seed_index: int,
    split_name: str,
    size: int | None = None,
) -> SyntheticSplit:
    """Generate one split from a stream independent of every other split."""
    if split_name not in SPLIT_IDS:
        raise ValueError(f"unknown split name {split_name!r}")
    split_size = config.split_sizes()[split_name] if size is None else int(size)
    seed_sequence = np.random.SeedSequence(
        [config.base_seed, int(seed_index), SPLIT_IDS[split_name]]
    )
    generator = np.random.default_rng(seed_sequence)
    difficulty = generator.normal(loc=0.0, scale=1.0, size=split_size)
    probabilities = true_marginals(difficulty, config)
    correctness = (generator.random(probabilities.shape) < probabilities).astype(np.int8)
    return SyntheticSplit(difficulty, probabilities, correctness)


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("correlation input must have at least two rows and columns")
    result = np.corrcoef(matrix, rowvar=False)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(result, 1.0)
    return result


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def greedy_orders(
    probabilities: np.ndarray,
    correlation: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Build query-specific nested orders using a transparent correlation penalty.

    At each depth, the next model minimizes its log conditional failure
    probability plus penalty times its average correlation with selected models.
    Penalty zero is exact factorized independence and therefore sorts by p_j(X).
    """
    probabilities = np.asarray(probabilities, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    n_queries, n_models = probabilities.shape
    if correlation.shape != (n_models, n_models):
        raise ValueError("correlation shape does not match probabilities")
    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")

    base_score = np.log(np.clip(1.0 - probabilities, 1e-12, 1.0))
    selected = np.zeros((n_queries, n_models), dtype=bool)
    orders = np.empty((n_queries, n_models), dtype=np.int64)
    selected_float = np.zeros((n_queries, n_models), dtype=float)

    for step in range(n_models):
        if step == 0 or penalty == 0.0:
            score = base_score.copy()
        else:
            average_correlation = (selected_float @ correlation) / float(step)
            score = base_score + penalty * average_correlation
        score[selected] = np.inf
        choice = np.argmin(score, axis=1)
        orders[:, step] = choice
        rows = np.arange(n_queries)
        selected[rows, choice] = True
        selected_float[rows, choice] = 1.0
    return orders


def cumulative_standard_loss(correctness: np.ndarray, orders: np.ndarray) -> np.ndarray:
    ordered = np.take_along_axis(np.asarray(correctness), orders, axis=1)
    return np.logical_and.accumulate(ordered == 0, axis=1)


def cumulative_population_risk(probabilities: np.ndarray, orders: np.ndarray) -> np.ndarray:
    ordered_failure = np.take_along_axis(1.0 - np.asarray(probabilities), orders, axis=1)
    return np.cumprod(ordered_failure, axis=1)


def choose_penalty(
    validation: SyntheticSplit,
    correlation: np.ndarray,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float]]:
    """Select a penalty by mean validation standard loss over matched depths."""
    candidates: list[tuple[float, float]] = []
    scores: dict[str, float] = {}
    for penalty in config.penalty_grid:
        orders = greedy_orders(validation.probabilities, correlation, penalty)
        losses = cumulative_standard_loss(validation.correctness, orders)
        depth_risks = [float(losses[:, depth - 1].mean()) for depth in config.fixed_depths]
        score = float(np.mean(depth_risks))
        scores[_format_penalty(penalty)] = score
        candidates.append((score, penalty))
    _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return float(selected), scores


def calibrated_depth(
    calibration_loss: np.ndarray,
    alpha: float,
) -> tuple[int, bool, float]:
    """Choose the first depth whose corrected standard-loss estimate meets alpha."""
    losses = np.asarray(calibration_loss, dtype=bool)
    n_queries, n_models = losses.shape
    for index in range(n_models):
        events = int(losses[:, index].sum())
        corrected = (events + 1.0) / (n_queries + 1.0)
        if corrected <= alpha:
            return index + 1, True, float(corrected)
    events = int(losses[:, -1].sum())
    corrected = (events + 1.0) / (n_queries + 1.0)
    return n_models, False, float(corrected)


def _binomial_standard_error(rate: float, n: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / float(n))


def _method_orders(
    split: SyntheticSplit,
    raw_correlation: np.ndarray,
    residual_correlation: np.ndarray,
    selected_penalties: dict[str, float],
) -> dict[str, np.ndarray]:
    identity = np.eye(split.probabilities.shape[1], dtype=float)
    return {
        "exact_factorized": greedy_orders(split.probabilities, identity, 0.0),
        "pooled_raw_correlation": greedy_orders(
            split.probabilities,
            raw_correlation,
            selected_penalties["pooled_raw_correlation"],
        ),
        "residual_correlation": greedy_orders(
            split.probabilities,
            residual_correlation,
            selected_penalties["residual_correlation"],
        ),
    }


def run_seed(config: ExperimentConfig, seed_index: int) -> dict[str, Any]:
    """Run all four splits and all methods for one seed."""
    config.validate()
    design = simulate_split(config, seed_index, "design")
    validation = simulate_split(config, seed_index, "validation")
    final_calibration = simulate_split(config, seed_index, "final_calibration")
    test = simulate_split(config, seed_index, "test")

    raw_correlation = correlation_matrix(design.correctness)
    residuals = design.correctness.astype(float) - design.probabilities
    residual_correlation = correlation_matrix(residuals)
    raw_off_diagonal = off_diagonal_values(raw_correlation)
    residual_off_diagonal = off_diagonal_values(residual_correlation)

    raw_penalty, raw_validation_scores = choose_penalty(
        validation, raw_correlation, config
    )
    residual_penalty, residual_validation_scores = choose_penalty(
        validation, residual_correlation, config
    )
    selected_penalties = {
        "exact_factorized": 0.0,
        "pooled_raw_correlation": raw_penalty,
        "residual_correlation": residual_penalty,
    }

    test_orders = _method_orders(
        test, raw_correlation, residual_correlation, selected_penalties
    )
    test_losses = {
        method: cumulative_standard_loss(test.correctness, orders)
        for method, orders in test_orders.items()
    }
    test_population = {
        method: cumulative_population_risk(test.probabilities, orders)
        for method, orders in test_orders.items()
    }

    fixed_depth: list[dict[str, Any]] = []
    exact_empirical: dict[int, float] = {}
    exact_population: dict[int, float] = {}
    for depth in config.fixed_depths:
        exact_empirical[depth] = float(
            test_losses["exact_factorized"][:, depth - 1].mean()
        )
        exact_population[depth] = float(
            test_population["exact_factorized"][:, depth - 1].mean()
        )

    for method in METHODS:
        for depth in config.fixed_depths:
            empirical = float(test_losses[method][:, depth - 1].mean())
            population = float(test_population[method][:, depth - 1].mean())
            population_excess = population - exact_population[depth]
            empirical_difference = empirical - exact_empirical[depth]
            false_win = bool(
                method != "exact_factorized"
                and empirical_difference < 0.0
                and population_excess >= -1e-12
            )
            fixed_depth.append(
                {
                    "method": method,
                    "depth": depth,
                    "calls": float(depth),
                    "empirical_standard_loss_risk": empirical,
                    "empirical_standard_loss_se": _binomial_standard_error(
                        empirical, config.test_size
                    ),
                    "conditional_population_risk": population,
                    "empirical_risk_difference_vs_exact": empirical_difference,
                    "population_risk_excess_vs_exact": population_excess,
                    "false_dependence_win": false_win,
                }
            )

    calibration_orders = _method_orders(
        final_calibration,
        raw_correlation,
        residual_correlation,
        selected_penalties,
    )
    calibrated: list[dict[str, Any]] = []
    for method in METHODS:
        calibration_loss = cumulative_standard_loss(
            final_calibration.correctness, calibration_orders[method]
        )
        depth, certified, corrected_calibration_risk = calibrated_depth(
            calibration_loss, config.target_alpha
        )
        empirical = float(test_losses[method][:, depth - 1].mean())
        population = float(test_population[method][:, depth - 1].mean())
        calibrated.append(
            {
                "method": method,
                "selected_depth": depth,
                "calls": float(depth),
                "certified": certified,
                "corrected_final_calibration_risk": corrected_calibration_risk,
                "test_empirical_standard_loss_risk": empirical,
                "test_empirical_standard_loss_se": _binomial_standard_error(
                    empirical, config.test_size
                ),
                "test_conditional_population_risk": population,
            }
        )

    return {
        "seed_index": int(seed_index),
        "diagnostics": {
            "pooled_raw_off_diagonal_correlation": float(raw_off_diagonal.mean()),
            "pooled_raw_off_diagonal_absolute_correlation": float(
                np.abs(raw_off_diagonal).mean()
            ),
            "residual_off_diagonal_correlation": float(
                residual_off_diagonal.mean()
            ),
            "residual_off_diagonal_absolute_correlation": float(
                np.abs(residual_off_diagonal).mean()
            ),
        },
        "selected_penalties": selected_penalties,
        "validation_scores": {
            "pooled_raw_correlation": raw_validation_scores,
            "residual_correlation": residual_validation_scores,
        },
        "fixed_depth": fixed_depth,
        "calibrated_standard_loss": calibrated,
    }


def _mean_interval(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    mean = float(array.mean())
    standard_error = (
        float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    )
    radius = 1.96 * standard_error
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
    }


def _wilson_interval(successes: int, trials: int) -> dict[str, float | int]:
    if trials < 1:
        raise ValueError("trials must be positive")
    z = 1.96
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    standard_error = math.sqrt(proportion * (1.0 - proportion) / trials)
    return {
        "n": int(trials),
        "mean": float(proportion),
        "standard_error": float(standard_error),
        "ci95_low": float(max(0.0, center - radius)),
        "ci95_high": float(min(1.0, center + radius)),
    }


def _summary_row(
    section: str,
    method: str,
    depth: int | str,
    metric: str,
    interval: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "section": section,
        "method": method,
        "depth": depth,
        "metric": metric,
        **interval,
    }


def summarize(seed_results: list[dict[str, Any]], config: ExperimentConfig) -> list[dict[str, Any]]:
    """Create a deterministic long-format summary with uncertainty."""
    rows: list[dict[str, Any]] = []
    diagnostic_metrics = (
        "pooled_raw_off_diagonal_correlation",
        "pooled_raw_off_diagonal_absolute_correlation",
        "residual_off_diagonal_correlation",
        "residual_off_diagonal_absolute_correlation",
    )
    for metric in diagnostic_metrics:
        rows.append(
            _summary_row(
                "correlation_diagnostic",
                "diagnostic",
                "",
                metric,
                _mean_interval(result["diagnostics"][metric] for result in seed_results),
            )
        )

    fixed_lookup = {
        (result["seed_index"], row["method"], row["depth"]): row
        for result in seed_results
        for row in result["fixed_depth"]
    }
    fixed_metrics = (
        "empirical_standard_loss_risk",
        "conditional_population_risk",
        "empirical_risk_difference_vs_exact",
        "population_risk_excess_vs_exact",
    )
    for method in METHODS:
        for depth in config.fixed_depths:
            records = [
                fixed_lookup[(result["seed_index"], method, depth)]
                for result in seed_results
            ]
            rows.append(
                _summary_row(
                    "fixed_matched_depth",
                    method,
                    depth,
                    "calls",
                    _mean_interval(record["calls"] for record in records),
                )
            )
            for metric in fixed_metrics:
                rows.append(
                    _summary_row(
                        "fixed_matched_depth",
                        method,
                        depth,
                        metric,
                        _mean_interval(record[metric] for record in records),
                    )
                )
            if method != "exact_factorized":
                wins = sum(record["false_dependence_win"] for record in records)
                rows.append(
                    _summary_row(
                        "fixed_matched_depth",
                        method,
                        depth,
                        "false_dependence_win_frequency",
                        _wilson_interval(wins, len(records)),
                    )
                )

    calibrated_lookup = {
        (result["seed_index"], row["method"]): row
        for result in seed_results
        for row in result["calibrated_standard_loss"]
    }
    calibrated_metrics = (
        "calls",
        "selected_depth",
        "corrected_final_calibration_risk",
        "test_empirical_standard_loss_risk",
        "test_conditional_population_risk",
    )
    for method in METHODS:
        records = [
            calibrated_lookup[(result["seed_index"], method)]
            for result in seed_results
        ]
        for metric in calibrated_metrics:
            rows.append(
                _summary_row(
                    "calibrated_standard_loss",
                    method,
                    "selected",
                    metric,
                    _mean_interval(record[metric] for record in records),
                )
            )
        certified = sum(record["certified"] for record in records)
        rows.append(
            _summary_row(
                "calibrated_standard_loss",
                method,
                "selected",
                "certification_rate",
                _wilson_interval(certified, len(records)),
            )
        )
        if method != "exact_factorized":
            penalties = [
                result["selected_penalties"][method] for result in seed_results
            ]
            rows.append(
                _summary_row(
                    "validation_selection",
                    method,
                    "",
                    "selected_penalty",
                    _mean_interval(penalties),
                )
            )
    return rows


def run_experiment(config: ExperimentConfig = FROZEN_CONFIG) -> dict[str, Any]:
    """Run every frozen seed and return the complete deterministic payload."""
    config.validate()
    seed_results = [run_seed(config, seed_index) for seed_index in range(config.seed_count)]
    summary = summarize(seed_results, config)
    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "definitions": {
            "conditional_independence": (
                "Given X, model correctness draws are independent Bernoulli variables "
                "with heterogeneous known probabilities p_j(X)."
            ),
            "pooled_raw_correlation": (
                "Mean off-diagonal Pearson correlation of design correctness labels, "
                "without conditioning on X."
            ),
            "residual_correlation": (
                "Mean off-diagonal Pearson correlation of design residuals Y - p_j(X)."
            ),
            "false_dependence_win": (
                "At a matched fixed depth, a heuristic has lower empirical test risk "
                "than exact factorization even though its conditional population risk "
                "is no lower."
            ),
            "uncertainty": (
                "Means use seed-level standard errors and normal 95 percent intervals. "
                "Frequencies use Wilson 95 percent intervals."
            ),
        },
        "seed_results": seed_results,
        "summary": summary,
    }


def _format_penalty(value: float) -> str:
    return format(value, ".12g")


def _round_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _round_for_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_for_output(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        rounded = round(float(value), 12)
        return 0.0 if rounded == 0.0 else rounded
    return value


def write_outputs(output_dir: Path, payload: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    """Write deterministic JSON and CSV using atomic whole-file replacement."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "summary.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output: {joined}")

    rounded_payload = _round_for_output(payload)
    json_text = json.dumps(
        rounded_payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    json_temporary = json_path.with_suffix(".json.tmp")
    json_temporary.write_text(json_text, encoding="utf-8")
    os.replace(json_temporary, json_path)

    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rounded_payload["summary"])
    os.replace(csv_temporary, csv_path)
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/synthetic_conditional_independence/v1"),
        help="directory for results.json and summary.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing v1 files with deterministic recomputation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(FROZEN_CONFIG)
    json_path, csv_path = write_outputs(args.output_dir, payload, force=args.force)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
