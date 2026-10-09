"""Scalable higher-order fixed-order estimators for ISSUE-71."""

from __future__ import annotations

from typing import Any

import numpy as np

from components.scale_selectors import EPSILON, _as_binary, _greedy_cover, _normal_quantile

TETRACHORIC_RIDGE_GRID = (0.0, 0.05, 0.10, 0.20)
VINE_LAPLACE_GRID = (0.5, 1.0, 2.0, 5.0)


def _probit_conditional_scores(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map binary failures to conditional latent-normal expectations."""

    binary = _as_binary(labels)
    failures = (~binary).astype(float)
    n_rows = failures.shape[0]
    failure_rate = (failures.sum(axis=0) + 0.5) / (n_rows + 1.0)
    thresholds = _normal_quantile(failure_rate)
    density = np.exp(-0.5 * thresholds * thresholds) / np.sqrt(2.0 * np.pi)
    below = -density / np.maximum(failure_rate, EPSILON)
    above = density / np.maximum(1.0 - failure_rate, EPSILON)
    latent = np.where(failures > 0.5, below[None, :], above[None, :])
    latent -= latent.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(latent * latent, axis=0)) + EPSILON
    return latent / scale, failure_rate


def _tetrachoric_factor_fit(
    labels: np.ndarray,
    *,
    ridge: float,
    maximum_rank: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a PSD low-rank latent correlation from probit normal scores."""

    standardized, failure_rate = _probit_conditional_scores(labels)
    n_rows, n_models = standardized.shape
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    rank = min(maximum_rank, max(1, n_rows - 1), n_models, len(singular))
    loadings = (right[:rank].T * singular[:rank]) / np.sqrt(max(n_rows, 1))
    shrinkage = float(np.clip(ridge, 0.0, 1.0))
    loadings *= np.sqrt(1.0 - shrinkage)
    norms = np.sqrt(np.sum(loadings * loadings, axis=1))
    cap = np.maximum(norms / 0.95, 1.0)
    loadings /= cap[:, None]
    explained = np.sum(loadings * loadings, axis=1)
    idiosyncratic = np.sqrt(np.maximum(1.0 - explained, 1e-6))
    retained = singular[:rank]
    condition = (
        float(retained[0] / retained[-1])
        if len(retained) and retained[-1] > EPSILON
        else None
    )
    diagnostics = {
        "estimator": "probit_conditional_normal_score_tetrachoric",
        "n_design_rows": int(n_rows),
        "n_models": int(n_models),
        "rows_per_model": float(n_rows / n_models),
        "effective_rank": int(rank),
        "ridge": shrinkage,
        "psd_strategy": "low_rank_svd_plus_positive_diagonal",
        "nearest_psd_projection_used": False,
        "minimum_idiosyncratic_variance": float(np.min(idiosyncratic**2)),
        "maximum_common_variance": float(np.max(explained)),
        "retained_singular_condition_number": condition,
        "degenerate_margin_count": int(
            np.count_nonzero((failure_rate <= 1.0 / (n_rows + 1.0)) | (failure_rate >= n_rows / (n_rows + 1.0)))
        ),
        "identifiability_note": (
            "Pairwise binary margins identify a tetrachoric latent correlation only under the "
            "probit threshold model. The low-rank SVD restriction is required when models exceed design rows."
        ),
    }
    return loadings, idiosyncratic, diagnostics


def tetrachoric_normal_score_order(
    design: np.ndarray,
    max_depth: int,
    *,
    ridge: float = 0.10,
    n_draws: int = 2000,
    seed: int = 0,
) -> tuple[int, ...]:
    """Build a fixed order from a scalable tetrachoric latent-factor approximation."""

    labels = _as_binary(design)
    loadings, idiosyncratic, _ = _tetrachoric_factor_fit(labels, ridge=ridge)
    failure_rate = np.clip((~labels).mean(axis=0), 1e-6, 1.0 - 1e-6)
    generator = np.random.default_rng(seed)
    common = generator.standard_normal((int(n_draws), loadings.shape[1]))
    unique = generator.standard_normal((int(n_draws), labels.shape[1]))
    latent = common @ loadings.T + unique * idiosyncratic[None, :]
    draws = latent <= _normal_quantile(failure_rate)[None, :]
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def tetrachoric_diagnostics(design: np.ndarray, *, ridge: float) -> dict[str, Any]:
    """Return the exact fit and identifiability record for a tetrachoric arm."""

    _, _, diagnostics = _tetrachoric_factor_fit(_as_binary(design), ridge=ridge)
    return diagnostics


def _mutual_information_tree(
    failures: np.ndarray,
    *,
    laplace: float,
    root: int,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Fit a deterministic Chow-Liu first-tree discrete vine."""

    matrix = np.asarray(failures, dtype=float)
    n_rows, n_models = matrix.shape
    smoothing = float(laplace)
    if smoothing <= 0.0:
        raise ValueError("vine Laplace smoothing must be positive")
    count_one = matrix.sum(axis=0)
    count_11 = matrix.T @ matrix
    count_10 = count_one[:, None] - count_11
    count_01 = count_one[None, :] - count_11
    count_00 = n_rows - count_11 - count_10 - count_01
    counts = np.stack((count_00, count_01, count_10, count_11), axis=0)
    probabilities = (counts + smoothing) / (n_rows + 4.0 * smoothing)
    p_parent_zero = probabilities[0] + probabilities[1]
    p_parent_one = probabilities[2] + probabilities[3]
    p_child_zero = probabilities[0] + probabilities[2]
    p_child_one = probabilities[1] + probabilities[3]
    denominators = np.stack((
        p_parent_zero * p_child_zero,
        p_parent_zero * p_child_one,
        p_parent_one * p_child_zero,
        p_parent_one * p_child_one,
    ))
    mutual_information = np.sum(
        probabilities * np.log(np.maximum(probabilities, EPSILON) / np.maximum(denominators, EPSILON)),
        axis=0,
    )
    np.fill_diagonal(mutual_information, -np.inf)

    parent = np.full(n_models, -1, dtype=int)
    selected = np.zeros(n_models, dtype=bool)
    selected[int(root)] = True
    traversal = [int(root)]
    best_weight = mutual_information[int(root)].copy()
    best_parent = np.full(n_models, int(root), dtype=int)
    best_weight[selected] = -np.inf
    while len(traversal) < n_models:
        maximum = float(np.max(best_weight))
        candidates = np.flatnonzero(np.isclose(best_weight, maximum, rtol=0.0, atol=1e-15))
        child = int(candidates[0])
        parent[child] = int(best_parent[child])
        selected[child] = True
        traversal.append(child)
        best_weight[child] = -np.inf
        for candidate in np.flatnonzero(~selected):
            weight = mutual_information[child, candidate]
            if weight > best_weight[candidate] + 1e-15:
                best_weight[candidate] = weight
                best_parent[candidate] = child
            elif abs(weight - best_weight[candidate]) <= 1e-15 and child < best_parent[candidate]:
                best_parent[candidate] = child

    off_diagonal = ~np.eye(n_models, dtype=bool)
    zero_fraction = float(np.mean(counts[:, off_diagonal] == 0.0)) if n_models > 1 else 0.0
    diagnostics = {
        "estimator": "discrete_bernoulli_chow_liu_first_tree_vine",
        "n_design_rows": int(n_rows),
        "n_models": int(n_models),
        "rows_per_model": float(n_rows / n_models),
        "laplace_smoothing": smoothing,
        "root_model_index": int(root),
        "tree_edges": int(n_models - 1),
        "zero_contingency_cell_fraction": zero_fraction,
        "higher_vine_levels_fit": 0,
        "identifiability_note": (
            "Only the first Chow-Liu tree is fit. Higher binary vine levels are not uniquely "
            "identifiable at 100 to 500 design rows without untestable pair-copula choices."
        ),
        "psd_strategy": "not_applicable_tree_factorization",
    }
    return parent, traversal, diagnostics


def _vine_fit(
    labels: np.ndarray,
    *,
    laplace: float,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    binary = _as_binary(labels)
    failures = (~binary).astype(float)
    failure_rate = failures.mean(axis=0)
    root = int(np.argmin(failure_rate))
    parent, traversal, diagnostics = _mutual_information_tree(
        failures, laplace=laplace, root=root
    )
    n_rows = failures.shape[0]
    count_one = failures.sum(axis=0)
    count_11 = failures.T @ failures
    conditional_zero = np.empty(failures.shape[1], dtype=float)
    conditional_one = np.empty(failures.shape[1], dtype=float)
    smoothing = float(laplace)
    for child in traversal[1:]:
        source = int(parent[child])
        conditional_one[child] = (count_11[source, child] + smoothing) / (
            count_one[source] + 2.0 * smoothing
        )
        conditional_zero[child] = (
            count_one[child] - count_11[source, child] + smoothing
        ) / (n_rows - count_one[source] + 2.0 * smoothing)
    root_rate = (count_one[root] + smoothing) / (n_rows + 2.0 * smoothing)
    return parent, traversal, conditional_zero, conditional_one, np.asarray([root_rate]), diagnostics


def discrete_vine_tree_order(
    design: np.ndarray,
    max_depth: int,
    *,
    laplace: float = 1.0,
    n_draws: int = 2000,
    seed: int = 0,
) -> tuple[int, ...]:
    """Build a fixed order from a first-tree discrete Bernoulli vine."""

    labels = _as_binary(design)
    parent, traversal, conditional_zero, conditional_one, root_rate, _ = _vine_fit(
        labels, laplace=laplace
    )
    generator = np.random.default_rng(seed)
    draws = np.zeros((int(n_draws), labels.shape[1]), dtype=bool)
    root = traversal[0]
    draws[:, root] = generator.random(int(n_draws)) < root_rate[0]
    for child in traversal[1:]:
        source = int(parent[child])
        probability = np.where(draws[:, source], conditional_one[child], conditional_zero[child])
        draws[:, child] = generator.random(int(n_draws)) < probability
    return _greedy_cover(draws, max_depth, tie_break=labels.mean(axis=0))


def vine_diagnostics(design: np.ndarray, *, laplace: float) -> dict[str, Any]:
    """Return fit sparsity and identifiability diagnostics for the vine arm."""

    *_, diagnostics = _vine_fit(_as_binary(design), laplace=laplace)
    return diagnostics


__all__ = [
    "TETRACHORIC_RIDGE_GRID",
    "VINE_LAPLACE_GRID",
    "discrete_vine_tree_order",
    "tetrachoric_diagnostics",
    "tetrachoric_normal_score_order",
    "vine_diagnostics",
]
