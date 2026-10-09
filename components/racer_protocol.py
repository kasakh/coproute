"""RACER scoring with standard real-model all-miss calibration.

The published augmented-null construction is retained for a separately named
control diagnostic. Paper-facing calibration and certification use only real
model scores and standard all-miss loss:

* an empty real-model set is a miss;
* rows that every real model fails remain unavoidable misses;
* virtual-null membership never suppresses real-model calls; and
* an infeasible corrected bound produces explicit non-certification, not a free
  null or human-review fallback.

The implementation operates on arrays only. It does not train a base router or
aggregate model responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


DEFAULT_SMOOTHING = 1e-6
RACER_G = "G"
RACER_P = "P"


@dataclass(frozen=True)
class CalibrationResult:
    """Result of finite-sample threshold calibration.

    ``threshold`` is a deployment threshold only when ``feasible`` is true.
    Otherwise it is negative infinity as a non-deployment sentinel,
    ``calibration_misses`` is the minimum attainable miss count, and
    ``non_certification_reason`` explains why no standard-risk certificate exists.
    """

    threshold: float
    feasible: bool
    alpha: float
    n_calibration: int
    max_misses: int
    calibration_misses: int
    crc_bound: float
    critical_scores: np.ndarray
    unavoidable_misses: int = 0
    non_certification_reason: str | None = None


@dataclass(frozen=True)
class RoutingResult:
    """Prediction sets and derived real-model routing decisions.

    The final column of ``augmented_mask`` is the virtual null used only for the
    published RACER diagnostic. ``standard_mask`` always contains every selected
    real model, including rows where the virtual null is also selected. ``size``
    counts those real calls. ``safe_action`` and ``abstain`` are compatibility
    aliases for an empty real-model set, whose standard all-miss loss is one.
    """

    augmented_scores: np.ndarray
    nonconformity_scores: np.ndarray
    augmented_mask: np.ndarray
    standard_mask: np.ndarray
    null_selected: np.ndarray
    safe_action: np.ndarray
    abstain: np.ndarray
    size: np.ndarray


@dataclass(frozen=True)
class SplitIndices:
    """Deterministic calibration, validation, and test indices."""

    calibration: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _as_score_matrix(base_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(base_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("base_scores must be a two-dimensional array")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("base_scores must contain at least one row and model")
    if not np.all(np.isfinite(scores)):
        raise ValueError("base_scores must contain only finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("base_scores must be probabilities in [0, 1]")
    return scores


def _as_ground_truth(
    ground_truth: np.ndarray, expected_shape: Tuple[int, int]
) -> np.ndarray:
    labels = np.asarray(ground_truth)
    if labels.shape != expected_shape:
        raise ValueError(
            "ground_truth must have the same shape as the standard model scores"
        )
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("ground_truth must contain only binary values")
    return labels.astype(bool, copy=False)


def _validate_seed(seed: int, name: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return seed


def augment_router_scores(base_scores: np.ndarray) -> np.ndarray:
    """Append the RACER null score to each row.

    For standard router probabilities ``f(x, m)``, the null score is
    ``1 - max_m f(x, m)``. The null model is always the last column.
    """

    scores = _as_score_matrix(base_scores)
    null_scores = 1.0 - np.max(scores, axis=1, keepdims=True)
    return np.concatenate((scores, null_scores), axis=1)


def augment_ground_truth(ground_truth: np.ndarray) -> np.ndarray:
    """Append null-model labels using RACER's all-model-failure rule."""

    labels = np.asarray(ground_truth)
    if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("ground_truth must be a non-empty two-dimensional array")
    labels = _as_ground_truth(labels, labels.shape)
    null_is_correct = ~np.any(labels, axis=1, keepdims=True)
    return np.concatenate((labels, null_is_correct), axis=1)


def raw_nonconformity_scores(
    augmented_scores: np.ndarray, variant: str
) -> np.ndarray:
    """Compute unsmoothed RACER-G or RACER-P scores."""

    scores = np.asarray(augmented_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("augmented_scores must contain rows and an augmented pool")
    if not np.all(np.isfinite(scores)):
        raise ValueError("augmented_scores must contain only finite values")

    normalized_variant = str(variant).upper()
    if normalized_variant == RACER_G:
        return np.max(scores, axis=1, keepdims=True) - scores
    if normalized_variant == RACER_P:
        return 1.0 - scores
    raise ValueError("variant must be 'G' or 'P'")


def smooth_nonconformity_scores(
    scores: np.ndarray,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
    seed: int = 0,
    split_seed: int = 0,
    preserve_prefixes: bool = False,
) -> np.ndarray:
    """Break score ties reproducibly, optionally preserving fitted prefixes.

    The legacy path adds independent uniform noise in ``[0, smoothing]`` and is
    retained as the default for exact reproduction of completed runs. With
    ``preserve_prefixes=True``, each row is stably ordered by raw score and model
    index, then receives a deterministic non-decreasing offset along that
    sequence. The resulting order cannot invert a strict raw-score comparison or
    cut a tied plateau out of stable sequence order.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("scores must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("smoothing must be a finite non-negative value")
    if not isinstance(preserve_prefixes, (bool, np.bool_)):
        raise TypeError("preserve_prefixes must be a boolean")

    seed = _validate_seed(seed, "seed")
    split_seed = _validate_seed(split_seed, "split_seed")
    if smoothing == 0.0:
        return values.copy()

    if preserve_prefixes:
        n_columns = values.shape[1]
        offsets_by_rank = smoothing * np.arange(n_columns, dtype=float) / n_columns
        result = np.empty_like(values)
        for row_index, row in enumerate(values):
            sequence = np.argsort(row, kind="stable")
            result[row_index, sequence] = row[sequence] + offsets_by_rank
        return result

    rng = np.random.default_rng(np.random.SeedSequence((seed, split_seed)))
    noise = rng.uniform(0.0, smoothing, size=values.shape)
    return values + noise


def nonconformity_scores(
    base_scores: np.ndarray,
    variant: str,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
    seed: int = 0,
    split_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return augmented router scores and final nonconformity scores."""

    augmented = augment_router_scores(base_scores)
    raw = raw_nonconformity_scores(augmented, variant)
    smoothed = smooth_nonconformity_scores(
        raw, smoothing=smoothing, seed=seed, split_seed=split_seed
    )
    return augmented, smoothed


def ground_truth_critical_scores(
    nonconformity: np.ndarray, augmented_ground_truth: np.ndarray
) -> np.ndarray:
    """Return critical scores for RACER's published augmented-null diagnostic."""

    scores = np.asarray(nonconformity, dtype=float)
    labels = np.asarray(augmented_ground_truth)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("nonconformity must be a non-empty two-dimensional array")
    labels = _as_ground_truth(labels, scores.shape)
    if not np.all(np.any(labels, axis=1)):
        raise ValueError("each augmented ground-truth row must be non-empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("nonconformity must contain only finite values")
    return np.min(np.where(labels, scores, np.inf), axis=1)


def standard_ground_truth_critical_scores(
    real_model_scores: np.ndarray, standard_ground_truth: np.ndarray
) -> np.ndarray:
    """Return critical scores for standard real-model all-miss loss.

    A row that no real model answers has critical score positive infinity. Such
    a row remains a miss at every finite threshold and therefore contributes to
    the structural risk floor instead of becoming a correct virtual-null action.
    """

    scores = np.asarray(real_model_scores, dtype=float)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("real_model_scores must be a non-empty two-dimensional array")
    labels = _as_ground_truth(standard_ground_truth, scores.shape)
    if not np.all(np.isfinite(scores)):
        raise ValueError("real_model_scores must contain only finite values")
    return np.min(np.where(labels, scores, np.inf), axis=1)


def select_crc_threshold(
    critical_scores: np.ndarray, alpha: float
) -> CalibrationResult:
    """Exactly scan critical-score thresholds under the corrected CRC bound.

    For a finite threshold, calibration misses are the critical scores strictly
    greater than that threshold. Positive-infinity critical scores represent
    rows unanswerable by every real model and are always counted as misses. The
    first finite threshold satisfying ``(1 + misses) / (n + 1) <= alpha`` is the
    smallest risk-certified operating point.

    If no finite threshold satisfies the bound, the result is explicitly
    non-certified. Negative infinity is only a non-deployment sentinel. It must
    never be interpreted as a free abstention policy or a risk-valid action.
    """

    values = np.asarray(critical_scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("critical_scores must be a non-empty one-dimensional array")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("critical_scores may contain finite values or positive infinity")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    n = int(values.size)
    miss_budget = alpha * (n + 1) - 1.0
    max_misses = int(np.floor(np.nextafter(miss_budget, np.inf)))
    unavoidable = int(np.count_nonzero(np.isposinf(values)))
    finite = np.sort(values[np.isfinite(values)])
    minimum_misses = unavoidable if finite.size else n
    minimum_bound = (minimum_misses + 1.0) / (n + 1.0)

    if max_misses < 0:
        reason = "below_finite_sample_resolution"
    elif unavoidable > max_misses or finite.size == 0:
        reason = "standard_risk_structurally_infeasible"
    else:
        reason = None
        for threshold in np.unique(finite):
            covered_finite = int(np.searchsorted(finite, threshold, side="right"))
            misses = unavoidable + int(finite.size) - covered_finite
            bound = (misses + 1.0) / (n + 1.0)
            if bound <= alpha:
                return CalibrationResult(
                    threshold=float(threshold),
                    feasible=True,
                    alpha=float(alpha),
                    n_calibration=n,
                    max_misses=max_misses,
                    calibration_misses=misses,
                    crc_bound=bound,
                    critical_scores=values.copy(),
                    unavoidable_misses=unavoidable,
                    non_certification_reason=None,
                )
        reason = "no_standard_threshold_satisfies_bound"

    return CalibrationResult(
        threshold=float("-inf"),
        feasible=False,
        alpha=float(alpha),
        n_calibration=n,
        max_misses=max_misses,
        calibration_misses=minimum_misses,
        crc_bound=minimum_bound,
        critical_scores=values.copy(),
        unavoidable_misses=unavoidable,
        non_certification_reason=reason,
    )


def select_standard_crc_threshold(
    real_model_scores: np.ndarray,
    standard_ground_truth: np.ndarray,
    alpha: float,
) -> CalibrationResult:
    """Calibrate a real-model threshold for standard all-miss loss."""

    critical = standard_ground_truth_critical_scores(
        real_model_scores, standard_ground_truth
    )
    return select_crc_threshold(critical, alpha)


def prediction_sets(
    augmented_scores: np.ndarray,
    nonconformity: np.ndarray,
    threshold: float,
) -> RoutingResult:
    """Construct variable-size RACER sets at a calibrated threshold."""

    router_scores = np.asarray(augmented_scores, dtype=float)
    scores = np.asarray(nonconformity, dtype=float)
    if router_scores.ndim != 2 or router_scores.shape != scores.shape:
        raise ValueError("augmented_scores and nonconformity must share a 2D shape")
    if router_scores.shape[0] == 0 or router_scores.shape[1] < 2:
        raise ValueError("arrays must contain rows and an augmented model pool")
    if not np.all(np.isfinite(router_scores)) or not np.all(np.isfinite(scores)):
        raise ValueError("score arrays must contain only finite values")
    if np.isnan(threshold):
        raise ValueError("threshold must not be NaN")

    augmented_mask = scores <= float(threshold)
    standard_mask = augmented_mask[:, :-1].copy()
    null_selected = augmented_mask[:, -1]
    size = np.sum(standard_mask, axis=1, dtype=int)
    empty_set = size == 0
    return RoutingResult(
        augmented_scores=router_scores.copy(),
        nonconformity_scores=scores.copy(),
        augmented_mask=augmented_mask,
        standard_mask=standard_mask,
        null_selected=null_selected,
        safe_action=empty_set.copy(),
        abstain=empty_set.copy(),
        size=size,
    )


def augmented_misrouting_loss(
    augmented_mask: np.ndarray, augmented_ground_truth: np.ndarray
) -> np.ndarray:
    """RACER's augmented-label loss, including the virtual null label."""

    selected = np.asarray(augmented_mask)
    labels = np.asarray(augmented_ground_truth)
    if selected.ndim != 2 or selected.shape != labels.shape:
        raise ValueError("prediction mask and ground truth must share a 2D shape")
    selected = selected.astype(bool, copy=False)
    labels = _as_ground_truth(labels, selected.shape)
    return ~np.any(selected & labels, axis=1)


def standard_all_miss_loss(
    standard_mask: np.ndarray, standard_ground_truth: np.ndarray
) -> np.ndarray:
    """Return whether all selected real models miss the query.

    An empty real-model set is always a miss. Virtual-null membership is not an
    input and cannot convert any standard miss into a success.
    """

    selected = np.asarray(standard_mask)
    labels = np.asarray(standard_ground_truth)
    if selected.ndim != 2 or selected.shape != labels.shape:
        raise ValueError("standard prediction mask and ground truth must share a 2D shape")
    selected = selected.astype(bool, copy=False)
    labels = _as_ground_truth(labels, selected.shape)
    return ~np.any(selected & labels, axis=1)


def safe_action_loss(
    routing: RoutingResult, standard_ground_truth: np.ndarray
) -> np.ndarray:
    """Legacy empty-set-exempt diagnostic, never a certifiable paper risk.

    Standard all-miss calibration must use :func:`standard_all_miss_loss`. This
    helper remains only for readers of older result schemas.
    """

    all_miss = standard_all_miss_loss(routing.standard_mask, standard_ground_truth)
    return all_miss & ~routing.safe_action


def misrouting_loss(
    augmented_mask: np.ndarray, augmented_ground_truth: np.ndarray
) -> np.ndarray:
    """Backward-compatible name for :func:`augmented_misrouting_loss`."""

    return augmented_misrouting_loss(augmented_mask, augmented_ground_truth)


def deterministic_split_indices(
    n_samples: int,
    *,
    seed: int,
    fractions: Iterable[float] = (0.5, 0.1, 0.4),
) -> SplitIndices:
    """Create a deterministic split using largest-remainder allocation.

    The default fractions are RACER's calibration, validation, and test split
    of the post-router evaluation pool. ARC-Challenge can use ``(0.4, 0.2,
    0.4)`` as specified by the paper.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError("n_samples must be a positive integer")
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    seed = _validate_seed(seed, "seed")

    parts = np.asarray(tuple(fractions), dtype=float)
    if parts.shape != (3,) or not np.all(np.isfinite(parts)):
        raise ValueError("fractions must contain three finite values")
    if np.any(parts < 0.0) or not np.isclose(np.sum(parts), 1.0):
        raise ValueError("fractions must be non-negative and sum to one")

    exact = parts * n_samples
    counts = np.floor(exact).astype(int)
    remainder = n_samples - int(np.sum(counts))
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1

    permutation = np.random.default_rng(seed).permutation(n_samples)
    first = counts[0]
    second = first + counts[1]
    return SplitIndices(
        calibration=permutation[:first],
        validation=permutation[first:second],
        test=permutation[second:],
    )


class RACERProtocol:
    """Configured exact RACER scorer, calibrator, and set constructor."""

    def __init__(
        self,
        variant: str,
        *,
        seed: int = 0,
        smoothing: float = DEFAULT_SMOOTHING,
    ) -> None:
        normalized_variant = str(variant).upper()
        if normalized_variant not in (RACER_G, RACER_P):
            raise ValueError("variant must be 'G' or 'P'")
        self.variant = normalized_variant
        self.seed = _validate_seed(seed, "seed")
        if not np.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("smoothing must be a finite non-negative value")
        self.smoothing = float(smoothing)

    def score(
        self, base_scores: np.ndarray, *, split_seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment and score one deterministic split."""

        return nonconformity_scores(
            base_scores,
            self.variant,
            smoothing=self.smoothing,
            seed=self.seed,
            split_seed=split_seed,
        )

    def calibrate(
        self,
        base_scores: np.ndarray,
        ground_truth: np.ndarray,
        alpha: float,
        *,
        split_seed: int = 0,
    ) -> CalibrationResult:
        """Calibrate a RACER threshold from one labeled split."""

        standard_scores = _as_score_matrix(base_scores)
        labels = _as_ground_truth(ground_truth, standard_scores.shape)
        _, scores = self.score(standard_scores, split_seed=split_seed)
        return select_standard_crc_threshold(scores[:, :-1], labels, alpha)

    def route(
        self,
        base_scores: np.ndarray,
        threshold: float | CalibrationResult,
        *,
        split_seed: int = 0,
    ) -> RoutingResult:
        """Route queries into variable-size model sets."""

        if isinstance(threshold, CalibrationResult):
            if not threshold.feasible:
                raise ValueError(
                    "cannot route a non-certified calibration result; no risk-valid "
                    "operating point exists"
                )
            calibrated_threshold = threshold.threshold
        else:
            calibrated_threshold = float(threshold)
        augmented, scores = self.score(base_scores, split_seed=split_seed)
        return prediction_sets(augmented, scores, calibrated_threshold)
