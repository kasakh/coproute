"""Task aware answer extraction and response aggregation for DACR.

RouterBench stores one correctness column per model plus matching
``<model>|model_response`` and ``<model>|total_cost`` columns. It does not
store a standalone gold answer. This module keeps answer aggregation separate
from correctness scoring so an answer group is never declared correct merely
because one supporting model has a positive label.

Concrete RouterBench style extraction examples:

Multiple choice:
    ``['C)']`` -> ``C``
    ``['\\nB']`` -> ``B``
    ``Answer: A`` -> ``A``

GSM8K / grade school math:
    ``48 + 24 = 72 clips`` -> ``72``
    ``The answer is USD 10.00`` -> ``10``
    ``work: 847 / 7; final answer: 121`` -> ``121``

MBPP / code uses conservative normalized exact text:
    ``['def f(x):\\n    return x']`` -> the unwrapped code
    CRLF line endings -> LF line endings
    surrounding whitespace -> stripped text

The four public aggregation methods share :func:`aggregate_responses`.
Majority voting follows RACER: vote count first, average router score only for
ties. Score weighted voting uses softmax router weights. Top router returns the
valid response from the highest scoring selected model. Full ensemble applies
RACER majority voting to every model, regardless of a supplied selected set.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union


STATUS_OK = "ok"
STATUS_SCORED = "scored"
STATUS_INVALID = "invalid"
STATUS_UNSUPPORTED = "unsupported"
STATUS_ABSTAIN = "abstain"


@dataclass(frozen=True)
class AnswerExtraction:
    """A canonical answer plus an explicit extraction status."""

    task: str
    family: Optional[str]
    value: Optional[str]
    status: str
    reason: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.status == STATUS_OK


@dataclass(frozen=True)
class AggregationResult:
    """The answer and optional correctness produced by one aggregator."""

    method: str
    task: str
    answer: Optional[str]
    raw_answer: Optional[str]
    status: str
    correct: Optional[bool]
    used_indices: Tuple[int, ...]
    supporter_indices: Tuple[int, ...]
    reason: Optional[str] = None
    vote_totals: Tuple[Tuple[str, float], ...] = ()

    @property
    def scored(self) -> bool:
        return self.status == STATUS_SCORED


_MULTIPLE_CHOICE_TASKS = {
    "arc",
    "arc-challenge",
    "arc_challenge",
    "hellaswag",
    "multiple-choice",
    "multiple_choice",
    "winogrande",
}
_GSM_TASKS = {"grade-school-math", "grade_school_math", "gsm8k", "gsm-8k"}
_CODE_TASKS = {
    "code",
    "code-generation",
    "code_generation",
    "humaneval",
    "human-eval",
    "mbpp",
}

_CHOICE_EXACT_RE = re.compile(r"^[\s\[\(]*([A-Za-z])[\s\]\)\.:]*$")
# Families whose canonical answer is a small comparable token, so equality with a
# reference answer is meaningful. A generated program is NOT in this set: two
# correct programs are textually different, so string equality between a chosen
# response and another model's response carries no information about
# correctness. Code correctness therefore comes from official per-model labels.
_REFERENCE_COMPARABLE_FAMILIES = frozenset({"multiple_choice", "gsm8k"})
_CHOICE_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|choice)\s*(?:is|=|:|-)?\s*[\[\(]?([A-Za-z])\b",
    re.IGNORECASE,
)
_NUMBER_TOKEN = (
    r"[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|(?:\.\d+))"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:(?:\d[\d,]*)(?:\.\d*)?|(?:\.\d+))"
    r"(?:[eE][-+]?\d+)?)?"
)
_NUMBER_RE = re.compile(_NUMBER_TOKEN)
_GSM_MARKER_RES = (
    re.compile(r"####\s*(" + _NUMBER_TOKEN + r")", re.IGNORECASE),
    re.compile(
        r"(?:final\s+answer|answer\s+is|answer)\s*(?:=|:)?\s*"
        r"(?:USD\s*)?[\u0024]?\s*(" + _NUMBER_TOKEN + r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:USD\s*|[\u0024]\s*)(" + _NUMBER_TOKEN + r")",
        re.IGNORECASE,
    ),
)


def task_family(task: str) -> Optional[str]:
    """Map RouterBench task names and common aliases to scoring families."""

    normalized = str(task).strip().lower()
    if normalized.startswith("mmlu") or normalized in _MULTIPLE_CHOICE_TASKS:
        return "multiple_choice"
    if normalized in _GSM_TASKS:
        return "gsm8k"
    if normalized in _CODE_TASKS or normalized.startswith("mbpp"):
        return "code"
    return None


def _unwrap_routerbench_cell(value: Any) -> Optional[str]:
    """Safely unwrap RouterBench's string encoded one element lists."""

    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    for _ in range(3):
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
            continue
        if not isinstance(value, str):
            return str(value)
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] not in "['\"":
            return stripped
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
        if parsed == value:
            return stripped
        value = parsed
    return value.strip() if isinstance(value, str) and value.strip() else None


def _canonical_number(token: str) -> Optional[str]:
    compact = token.replace(",", "").replace(" ", "")
    try:
        if "/" in compact:
            numerator, denominator = compact.split("/", 1)
            number = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            number = Fraction(Decimal(compact))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    if number.denominator == 1:
        return str(number.numerator)
    return f"{number.numerator}/{number.denominator}"


def canonicalize_answer(
    task: str,
    response: Any,
    *,
    allowed_choices: Iterable[str] = ("A", "B", "C", "D"),
) -> AnswerExtraction:
    """Extract a task appropriate canonical answer from a model response.

    Unsupported tasks and malformed responses are explicit. Code is only
    normalized for exact comparison. It is never parsed or executed.
    """

    family = task_family(task)
    if family is None:
        return AnswerExtraction(
            str(task), None, None, STATUS_UNSUPPORTED, "unsupported task"
        )

    text = _unwrap_routerbench_cell(response)
    if text is None:
        return AnswerExtraction(
            str(task), family, None, STATUS_INVALID, "empty response"
        )

    if family == "multiple_choice":
        choices = {str(choice).upper() for choice in allowed_choices}
        exact = _CHOICE_EXACT_RE.fullmatch(text)
        match = exact or _CHOICE_CUE_RE.search(text)
        if not match or match.group(1).upper() not in choices:
            return AnswerExtraction(
                str(task), family, None, STATUS_INVALID,
                "no unambiguous multiple choice answer",
            )
        return AnswerExtraction(
            str(task), family, match.group(1).upper(), STATUS_OK
        )

    if family == "gsm8k":
        token = None
        for marker_re in _GSM_MARKER_RES:
            matches = list(marker_re.finditer(text))
            if matches:
                token = matches[-1].group(1)
                break
        if token is None:
            matches = list(_NUMBER_RE.finditer(text))
            if matches:
                token = matches[-1].group(0)
        value = _canonical_number(token) if token is not None else None
        if value is None:
            return AnswerExtraction(
                str(task), family, None, STATUS_INVALID,
                "no valid numeric final answer",
            )
        return AnswerExtraction(str(task), family, value, STATUS_OK)

    code = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not code:
        return AnswerExtraction(
            str(task), family, None, STATUS_INVALID, "empty code response"
        )
    return AnswerExtraction(str(task), family, code, STATUS_OK)


def extract_canonical_answer(task: str, response: Any) -> Optional[str]:
    """Convenience adapter returning only the canonical value."""

    return canonicalize_answer(task, response).value


def reference_comparison_supported(task: str) -> bool:
    """Whether equality with a reference answer is meaningful for this task.

    True for tasks whose canonical answer is a small comparable token, namely
    multiple choice and GSM8K. False for code generation: two correct programs
    are textually different, so equality between a chosen response and a
    reference response carries no information about correctness. Callers that
    score by reference comparison MUST consult this predicate first and report
    an unsupported result rather than a silent incorrect one.
    """

    return task_family(task) in _REFERENCE_COMPARABLE_FAMILIES


def _coerce_indices(selected_indices: Optional[Iterable[int]], size: int) -> Tuple[int, ...]:
    indices = range(size) if selected_indices is None else selected_indices
    result = tuple(dict.fromkeys(int(index) for index in indices))
    if any(index < 0 or index >= size for index in result):
        raise IndexError("selected model index is out of range")
    return result


def _coerce_scores(
    router_scores: Optional[Union[Sequence[float], Mapping[int, float]]], size: int
) -> Tuple[float, ...]:
    if router_scores is None:
        return (0.0,) * size
    if isinstance(router_scores, Mapping):
        return tuple(float(router_scores.get(index, 0.0)) for index in range(size))
    if len(router_scores) != size:
        raise ValueError("router_scores and responses must have equal length")
    scores = tuple(float(score) for score in router_scores)
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("router_scores must be finite")
    return scores


def _label_to_bool(label: Any, threshold: float) -> Optional[bool]:
    if label is None:
        return None
    try:
        numeric = float(label)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric >= threshold


def _score_answer(
    *,
    task: str,
    answer: str,
    supporters: Tuple[int, ...],
    reference_answer: Any,
    labels: Optional[Sequence[Any]],
    label_threshold: float,
) -> Tuple[str, Optional[bool], Optional[str]]:
    reference_is_comparable = (
        reference_answer is not None and reference_comparison_supported(task)
    )
    if reference_is_comparable:
        reference = canonicalize_answer(task, reference_answer)
        if not reference.supported:
            return STATUS_UNSUPPORTED, None, "reference answer could not be extracted"
        return STATUS_SCORED, answer == reference.value, None

    if labels is None:
        if reference_answer is not None:
            return (
                STATUS_UNSUPPORTED,
                None,
                "task has no canonically comparable answer and no labels were given",
            )
        return STATUS_UNSUPPORTED, None, "no reference answer or labels"
    if len(labels) <= max(supporters, default=-1):
        raise ValueError("labels must cover every response")

    supporter_labels = tuple(
        _label_to_bool(labels[index], label_threshold) for index in supporters
    )
    if not supporter_labels or any(label is None for label in supporter_labels):
        return STATUS_UNSUPPORTED, None, "missing supporter label"
    if len(set(supporter_labels)) != 1:
        return (
            STATUS_UNSUPPORTED,
            None,
            "supporter labels conflict; correctness is not inferred from any one label",
        )
    return STATUS_SCORED, supporter_labels[0], "unanimous supporter label fallback"


def aggregate_responses(
    task: str,
    responses: Sequence[Any],
    router_scores: Optional[Union[Sequence[float], Mapping[int, float]]] = None,
    *,
    method: str = "majority",
    selected_indices: Optional[Iterable[int]] = None,
    labels: Optional[Sequence[Any]] = None,
    reference_answer: Any = None,
    temperature: float = 1.0,
    label_threshold: float = 0.5,
) -> AggregationResult:
    """Aggregate model responses through one API and conservatively score it.

    ``correct`` is populated only by comparison with a canonical reference or
    by unanimous correctness labels for every model supporting the selected
    exact answer. A mixture of positive and negative supporter labels yields
    ``status='unsupported'`` and ``correct=None``.
    """

    method_key = str(method).strip().lower().replace("-", "_")
    aliases = {
        "majority_vote": "majority",
        "weighted": "score_weighted",
        "weighted_vote": "score_weighted",
        "top": "top_router",
        "router": "top_router",
        "full": "full_ensemble",
    }
    method_key = aliases.get(method_key, method_key)
    valid_methods = {"majority", "score_weighted", "top_router", "full_ensemble"}
    if method_key not in valid_methods:
        raise ValueError(f"unknown aggregation method: {method}")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be a positive finite number")
    if labels is not None and len(labels) != len(responses):
        raise ValueError("labels and responses must have equal length")

    scores = _coerce_scores(router_scores, len(responses))
    requested = _coerce_indices(selected_indices, len(responses))
    used = tuple(range(len(responses))) if method_key == "full_ensemble" else requested
    if not used:
        return AggregationResult(
            method_key, str(task), None, None, STATUS_ABSTAIN, None, (), (),
            "no models selected",
        )

    extracted = {
        index: canonicalize_answer(task, responses[index]) for index in used
    }
    valid = tuple(index for index in used if extracted[index].supported)
    if not valid:
        family = task_family(task)
        reason = "unsupported task" if family is None else "no valid selected responses"
        return AggregationResult(
            method_key, str(task), None, None, STATUS_UNSUPPORTED, None, used, (),
            reason,
        )

    groups: dict[str, list[int]] = {}
    for index in valid:
        groups.setdefault(extracted[index].value or "", []).append(index)

    vote_totals: dict[str, float]
    if method_key == "top_router":
        winner_index = max(valid, key=lambda index: (scores[index], -index))
        winner = extracted[winner_index].value or ""
        vote_totals = {answer: max(scores[i] for i in members)
                       for answer, members in groups.items()}
    else:
        effective_method = "majority" if method_key == "full_ensemble" else method_key
        if effective_method == "majority":
            vote_totals = {
                answer: float(len(members)) for answer, members in groups.items()
            }
            max_votes = max(vote_totals.values())
            candidates = [
                answer for answer, total in vote_totals.items() if total == max_votes
            ]
        else:
            maximum = max(scores[index] / temperature for index in valid)
            unnormalized = {
                index: math.exp(scores[index] / temperature - maximum)
                for index in valid
            }
            normalizer = sum(unnormalized.values())
            weights = {
                index: weight / normalizer
                for index, weight in unnormalized.items()
            }
            vote_totals = {
                answer: sum(weights[index] for index in members)
                for answer, members in groups.items()
            }
            max_votes = max(vote_totals.values())
            candidates = [
                answer for answer, total in vote_totals.items()
                if math.isclose(total, max_votes, rel_tol=1e-12, abs_tol=1e-15)
            ]

        def tie_key(answer: str) -> Tuple[float, int]:
            members = groups[answer]
            average_score = sum(scores[index] for index in members) / len(members)
            return average_score, -min(members)

        winner = max(candidates, key=tie_key)

    supporters = tuple(groups[winner])
    representative = supporters[0]
    raw_answer = _unwrap_routerbench_cell(responses[representative])
    status, correct, reason = _score_answer(
        task=task,
        answer=winner,
        supporters=supporters,
        reference_answer=reference_answer,
        labels=labels,
        label_threshold=label_threshold,
    )
    return AggregationResult(
        method=method_key,
        task=str(task),
        answer=winner,
        raw_answer=raw_answer,
        status=status,
        correct=correct,
        used_indices=used,
        supporter_indices=supporters,
        reason=reason,
        vote_totals=tuple(sorted(vote_totals.items())),
    )


def majority_aggregate(task: str, responses: Sequence[Any], router_scores=None, **kwargs) -> AggregationResult:
    return aggregate_responses(
        task, responses, router_scores, method="majority", **kwargs
    )


def score_weighted_aggregate(task: str, responses: Sequence[Any], router_scores=None, **kwargs) -> AggregationResult:
    return aggregate_responses(
        task, responses, router_scores, method="score_weighted", **kwargs
    )


def top_router_aggregate(task: str, responses: Sequence[Any], router_scores=None, **kwargs) -> AggregationResult:
    return aggregate_responses(
        task, responses, router_scores, method="top_router", **kwargs
    )


def full_ensemble_aggregate(task: str, responses: Sequence[Any], router_scores=None, **kwargs) -> AggregationResult:
    return aggregate_responses(
        task, responses, router_scores, method="full_ensemble", **kwargs
    )


aggregate = aggregate_responses


__all__ = [
    "AggregationResult",
    "AnswerExtraction",
    "STATUS_ABSTAIN",
    "STATUS_INVALID",
    "STATUS_OK",
    "STATUS_SCORED",
    "STATUS_UNSUPPORTED",
    "aggregate",
    "aggregate_responses",
    "canonicalize_answer",
    "reference_comparison_supported",
    "extract_canonical_answer",
    "full_ensemble_aggregate",
    "majority_aggregate",
    "score_weighted_aggregate",
    "task_family",
    "top_router_aggregate",
]
