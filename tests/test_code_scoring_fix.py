"""Regression tests for code-task aggregation scoring.

Defect found 2026-09-03: a chosen program was scored by exact string equality
against another model's program, which is uninformative. On RouterBench MBPP this
produced aggregation accuracy 0.0882 against 0.5330 for a uniform random model
pick and 0.7255 for the best single model, so the metric was far below chance.

Fix: only multiple choice and GSM8K have canonically comparable answers. Code
correctness comes from the official per-model labels of the supporters.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.scoring import (  # noqa: E402
    STATUS_SCORED,
    STATUS_UNSUPPORTED,
    aggregate_responses,
)

CODE_A = "def f(x):\n    return x + 1\n"
CODE_B = "def f(x):\n    y = x\n    return y + 1\n"


def test_code_top_router_uses_official_label_not_string_equality() -> None:
    """The chosen program is scored by its own model's label."""

    result = aggregate_responses(
        "mbpp",
        [CODE_A, CODE_B],
        router_scores=[0.9, 0.1],
        method="top_router",
        selected_indices=[0, 1],
        labels=[1.0, 0.0],
        reference_answer=CODE_B,
    )
    assert result.status == STATUS_SCORED
    assert result.correct is True, "winner's own label must decide, not equality with CODE_B"


def test_code_losing_model_is_scored_incorrect() -> None:
    result = aggregate_responses(
        "mbpp",
        [CODE_A, CODE_B],
        router_scores=[0.1, 0.9],
        method="top_router",
        selected_indices=[0, 1],
        labels=[1.0, 0.0],
        reference_answer=CODE_A,
    )
    assert result.status == STATUS_SCORED
    assert result.correct is False


def test_code_without_labels_is_unsupported_not_silently_wrong() -> None:
    """A reference program alone must never be used to score code."""

    result = aggregate_responses(
        "mbpp",
        [CODE_A, CODE_B],
        router_scores=[0.9, 0.1],
        method="top_router",
        selected_indices=[0, 1],
        labels=None,
        reference_answer=CODE_B,
    )
    assert result.status == STATUS_UNSUPPORTED
    assert result.correct is None
    assert "canonically comparable" in (result.reason or "")


def test_multiple_choice_reference_comparison_is_preserved() -> None:
    """The fix must not change multiple-choice behaviour."""

    result = aggregate_responses(
        "mmlu",
        ["The answer is B", "The answer is C"],
        router_scores=[0.9, 0.1],
        method="top_router",
        selected_indices=[0, 1],
        labels=[0.0, 0.0],
        reference_answer="B",
    )
    assert result.status == STATUS_SCORED
    assert result.correct is True, "reference comparison must still win for choices"


def test_gsm8k_reference_comparison_is_preserved() -> None:
    result = aggregate_responses(
        "gsm8k",
        ["The final answer is 42", "The final answer is 7"],
        router_scores=[0.9, 0.1],
        method="top_router",
        selected_indices=[0, 1],
        labels=[0.0, 0.0],
        reference_answer="#### 42",
    )
    assert result.status == STATUS_SCORED
    assert result.correct is True


def test_evaluation_reference_path_refuses_code() -> None:
    """The evaluation scoring path must not compare two programs for equality.

    Regression for the SECOND copy of the defect, in components/evaluation.py,
    which duplicated the equality comparison and bypassed the scoring-module gate.
    """

    from components.evaluation import _score_against_reference

    result = _score_against_reference(
        task="mbpp",
        responses=[CODE_A, CODE_B],
        router_scores=[0.9, 0.1],
        selected=(0, 1),
        aggregator="top_router",
        reference_answer=CODE_B,
    )
    assert result is not None
    assert result.status == STATUS_UNSUPPORTED
    assert result.correct is None
    assert "uninformative" in (result.reason or "")


def test_evaluation_reference_path_still_scores_choices() -> None:
    from components.evaluation import _score_against_reference

    result = _score_against_reference(
        task="mmlu",
        responses=["The answer is B", "The answer is C"],
        router_scores=[0.9, 0.1],
        selected=(0, 1),
        aggregator="top_router",
        reference_answer="B",
    )
    assert result is not None
    assert result.status == STATUS_SCORED
    assert result.correct is True


def test_reference_comparison_predicate_is_single_source_of_truth() -> None:
    from components.scoring import reference_comparison_supported

    assert reference_comparison_supported("mmlu") is True
    assert reference_comparison_supported("winogrande") is True
    assert reference_comparison_supported("gsm8k") is True
    assert reference_comparison_supported("mbpp") is False
    assert reference_comparison_supported("humaneval") is False
