import math
from pathlib import Path
import sys
import unittest


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
if str(CORRECTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CORRECTED_ROOT))

from components.scoring import (  # noqa: E402
    STATUS_ABSTAIN,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_SCORED,
    STATUS_UNSUPPORTED,
    aggregate_responses,
    canonicalize_answer,
    extract_canonical_answer,
    full_ensemble_aggregate,
    majority_aggregate,
    score_weighted_aggregate,
    task_family,
    top_router_aggregate,
)


class ExtractionTests(unittest.TestCase):
    def test_multiple_choice_concrete_examples(self):
        examples = [
            ("arc-challenge", "['C)']", "C"),
            ("winogrande", "['\\nB']", "B"),
            ("mmlu-abstract-algebra", "Answer: A", "A"),
        ]
        for task, response, expected in examples:
            with self.subTest(task=task, response=response):
                result = canonicalize_answer(task, response)
                self.assertEqual(result.status, STATUS_OK)
                self.assertEqual(result.value, expected)

    def test_gsm8k_concrete_examples(self):
        examples = [
            ("Natalia sold 48 + 24 = 72 clips.", "72"),
            ("50/60 of an hour. The answer is USD 10.00.", "10"),
            ("work: 847 / 7; final answer: 121", "121"),
        ]
        for response, expected in examples:
            with self.subTest(response=response):
                result = canonicalize_answer("grade-school-math", response)
                self.assertEqual(result.status, STATUS_OK)
                self.assertEqual(result.value, expected)

    def test_mbpp_concrete_exact_examples(self):
        examples = [
            (
                "['def find_shared_elements(a, b):\\n    return list(set(a) & set(b))']",
                "def find_shared_elements(a, b):\n    return list(set(a) & set(b))",
            ),
            (
                "  def is_non_prime(n):\r\n    return n < 2\r\n  ",
                "def is_non_prime(n):\n    return n < 2",
            ),
            (
                "```python\ndef check_string(value):\n    return '_' in value\n```",
                "```python\ndef check_string(value):\n    return '_' in value\n```",
            ),
        ]
        for response, expected in examples:
            with self.subTest(response=response):
                result = canonicalize_answer("mbpp", response)
                self.assertEqual(result.status, STATUS_OK)
                self.assertEqual(result.value, expected)

    def test_parenthesized_code_is_not_treated_as_a_cell_wrapper(self):
        response = "(lambda value: value + 1)"
        self.assertEqual(extract_canonical_answer("mbpp", response), response)

    def test_task_families_and_unsupported_status_are_explicit(self):
        self.assertEqual(task_family("mmlu-professional-law"), "multiple_choice")
        self.assertEqual(task_family("gsm8k"), "gsm8k")
        self.assertEqual(task_family("HumanEval"), "code")

        unsupported = canonicalize_answer("consensus_summary", "some answer")
        self.assertEqual(unsupported.status, STATUS_UNSUPPORTED)
        self.assertIsNone(unsupported.value)
        self.assertIsNone(
            extract_canonical_answer("consensus_summary", "some answer")
        )

    def test_invalid_answers_do_not_become_votes(self):
        invalid_choice = canonicalize_answer("hellaswag", "I cannot decide")
        invalid_number = canonicalize_answer("gsm8k", "No numeric result")
        self.assertEqual(invalid_choice.status, STATUS_INVALID)
        self.assertEqual(invalid_number.status, STATUS_INVALID)

    def test_gsm8k_prefers_markers_and_normalizes_numbers(self):
        self.assertEqual(
            extract_canonical_answer("gsm8k", "First 1,200. #### 1,234.50"),
            "2469/2",
        )
        self.assertEqual(
            extract_canonical_answer("gsm8k", "final answer: 1/2"), "1/2"
        )
        self.assertEqual(
            extract_canonical_answer("gsm8k", "answer is 0.5"), "1/2"
        )

    def test_gsm8k_currency_answer_beats_trailing_context_number(self):
        response = "Weng earned USD 10 for 50 minutes of babysitting."
        self.assertEqual(extract_canonical_answer("gsm8k", response), "10")


class AggregationTests(unittest.TestCase):
    def test_majority_does_not_use_any_correct_model_label(self):
        result = aggregate_responses(
            "arc-challenge",
            ["A", "A", "B"],
            [0.8, 0.7, 0.1],
            method="majority",
            labels=[0, 1, 0],
        )
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.supporter_indices, (0, 1))
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIsNone(result.correct)
        self.assertIn("conflict", result.reason)

    def test_reference_scores_aggregated_answer_directly(self):
        result = majority_aggregate(
            "mmlu",
            ["A", "A", "B"],
            [0.2, 0.3, 0.9],
            reference_answer="B",
            labels=[1, 1, 0],
        )
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.status, STATUS_SCORED)
        self.assertFalse(result.correct)

    def test_majority_tie_uses_average_router_score(self):
        result = majority_aggregate(
            "hellaswag",
            ["A", "A", "B", "B"],
            [0.2, 0.2, 0.1, 0.8],
            reference_answer="B",
        )
        self.assertEqual(result.answer, "B")
        self.assertTrue(result.correct)
        self.assertEqual(result.vote_totals, (("A", 2.0), ("B", 2.0)))

    def test_score_weighted_can_differ_from_majority(self):
        responses = ["A", "A", "B"]
        scores = [0.0, 0.0, 2.0]
        majority = majority_aggregate(
            "arc-challenge", responses, scores, reference_answer="B"
        )
        weighted = score_weighted_aggregate(
            "arc-challenge", responses, scores, reference_answer="B"
        )
        self.assertEqual(majority.answer, "A")
        self.assertFalse(majority.correct)
        self.assertEqual(weighted.answer, "B")
        self.assertTrue(weighted.correct)
        totals = dict(weighted.vote_totals)
        self.assertGreater(totals["B"], totals["A"])
        self.assertAlmostEqual(sum(totals.values()), 1.0)

    def test_score_weighted_is_stable_for_large_scores(self):
        result = score_weighted_aggregate(
            "winogrande",
            ["A", "B"],
            [10_000.0, 9_999.0],
            temperature=0.5,
            reference_answer="A",
        )
        self.assertEqual(result.answer, "A")
        self.assertTrue(result.correct)
        self.assertTrue(
            all(math.isfinite(total) for _, total in result.vote_totals)
        )

    def test_top_router_uses_highest_scoring_valid_response(self):
        result = top_router_aggregate(
            "gsm8k",
            ["not numeric", "answer: 10", "answer: 12"],
            [10.0, 0.8, 0.7],
            selected_indices=[0, 1, 2],
            reference_answer="10",
        )
        self.assertEqual(result.answer, "10")
        self.assertEqual(result.supporter_indices, (1,))
        self.assertTrue(result.correct)

    def test_full_ensemble_ignores_subset(self):
        result = full_ensemble_aggregate(
            "arc-challenge",
            ["A", "B", "B"],
            [0.9, 0.2, 0.1],
            selected_indices=[0],
            reference_answer="B",
        )
        self.assertEqual(result.used_indices, (0, 1, 2))
        self.assertEqual(result.answer, "B")
        self.assertTrue(result.correct)

    def test_every_aggregator_uses_common_api(self):
        for method in (
            "majority",
            "score_weighted",
            "top_router",
            "full_ensemble",
        ):
            with self.subTest(method=method):
                result = aggregate_responses(
                    "mmlu",
                    ["A", "B"],
                    [0.9, 0.1],
                    method=method,
                    selected_indices=[0],
                    reference_answer="A",
                )
                self.assertEqual(result.method, method)
                self.assertEqual(result.status, STATUS_SCORED)
                self.assertTrue(result.correct)

    def test_empty_selection_abstains(self):
        result = aggregate_responses(
            "gsm8k", ["answer: 72"], [0.9], selected_indices=[]
        )
        self.assertEqual(result.status, STATUS_ABSTAIN)
        self.assertIsNone(result.answer)
        self.assertIsNone(result.correct)


class CodeFallbackTests(unittest.TestCase):
    def test_mbpp_exact_answer_can_use_unanimous_labels(self):
        code = "def shared(a, b):\n    return list(set(a) & set(b))"
        result = majority_aggregate(
            "mbpp",
            [code, [code], "def shared(a, b):\n    return []"],
            [0.5, 0.4, 0.9],
            labels=[1, 1, 0],
        )
        self.assertEqual(result.answer, code)
        self.assertEqual(result.supporter_indices, (0, 1))
        self.assertEqual(result.status, STATUS_SCORED)
        self.assertTrue(result.correct)
        self.assertEqual(result.reason, "unanimous supporter label fallback")

    def test_mbpp_conflicting_labels_are_unsupported(self):
        code = "def f(x):\n    return x"
        result = majority_aggregate(
            "mbpp",
            [code, code, "def f(x):\n    return None"],
            [0.9, 0.8, 0.1],
            labels=[1, 0, 0],
        )
        self.assertEqual(result.answer, code)
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIsNone(result.correct)
        self.assertIn("conflict", result.reason)

    def test_mbpp_without_reference_or_labels_is_unsupported(self):
        result = top_router_aggregate(
            "mbpp", ["def f():\n    return 1"], [0.9]
        )
        self.assertEqual(result.answer, "def f():\n    return 1")
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIsNone(result.correct)
        self.assertEqual(result.reason, "no reference answer or labels")

    def test_fractional_routerbench_labels_use_threshold(self):
        result = majority_aggregate(
            "grade-school-math",
            ["answer: 72", "72 clips"],
            [0.9, 0.8],
            labels=[0.5, 0.75],
        )
        self.assertEqual(result.status, STATUS_SCORED)
        self.assertTrue(result.correct)


class ValidationTests(unittest.TestCase):
    def test_argument_validation(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            aggregate_responses("mmlu", ["A"], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "temperature"):
            aggregate_responses("mmlu", ["A"], [0.1], temperature=0)
        with self.assertRaisesRegex(ValueError, "unknown aggregation"):
            aggregate_responses("mmlu", ["A"], [0.1], method="oracle")
        with self.assertRaisesRegex(IndexError, "out of range"):
            aggregate_responses("mmlu", ["A"], [0.1], selected_indices=[2])


if __name__ == "__main__":
    unittest.main()
