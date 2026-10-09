"""Regression tests for the prompt degeneracy guard.

Discovered 2026-09-03, after the user pushed back on a pessimistic claim: the
released RouterBench prompt field is TRUNCATED for `grade-school-math`, so all
7,450 rows share one identical prompt and the router had zero query signal on
that task. Twelve families collapse the same way, covering 8,967 rows, and the
overall duplicate rate is 24.6 percent.

The loader validated row accounting, hashes, split disjointness and label ranges
and never checked that its input FEATURES were distinct. These tests pin that
check so no future dataset reaches an experiment with collapsed features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.data_protocol import load_routerbench  # noqa: E402

MODELS = ("model-a", "model-b")


def _frame(prompts: list[str], tasks: list[str]) -> pd.DataFrame:
    rows = []
    for index, (prompt, task) in enumerate(zip(prompts, tasks)):
        row = {
            "sample_id": f"{task}.dev.{index}",
            "prompt": prompt,
            "eval_name": task,
        }
        for position, model in enumerate(MODELS):
            row[model] = 1.0 if (index + position) % 2 == 0 else 0.0
            row[f"{model}|model_response"] = f"answer {index}"
            row[f"{model}|total_cost"] = 0.001
        rows.append(row)
    return pd.DataFrame(rows)


def test_degenerate_task_is_detected() -> None:
    frame = _frame(["same prompt"] * 6, ["winogrande"] * 6)
    data = load_routerbench(frame, shot="0shot", model_columns=MODELS)
    assert "winogrande" in data.degenerate_tasks
    record = next(r for r in data.prompt_degeneracy if r["task"] == "winogrande")
    assert record["unique_prompts"] == 1
    assert record["rows"] == 6
    assert record["degenerate"] is True
    assert record["duplicate_rate"] > 0.8


def test_distinct_task_is_clean() -> None:
    frame = _frame([f"prompt {i}" for i in range(6)], ["winogrande"] * 6)
    data = load_routerbench(frame, shot="0shot", model_columns=MODELS)
    assert data.degenerate_tasks == ()
    assert data.tasks_with_duplicate_prompts == ()
    record = next(r for r in data.prompt_degeneracy if r["task"] == "winogrande")
    assert record["unique_prompts"] == 6
    assert record["duplicate_rate"] == 0.0


def test_partial_duplication_is_reported_without_being_called_degenerate() -> None:
    prompts = ["a", "a", "b", "c", "d", "e"]
    frame = _frame(prompts, ["winogrande"] * 6)
    data = load_routerbench(frame, shot="0shot", model_columns=MODELS)
    assert data.degenerate_tasks == ()
    assert "winogrande" in data.tasks_with_duplicate_prompts
    record = next(r for r in data.prompt_degeneracy if r["task"] == "winogrande")
    assert record["unique_prompts"] == 5
    assert 0.0 < record["duplicate_rate"] < 0.2


def test_degeneracy_is_per_task_not_global() -> None:
    """The loader normalizes eval_name to a task family, so assert on that name."""

    prompts = ["same"] * 4 + [f"distinct {i}" for i in range(4)]
    tasks = ["grade-school-math"] * 4 + ["winogrande"] * 4
    frame = _frame(prompts, tasks)
    data = load_routerbench(frame, shot="0shot", model_columns=MODELS)
    assert data.degenerate_tasks == ("gsm8k",)
    assert "winogrande" not in data.tasks_with_duplicate_prompts


def test_list_wrapped_prompts_are_unwrapped_before_counting() -> None:
    """RouterBench stores prompts as single-element list strings."""

    prompts = ["['same prompt']"] * 5
    frame = _frame(prompts, ["winogrande"] * 5)
    data = load_routerbench(frame, shot="0shot", model_columns=MODELS)
    assert "winogrande" in data.degenerate_tasks
