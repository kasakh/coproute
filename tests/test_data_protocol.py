"""Tests for the corrected RouterBench data and split protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORRECTED_ROOT))

from components.data_protocol import (  # noqa: E402
    GSM8K_GE_0_75_COVERAGE_POLICY,
    GSM8K_POSITIVE_COVERAGE_POLICY,
    GSM8K_STRICT_COVERAGE_POLICY,
    PRIMARY_COVERAGE_POLICY,
    SPLIT_NAMES,
    assign_racer_splits,
    augment_null_ground_truth,
    build_split_manifest,
    derive_coverage_labels,
    load_routerbench,
    load_split_manifest,
    normalize_task,
    save_split_manifest,
    shot_overlap_report,
    split_fractions_for_task,
    stable_query_id,
)


MODELS = ("model-a", "model-b")


def make_frame(rows):
    """Create a minimal RouterBench-shaped frame from compact row mappings."""

    materialized = []
    for number, supplied in enumerate(rows):
        row = {
            "sample_id": "sample." + str(number),
            "prompt": "prompt " + str(number),
            "eval_name": "hellaswag",
            "model-a": 1.0,
            "model-b": 0.0,
            "model-a|model_response": "A",
            "model-b|model_response": "B",
            "model-a|total_cost": 0.1,
            "model-b|total_cost": 0.2,
            "oracle_model_to_route_to": "model-a",
        }
        row.update(supplied)
        materialized.append(row)
    return pd.DataFrame(materialized)


def load_synthetic(rows, *, shot="0shot"):
    return load_routerbench(
        make_frame(rows), shot=shot, source_name="synthetic_" + shot,
        model_columns=MODELS,
    )


def test_task_normalization_is_explicit_and_mmlu_aware():
    assert normalize_task("mmlu-professional-law") == "mmlu"
    assert normalize_task("grade-school-math") == "gsm8k"
    assert normalize_task("arc_challenge") == "arc-challenge"
    assert normalize_task("hellaswag") == "hellaswag"
    assert normalize_task("unknown-benchmark") is None


def test_query_id_is_stable_across_prompt_and_shot_changes():
    first = stable_query_id("arc-challenge", "arc.val.3")
    second = stable_query_id("arc-challenge", "arc.val.3")
    other_task = stable_query_id("hellaswag", "arc.val.3")

    assert first == second
    assert first.startswith("rbq_")
    assert len(first) == 28
    assert first != other_task


def test_loader_accounts_for_every_row_and_names_unsupported_tasks():
    data = load_synthetic([
        {},
        {"sample_id": "unsupported.1", "eval_name": "custom-task"},
        {"sample_id": "missing-prompt", "prompt": None},
    ])

    assert data.raw_row_count == 3
    assert data.included_query_count == 1
    assert len(data.row_audit) == 3
    assert data.row_audit["reason"].tolist() == [
        "included_supported_task",
        "excluded_unsupported_task:custom-task",
        "excluded_missing_required_metadata",
    ]
    assert data.row_audit["included"].tolist() == [True, False, False]


def test_loader_preserves_fractional_official_scores_without_binarizing():
    data = load_synthetic([{
        "eval_name": "grade-school-math",
        "model-a": 0.6,
        "model-b": 0.25,
    }])

    assert data.frame.loc[0, "model-a"] == 0.6
    assert data.frame.loc[0, "model-b"] == 0.25
    assert data.frame["model-a"].dtype == float
    assert data.frame["model-b"].dtype == float


def test_complementary_duplicate_fragments_merge_without_losing_scores():
    data = load_synthetic([
        {
            "sample_id": "arc.val.1",
            "eval_name": "arc-challenge",
            "prompt": "one-model prompt",
            "model-a": 1.0,
            "model-b": np.nan,
            "model-b|model_response": "stale B",
            "model-b|total_cost": 9.9,
        },
        {
            "sample_id": "arc.val.1",
            "eval_name": "arc-challenge",
            "prompt": "majority prompt",
            "model-a": np.nan,
            "model-b": 1.0,
            "model-a|model_response": "stale A",
            "model-a|total_cost": 8.8,
        },
    ], shot="5shot")

    assert data.included_query_count == 1
    np.testing.assert_allclose(
        data.frame.loc[0, list(MODELS)].to_numpy(dtype=float), [1.0, 1.0]
    )
    np.testing.assert_allclose(
        data.frame.loc[0, ["model-a|total_cost", "model-b|total_cost"]].to_numpy(dtype=float),
        [0.1, 0.2],
    )
    assert data.frame.loc[0, "model-a|model_response"] == "A"
    assert data.frame.loc[0, "model-b|model_response"] == "B"
    assert data.frame.loc[0, "source_rows"] == (0, 1)
    assert data.row_audit["included"].tolist() == [True, True]
    assert set(data.row_audit["reason"]) == {"included_merged_complementary_rows"}


def test_duplicate_merge_is_stable_under_source_row_reordering():
    rows = [
        {
            "sample_id": "arc.val.1", "eval_name": "arc-challenge",
            "prompt": "sparse", "model-a": 1.0, "model-b": np.nan,
            "model-b|model_response": np.nan, "model-b|total_cost": np.nan,
        },
        {
            "sample_id": "arc.val.1", "eval_name": "arc-challenge",
            "prompt": "complete", "model-a": np.nan, "model-b": 0.0,
            "model-a|model_response": np.nan, "model-a|total_cost": np.nan,
        },
    ]
    left = load_synthetic(rows, shot="5shot")
    right = load_synthetic(list(reversed(rows)), shot="5shot")

    assert left.query_ids == right.query_ids
    assert left.frame.loc[0, "prompt"] == right.frame.loc[0, "prompt"]
    assert left.frame.loc[0, "prompt"] in {"sparse", "complete"}
    np.testing.assert_array_equal(
        left.frame.loc[:, list(MODELS)].to_numpy(),
        right.frame.loc[:, list(MODELS)].to_numpy(),
    )


def test_conflicting_duplicate_score_excludes_every_contributing_row():
    data = load_synthetic([
        {"sample_id": "same", "model-a": 1.0},
        {"sample_id": "same", "model-a": 0.0},
    ])

    assert data.included_query_count == 0
    assert data.row_audit["included"].tolist() == [False, False]
    assert set(data.row_audit["reason"]) == {"excluded_conflicting_duplicate_scores"}


def test_incomplete_supported_query_is_explicitly_excluded():
    data = load_synthetic([{"model-b": np.nan}])

    assert data.included_query_count == 0
    assert data.row_audit.loc[0, "reason"] == "excluded_missing_correctness"


def test_out_of_range_correctness_is_explicitly_excluded():
    data = load_synthetic([{"model-b": 1.1}])

    assert data.included_query_count == 0
    assert data.row_audit.loc[0, "reason"] == "excluded_invalid_correctness"


def test_coverage_policy_must_be_explicit_and_named():
    data = load_synthetic([{}])

    with pytest.raises(TypeError):
        derive_coverage_labels(data)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="coverage policy must be explicit"):
        derive_coverage_labels(data, policy="implicit-default")


def test_primary_gsm8k_policy_uses_documented_half_threshold():
    data = load_synthetic([{
        "eval_name": "grade-school-math", "model-a": 0.5, "model-b": 0.4,
    }])

    result = derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY)

    np.testing.assert_array_equal(result.standard, [[True, False]])
    np.testing.assert_array_equal(result.augmented, [[True, False, False]])
    assert result.policy.name == "task_primary_gsm8k_ge_0_5"


def test_gsm8k_sensitivity_variants_produce_distinct_labels():
    data = load_synthetic([
        {"eval_name": "grade-school-math", "model-a": 0.8, "model-b": 0.6},
        {"eval_name": "grade-school-math", "model-a": 0.1, "model-b": 1.0},
    ])

    strict = derive_coverage_labels(data, policy=GSM8K_STRICT_COVERAGE_POLICY)
    medium = derive_coverage_labels(data, policy=GSM8K_GE_0_75_COVERAGE_POLICY)
    positive = derive_coverage_labels(data, policy=GSM8K_POSITIVE_COVERAGE_POLICY)

    strict_by_sample = dict(zip(data.frame["sample_id"], strict.standard.tolist()))
    medium_by_sample = dict(zip(data.frame["sample_id"], medium.standard.tolist()))
    positive_by_sample = dict(zip(data.frame["sample_id"], positive.standard.tolist()))
    assert strict_by_sample == {
        "sample.0": [False, False], "sample.1": [False, True]
    }
    assert medium_by_sample == {
        "sample.0": [True, False], "sample.1": [False, True]
    }
    assert positive_by_sample == {
        "sample.0": [True, True], "sample.1": [True, True]
    }


def test_binary_task_rejects_fractional_scores_instead_of_thresholding():
    data = load_synthetic([{"eval_name": "hellaswag", "model-a": 0.8}])

    with pytest.raises(ValueError, match="binary-only task hellaswag"):
        derive_coverage_labels(data, policy=PRIMARY_COVERAGE_POLICY)


def test_null_ground_truth_is_true_only_when_every_standard_model_misses():
    standard = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.int8)

    augmented = augment_null_ground_truth(standard)

    np.testing.assert_array_equal(
        augmented,
        [[True, False, False], [False, False, True], [False, True, False]],
    )


def test_null_ground_truth_rejects_nonbinary_input():
    with pytest.raises(ValueError, match="must be binary"):
        augment_null_ground_truth(np.array([[0.2, 0.0]]))


def test_published_racer_fractions_are_preserved_with_design_carveout():
    assert split_fractions_for_task("gsm8k") == {
        "train": 0.25,
        "design": 0.25,
        "calibration": 0.25,
        "validation": 0.05,
        "test": 0.2,
    }
    assert split_fractions_for_task("mmlu") == {
        "train": 0.2,
        "design": 0.2,
        "calibration": 0.3,
        "validation": 0.06,
        "test": 0.24,
    }
    assert split_fractions_for_task("arc-challenge") == {
        "train": 0.2,
        "design": 0.2,
        "calibration": 0.24,
        "validation": 0.12,
        "test": 0.24,
    }


def test_custom_design_share_preserves_total_racer_training_budget():
    fractions = split_fractions_for_task(
        "hellaswag", design_share_of_racer_train=0.25
    )

    assert fractions["train"] == pytest.approx(0.3)
    assert fractions["design"] == pytest.approx(0.1)
    assert fractions["train"] + fractions["design"] == pytest.approx(0.4)
    assert sum(fractions.values()) == pytest.approx(1.0)


def many_task_rows(task, count):
    eval_name = {
        "gsm8k": "grade-school-math",
        "arc-challenge": "arc-challenge",
    }.get(task, task)
    return [
        {"sample_id": task + "." + str(index), "eval_name": eval_name}
        for index in range(count)
    ]


def test_split_counts_match_racer_fractions_at_one_hundred_queries():
    rows = many_task_rows("gsm8k", 100)
    rows += many_task_rows("mmlu", 100)
    rows += many_task_rows("arc-challenge", 100)
    data = load_synthetic(rows)

    assignment = assign_racer_splits(data, seed=17)
    counts = assignment.assignments.groupby(["task", "split"]).size()

    assert [counts["gsm8k", split] for split in SPLIT_NAMES] == [25, 25, 25, 5, 20]
    assert [counts["mmlu", split] for split in SPLIT_NAMES] == [20, 20, 30, 6, 24]
    assert [counts["arc-challenge", split] for split in SPLIT_NAMES] == [20, 20, 24, 12, 24]


def test_split_allocation_preserves_racer_budget_before_subsplitting():
    data = load_synthetic(many_task_rows("gsm8k", 11))

    assignment = assign_racer_splits(data, seed=17)
    counts = assignment.assignments["split"].value_counts()

    assert counts["train"] + counts["design"] == 6
    assert counts["calibration"] + counts.get("validation", 0) + counts["test"] == 5
    assert [counts.get(split, 0) for split in SPLIT_NAMES] == [3, 3, 3, 0, 2]


def test_splits_are_complete_disjoint_seeded_and_order_invariant():
    rows = many_task_rows("hellaswag", 101)
    left_data = load_synthetic(rows)
    right_data = load_synthetic(list(reversed(rows)))

    left = assign_racer_splits(left_data, seed=123)
    repeated = assign_racer_splits(left_data, seed=123)
    reordered = assign_racer_splits(right_data, seed=123)
    other_seed = assign_racer_splits(left_data, seed=124)

    left_map = dict(zip(left.assignments.query_id, left.assignments.split))
    assert left_map == dict(zip(repeated.assignments.query_id, repeated.assignments.split))
    assert left_map == dict(zip(reordered.assignments.query_id, reordered.assignments.split))
    assert left_map != dict(zip(other_seed.assignments.query_id, other_seed.assignments.split))
    assert len(left_map) == 101
    assert set(left_map) == set(left_data.query_ids)


def test_shot_overlap_reports_correlated_reuse_not_independent_replication():
    zero = load_synthetic([
        {"sample_id": "shared", "prompt": "zero prompt"},
        {"sample_id": "zero-only"},
    ], shot="0shot")
    five = load_synthetic([
        {"sample_id": "shared", "prompt": "five prompt", "model-b": 1.0},
        {"sample_id": "five-only"},
    ], shot="5shot")

    report = shot_overlap_report(zero, five)

    assert report["included_queries"]["overlap"] == 1
    assert report["included_queries"]["left_only"] == 1
    assert report["included_queries"]["right_only"] == 1
    assert report["shared_included_prompt_equal"] == 0
    assert report["shared_included_prompt_different"] == 1
    assert report["shared_included_official_scores_different"] == 1


def test_overlap_counts_duplicate_raw_rows_once_as_query_identity():
    zero = load_synthetic([{"sample_id": "shared"}], shot="0shot")
    five = load_synthetic([
        {
            "sample_id": "shared", "model-a": 1.0, "model-b": np.nan,
            "model-b|model_response": np.nan, "model-b|total_cost": np.nan,
        },
        {
            "sample_id": "shared", "model-a": np.nan, "model-b": 0.0,
            "model-a|model_response": np.nan, "model-a|total_cost": np.nan,
        },
    ], shot="5shot")

    report = shot_overlap_report(zero, five)

    assert report["raw_rows"] == {"left": 1, "right": 2}
    assert report["all_query_identities"]["left_unique"] == 1
    assert report["all_query_identities"]["right_unique"] == 1
    assert report["included_queries"]["overlap"] == 1


def test_manifest_contains_row_accounting_split_ids_and_hashes(tmp_path):
    data = load_synthetic(
        many_task_rows("hellaswag", 20)
        + [{"sample_id": "excluded", "eval_name": "custom"}]
    )
    assignment = assign_racer_splits(data, seed=29)
    manifest = build_split_manifest(
        data, assignment, coverage_policy=PRIMARY_COVERAGE_POLICY
    )

    assert manifest["source"]["raw_rows"] == 21
    assert manifest["source"]["included_queries"] == 20
    assert manifest["source"]["excluded_source_rows"] == 1
    assert len(manifest["row_accounting"]) == 21
    assert set(manifest["splits"]["hellaswag"]) == set(SPLIT_NAMES)
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["all_included_query_ids_sha256"]) == 64
    assert manifest["coverage_policy"]["name"] == PRIMARY_COVERAGE_POLICY.name

    path = tmp_path / "split_manifest.json"
    saved_hash = save_split_manifest(manifest, path)
    loaded = load_split_manifest(path)

    assert len(saved_hash) == 64
    assert loaded == manifest


def test_manifest_tampering_is_detected(tmp_path):
    data = load_synthetic(many_task_rows("hellaswag", 10))
    assignment = assign_racer_splits(data, seed=7)
    manifest = build_split_manifest(data, assignment)
    path = tmp_path / "split_manifest.json"
    save_split_manifest(manifest, path)

    payload = json.loads(path.read_text())
    payload["seed"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_split_manifest(path)


def test_split_hash_tampering_is_detected_even_with_recomputed_manifest_hash(tmp_path):
    data = load_synthetic(many_task_rows("hellaswag", 10))
    assignment = assign_racer_splits(data, seed=7)
    manifest = build_split_manifest(data, assignment)
    manifest["splits"]["hellaswag"]["test"].append("fabricated-query")

    with pytest.raises(ValueError, match="split hash mismatch"):
        save_split_manifest(manifest, tmp_path / "bad.json")


def test_manifest_can_verify_original_pickle_bytes(tmp_path):
    frame = make_frame(many_task_rows("hellaswag", 10))
    source_path = tmp_path / "routerbench.pkl"
    frame.to_pickle(source_path)
    data = load_routerbench(source_path, model_columns=MODELS)
    assignment = assign_racer_splits(data, seed=3)
    manifest_path = tmp_path / "manifest.json"
    save_split_manifest(build_split_manifest(data, assignment), manifest_path)

    loaded = load_split_manifest(manifest_path, source_path=source_path)
    assert loaded["source"]["sha256"] == data.source_sha256

    changed = frame.copy()
    changed.loc[0, "prompt"] = "changed"
    changed.to_pickle(source_path)
    with pytest.raises(ValueError, match="source file hash mismatch"):
        load_split_manifest(manifest_path, source_path=source_path)


def test_loader_does_not_mutate_caller_dataframe():
    frame = make_frame([{"eval_name": "grade-school-math", "model-a": 0.6}])
    before = frame.copy(deep=True)

    load_routerbench(frame, model_columns=MODELS)

    pd.testing.assert_frame_equal(frame, before)


def test_loader_records_named_source_index_without_schema_collision():
    frame = make_frame([{}])
    frame.index = pd.Index(["source-row-a"], name="custom_index")

    data = load_routerbench(frame, model_columns=MODELS)

    assert data.row_audit.loc[0, "source_index"] == repr("source-row-a")
    assert data.frame.loc[0, "_source_index"] == "source-row-a"


def test_manifest_build_rejects_assignment_from_another_dataset():
    left = load_synthetic(many_task_rows("hellaswag", 10))
    right = load_synthetic(many_task_rows("hellaswag", 11))
    assignment = assign_racer_splits(left, seed=1)

    with pytest.raises(ValueError, match="query IDs differ"):
        build_split_manifest(right, assignment)


def test_manifest_roundtrip_does_not_depend_on_mapping_insertion_order(tmp_path):
    data = load_synthetic(many_task_rows("hellaswag", 10))
    assignment = assign_racer_splits(data, seed=5)
    original = build_split_manifest(data, assignment)
    reordered = {key: copy.deepcopy(original[key]) for key in reversed(original)}
    path = tmp_path / "manifest.json"

    save_split_manifest(reordered, path)
    loaded = load_split_manifest(path)

    assert loaded == original
