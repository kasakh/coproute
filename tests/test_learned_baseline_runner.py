"""Synthetic protocol tests for the learned-baseline runner and summarizer."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.barp_baseline import BaRPBaseline, BaRPConfig  # noqa: E402
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402
from figures_src import summarize_learned_baselines as summary  # noqa: E402
from scripts import run_learned_baselines as learned  # noqa: E402


class _SavedFakeModel:
    def __init__(self) -> None:
        self.training_history = [{"epoch": 0, "validation": 1.0}]
        self.best_epoch = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"deterministic-checkpoint")


def _task(task: str, rows: int, models: int, *, label_value: int) -> learned.LearnedTaskData:
    embeddings = np.arange(1, rows * 4 + 1, dtype=float).reshape(rows, 4)
    embeddings = learned.l2_normalize_rows(embeddings)
    correctness = np.zeros((rows, models), dtype=np.int8)
    correctness[:, label_value % models] = 1
    return learned.LearnedTaskData(
        task=task,
        query_ids=tuple(f"{task}:{index}" for index in range(rows)),
        embeddings=embeddings,
        correctness=correctness,
        model_ids=tuple(f"model-{index}" for index in range(models)),
        prompts=tuple(f"prompt {task} {index}" for index in range(rows)),
        feature_audit={"degenerate": False, "duplicate_rate": 0.0},
        source_hashes={"fixture": task * 8},
    )


def _routereval_tasks(*targets: str) -> dict[str, learned.LearnedTaskData]:
    requested = targets or ("ifeval",)
    tasks = {
        "bbh": _task("bbh", 20, 12, label_value=0),
        "math": _task("math", 20, 12, label_value=1),
        "gpqa": _task("gpqa", 20, 12, label_value=2),
    }
    if "ifeval" in requested:
        tasks["ifeval"] = _task("ifeval", 40, 12, label_value=9)
    if "musr" in requested:
        tasks["musr"] = _task("musr", 44, 12, label_value=10)
    return tasks


def _frozen(rows: int, models: int) -> learned.FrozenScores:
    depth = min(10, models)
    orders = tuple(tuple(range(depth)) for _ in range(rows))
    scores = np.full((rows, models), np.inf)
    scores[:, :depth] = np.linspace(0.0, 0.9, depth)
    auxiliary = tuple(
        {
            "gains": list(np.linspace(1.0, 0.1, depth)),
            "relative_gains": list(np.linspace(1.0, 0.1, depth)),
        }
        for _ in range(rows)
    )
    return learned.FrozenScores(orders, scores, auxiliary)


def test_target_redraw_has_exact_registered_roles_and_is_deterministic() -> None:
    first = learned.target_redraw_split(101, 3)
    second = learned.target_redraw_split(101, 3)
    assert {name: len(rows) for name, rows in first.items()} == {
        "design": 30,
        "router": 20,
        "final_calibration": 20,
        "validation": 15,
        "test": 16,
    }
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    joined = np.concatenate(tuple(first.values()))
    assert len(np.unique(joined)) == 101
    assert set(joined) == set(range(101))


def test_external_hash_split_is_stable_and_largest_remainder() -> None:
    ids = [f"q{index}" for index in range(13)]
    first = learned.stable_hash_split(
        ids,
        {"fit": 0.8, "validation": 0.2},
        seed=learned.EXTERNAL_SPLIT_SEED,
        namespace="unit",
    )
    repeated = learned.stable_hash_split(
        ids,
        {"fit": 0.8, "validation": 0.2},
        seed=learned.EXTERNAL_SPLIT_SEED,
        namespace="unit",
    )
    assert len(first["fit"]) == 10
    assert len(first["validation"]) == 3
    np.testing.assert_array_equal(first["fit"], repeated["fit"])
    assert not set(first["fit"]) & set(first["validation"])


def test_routereval_training_never_receives_target_labels_and_pool_is_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _routereval_tasks()
    captured: dict[str, np.ndarray | tuple[str, ...]] = {}

    def fake_train(method, train_embeddings, train_labels, validation_embeddings, validation_labels, model_ids, **kwargs):
        captured["train_labels"] = np.asarray(train_labels).copy()
        captured["validation_labels"] = np.asarray(validation_labels).copy()
        captured["model_ids"] = tuple(model_ids)
        return _SavedFakeModel()

    def fake_score(model, method, embeddings, n_models, **kwargs):
        return _frozen(len(embeddings), n_models)

    monkeypatch.setattr(learned, "train_learned_model", fake_train)
    monkeypatch.setattr(learned, "score_learned_model", fake_score)
    output = tmp_path / "run"
    result = learned.run_routereval(
        tasks=tasks,
        target_task="ifeval",
        method="flexrouter",
        pool_size=10,
        pool_seed=7001,
        training_seed=9101,
        output_dir=output,
        redraws=2,
        alphas=(0.5,),
        max_epochs=1,
        patience=1,
        expected_embedding_dim=None,
    )

    assert np.all(np.asarray(captured["train_labels"])[:, 9] == 0)
    assert np.all(np.asarray(captured["validation_labels"])[:, 9] == 0)
    assert result["training"]["target_labels_used"] is False
    expected_ids = tuple(result["model_pool"]["model_ids"])
    assert captured["model_ids"] == expected_ids
    assert len(expected_ids) == 10
    assert result["model_pool"]["fixed_across_tasks"] is True
    assert result["native_and_adapted_separated"] is True
    for redraw in result["per_redraw"]:
        assert redraw["target_label_access"] == {
            "design": False,
            "router": False,
            "validation": False,
            "final_calibration": True,
            "test": "evaluation_only_after_freeze",
        }
        assert redraw["native_fixed_depth"]["risk_certificate"] is False
        assert all(point["risk_certificate"] is False for point in redraw["native_adaptive"])
        assert all(point["final_calibration_count"] == 1 for point in redraw["adapted_calibrated"])
    assert (output / "checkpoint.pt").is_file()
    assert (output / "training_history.json").is_file()
    assert (output / "results.json").is_file()
    assert (output / "manifest.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        learned.run_routereval(
            tasks=tasks,
            target_task="ifeval",
            method="flexrouter",
            pool_size=10,
            pool_seed=7001,
            training_seed=9101,
            output_dir=output,
            redraws=1,
            alphas=(0.5,),
            max_epochs=1,
            patience=1,
            expected_embedding_dim=None,
        )



def test_routereval_all_trains_once_without_target_labels_and_stores_both_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _routereval_tasks("ifeval", "musr")
    tasks["ifeval"].correctness[:] = 7
    tasks["musr"].correctness[:] = 8
    training_calls = []
    scoring_labels = []

    def fake_train(method, train_embeddings, train_labels, validation_embeddings, validation_labels, model_ids, **kwargs):
        training_calls.append(
            (np.asarray(train_labels).copy(), np.asarray(validation_labels).copy())
        )
        return _SavedFakeModel()

    def fake_score(model, method, embeddings, n_models, **kwargs):
        scoring_labels.append(id(embeddings))
        return _frozen(len(embeddings), n_models)

    monkeypatch.setattr(learned, "train_learned_model", fake_train)
    monkeypatch.setattr(learned, "score_learned_model", fake_score)
    output = tmp_path / "shared"
    result = learned.run_routereval_all(
        tasks=tasks,
        method="flexrouter",
        pool_size=10,
        pool_seed=7001,
        training_seed=9101,
        output_dir=output,
        redraws=2,
        alphas=(0.5,),
        max_epochs=1,
        patience=1,
        expected_embedding_dim=None,
    )

    assert len(training_calls) == 1
    train_labels, validation_labels = training_calls[0]
    assert set(np.unique(train_labels)) <= {0, 1}
    assert set(np.unique(validation_labels)) <= {0, 1}
    assert scoring_labels == [id(tasks[task].embeddings) for task in learned.TARGET_TASKS]
    assert result["training_metadata"]["training"]["target_labels_used"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "checkpoint.pt",
        "manifest.json",
        "results_ifeval.json",
        "results_musr.json",
        "training_history.json",
        "training_metadata.json",
    ]
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        "checkpoint.pt",
        "results_ifeval.json",
        "results_musr.json",
        "training_history.json",
        "training_metadata.json",
    }
    for filename, digest in manifest["files"].items():
        assert learned.sha256_file(output / filename) == digest
    for target_task in learned.TARGET_TASKS:
        target_result = json.loads((output / f"results_{target_task}.json").read_text())
        assert target_result["identity"]["target_task"] == target_task
        assert len(target_result["per_redraw"]) == 2
        for redraw in target_result["per_redraw"]:
            assert set(redraw["split_ids"]) == {
                "design", "router", "final_calibration", "validation", "test"
            }
            assert all(
                query_id.startswith(target_task + ":")
                for split_ids in redraw["split_ids"].values()
                for query_id in split_ids
            )


def test_multi_target_outputs_equal_eval_only_references_from_frozen_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _routereval_tasks("ifeval", "musr")

    def fake_train(*args, **kwargs):
        return _SavedFakeModel()

    def fake_score(model, method, embeddings, n_models, **kwargs):
        return _frozen(len(embeddings), n_models)

    monkeypatch.setattr(learned, "train_learned_model", fake_train)
    monkeypatch.setattr(learned, "score_learned_model", fake_score)
    checkpoint = learned.train_routereval_checkpoint(
        tasks={name: tasks[name] for name in learned.EXTERNAL_TASKS},
        method="flexrouter",
        pool_size=10,
        pool_seed=7001,
        training_seed=9101,
        max_epochs=1,
        patience=1,
        expected_embedding_dim=None,
    )
    references = {
        target_task: learned.evaluate_routereval_target(
            checkpoint=checkpoint,
            target=tasks[target_task],
            redraws=2,
            alphas=(0.5,),
        )
        for target_task in learned.TARGET_TASKS
    }
    monkeypatch.setattr(learned, "train_routereval_checkpoint", lambda **kwargs: checkpoint)
    output = tmp_path / "shared-reference"
    learned.run_routereval_all(
        tasks=tasks,
        method="flexrouter",
        pool_size=10,
        pool_seed=7001,
        training_seed=9101,
        output_dir=output,
        redraws=2,
        alphas=(0.5,),
        max_epochs=1,
        patience=1,
        expected_embedding_dim=None,
    )
    for target_task, reference in references.items():
        stored = json.loads((output / f"results_{target_task}.json").read_text())
        assert stored == reference


def test_summary_discovers_legacy_and_multi_target_layouts(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    shared = tmp_path / "shared"
    legacy.mkdir()
    shared.mkdir()
    (legacy / "results.json").write_text(json.dumps(_minimal_result(9101, 0.8, 2.0)))
    ifeval = _minimal_result(9102, 0.9, 2.5)
    musr = _minimal_result(9102, 0.7, 3.5)
    musr["identity"]["target_task"] = "musr"
    (shared / "results_ifeval.json").write_text(json.dumps(ifeval))
    (shared / "results_musr.json").write_text(json.dumps(musr))
    (shared / "training_metadata.json").write_text(
        json.dumps({"protocol": learned.PROTOCOL_NAME, "mode": "routereval_shared_training"})
    )

    discovered = summary.discover_results([legacy, shared])
    assert [Path(source).name for source, _ in discovered] == [
        "results.json", "results_ifeval.json", "results_musr.json"
    ]
    result = summary.summarize_results(discovered)
    assert {row["target_task"] for row in result["rows"]} == {"ifeval", "musr"}

def test_calibration_is_called_once_and_noncertification_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _frozen(10, 10)
    labels = np.zeros((10, 10), dtype=np.int8)
    labels[:, 9] = 1
    calls = 0
    original = learned.select_crc_threshold

    def counted(critical, alpha):
        nonlocal calls
        calls += 1
        return original(critical, alpha)

    monkeypatch.setattr(learned, "select_crc_threshold", counted)
    record = learned.calibrated_record(
        frozen,
        labels,
        np.arange(4),
        np.arange(4, 10),
        tuple(f"q{index}" for index in range(10)),
        tuple(f"m{index}" for index in range(10)),
        0.02,
    )
    assert calls == 1
    assert record["final_calibration_count"] == 1
    assert record["certified"] is False
    assert record["non_certification_reason"] == "below_finite_sample_resolution"
    assert record["mean_calls"] is None
    assert record["risk_validity_claimed"] is False


def test_fixed_and_adaptive_views_use_distinct_registered_rules() -> None:
    frozen = _frozen(4, 10)
    labels = np.zeros((4, 10), dtype=np.int8)
    labels[:, 5] = 1
    ids = tuple(f"q{index}" for index in range(4))
    models = tuple(f"m{index}" for index in range(10))
    fixed = learned.fixed_depth_record(frozen, labels, np.arange(4), ids, models)
    adaptive = learned.flex_adaptive_record(frozen, labels, np.arange(4), ids, models)
    assert fixed["depths"][4]["success"] == 0.0
    assert fixed["depths"][5]["success"] == 1.0
    assert [point["tau"] for point in adaptive] == list(learned.FLEX_TAU_GRID)
    assert fixed["comparison_kind"] == "native_fixed_depth"
    assert all("tau" in point and point["risk_certificate"] is False for point in adaptive)


def test_nondominated_frontier_is_mechanical_and_keeps_ties() -> None:
    points = [
        {"mean_official_quality": 0.7, "mean_cost": 3.0},
        {"mean_official_quality": 0.8, "mean_cost": 2.0},
        {"mean_official_quality": 0.9, "mean_cost": 4.0},
        {"mean_official_quality": 0.8, "mean_cost": 2.0},
    ]
    assert learned.nondominated_frontier(points) == [1, 2, 3]


def _routerbench_fixture() -> SimpleNamespace:
    rows = []
    counter = 0
    task_to_eval = {
        "mmlu": "mmlu-anatomy",
        "arc-challenge": "arc-challenge",
        "winogrande": "winogrande",
        "gsm8k": "grade-school-math",
        "mbpp": "mbpp",
        "hellaswag": "hellaswag",
    }
    for task, eval_name in task_to_eval.items():
        for index in range(20):
            query_id = f"rb:{task}:{index}"
            row = {
                "query_id": query_id,
                "task": task,
                "prompt": f"prompt {task} {index}",
                "eval_name": eval_name,
                "sample_id": str(counter),
            }
            for model_index, model in enumerate(ROUTERBENCH_MODEL_COLUMNS):
                row[model] = float(model_index == (index % 2 if task in ("mbpp", "hellaswag") else 0))
                row[model + "|total_cost"] = float(model_index + 1)
                row[model + "|model_response"] = "response"
            rows.append(row)
            counter += 1
    frame = pd.DataFrame(rows)
    audits = tuple(
        {
            "task": task,
            "rows": 20,
            "unique_prompts": 20,
            "duplicate_rate": 0.0,
            "degenerate": False,
        }
        for task in task_to_eval
    )
    return SimpleNamespace(
        frame=frame,
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
        query_ids=tuple(frame["query_id"]),
        prompt_degeneracy=audits,
        shot="0shot",
        source_sha256="a" * 64,
    )


def test_routerbench_native_barp_uses_train_cost_tau_and_all_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _routerbench_fixture()
    embeddings = learned.l2_normalize_rows(np.arange(1, len(data.frame) * 4 + 1).reshape(len(data.frame), 4))
    captured = {}

    def fake_train(method, train_embeddings, train_labels, validation_embeddings, validation_labels, model_ids, **kwargs):
        captured["train_labels"] = np.asarray(train_labels).copy()
        captured["tau"] = kwargs["tau"]
        model = BaRPBaseline(
            4,
            11,
            config=BaRPConfig(tau=kwargs["tau"], max_epochs=1, seed=kwargs["training_seed"]),
            model_ids=model_ids,
        )
        model.is_fitted = True
        model.training_history = [{"epoch": 0, "validation_mean_reward": 0.0}]
        model.best_epoch = 0

        def predict(features, preference=None):
            preference = [1.0, 0.0] if preference is None else preference
            action = 0 if float(preference[0]) >= float(preference[1]) else 1
            return torch.full((len(features),), action, dtype=torch.long)

        model.predict = predict  # type: ignore[method-assign]
        return model

    monkeypatch.setattr(learned, "train_learned_model", fake_train)
    result = learned.run_routerbench_native_barp(
        data=data,
        embeddings=embeddings,
        embedding_audit={"source_sha256": "b" * 64},
        training_seed=9101,
        output_dir=tmp_path / "barp",
        max_epochs=1,
        patience=1,
    )
    assert np.all(captured["train_labels"][:, 0] == 1.0)
    assert captured["tau"] == pytest.approx(result["configuration"]["tau"])
    assert len(result["native_barp"]["macro_average"]) == 11
    assert all(len(points) == 11 for points in result["native_barp"]["tasks"].values())
    assert result["model_pool"] == {
        "model_ids": list(ROUTERBENCH_MODEL_COLUMNS),
        "fixed_complete_pool": True,
        "pool_size": 11,
    }
    assert result["native_barp"]["risk_certificate"] is False


def _set_gsm8k_degenerate(data: SimpleNamespace) -> None:
    data.prompt_degeneracy = tuple(
        {
            **record,
            "unique_prompts": 1,
            "duplicate_rate": 0.95,
            "degenerate": True,
        }
        if record["task"] == "gsm8k"
        else record
        for record in data.prompt_degeneracy
    )


def test_routerbench_primary_writes_unavailable_artifact_for_degenerate_gsm8k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _routerbench_fixture()
    _set_gsm8k_degenerate(data)
    embeddings = learned.l2_normalize_rows(
        np.arange(1, len(data.frame) * 4 + 1).reshape(len(data.frame), 4)
    )

    def forbidden_train(*args, **kwargs):
        raise AssertionError("primary unavailable cell must not start training")

    monkeypatch.setattr(learned, "train_learned_model", forbidden_train)
    output = tmp_path / "primary-unavailable"
    result = learned.run_routerbench_native_barp(
        data=data,
        embeddings=embeddings,
        embedding_audit={"source_sha256": "b" * 64, "manifest_sha256": "c" * 64},
        training_seed=9101,
        output_dir=output,
        max_epochs=1,
        patience=1,
    )

    assert result["status"] == "unavailable"
    assert result["analysis_label"] == "primary_four_training_tasks"
    assert result["configuration"]["training_tasks"] == list(learned.ROUTERBENCH_TRAIN_TASKS)
    assert result["configuration"]["substituted_tasks"] == []
    assert result["failure"]["stage"] == "prompt_feature_audit"
    assert any("gsm8k" in reason for reason in result["failure"]["failures"])
    assert result["training"]["ood_target_labels_used"] is False
    assert sorted(path.name for path in output.iterdir()) == ["manifest.json", "results.json"]
    stored = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert stored["status"] == "unavailable"
    assert not (output / "checkpoint.pt").exists()


def test_routerbench_three_task_sensitivity_uses_exact_tasks_and_no_target_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _routerbench_fixture()
    _set_gsm8k_degenerate(data)
    embeddings = learned.l2_normalize_rows(
        np.arange(1, len(data.frame) * 4 + 1).reshape(len(data.frame), 4)
    )
    captured: dict[str, np.ndarray] = {}

    def fake_train(method, train_embeddings, train_labels, validation_embeddings, validation_labels, model_ids, **kwargs):
        captured["train_labels"] = np.asarray(train_labels).copy()
        captured["validation_labels"] = np.asarray(validation_labels).copy()
        model = BaRPBaseline(
            4,
            11,
            config=BaRPConfig(tau=kwargs["tau"], max_epochs=1, seed=kwargs["training_seed"]),
            model_ids=model_ids,
        )
        model.is_fitted = True
        model.training_history = [{"epoch": 0, "validation_mean_reward": 0.0}]
        model.best_epoch = 0
        model.predict = lambda features, preference=None: torch.zeros(  # type: ignore[method-assign]
            len(features), dtype=torch.long
        )
        return model

    monkeypatch.setattr(learned, "train_learned_model", fake_train)
    result = learned.run_routerbench_native_barp(
        data=data,
        embeddings=embeddings,
        embedding_audit={"source_sha256": "b" * 64, "manifest_sha256": "c" * 64},
        training_seed=9101,
        output_dir=tmp_path / "sensitivity",
        max_epochs=1,
        patience=1,
        three_task_sensitivity=True,
    )

    assert result["status"] == "available"
    assert result["analysis_label"] == "sensitivity_without_degenerate_gsm8k"
    assert result["configuration"]["training_tasks"] == [
        "mmlu", "arc-challenge", "winogrande"
    ]
    assert result["configuration"]["substituted_tasks"] == []
    assert result["configuration"]["target_tasks"] == ["mbpp", "hellaswag"]
    assert set(result["native_barp"]["tasks"]) == {"mbpp", "hellaswag"}
    assert result["training"]["ood_target_labels_used"] is False
    assert len(captured["train_labels"]) == result["training"]["fit_query_count"]
    assert len(captured["validation_labels"]) == result["training"]["validation_query_count"]
    assert np.all(captured["train_labels"][:, 0] == 1.0)
    assert np.all(captured["train_labels"][:, 1:] == 0.0)
    assert np.all(captured["validation_labels"][:, 0] == 1.0)
    assert np.all(captured["validation_labels"][:, 1:] == 0.0)


def _minimal_result(seed: int, success: float, calls: float) -> dict:
    fixed_point = {
        "depth": 1,
        "success": success,
        "zero_correct_rate": 1.0 - success,
        "mean_calls": 1.0,
    }
    return {
        "protocol": learned.PROTOCOL_NAME,
        "mode": "routereval",
        "identity": {
            "method": "barp",
            "target_task": "ifeval",
            "pool_size": 100,
            "pool_seed": 7001,
            "training_seed": seed,
        },
        "per_redraw": [
            {
                "redraw": 0,
                "native_fixed_depth": {"success_auc": success, "depths": [fixed_point]},
                "native_adaptive": [],
                "adapted_calibrated": [
                    {
                        "alpha": 0.1,
                        "certified": True,
                        "standard_all_miss_risk": 1.0 - success,
                        "mean_calls": calls,
                        "median_calls": calls,
                        "p90_calls": calls,
                    }
                ],
            }
        ],
    }


def test_summary_aggregates_training_seeds_and_keeps_blocks_separate() -> None:
    records = [
        ("seed9101/results.json", _minimal_result(9101, 0.8, 2.0)),
        ("seed9102/results.json", _minimal_result(9102, 1.0, 4.0)),
    ]
    result = summary.summarize_results(records)
    assert set(result["blocks"]) == {
        "native_fixed_depth",
        "native_adaptive",
        "adapted_calibrated",
        "native_barp",
        "native_barp_sensitivity",
        "unavailable",
    }
    fixed = result["blocks"]["native_fixed_depth"][0]
    calibrated = result["blocks"]["adapted_calibrated"][0]
    assert fixed["training_seed_count"] == 2
    assert fixed["success_mean"] == pytest.approx(0.9)
    assert calibrated["mean_calls"] == pytest.approx(3.0)
    assert calibrated["certification_rate"] == 1.0
    assert calibrated["block"] == "adapted_calibrated"


def test_summary_separates_routerbench_unavailable_and_sensitivity_rows() -> None:
    unavailable = {
        "protocol": learned.PROTOCOL_NAME,
        "mode": "routerbench_native_barp",
        "status": "unavailable",
        "analysis_label": "primary_four_training_tasks",
        "identity": {
            "method": "barp",
            "training_seed": 9101,
            "shot": "0shot",
            "variant": "primary_four_training_tasks",
        },
        "configuration": {
            "training_tasks": list(learned.ROUTERBENCH_TRAIN_TASKS),
            "target_tasks": list(learned.ROUTERBENCH_TARGET_TASKS),
        },
        "failure": {
            "stage": "prompt_feature_audit",
            "reason": "RouterBench feature audit failed: gsm8k",
        },
    }
    point = {
        "preference_index": 0,
        "w_quality": 1.0,
        "w_cost": 0.0,
        "mean_official_quality": 0.8,
        "mean_cost": 1.0,
        "mean_scalar_reward": 0.8,
        "binary_coverage": 0.8,
        "nondominated": True,
    }
    sensitivity = {
        "protocol": learned.PROTOCOL_NAME,
        "mode": "routerbench_native_barp",
        "status": "available",
        "analysis_label": "sensitivity_without_degenerate_gsm8k",
        "identity": {
            "method": "barp",
            "training_seed": 9101,
            "shot": "0shot",
            "variant": "sensitivity_without_degenerate_gsm8k",
        },
        "configuration": {
            "training_tasks": list(learned.ROUTERBENCH_THREE_TASK_SENSITIVITY),
            "target_tasks": list(learned.ROUTERBENCH_TARGET_TASKS),
        },
        "native_barp": {
            "tasks": {task: [dict(point)] for task in learned.ROUTERBENCH_TARGET_TASKS},
            "macro_average": [dict(point)],
        },
    }

    result = summary.summarize_results(
        [("primary/results.json", unavailable), ("sensitivity/results.json", sensitivity)]
    )
    assert result["blocks"]["native_barp"] == []
    assert len(result["blocks"]["native_barp_sensitivity"]) == 3
    assert {
        row["target_task"] for row in result["blocks"]["native_barp_sensitivity"]
    } == {"mbpp", "hellaswag", "macro_average"}
    assert len(result["blocks"]["unavailable"]) == 2
    assert {row["target_task"] for row in result["blocks"]["unavailable"]} == {
        "mbpp", "hellaswag"
    }
    assert all(row["run_status"] == "unavailable" for row in result["blocks"]["unavailable"])
    assert all(
        row["analysis_label"] == "sensitivity_without_degenerate_gsm8k"
        for row in result["blocks"]["native_barp_sensitivity"]
    )


def test_immutable_json_output_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    payload = {"schema_version": 1, "rows": [{"a": 1, "b": 2}]}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    summary.write_json(payload, first)
    summary.write_json(payload, second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summary.write_json(payload, first)
