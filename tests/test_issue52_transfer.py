"""Focused synthetic tests for the ISSUE-52 transfer runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_issue52_transfer as transfer
from scripts import run_paired_main_suite as paired


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, item):
        return FakeTensor(self.values[item])


class FakePath:
    def __init__(self, order):
        self.order = tuple(int(value) for value in order)


class FakeFlexRouter:
    def __init__(self, n_models: int, offset: int):
        self.n_models = n_models
        self.offset = offset

    def predict_quality(self, embeddings):
        rows = len(embeddings)
        base = np.arange(self.n_models, dtype=float)
        values = np.tile(np.roll(base, self.offset), (rows, 1))
        return FakeTensor(values)

    def select_from_qualities(self, qualities, **kwargs):
        assert kwargs == {
            "max_size": self.n_models,
            "relative_threshold": 0.0,
            "top_k0": None,
        }
        order = np.argsort(-qualities.numpy(), kind="stable")
        return FakePath(order)


class FakeBaRP:
    def __init__(self, model_ids: list[str], offset: int):
        self.n_models = len(model_ids)
        self.model_ids = list(model_ids)
        self.offset = offset

    def rank_models(self, embeddings, preference, *, top_k):
        assert preference == [1.0, 0.0]
        assert top_k == self.n_models
        base = np.roll(np.arange(self.n_models), self.offset)
        return FakeTensor(np.tile(base, (len(embeddings), 1)))


def _task(name: str, *, rows: int = 120, model_ids: tuple[str, ...] | None = None):
    model_ids = model_ids or tuple(f"model-{index}" for index in range(4))
    row_index = np.arange(rows)
    embeddings = np.column_stack(
        (
            row_index / max(rows - 1, 1),
            np.sin(row_index),
            np.cos(row_index),
        )
    )
    labels = np.zeros((rows, len(model_ids)), dtype=np.int8)
    for row in range(rows):
        labels[row, row % len(model_ids)] = 1
        labels[row, (row + 1) % len(model_ids)] = 1
    return SimpleNamespace(
        task=name,
        query_ids=tuple(f"{name}:q{row}" for row in range(rows)),
        embeddings=embeddings,
        correctness=labels,
        model_ids=model_ids,
        prompts=tuple(f"prompt {name} {row}" for row in range(rows)),
        source_hashes={"labels": name * 4, "embeddings": name * 3},
        feature_audit={"degenerate": False},
    )


def _ensembles(model_ids: list[str], target_family: str):
    output = {"flexrouter": [], "barp": []}
    for offset, seed in enumerate(paired.TRAINING_SEEDS):
        common = {
            "pool_size": len(model_ids),
            "pool_seed": paired.BASE_SEED,
            "training_seed": seed,
            "model_ids": list(model_ids),
            "source": "routereval",
            "target_family": target_family,
            "excluded_family": target_family,
            "checkpoint_sha256": f"{seed:064d}"[-64:],
        }
        output["flexrouter"].append(
            {
                "identity": {**common, "method": "flexrouter"},
                "model": FakeFlexRouter(len(model_ids), offset),
            }
        )
        output["barp"].append(
            {
                "identity": {**common, "method": "barp"},
                "model": FakeBaRP(model_ids, offset),
            }
        )
    return output


def _fake_knn(train_x, train_y, query_x, k):
    del train_x, k
    n_models = train_y.shape[1]
    base = np.linspace(0.72, 0.42, n_models)
    variation = np.asarray(query_x)[:, :1] * 0.08
    probabilities = np.clip(base[None, :] + variation, 1e-4, 1.0 - 1e-4)
    support = np.zeros((len(query_x), 1, n_models), dtype=np.int8)
    return probabilities, support


def _run(source, target, *, fit_mode="source_fitted", source_family="source"):
    model_ids = list(source.model_ids)
    return transfer.run_transfer_pair(
        source,
        target,
        protocol="routereval",
        fit_mode=fit_mode,
        source_family=source_family,
        target_family="target",
        pool_size=len(model_ids),
        pool_seed=paired.BASE_SEED,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        learned_ensembles=_ensembles(model_ids, "target"),
        knn_function=_fake_knn,
    )


def test_source_fitted_path_never_reads_target_fit_labels() -> None:
    source = _task("source_task")
    target = _task("target_task")
    split = paired._target_split(len(target.query_ids), 0, paired.BASE_SEED)
    forbidden = np.concatenate((split["design"], split["router"], split["validation"]))

    first = copy.deepcopy(target)
    second = copy.deepcopy(target)
    first.correctness[forbidden] = 7
    second.correctness[forbidden] = 9

    first_payload = _run(source, first)
    second_payload = _run(source, second)
    assert first_payload["cells"] == second_payload["cells"]
    access = first_payload["cells"][0]["redraws"][0]["fit_label_access"]
    assert access["target_fit_label_roles"] == []
    assert access["target_label_roles"] == ["final_calibration", "test"]


def test_calibration_twins_share_one_frozen_target_score_matrix() -> None:
    payload = _run(_task("source_task"), _task("target_task"))
    assert payload["target_grid"] == list(paired.ALPHA_GRID)
    assert payload["redraw_range"]["registered_total"] == 20
    assert payload["preserve_prefixes"] is True
    assert len(payload["method_identities"]) == 14

    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        for method in redraw["methods"].values():
            twins = method["calibrations"]
            assert set(twins) == set(transfer.CALIBRATION_MODES)
            assert (
                twins["source_calibrated"]["frozen_target_scores_sha256"]
                == twins["target_recalibrated"]["frozen_target_scores_sha256"]
                == method["frozen_target_scores_sha256"]
            )
            assert (
                twins["source_calibrated"]["target_test_query_ids"]
                == twins["target_recalibrated"]["target_test_query_ids"]
                == redraw["target_split_ids"]["test"]
            )
            for twin in twins.values():
                if twin["certified"]:
                    assert len(twin["target_test_calls_by_query"]) == len(
                        redraw["target_split_ids"]["test"]
                    )
                    assert twin["target_test_event_count"] == sum(
                        twin["target_test_miss_by_query"]
                    )


def test_source_equals_target_reproduces_identical_calibration_results() -> None:
    task = _task("target_task")
    payload = _run(task, task, fit_mode="in_domain", source_family="target")
    assert payload["source_equals_target"] is True
    for cell in payload["cells"]:
        redraw = cell["redraws"][0]
        assert redraw["source_split_ids"] == redraw["target_split_ids"]
        for method in redraw["methods"].values():
            source = method["calibrations"]["source_calibrated"]
            target = method["calibrations"]["target_recalibrated"]
            for field in (
                "certified",
                "threshold",
                "corrected_calibration_bound",
                "calibration_event_count",
                "target_test_calls_by_query",
                "target_test_miss_by_query",
            ):
                assert source[field] == target[field]


def test_chain_discount_off_endpoint_reproduces_fixed_tally_exactly() -> None:
    payload = _run(_task("source_task"), _task("target_task"))
    for cell in payload["cells"]:
        methods = cell["redraws"][0]["methods"]
        chain = methods[paired.CHAIN_DISCOUNT_METHOD]
        fixed = methods["fixed_tally_factorized"]
        assert chain["endpoint_reproduction_exact"] is True
        assert chain["endpoint_reference_method"] == "fixed_tally_factorized"
        assert chain["discount_off_endpoint_sha256"] == fixed["frozen_target_scores_sha256"]
        assert chain["hyperparameter_selection_origin"].endswith(":source_task:validation")


def test_wrong_model_columns_and_wrong_checkpoint_family_fail() -> None:
    source = _task("source_task")
    reversed_target = _task("target_task", model_ids=tuple(reversed(source.model_ids)))
    with pytest.raises(ValueError, match="model columns must be exactly identical"):
        _run(source, reversed_target)

    ensembles = _ensembles(list(source.model_ids), "wrong_family")
    with pytest.raises(ValueError, match="checkpoint target family is wrong"):
        transfer.run_transfer_pair(
            source,
            _task("target_task"),
            protocol="routereval",
            fit_mode="source_fitted",
            source_family="source",
            target_family="target",
            pool_size=4,
            pool_seed=paired.BASE_SEED,
            redraw_start=0,
            redraw_end=1,
            knn_k=8,
            learned_ensembles=ensembles,
            knn_function=_fake_knn,
        )


def test_missing_arm_redraw_bounds_and_atomic_output_fail_loudly(tmp_path: Path) -> None:
    payload = _run(_task("source_task"), _task("target_task"))
    broken = copy.deepcopy(payload["cells"])
    del broken[0]["redraws"][0]["methods"]["racer_g"]
    with pytest.raises(RuntimeError, match="registered arm mismatch"):
        transfer.assert_registered_arm_presence(broken, payload["method_identities"])

    with pytest.raises(ValueError, match="within 0 through 20"):
        transfer.run_transfer_pair(
            _task("source_task"),
            _task("target_task"),
            protocol="routereval",
            fit_mode="source_fitted",
            source_family="source",
            target_family="target",
            pool_size=4,
            pool_seed=paired.BASE_SEED,
            redraw_start=20,
            redraw_end=21,
            knn_k=8,
            learned_ensembles=_ensembles([f"model-{i}" for i in range(4)], "target"),
            knn_function=_fake_knn,
        )

    output = tmp_path / "issue52_testhost_20260920T020000Z" / "shard.json"
    transfer._validate_output_identity(output, "testhost", "20260920T020000Z")
    paired.atomic_write_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired.atomic_write_json(output, {"ok": False})


def test_explicit_legacy_checkpoint_identity_path() -> None:
    source = _task("source_task")
    target = _task("target_task")
    ensembles = _ensembles(list(source.model_ids), "target")
    for members in ensembles.values():
        for member in members:
            member["identity"]["source"] = None
            member["identity"]["target_family"] = None
            member["identity"]["excluded_family"] = None
    payload = transfer.run_transfer_pair(
        source,
        target,
        protocol="routereval",
        fit_mode="source_fitted",
        source_family="source",
        target_family="target",
        pool_size=4,
        pool_seed=paired.BASE_SEED,
        redraw_start=0,
        redraw_end=1,
        knn_k=8,
        learned_ensembles=ensembles,
        knn_function=_fake_knn,
        checkpoint_identity_target_family=None,
    )
    assert payload["target_benchmark"] == "target_task"
