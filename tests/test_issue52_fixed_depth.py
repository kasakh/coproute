"""Focused tests for the ISSUE-52 fixed-depth transfer runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_issue52_fixed_depth as fixed_depth
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
        base = np.roll(np.arange(self.n_models, dtype=float), self.offset)
        return FakeTensor(np.tile(base, (len(embeddings), 1)))

    def select_from_qualities(self, qualities, **kwargs):
        assert kwargs == {
            "max_size": self.n_models,
            "relative_threshold": 0.0,
            "top_k0": None,
        }
        return FakePath(np.argsort(-qualities.numpy(), kind="stable"))


class FakeBaRP:
    def __init__(self, model_ids: list[str], offset: int):
        self.n_models = len(model_ids)
        self.model_ids = list(model_ids)
        self.offset = offset

    def rank_models(self, embeddings, preference, *, top_k):
        assert preference == [1.0, 0.0]
        assert top_k == self.n_models
        order = np.roll(np.arange(self.n_models), self.offset)
        return FakeTensor(np.tile(order, (len(embeddings), 1)))


def _task(name: str, *, rows: int = 120, model_ids: tuple[str, ...] | None = None):
    model_ids = model_ids or tuple(f"model-{index}" for index in range(4))
    row_index = np.arange(rows)
    embeddings = np.column_stack(
        (row_index / max(rows - 1, 1), np.sin(row_index), np.cos(row_index))
    )
    labels = np.zeros((rows, len(model_ids)), dtype=np.int8)
    for row in range(rows):
        labels[row, row % len(model_ids)] = 1
        if row % 7:
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
    return fixed_depth.run_fixed_depth_transfer(
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


def test_source_fitted_never_reads_target_fit_labels() -> None:
    source = _task("source_task")
    target = _task("target_task")
    split = paired._target_split(len(target.query_ids), 0, paired.BASE_SEED)
    forbidden = np.concatenate(
        (
            split["design"],
            split["router"],
            split["final_calibration"],
            split["validation"],
        )
    )

    first = copy.deepcopy(target)
    second = copy.deepcopy(target)
    first.correctness[forbidden] = 7
    second.correctness[forbidden] = 9

    first_payload = _run(source, first)
    second_payload = _run(source, second)
    assert first_payload["redraws"] == second_payload["redraws"]
    access = first_payload["redraws"][0]["fit_label_access"]
    assert access["target_fit_label_roles"] == []
    assert access["target_evaluation_label_roles"] == ["test"]


def test_all_arms_have_nested_stable_prefixes_and_exact_arithmetic() -> None:
    payload = _run(_task("source_task"), _task("target_task"))
    assert len(payload["method_identities"]) == 14
    assert payload["redraw_range"]["registered_total"] == 20
    assert payload["stable_tie_rule"] == "sampled_model_column_order"
    redraw = payload["redraws"][0]
    assert set(redraw["methods"]) == set(payload["method_identities"])

    for method in redraw["methods"].values():
        records = method["fixed_depth"]["depths"]
        assert [record["depth"] for record in records] == [1, 2, 3, 4]
        for index, record in enumerate(records):
            assert record["target_success_count"] + record["target_miss_count"] == record[
                "target_trial_count"
            ]
            assert record["target_success"] == pytest.approx(
                record["target_success_count"] / record["target_trial_count"]
            )
            assert record["target_all_miss_risk"] == pytest.approx(
                record["target_miss_count"] / record["target_trial_count"]
            )
            assert record["target_miss_count"] == sum(record["target_miss_by_query"])
            assert record["target_success_count"] == sum(record["target_success_by_query"])
            if index:
                previous = records[index - 1]["selected_model_ids_by_query"]
                current = record["selected_model_ids_by_query"]
                assert all(new[:-1] == old for old, new in zip(previous, current))


def test_stable_ties_follow_sampled_model_column_order() -> None:
    model_ids = ["m0", "m1", "m2"]
    scores = np.zeros((2, 3), dtype=float)
    labels = np.asarray([[0, 1, 0], [1, 0, 0]], dtype=np.int8)
    record = fixed_depth._fixed_depth_record(
        scores,
        labels,
        np.asarray([0, 1]),
        ["q0", "q1"],
        model_ids,
    )
    assert record["depths"][0]["selected_model_ids_by_query"] == [["m0"], ["m0"]]
    assert record["depths"][1]["selected_model_ids_by_query"] == [
        ["m0", "m1"],
        ["m0", "m1"],
    ]


def test_full_pool_endpoint_reproduces_direct_misses_and_in_domain_identity() -> None:
    task = _task("target_task")
    payload = _run(task, task, fit_mode="in_domain", source_family="target")
    redraw = payload["redraws"][0]
    direct = redraw["target_complete_pool"]
    assert payload["source_equals_target"] is True
    assert redraw["source_split_ids"] == redraw["target_split_ids"]
    assert redraw["fit_label_access"]["target_fit_label_roles"] == ["design", "router"]

    for method in redraw["methods"].values():
        endpoint = method["fixed_depth"]["endpoint"]
        assert endpoint["reproduces_complete_pool"] is True
        assert endpoint["target_miss_by_query"] == direct["miss_by_query"]
        assert endpoint["target_miss_count"] == direct["event_count"]
        assert endpoint["target_trial_count"] == direct["trial_count"]
        assert endpoint["model_ids_sha256"] == payload["model_ids_sha256"]

    fixed = redraw["methods"]["fixed_tally_factorized"]
    chain = redraw["methods"][paired.CHAIN_DISCOUNT_METHOD]
    assert chain["endpoint_reproduction_exact"] is True
    assert chain["rank_identity_across_shrinkage_grid"] is True
    assert chain["frozen_target_scores_sha256"] == fixed["frozen_target_scores_sha256"]
    assert chain["fixed_depth"] == fixed["fixed_depth"]


def test_wrong_model_columns_and_checkpoint_family_fail() -> None:
    source = _task("source_task")
    reversed_target = _task("target_task", model_ids=tuple(reversed(source.model_ids)))
    with pytest.raises(ValueError, match="model columns must be exactly identical"):
        _run(source, reversed_target)

    ensembles = _ensembles(list(source.model_ids), "wrong_family")
    with pytest.raises(ValueError, match="checkpoint target family is wrong"):
        fixed_depth.run_fixed_depth_transfer(
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


def test_redraw_bounds_arm_presence_and_atomic_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="within 0 through 20"):
        fixed_depth.run_fixed_depth_transfer(
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

    payload = _run(_task("source_task"), _task("target_task"))
    broken = copy.deepcopy(payload["redraws"])
    del broken[0]["methods"]["racer_g"]
    with pytest.raises(RuntimeError, match="registered arm mismatch"):
        fixed_depth.assert_registered_arm_presence(broken, payload["method_identities"])

    output = tmp_path / "issue52_testhost_20260920T025000Z" / "fixed_depth" / "shard.json"
    fixed_depth._validate_output_identity(output, "testhost", "20260920T025000Z")
    paired.atomic_write_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired.atomic_write_json(output, {"ok": False})
