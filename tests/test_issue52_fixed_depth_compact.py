"""Focused tests for the compact ISSUE-52 fixed-depth schema."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_issue52_fixed_depth as v1
from scripts import run_issue52_fixed_depth_compact as compact
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


def _run(
    source,
    target,
    *,
    fit_mode="source_fitted",
    source_family="source",
    redraw_end=1,
):
    model_ids = list(source.model_ids)
    return compact.run_fixed_depth_transfer(
        source,
        target,
        protocol="routereval",
        fit_mode=fit_mode,
        source_family=source_family,
        target_family="target",
        pool_size=len(model_ids),
        pool_seed=paired.BASE_SEED,
        redraw_start=0,
        redraw_end=redraw_end,
        knn_k=8,
        learned_ensembles=_ensembles(model_ids, "target"),
        knn_function=_fake_knn,
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_compact_vectors_and_counts_are_scientifically_equivalent_to_v1() -> None:
    scores = np.asarray(
        [
            [0.4, 0.1, 0.3, 0.2],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.2, 0.1, 0.3],
        ],
        dtype=float,
    )
    labels = np.asarray(
        [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int8
    )
    rows = np.arange(3)
    query_ids = ["q0", "q1", "q2"]
    model_ids = ["m0", "m1", "m2", "m3"]

    old = v1._fixed_depth_record(scores, labels, rows, query_ids, model_ids)
    new = compact._fixed_depth_record(scores, labels, rows, query_ids, model_ids)
    fields = (
        "depth",
        "target_miss_by_query",
        "target_success_count",
        "target_miss_count",
        "target_trial_count",
        "target_success",
        "target_all_miss_risk",
        "mean_calls",
    )
    for old_depth, new_depth in zip(old["depths"], new["depths"]):
        assert {key: old_depth[key] for key in fields} == {
            key: new_depth[key] for key in fields
        }
    endpoint_fields = (
        "depth",
        "target_miss_by_query",
        "target_miss_count",
        "target_trial_count",
        "target_all_miss_risk",
        "reproduces_complete_pool",
        "model_ids_sha256",
    )
    assert {key: old["endpoint"][key] for key in endpoint_fields} == {
        key: new["endpoint"][key] for key in endpoint_fields
    }
    assert "selected_model_ids_by_query" not in set(_walk_keys(new))


def test_source_fitted_has_no_target_fit_label_leakage() -> None:
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


def test_schema_has_all_arms_exact_vectors_and_no_selected_id_field() -> None:
    payload = _run(_task("source_task"), _task("target_task"))
    assert payload["schema_version"] == "issue52_fixed_depth_v2"
    assert len(payload["method_identities"]) == 14
    assert payload["redraw_range"]["registered_total"] == 20
    assert payload["memory_contract"] == {
        "redraw_processing": "one_at_a_time",
        "score_array_retention": "current_redraw_only",
        "model_selection_identity_storage": "omitted",
        "target_query_ids_storage": "once_per_redraw_method",
    }
    assert "selected_model_ids_by_query" not in set(_walk_keys(payload))
    assert "selected_model_ids_by_query" not in json.dumps(payload, sort_keys=True)

    redraw = payload["redraws"][0]
    assert set(redraw["methods"]) == set(payload["method_identities"])
    for method in redraw["methods"].values():
        assert method["target_query_ids_sha256"] == paired.hash_json(
            method["target_query_ids"]
        )
        assert method["preserve_prefixes"] is True
        records = method["fixed_depth"]["depths"]
        assert [record["depth"] for record in records] == [1, 2, 3, 4]
        assert all("target_query_ids" not in record for record in records)
        for record in records:
            assert record["target_success_count"] + record["target_miss_count"] == record[
                "target_trial_count"
            ]
            assert record["target_miss_count"] == sum(record["target_miss_by_query"])
            assert record["target_all_miss_risk"] == pytest.approx(
                record["target_miss_count"] / record["target_trial_count"]
            )
            assert record["target_success"] == pytest.approx(
                record["target_success_count"] / record["target_trial_count"]
            )


def test_twenty_redraw_output_is_bounded_and_keeps_query_cluster_vectors() -> None:
    payload = _run(
        _task("source_task", rows=80),
        _task("target_task", rows=80),
        redraw_end=20,
    )
    assert len(payload["redraws"]) == 20
    assert payload["redraw_range"] == {
        "start_inclusive": 0,
        "end_exclusive": 20,
        "count": 20,
        "registered_total": 20,
    }
    encoded = json.dumps(payload, sort_keys=True)
    assert len(encoded.encode("utf-8")) < 2_000_000
    assert "selected_model_ids_by_query" not in encoded

    for redraw in payload["redraws"]:
        for method in redraw["methods"].values():
            query_count = len(method["target_query_ids"])
            assert query_count > 0
            for depth in method["fixed_depth"]["depths"]:
                assert len(depth["target_miss_by_query"]) == query_count
            assert len(method["fixed_depth"]["endpoint"]["target_miss_by_query"]) == query_count


def test_endpoint_identity_in_domain_identity_and_checkpoint_guard() -> None:
    task = _task("target_task")
    payload = _run(task, task, fit_mode="in_domain", source_family="target")
    redraw = payload["redraws"][0]
    direct = redraw["target_complete_pool"]
    assert payload["source_equals_target"] is True
    assert redraw["source_split_identity"] == redraw["target_split_identity"]
    assert redraw["fit_label_access"]["target_fit_label_roles"] == ["design", "router"]

    for method in redraw["methods"].values():
        endpoint = method["fixed_depth"]["endpoint"]
        assert endpoint["reproduces_complete_pool"] is True
        assert endpoint["target_miss_by_query"] == direct["miss_by_query"]
        assert endpoint["target_miss_count"] == direct["event_count"]
        assert endpoint["target_trial_count"] == direct["trial_count"]
        assert endpoint["model_ids_sha256"] == payload["model_ids_sha256"]

    wrong = _ensembles(list(task.model_ids), "wrong_family")
    with pytest.raises(ValueError, match="checkpoint target family is wrong"):
        compact.run_fixed_depth_transfer(
            _task("source_task"),
            task,
            protocol="routereval",
            fit_mode="source_fitted",
            source_family="source",
            target_family="target",
            pool_size=4,
            pool_seed=paired.BASE_SEED,
            redraw_start=0,
            redraw_end=1,
            knn_k=8,
            learned_ensembles=wrong,
            knn_function=_fake_knn,
        )


def test_bounds_arm_presence_exact_naming_and_atomic_no_overwrite(tmp_path: Path) -> None:
    source = _task("source_task")
    target = _task("target_task")
    with pytest.raises(ValueError, match="within 0 through 20"):
        compact.run_fixed_depth_transfer(
            source,
            target,
            protocol="routereval",
            fit_mode="source_fitted",
            source_family="source",
            target_family="target",
            pool_size=4,
            pool_seed=paired.BASE_SEED,
            redraw_start=20,
            redraw_end=21,
            knn_k=8,
            learned_ensembles=_ensembles(list(source.model_ids), "target"),
            knn_function=_fake_knn,
        )

    payload = _run(source, target)
    broken = copy.deepcopy(payload["redraws"])
    del broken[0]["methods"]["racer_g"]
    with pytest.raises(RuntimeError, match="registered arm mismatch"):
        compact.assert_registered_arm_presence(broken, payload["method_identities"])

    token = "issue52_testhost_20260920T034700Z"
    output = tmp_path / token / "fixed_depth" / "shard.json"
    compact._validate_output_identity(output, "testhost", "20260920T034700Z")
    with pytest.raises(ValueError, match="exact path token"):
        compact._validate_output_identity(
            tmp_path / f"prefix_{token}" / "shard.json",
            "testhost",
            "20260920T034700Z",
        )
    paired.atomic_write_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        paired.atomic_write_json(output, {"ok": False})
