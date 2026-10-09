"""Focused synthetic tests for the ISSUE-50 paired-suite shard merger."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import merge_paired_suite_shards as merge  # noqa: E402
from scripts import run_paired_main_suite as paired  # noqa: E402


def _task() -> SimpleNamespace:
    rng = np.random.default_rng(50)
    rows, models = 100, 5
    labels = np.zeros((rows, models), dtype=np.int8)
    for row in range(rows):
        labels[row, row % models] = 1
        labels[row, (row + 1) % models] = 1
    return SimpleNamespace(
        task="ifeval",
        query_ids=tuple(f"query-{row}" for row in range(rows)),
        embeddings=rng.normal(size=(rows, 4)),
        correctness=labels,
        model_ids=tuple(f"model-{index}" for index in range(models)),
        source_hashes={"leaderboard": "1" * 64, "router_dataset": "2" * 64},
        feature_audit={"degenerate": False},
    )


def _knn(train_x, train_y, query_x, k):
    del train_x, k
    rows, models = len(query_x), train_y.shape[1]
    probabilities = np.broadcast_to(np.linspace(0.2, 0.8, models), (rows, models)).copy()
    probabilities += (np.arange(rows)[:, None] % 3) * 0.005
    support = np.broadcast_to(train_y[:1], (rows, 1, models)).copy()
    return probabilities, support


def _run(start: int, end: int) -> dict:
    return paired.run_paired_suite(
        _task(),
        pool_size=5,
        pool_seed=7001,
        redraw_start=start,
        redraw_end=end,
        knn_k=10,
        include_e3=True,
        knn_function=_knn,
    )


def test_shards_merge_byte_identically_to_monolithic_output(tmp_path: Path) -> None:
    monolithic = _run(0, 4)
    shards = []
    for start, end in ((0, 1), (1, 3), (3, 4)):
        path = tmp_path / f"shard-{start}-{end}.json"
        payload = _run(start, end)
        paired.atomic_write_json(path, payload)
        shards.append((path, payload))

    merged = merge.merge_payloads(list(reversed(shards)))
    assert merged == monolithic
    assert paired.canonical_json_bytes(merged) == paired.canonical_json_bytes(monolithic)
    assert merged["aggregates"] == paired.recompute_aggregates(merged["cells"])

    output = tmp_path / "merged.json"
    paired.atomic_write_json(output, merged)
    reference = tmp_path / "monolithic.json"
    paired.atomic_write_json(reference, monolithic)
    assert output.read_bytes() == reference.read_bytes()
    assert not list(tmp_path.glob("*.tmp.*"))


def test_merger_rejects_gap_overlap_and_immutable_identity_changes() -> None:
    first = _run(0, 1)
    second = _run(1, 2)
    with pytest.raises(ValueError, match="duplicate redraws"):
        merge.merge_payloads([(Path("a.json"), first), (Path("again.json"), first)])

    gap = _run(2, 3)
    with pytest.raises(ValueError, match="leaves a gap"):
        merge.merge_payloads([(Path("a.json"), first), (Path("gap.json"), gap)])

    changed = copy.deepcopy(second)
    changed["source_hashes"]["leaderboard"] = "9" * 64
    with pytest.raises(ValueError, match="immutable identities differ"):
        merge.merge_payloads([(Path("a.json"), first), (Path("changed.json"), changed)])

    changed = copy.deepcopy(second)
    changed["checkpoint_identities"] = {"flexrouter": [{"training_seed": 9101}]}
    with pytest.raises(ValueError, match="immutable identities differ"):
        merge.merge_payloads([(Path("a.json"), first), (Path("changed.json"), changed)])


def test_merger_rejects_corrupt_query_vectors_endpoints_splits_and_aggregates() -> None:
    payload = _run(0, 1)
    alpha_cell = payload["cells"][-1]
    redraw = alpha_cell["redraws"][0]
    method = redraw["methods"]["exact_independence"]
    assert method["certified"]

    corrupt = copy.deepcopy(payload)
    record = corrupt["cells"][-1]["redraws"][0]["methods"]["exact_independence"]
    record["test_calls_by_query"][0] += 1
    with pytest.raises(ValueError, match="selected IDs disagree|mean calls disagree"):
        merge.validate_payload(corrupt, Path("calls.json"))

    corrupt = copy.deepcopy(payload)
    record = corrupt["cells"][-1]["redraws"][0]["methods"]["exact_independence"]
    record["endpoint"]["model_ids"] = list(reversed(payload["model_ids"]))
    with pytest.raises(ValueError, match="endpoint model identity differs"):
        merge.validate_payload(corrupt, Path("endpoint.json"))

    corrupt = copy.deepcopy(payload)
    split = corrupt["cells"][0]["redraws"][0]["split_ids"]
    split["test"][0] = split["design"][0]
    with pytest.raises(ValueError, match="overlapping or incomplete"):
        merge.validate_payload(corrupt, Path("split.json"))

    corrupt = copy.deepcopy(payload)
    corrupt["aggregates"][str(paired.ALPHA_GRID[-1])]["methods"][
        "exact_independence"
    ]["test_miss_count"] += 1
    with pytest.raises(ValueError, match="stored aggregates disagree"):
        merge.validate_payload(corrupt, Path("aggregate.json"))


def test_load_and_merge_rejects_invalid_json_and_output_is_immutable(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read valid JSON"):
        merge.load_and_merge([invalid])

    payload = _run(0, 1)
    shard = tmp_path / "shard.json"
    paired.atomic_write_json(shard, payload)
    output = tmp_path / "output.json"
    assert merge.main([str(shard), "--out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        merge.main([str(shard), "--out", str(output)])
