"""End-to-end tests for deterministic E2 redraw sharding and merging."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import design_bprime_e2_run as e2  # noqa: E402
from scripts import merge_e2_shards as merge  # noqa: E402


def _synthetic_aligned_data() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260918)
    n_queries = 180
    latent = generator.normal(size=(n_queries, 8))
    weights = generator.normal(size=(8, 5))
    logits = latent @ weights + np.linspace(-0.8, 0.8, 5)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    correctness = (generator.random(probabilities.shape) < probabilities).astype(int)
    embeddings = latent + 0.01 * generator.normal(size=latent.shape)
    return correctness, embeddings


def _run_e2(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    *,
    redraw_start: int,
    redraws: int,
) -> None:
    correctness, embeddings = _synthetic_aligned_data()
    monkeypatch.setattr(
        e2,
        "_load_aligned",
        lambda router_dir, leaderboard_path, benchmark, threshold: (
            correctness.copy(),
            embeddings.copy(),
        ),
    )
    arguments = [
        "design_bprime_e2_run.py",
        "--router-dir",
        "synthetic-router",
        "--leaderboard",
        "synthetic-leaderboard.pkl",
        "--benchmark",
        "ifeval",
        "--pool-size",
        "5",
        "--max-depth",
        "5",
        "--alphas",
        "0.10,0.20",
        "--redraws",
        str(redraws),
        "--redraw-start",
        str(redraw_start),
        "--knn-k",
        "12",
        "--seed",
        "8119",
        "--design-permutations",
        "1",
        "--out",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert e2.main() == 0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_four_shards_merge_to_exact_monolithic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monolithic_path = tmp_path / "monolithic.json"
    _run_e2(monkeypatch, monolithic_path, redraw_start=0, redraws=20)

    shard_paths = []
    for start in (0, 5, 10, 15):
        shard_path = tmp_path / f"shard_{start}_{start + 5}.json"
        _run_e2(monkeypatch, shard_path, redraw_start=start, redraws=5)
        shard_paths.append(shard_path)

    merged_path = tmp_path / "merged.json"
    merged = merge.load_and_merge(list(reversed(shard_paths)))
    merge.write_merged_output(merged_path, merged)

    monolithic = _load(monolithic_path)
    assert _load(merged_path) == monolithic
    assert merged_path.read_bytes() == monolithic_path.read_bytes()
    assert monolithic["redraw_range"] == {
        "start_inclusive": 0,
        "end_exclusive": 20,
        "count": 20,
    }
    for alpha_records in monolithic["per_redraw"].values():
        for records in alpha_records.values():
            assert [record["redraw"] for record in records] == list(range(20))
    assert [
        record["redraw"]
        for record in monolithic["information_destruction_control"]["per_redraw"]
    ] == list(range(20))

    second_path = tmp_path / "merged_second_order.json"
    second = merge.load_and_merge(shard_paths)
    merge.write_merged_output(second_path, second)
    assert second_path.read_bytes() == merged_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        merge.write_merged_output(merged_path, merged)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _run_e2(monkeypatch, monolithic_path, redraw_start=0, redraws=20)


def test_merge_rejects_configuration_gaps_overlaps_and_bad_redraws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for start in (0, 1):
        path = tmp_path / f"shard_{start}.json"
        _run_e2(monkeypatch, path, redraw_start=start, redraws=1)
        paths.append(path)
    payloads = [(path, _load(path)) for path in paths]

    changed = copy.deepcopy(payloads[1][1])
    changed["threshold"] = 0.75
    with pytest.raises(ValueError, match="immutable configuration differs"):
        merge.merge_payloads([payloads[0], (Path("changed.json"), changed)])

    gap = copy.deepcopy(payloads[1][1])
    gap["redraw_start"] = 2
    gap["redraw_end"] = 3
    gap["redraw_range"] = {
        "start_inclusive": 2,
        "end_exclusive": 3,
        "count": 1,
    }
    for by_arm in gap["per_redraw"].values():
        for records in by_arm.values():
            records[0]["redraw"] = 2
    information_record = gap["information_destruction_control"]["per_redraw"][0]
    information_record["redraw"] = 2
    for index, permutation in enumerate(information_record["permutations"]):
        permutation["permutation_seed"] = e2._design_permutation_seed(
            gap["seed"], 2, index
        )
    with pytest.raises(ValueError, match="leaves a gap"):
        merge.merge_payloads([payloads[0], (Path("gap.json"), gap)])

    with pytest.raises(ValueError, match="duplicate redraws"):
        merge.merge_payloads([payloads[0], payloads[0]])

    bad_identity = copy.deepcopy(payloads[1][1])
    first_alpha = next(iter(bad_identity["per_redraw"].values()))
    first_arm = next(iter(first_alpha.values()))
    first_arm[0]["redraw"] = 0
    with pytest.raises(ValueError, match="invalid redraw identities"):
        merge.merge_payloads(
            [payloads[0], (Path("bad_identity.json"), bad_identity)]
        )
