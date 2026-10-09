"""Tests for the auditable RouterBench embedding archive builder."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.data_protocol import load_routerbench, stable_query_id  # noqa: E402
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402
from scripts import build_routerbench_embedding_archive as builder  # noqa: E402


def _row(sample_id: str, eval_name: str, prompt: str) -> dict:
    row = {"sample_id": sample_id, "eval_name": eval_name, "prompt": prompt}
    for index, model in enumerate(ROUTERBENCH_MODEL_COLUMNS):
        row[model] = float(index == 0)
        row[model + "|model_response"] = "response"
        row[model + "|total_cost"] = float(index + 1)
    return row


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    frame = pd.DataFrame(
        [
            _row("merged", "mbpp", "merged prompt"),
            _row("merged", "mbpp", "merged prompt"),
            _row("single", "hellaswag", "single prompt"),
        ]
    )
    pickle_path = tmp_path / "routerbench_0shot.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(frame, handle)
    raw = np.asarray([[3.0, 0.0], [0.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    raw_path = tmp_path / "embeddings_0shot_mean.npy"
    np.save(raw_path, raw)
    metadata = {
        "num_rows": 3,
        "dim": 2,
        "mean_sha256": builder.sha256_file(raw_path),
        "source_shards": [
            {
                "input": str(pickle_path),
                "input_sha256": builder.sha256_file(pickle_path),
                "shard_index": 0,
                "num_shards": 1,
                "rows": 3,
            }
        ],
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return pickle_path, raw_path, meta_path, raw


def test_builder_rejects_routerbench_hash_mismatch(tmp_path: Path) -> None:
    pickle_path, raw_path, meta_path, _ = _inputs(tmp_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["source_shards"][0]["input_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="RouterBench input hash"):
        builder.build_routerbench_embedding_archive(
            routerbench_pickle=pickle_path,
            raw_embeddings=raw_path,
            embedding_meta=meta_path,
            output_dir=tmp_path / "archive",
        )
    assert not (tmp_path / "archive").exists()


def test_builder_averages_exact_source_rows_and_preserves_cleaned_ids(tmp_path: Path) -> None:
    pickle_path, raw_path, meta_path, _ = _inputs(tmp_path)
    output = tmp_path / "archive"
    manifest = builder.build_routerbench_embedding_archive(
        routerbench_pickle=pickle_path,
        raw_embeddings=raw_path,
        embedding_meta=meta_path,
        output_dir=output,
    )

    cleaned = load_routerbench(
        pickle_path,
        shot="0shot",
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    with np.load(output / builder.ARCHIVE_FILENAME, allow_pickle=False) as archive:
        ids = tuple(str(value) for value in archive["query_ids"].tolist())
        embeddings = np.asarray(archive["embeddings"])
    assert ids == cleaned.query_ids
    merged_id = stable_query_id("mbpp", "merged")
    merged_row = ids.index(merged_id)
    np.testing.assert_allclose(embeddings[merged_row], np.asarray([0.6, 0.8]))
    single_id = stable_query_id("hellaswag", "single")
    single_row = ids.index(single_id)
    np.testing.assert_allclose(embeddings[single_row], np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert tuple(cleaned.frame.loc[cleaned.frame["query_id"] == merged_id, "source_rows"].iloc[0]) == (0, 1)
    assert manifest["reconstruction"]["query_order"] == "exact_cleaned_routerbench_query_id_order"
    assert manifest["output"]["archive_sha256"] == builder.sha256_file(
        output / builder.ARCHIVE_FILENAME
    )
    loaded_manifest = builder.load_archive_manifest(
        output / builder.MANIFEST_FILENAME,
        archive_path=output / builder.ARCHIVE_FILENAME,
    )
    assert loaded_manifest == manifest


def test_builder_refuses_to_overwrite_immutable_archive(tmp_path: Path) -> None:
    pickle_path, raw_path, meta_path, _ = _inputs(tmp_path)
    output = tmp_path / "archive"
    builder.build_routerbench_embedding_archive(
        routerbench_pickle=pickle_path,
        raw_embeddings=raw_path,
        embedding_meta=meta_path,
        output_dir=output,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_routerbench_embedding_archive(
            routerbench_pickle=pickle_path,
            raw_embeddings=raw_path,
            embedding_meta=meta_path,
            output_dir=output,
        )
