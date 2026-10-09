"""Build an auditable cleaned-query RouterBench embedding archive.

The raw embedding matrix is aligned to raw RouterBench pickle rows. The
RouterBench loader can merge duplicate fragments into one cleaned query, so a
cleaned embedding is reconstructed by averaging exactly the loader-recorded
``source_rows`` and then applying row-wise L2 normalization.

The output is an immutable directory containing ``embeddings.npz`` and
``manifest.json``. Existing output directories are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.data_protocol import load_routerbench  # noqa: E402
from components.routerbench_protocol import ROUTERBENCH_MODEL_COLUMNS  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL_NAME = "routerbench_embedding_archive_v1"
ARCHIVE_FILENAME = "embeddings.npz"
MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_integer(metadata: Mapping[str, Any], *names: str) -> int:
    values = [metadata[name] for name in names if name in metadata]
    if len(values) != 1 or isinstance(values[0], bool):
        raise ValueError("embedding metadata must contain exactly one of " + ", ".join(names))
    value = int(values[0])
    if value <= 0:
        raise ValueError(names[0] + " must be positive")
    return value


def _expected_routerbench_hash(metadata: Mapping[str, Any]) -> str:
    hashes = set()
    for key in ("input_sha256", "routerbench_sha256", "source_sha256"):
        value = metadata.get(key)
        if isinstance(value, str):
            hashes.add(value)
    shards = metadata.get("source_shards")
    if shards is not None:
        if not isinstance(shards, list) or not shards:
            raise ValueError("source_shards must be a non-empty list")
        shard_rows = 0
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise ValueError("each source shard must be a JSON object")
            value = shard.get("input_sha256")
            if not isinstance(value, str):
                raise ValueError("each source shard must record input_sha256")
            hashes.add(value)
            shard_rows += int(shard.get("rows", 0))
        if shard_rows != _metadata_integer(metadata, "num_rows", "raw_row_count"):
            raise ValueError("source shard row counts do not equal metadata raw row count")
    if len(hashes) != 1:
        raise ValueError("embedding metadata must identify one RouterBench input hash")
    return next(iter(hashes))


def _expected_embedding_hash(metadata: Mapping[str, Any], hash_key: str) -> str:
    candidates = [hash_key]
    if hash_key == "mean_sha256":
        candidates.append("embedding_sha256")
    values = [metadata[name] for name in candidates if name in metadata]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("embedding metadata is missing " + hash_key)
    return values[0]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("reconstructed embeddings must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("reconstructed embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("reconstructed embeddings contain a zero-norm row")
    return values / norms


def reconstruct_cleaned_embeddings(raw_embeddings: Any, source_rows: Sequence[Any]) -> np.ndarray:
    """Average exact raw source rows for each cleaned query and L2 normalize."""

    raw = np.asarray(raw_embeddings)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("raw embeddings must be a non-empty matrix")
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise ValueError("raw embeddings must be finite numeric values")
    reconstructed = np.empty((len(source_rows), raw.shape[1]), dtype=np.float64)
    for query_row, values in enumerate(source_rows):
        indices = tuple(values) if isinstance(values, (tuple, list, np.ndarray)) else ()
        if not indices or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in indices):
            raise ValueError(f"cleaned query row {query_row} has invalid source_rows")
        integer_indices = np.asarray(indices, dtype=int)
        if len(np.unique(integer_indices)) != len(integer_indices):
            raise ValueError(f"cleaned query row {query_row} repeats a source row")
        if np.any(integer_indices < 0) or np.any(integer_indices >= raw.shape[0]):
            raise ValueError(f"cleaned query row {query_row} has out-of-range source_rows")
        reconstructed[query_row] = np.mean(raw[integer_indices], axis=0, dtype=np.float64)
    return _normalize_rows(reconstructed)


def load_archive_manifest(
    manifest_path: str | Path,
    *,
    archive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a builder manifest and verify its content and optional archive hash."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("embedding archive manifest must contain a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("protocol") != PROTOCOL_NAME:
        raise ValueError("unsupported embedding archive manifest")
    expected = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(expected, str) or _hash_json(unsigned) != expected:
        raise ValueError("embedding archive manifest content hash mismatch")
    if archive_path is not None:
        actual = sha256_file(archive_path)
        if actual != manifest.get("output", {}).get("archive_sha256"):
            raise ValueError("embedding archive file hash mismatch")
    return manifest


def build_routerbench_embedding_archive(
    *,
    routerbench_pickle: str | Path,
    raw_embeddings: str | Path,
    embedding_meta: str | Path,
    output_dir: str | Path,
    shot: str = "0shot",
    embedding_hash_key: str = "mean_sha256",
) -> dict[str, Any]:
    """Build and atomically publish one immutable cleaned-query archive."""

    pickle_path = Path(routerbench_pickle)
    raw_path = Path(raw_embeddings)
    meta_path = Path(embedding_meta)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output directory {destination}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("embedding metadata must contain a JSON object")
    expected_rows = _metadata_integer(metadata, "num_rows", "raw_row_count")
    expected_dim = _metadata_integer(metadata, "dim", "embedding_dim")
    pickle_hash = sha256_file(pickle_path)
    if pickle_hash != _expected_routerbench_hash(metadata):
        raise ValueError("RouterBench input hash does not match embedding metadata")
    raw_hash = sha256_file(raw_path)
    if raw_hash != _expected_embedding_hash(metadata, embedding_hash_key):
        raise ValueError("raw embedding hash does not match embedding metadata")

    raw = np.load(raw_path, allow_pickle=False)
    if not isinstance(raw, np.ndarray) or raw.ndim != 2:
        raise ValueError("raw embedding NPY must contain one two-dimensional array")
    if raw.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"raw embedding shape {raw.shape} does not match metadata {(expected_rows, expected_dim)}"
        )
    data = load_routerbench(
        pickle_path,
        shot=shot,
        model_columns=ROUTERBENCH_MODEL_COLUMNS,
    )
    if data.raw_row_count != expected_rows:
        raise ValueError("RouterBench raw row count does not match embedding metadata")
    cleaned = reconstruct_cleaned_embeddings(raw, tuple(data.frame["source_rows"]))
    query_ids = np.asarray(data.query_ids, dtype=str)
    if len(query_ids) != cleaned.shape[0] or len(set(query_ids.tolist())) != len(query_ids):
        raise RuntimeError("cleaned query IDs are not unique and row-aligned")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        archive_path = temporary / ARCHIVE_FILENAME
        np.savez(archive_path, embeddings=cleaned, query_ids=query_ids)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "immutable": True,
            "source": {
                "routerbench_pickle_sha256": pickle_hash,
                "raw_embeddings_sha256": raw_hash,
                "embedding_meta_sha256": sha256_file(meta_path),
                "embedding_hash_key": embedding_hash_key,
                "raw_rows": expected_rows,
                "embedding_dim": expected_dim,
            },
            "reconstruction": {
                "cleaned_queries": len(query_ids),
                "source_rows_rule": "arithmetic_mean_of_exact_loader_source_rows",
                "row_normalization": "l2_after_source_row_mean",
                "query_order": "exact_cleaned_routerbench_query_id_order",
                "query_ids_sha256": _hash_json(query_ids.tolist()),
                "embeddings_content_sha256": hashlib.sha256(
                    np.ascontiguousarray(cleaned).tobytes()
                ).hexdigest(),
            },
            "output": {
                "archive_filename": ARCHIVE_FILENAME,
                "archive_sha256": sha256_file(archive_path),
                "n_rows": int(cleaned.shape[0]),
                "embedding_dim": int(cleaned.shape[1]),
            },
        }
        manifest["manifest_sha256"] = _hash_json(manifest)
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        load_archive_manifest(temporary / MANIFEST_FILENAME, archive_path=archive_path)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routerbench-pickle", required=True)
    parser.add_argument("--raw-embeddings", required=True)
    parser.add_argument("--embedding-meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shot", default="0shot")
    parser.add_argument("--embedding-hash-key", default="mean_sha256")
    arguments = parser.parse_args(argv)
    manifest = build_routerbench_embedding_archive(
        routerbench_pickle=arguments.routerbench_pickle,
        raw_embeddings=arguments.raw_embeddings,
        embedding_meta=arguments.embedding_meta,
        output_dir=arguments.out_dir,
        shot=arguments.shot,
        embedding_hash_key=arguments.embedding_hash_key,
    )
    print(
        "wrote immutable RouterBench embedding archive with "
        f"{manifest['output']['n_rows']} cleaned rows to {arguments.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
