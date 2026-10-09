"""Durable, collision-safe experiment artifact utilities.

A run directory has an immutable identity. Writers may atomically replace files
inside a run while it is active, but a completed run must never be reused.
Completion is represented only by a validated ``DONE`` sentinel.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DONE_SENTINEL = "DONE"
MANIFEST_FILE = "manifest.json"
RAW_ARTIFACTS_FILE = "raw_artifacts.json"
DEFAULT_REQUIRED_ARTIFACTS = (MANIFEST_FILE, RAW_ARTIFACTS_FILE)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RERUN_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{4}(?:\d{2})?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Base class for artifact lifecycle errors."""


class RunCollisionError(FileExistsError, ArtifactError):
    """Raised when a caller attempts to reuse a run directory."""


class ArtifactValidationError(ValueError, ArtifactError):
    """Raised when an artifact or manifest violates its schema."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a regular file using bounded memory."""

    source = Path(path)
    if not source.is_file():
        raise ArtifactValidationError(f"not a regular file: {source}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON independently of mapping insertion order or whitespace."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> Path:
    """Durably replace a JSON file using a temporary file in the same directory."""

    target = Path(path)
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None

        directory_flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
        try:
            directory_fd = os.open(target.parent, directory_flags)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return target


def _normalize_rerun_timestamp(value: str | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("rerun datetime must be timezone aware")
        value = value.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    value = value.removeprefix("rerun_")
    if not _RERUN_TIMESTAMP_RE.fullmatch(value):
        raise ValueError("rerun timestamp must use YYYYMMDD_HHMM or YYYYMMDD_HHMMSS")
    return value


def create_run_directory(
    runs_root: os.PathLike[str] | str,
    run_id: str,
    *,
    rerun_timestamp: str | datetime | None = None,
) -> Path:
    """Create a new run directory without ever reusing an existing path.

    A collision is never resolved automatically. The caller must provide a
    timestamp for a new ``<run_id>_rerun_<timestamp>`` directory.
    """

    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(f"unsafe run_id: {run_id!r}")
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    directory_name = run_id
    if rerun_timestamp is not None:
        directory_name = f"{run_id}_rerun_{_normalize_rerun_timestamp(rerun_timestamp)}"
    destination = root / directory_name
    try:
        destination.mkdir()
    except FileExistsError as exc:
        suffix = " with a new rerun_timestamp" if rerun_timestamp is None else ""
        raise RunCollisionError(
            f"run directory already exists and is immutable: {destination}; retry{suffix}"
        ) from exc
    return destination


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{key} must be a list")
    return value


def _stable_identity(value: Any) -> bytes:
    return _canonical_json_bytes(value)


def validate_raw_artifacts(payload: Mapping[str, Any]) -> None:
    """Validate the raw per-query experiment artifact schema.

    ``selected_sets``, ``thresholds``, ``predictions``, ``risks``, and
    ``costs`` have one entry per test query. Empty selected sets are valid and
    represent abstention. ``split_ids`` must contain disjoint design,
    calibration, and test identifiers.
    """

    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("raw artifacts must be a mapping")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ArtifactValidationError(f"unsupported raw artifact schema_version: {payload.get('schema_version')}")

    selected_sets = _require_list(payload, "selected_sets")
    thresholds = _require_list(payload, "thresholds")
    predictions = _require_list(payload, "predictions")
    risks = _require_list(payload, "risks")
    costs = _require_list(payload, "costs")
    correlations = _require_list(payload, "correlations")
    if not selected_sets:
        raise ArtifactValidationError("selected_sets must not be empty")

    row_count = len(selected_sets)
    for key, values in (
        ("thresholds", thresholds),
        ("predictions", predictions),
        ("risks", risks),
        ("costs", costs),
    ):
        if len(values) != row_count:
            raise ArtifactValidationError(
                f"{key} length {len(values)} does not match selected_sets length {row_count}"
            )

    for row_index, selected in enumerate(selected_sets):
        if not isinstance(selected, list):
            raise ArtifactValidationError(f"selected_sets[{row_index}] must be a list")
        if any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in selected):
            raise ArtifactValidationError(f"selected_sets[{row_index}] contains an invalid model id")
        if len({_stable_identity(item) for item in selected}) != len(selected):
            raise ArtifactValidationError(f"selected_sets[{row_index}] contains duplicate model ids")

    for key, values in (("thresholds", thresholds), ("risks", risks), ("costs", costs)):
        for row_index, value in enumerate(values):
            if not _is_number(value):
                raise ArtifactValidationError(f"{key}[{row_index}] must be finite numeric")
            if key in {"risks", "costs"} and value < 0:
                raise ArtifactValidationError(f"{key}[{row_index}] must be nonnegative")

    try:
        _canonical_json_bytes(predictions)
    except ArtifactValidationError as exc:
        raise ArtifactValidationError(f"predictions are not valid JSON: {exc}") from exc

    if not correlations or any(not isinstance(row, list) for row in correlations):
        raise ArtifactValidationError("correlations must be a nonempty square matrix")
    matrix_size = len(correlations)
    if any(len(row) != matrix_size for row in correlations):
        raise ArtifactValidationError("correlations must be square")
    for row_index, row in enumerate(correlations):
        for column_index, value in enumerate(row):
            if not _is_number(value) or value < -1 or value > 1:
                raise ArtifactValidationError(
                    f"correlations[{row_index}][{column_index}] must be finite and within [-1, 1]"
                )
            if row_index == column_index and not math.isclose(value, 1.0, abs_tol=1e-8):
                raise ArtifactValidationError("correlation diagonal must equal 1")
            if not math.isclose(value, correlations[column_index][row_index], abs_tol=1e-8):
                raise ArtifactValidationError("correlations must be symmetric")

    seeds = payload.get("seeds")
    if isinstance(seeds, Mapping):
        seed_values = list(seeds.values())
        if not seeds or any(not isinstance(key, str) or not key for key in seeds):
            raise ArtifactValidationError("seed names must be nonempty strings")
    elif isinstance(seeds, list):
        seed_values = seeds
    else:
        raise ArtifactValidationError("seeds must be a nonempty mapping or list")
    if not seed_values or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_values):
        raise ArtifactValidationError("all seeds must be integers")

    split_ids = payload.get("split_ids")
    if not isinstance(split_ids, Mapping):
        raise ArtifactValidationError("split_ids must be a mapping")
    required_splits = ("design", "calibration", "test")
    all_identities: set[bytes] = set()
    for split_name in required_splits:
        identifiers = split_ids.get(split_name)
        if not isinstance(identifiers, list) or not identifiers:
            raise ArtifactValidationError(f"split_ids.{split_name} must be a nonempty list")
        identities = [_stable_identity(identifier) for identifier in identifiers]
        if len(set(identities)) != len(identities):
            raise ArtifactValidationError(f"split_ids.{split_name} contains duplicates")
        overlap = all_identities.intersection(identities)
        if overlap:
            raise ArtifactValidationError("design, calibration, and test split_ids must be disjoint")
        all_identities.update(identities)
    if len(split_ids["test"]) != row_count:
        raise ArtifactValidationError("split_ids.test length must match per-query artifact lengths")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid JSON artifact {path}: {exc}") from exc


def _require_new_file(path: Path) -> None:
    if path.exists():
        raise RunCollisionError(f"refusing to overwrite immutable run artifact: {path}")


def write_run_manifest(
    run_directory: os.PathLike[str] | str,
    *,
    run_id: str,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    gpu: Mapping[str, Any],
    data_files: Mapping[str, os.PathLike[str] | str],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Snapshot run metadata and write a manifest containing all SHA256 hashes."""

    directory = Path(run_directory)
    if not directory.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {directory}")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    if not data_files:
        raise ArtifactValidationError("data_files must not be empty")
    if is_run_complete(directory):
        raise RunCollisionError(f"completed run is immutable: {directory}")

    snapshots = {
        "config": (directory / "config.json", config),
        "environment": (directory / "environment.json", environment),
        "gpu": (directory / "gpu.json", gpu),
    }
    for path, _ in snapshots.values():
        _require_new_file(path)
    manifest_path = directory / MANIFEST_FILE
    _require_new_file(manifest_path)

    data_records: dict[str, dict[str, str]] = {}
    for name, source_value in sorted(data_files.items()):
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("data file names must be nonempty strings")
        source = Path(source_value)
        data_records[name] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
        }

    for path, value in snapshots.values():
        atomic_write_json(path, value)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "config": "config.json",
            "environment": "environment.json",
            "gpu": "gpu.json",
        },
        "hashes": {
            "config": sha256_file(snapshots["config"][0]),
            "environment": sha256_file(snapshots["environment"][0]),
            "gpu": sha256_file(snapshots["gpu"][0]),
            "data": {name: record["sha256"] for name, record in data_records.items()},
        },
        "data": data_records,
    }
    if extra is not None:
        _canonical_json_bytes(extra)
        manifest["extra"] = dict(extra)
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def validate_run_manifest(
    run_directory: os.PathLike[str] | str,
    manifest: Mapping[str, Any] | None = None,
    *,
    verify_data_files: bool = True,
) -> Mapping[str, Any]:
    """Validate manifest structure and every recorded metadata or data hash."""

    directory = Path(run_directory)
    loaded = _read_json(directory / MANIFEST_FILE) if manifest is None else manifest
    if not isinstance(loaded, Mapping):
        raise ArtifactValidationError("manifest must be a mapping")
    if loaded.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported manifest schema_version")
    if not isinstance(loaded.get("run_id"), str) or not loaded["run_id"]:
        raise ArtifactValidationError("manifest run_id must be a nonempty string")

    files = loaded.get("files")
    hashes = loaded.get("hashes")
    data = loaded.get("data")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping) or not isinstance(data, Mapping):
        raise ArtifactValidationError("manifest files, hashes, and data must be mappings")
    for key in ("config", "environment", "gpu"):
        relative = files.get(key)
        expected = hashes.get(key)
        if relative != f"{key}.json" or not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise ArtifactValidationError(f"invalid manifest {key} metadata")
        if sha256_file(directory / relative) != expected:
            raise ArtifactValidationError(f"manifest {key} hash mismatch")

    data_hashes = hashes.get("data")
    if not isinstance(data_hashes, Mapping) or set(data_hashes) != set(data):
        raise ArtifactValidationError("manifest data hash keys do not match data records")
    for name, record in data.items():
        if not isinstance(record, Mapping):
            raise ArtifactValidationError(f"manifest data record {name!r} must be a mapping")
        expected = record.get("sha256")
        path_value = record.get("path")
        if (
            not isinstance(expected, str)
            or not _SHA256_RE.fullmatch(expected)
            or data_hashes.get(name) != expected
            or not isinstance(path_value, str)
            or not path_value
        ):
            raise ArtifactValidationError(f"invalid manifest data record: {name}")
        if verify_data_files and sha256_file(path_value) != expected:
            raise ArtifactValidationError(f"manifest data hash mismatch: {name}")
    return loaded


def _safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ArtifactValidationError(f"required artifact path must stay inside the run: {value!r}")
    return candidate


def finalize_run(
    run_directory: os.PathLike[str] | str,
    *,
    required_artifacts: Iterable[str] = (),
) -> Path:
    """Validate required artifacts and write ``DONE`` as the final operation."""

    directory = Path(run_directory)
    if not directory.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {directory}")
    sentinel = directory / DONE_SENTINEL
    if sentinel.exists():
        raise RunCollisionError(f"run is already complete: {directory}")

    required_names = list(DEFAULT_REQUIRED_ARTIFACTS)
    for name in required_artifacts:
        if name not in required_names:
            required_names.append(name)

    artifact_hashes: dict[str, str] = {}
    for name in required_names:
        relative = _safe_relative_path(name)
        artifact_path = directory / relative
        if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
            raise ArtifactValidationError(f"required artifact is missing or empty: {name}")
        if artifact_path.suffix == ".json":
            value = _read_json(artifact_path)
            if name == MANIFEST_FILE:
                validate_run_manifest(directory, value)
            elif name == RAW_ARTIFACTS_FILE:
                validate_raw_artifacts(value)
        artifact_hashes[name] = sha256_file(artifact_path)

    done_payload = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "required_artifacts": artifact_hashes,
    }
    atomic_write_json(sentinel, done_payload)
    return sentinel


def is_run_complete(run_directory: os.PathLike[str] | str, *, strict: bool = False) -> bool:
    """Return true only for a valid sentinel whose artifact hashes still match."""

    directory = Path(run_directory)
    sentinel = directory / DONE_SENTINEL
    if not sentinel.is_file():
        return False
    try:
        payload = _read_json(sentinel)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactValidationError("invalid DONE schema_version")
        required = payload.get("required_artifacts")
        if not isinstance(required, Mapping) or not required:
            raise ArtifactValidationError("DONE required_artifacts must be a nonempty mapping")
        for name, expected in required.items():
            if not isinstance(name, str) or not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                raise ArtifactValidationError("invalid DONE artifact hash entry")
            path = directory / _safe_relative_path(name)
            if sha256_file(path) != expected:
                raise ArtifactValidationError(f"completed artifact hash mismatch: {name}")
        return True
    except ArtifactValidationError:
        if strict:
            raise
        return False


def aggregate_run_index(runs_root: os.PathLike[str] | str) -> dict[str, Any]:
    """Build an in-memory run index without creating or modifying any file."""

    root = Path(runs_root)
    if not root.exists():
        return {"schema_version": SCHEMA_VERSION, "runs": []}
    if not root.is_dir():
        raise ArtifactValidationError(f"runs root is not a directory: {root}")

    rows: list[dict[str, Any]] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        row: dict[str, Any] = {
            "directory": directory.name,
            "complete": is_run_complete(directory),
        }
        manifest_path = directory / MANIFEST_FILE
        if manifest_path.is_file():
            try:
                manifest = _read_json(manifest_path)
                if not isinstance(manifest, Mapping):
                    raise ArtifactValidationError("manifest must be a mapping")
                row.update(
                    run_id=manifest.get("run_id"),
                    created_at=manifest.get("created_at"),
                    hashes=manifest.get("hashes"),
                )
            except ArtifactValidationError as exc:
                row["manifest_error"] = str(exc)
        sentinel_path = directory / DONE_SENTINEL
        if sentinel_path.is_file():
            try:
                done = _read_json(sentinel_path)
                if isinstance(done, Mapping):
                    row["completed_at"] = done.get("completed_at")
            except ArtifactValidationError as exc:
                row["completion_error"] = str(exc)
        rows.append(row)
    return {"schema_version": SCHEMA_VERSION, "runs": rows}


# Clear aliases for callers that use lifecycle verbs.
create_run_dir = create_run_directory
complete_run = finalize_run
mark_run_complete = finalize_run
build_run_index = aggregate_run_index
validate_raw_artifact_schema = validate_raw_artifacts
