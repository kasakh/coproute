"""Focused tests for durable experiment artifacts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.artifacts import (  # noqa: E402
    ArtifactValidationError,
    RunCollisionError,
    aggregate_run_index,
    atomic_write_json,
    create_run_directory,
    finalize_run,
    is_run_complete,
    sha256_file,
    sha256_json,
    validate_raw_artifacts,
    write_run_manifest,
)


def valid_raw_artifacts() -> dict:
    return {
        "schema_version": 1,
        "selected_sets": [["model-a"], []],
        "thresholds": [0.8, 0.9],
        "predictions": [{"answer": "a"}, {"answer": "b"}],
        "risks": [0.05, 0.08],
        "costs": [1.25, 0.0],
        "correlations": [[1.0, 0.25], [0.25, 1.0]],
        "seeds": {"design": 11, "calibration": 12, "test": 13},
        "split_ids": {
            "design": ["d-1", "d-2"],
            "calibration": ["c-1", "c-2"],
            "test": ["t-1", "t-2"],
        },
    }


class ArtifactUtilityTests(unittest.TestCase):
    def test_run_collision_requires_explicit_rerun_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            first = create_run_directory(root, "experiment")
            self.assertEqual(first.name, "experiment")

            with self.assertRaises(RunCollisionError):
                create_run_directory(root, "experiment")

            rerun = create_run_directory(
                root,
                "experiment",
                rerun_timestamp="20260903_0145",
            )
            self.assertEqual(rerun.name, "experiment_rerun_20260903_0145")
            with self.assertRaises(RunCollisionError):
                create_run_directory(
                    root,
                    "experiment",
                    rerun_timestamp="20260903_0145",
                )
            self.assertTrue(first.is_dir())

    def test_partial_run_is_never_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = create_run_directory(root / "runs", "partial")
            data = root / "input.jsonl"
            data.write_text('{"id": 1}\n', encoding="utf-8")
            write_run_manifest(
                run,
                run_id="partial",
                config={"alpha": 0.1},
                environment={"python": "3.11"},
                gpu={"indices": [2, 3]},
                data_files={"input": data},
            )

            self.assertFalse(is_run_complete(run))
            with self.assertRaises(ArtifactValidationError):
                finalize_run(run)
            self.assertFalse((run / "DONE").exists())
            self.assertFalse(is_run_complete(run))

    def test_atomic_write_uses_replace_and_preserves_old_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state.json"
            atomic_write_json(target, {"generation": 1})

            real_replace = os.replace
            with mock.patch("components.artifacts.os.replace", wraps=real_replace) as replace:
                atomic_write_json(target, {"generation": 2})
            replace.assert_called_once()
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"generation": 2})

            with mock.patch("components.artifacts.os.replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    atomic_write_json(target, {"generation": 3})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"generation": 2})
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_schema_errors_are_rejected(self) -> None:
        cases = {}

        missing = valid_raw_artifacts()
        del missing["costs"]
        cases["missing costs"] = missing

        mismatched = valid_raw_artifacts()
        mismatched["thresholds"] = [0.8]
        cases["row count mismatch"] = mismatched

        overlapping = valid_raw_artifacts()
        overlapping["split_ids"]["test"][0] = "d-1"
        cases["split overlap"] = overlapping

        asymmetric = valid_raw_artifacts()
        asymmetric["correlations"] = [[1.0, 0.25], [0.5, 1.0]]
        cases["asymmetric correlations"] = asymmetric

        invalid_seed = valid_raw_artifacts()
        invalid_seed["seeds"]["test"] = True
        cases["boolean seed"] = invalid_seed

        for name, payload in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ArtifactValidationError):
                    validate_raw_artifacts(payload)

    def test_sha256_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.bin"
            second = Path(temporary) / "second.bin"
            content = b"stable experiment bytes\x00\xff\n"
            first.write_bytes(content)
            second.write_bytes(content)

            self.assertEqual(sha256_file(first), sha256_file(first))
            self.assertEqual(sha256_file(first), sha256_file(second))
            self.assertEqual(
                sha256_json({"b": [2, 3], "a": 1}),
                sha256_json({"a": 1, "b": [2, 3]}),
            )

    def test_completion_validates_artifacts_and_index_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            run = create_run_directory(runs, "complete")
            data = root / "dataset.bin"
            data.write_bytes(b"dataset-v1")
            manifest = write_run_manifest(
                run,
                run_id="complete",
                config={"alpha": 0.1, "selector": "copula"},
                environment={"python": "3.11", "packages": {"numpy": "2.0"}},
                gpu={"indices": [2], "name": "L40S"},
                data_files={"routerbench": data},
            )
            atomic_write_json(run / "raw_artifacts.json", valid_raw_artifacts())

            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertRegex(manifest_payload["hashes"]["config"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest_payload["hashes"]["environment"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest_payload["hashes"]["gpu"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                manifest_payload["hashes"]["data"]["routerbench"],
                sha256_file(data),
            )
            self.assertFalse(is_run_complete(run))

            sentinel = finalize_run(run)
            self.assertEqual(sentinel.name, "DONE")
            self.assertTrue(is_run_complete(run, strict=True))

            before = {
                path.relative_to(runs): (path.stat().st_mtime_ns, path.stat().st_size)
                for path in runs.rglob("*")
            }
            index = aggregate_run_index(runs)
            after = {
                path.relative_to(runs): (path.stat().st_mtime_ns, path.stat().st_size)
                for path in runs.rglob("*")
            }
            self.assertEqual(before, after)
            self.assertEqual(len(index["runs"]), 1)
            self.assertEqual(index["runs"][0]["run_id"], "complete")
            self.assertTrue(index["runs"][0]["complete"])
            self.assertFalse((runs / "run_index.json").exists())

            with self.assertRaises(RunCollisionError):
                finalize_run(run)

            atomic_write_json(run / "raw_artifacts.json", {"tampered": True})
            self.assertFalse(is_run_complete(run))
            with self.assertRaises(ArtifactValidationError):
                is_run_complete(run, strict=True)


if __name__ == "__main__":
    unittest.main()
