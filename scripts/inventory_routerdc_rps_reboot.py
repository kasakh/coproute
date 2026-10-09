"""Inventory durable RouterDC checkpoints after an authorized the compute host reboot.

This tool is preparation only. Run it on the compute host after reboot with the sealed
reservation. It validates every completed artifact and writes an immutable
remaining-task manifest. It never deletes partial temporary directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inventory(root: Path, reservation_path: Path) -> dict[str, Any]:
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if reservation.get("task_count") != 32:
        raise ValueError("reservation must contain exactly 32 tasks")
    completed = []
    remaining = []
    for task in reservation["tasks"]:
        source = task["source"]
        family = task["target_family"]
        seed = int(task["training_seed"])
        directory = root / "checkpoints" / source / family / f"seed{seed}"
        manifest_path = directory / "manifest.json"
        done_path = directory / "DONE"
        reason = None
        if not done_path.is_file():
            reason = "missing_DONE"
        elif not manifest_path.is_file():
            reason = "missing_manifest"
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest.get("identity", {})
            expected = {
                "source": source,
                "target_family": family,
                "excluded_family": family,
                "training_seed": seed,
                "host": "ip-172-31-37-64.us-east-2.compute.internal",
                "run_stamp": "20260920T0802Z",
            }
            mismatch = [key for key, value in expected.items() if identity.get(key) != value]
            if mismatch:
                reason = "identity_mismatch:" + ",".join(mismatch)
            else:
                for name, expected_hash in manifest.get("files", {}).items():
                    artifact = directory / name
                    if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                        reason = f"artifact_hash_mismatch:{name}"
                        break
        record = {**task, "directory": str(directory)}
        if reason is None:
            record["manifest_sha256"] = sha256_file(manifest_path)
            completed.append(record)
        else:
            record["reason"] = reason
            remaining.append(record)
    count = len(remaining)
    waves = (count + 7) // 8
    return {
        "protocol": "issue63_the compute host_post_reboot_inventory_v1",
        "source_root": str(root),
        "reservation": str(reservation_path),
        "expected_task_count": 32,
        "completed_count": len(completed),
        "remaining_count": count,
        "completed": completed,
        "remaining": remaining,
        "estimated_relaunch_waves": waves,
        "estimated_minutes_low": 10 + 35 * waves if count else 10,
        "estimated_minutes_high": 10 + 40 * waves if count else 10,
        "partial_temporary_directories_deleted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reservation", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = inventory(args.root, args.reservation)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in (
        "completed_count", "remaining_count", "estimated_minutes_low", "estimated_minutes_high"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
