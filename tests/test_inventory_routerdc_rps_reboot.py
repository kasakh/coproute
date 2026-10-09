from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.inventory_routerdc_rps_reboot import inventory


def test_inventory_preserves_valid_checkpoint_and_requeues_missing(tmp_path: Path) -> None:
    tasks = []
    for index in range(32):
        tasks.append({
            "source": "routereval",
            "target_family": "musr",
            "training_seed": 9101 + index,
            "claim_tag": f"task{index}",
        })
    reservation = tmp_path / "reservation.json"
    reservation.write_text(json.dumps({"task_count": 32, "tasks": tasks}))
    directory = tmp_path / "root/checkpoints/routereval/musr/seed9101"
    directory.mkdir(parents=True)
    artifact = directory / "checkpoint.pt"
    artifact.write_bytes(b"complete")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "identity": {
            "source": "routereval",
            "target_family": "musr",
            "excluded_family": "musr",
            "training_seed": 9101,
            "host": "ip-172-31-37-64.us-east-2.compute.internal",
            "run_stamp": "20260920T0802Z",
        },
        "files": {"checkpoint.pt": digest},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "DONE").touch()
    result = inventory(tmp_path / "root", reservation)
    assert result["completed_count"] == 1
    assert result["remaining_count"] == 31
    assert result["estimated_relaunch_waves"] == 4
    assert result["estimated_minutes_low"] == 150
    assert result["estimated_minutes_high"] == 170
    assert result["partial_temporary_directories_deleted"] is False
