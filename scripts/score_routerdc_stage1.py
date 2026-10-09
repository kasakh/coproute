"""Generate immutable full-pool RouterDC logits for one registered target task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from components.routerdc_baseline import RouterDCBaseline
from scripts.run_learned_baselines import sha256_file
from scripts.run_paired_main_suite import _load_task, hash_json


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(arguments: argparse.Namespace) -> Path:
    destination = Path(arguments.out_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    checkpoint_dir = Path(arguments.checkpoint_dir)
    checkpoint_manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_metadata = json.loads(
        (checkpoint_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    task = _load_task(
        arguments.protocol,
        arguments.benchmark,
        leaderboard=Path(arguments.leaderboard) if arguments.leaderboard else None,
        router_dir=Path(arguments.router_dir) if arguments.router_dir else None,
        routerbench_pickle=Path(arguments.routerbench_pickle) if arguments.routerbench_pickle else None,
        embedding_archive=Path(arguments.embedding_archive) if arguments.embedding_archive else None,
        embedding_manifest=Path(arguments.embedding_manifest) if arguments.embedding_manifest else None,
    )
    model = RouterDCBaseline.load(
        checkpoint_dir / "checkpoint.pt",
        backbone_path=arguments.backbone_path,
        device=arguments.device,
    )
    if list(model.model_ids) != list(task.model_ids):
        raise ValueError("RouterDC full model IDs differ from target task model IDs")
    identity = checkpoint_metadata["identity"]
    if identity["target_family"] != arguments.target_family:
        raise ValueError("RouterDC checkpoint target family differs from score request")
    if identity["excluded_family"] != arguments.target_family:
        raise ValueError("RouterDC checkpoint exclusion differs from score request")
    if arguments.target_family in checkpoint_metadata["training_families"]:
        raise ValueError("RouterDC target family entered checkpoint training")
    logits = model.predict_logits(task.prompts)
    if logits.shape != task.correctness.shape or not np.all(np.isfinite(logits)):
        raise ValueError("RouterDC logits do not align with the target task")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
    try:
        np.savez_compressed(
            temporary / "scores.npz",
            logits=logits,
            query_ids=np.asarray(task.query_ids),
            model_ids=np.asarray(task.model_ids),
        )
        metadata = {
            "protocol": "issue63_routerdc_frozen_scores_v1",
            "identity": {
                "source": arguments.protocol,
                "benchmark": arguments.benchmark,
                "target_family": arguments.target_family,
                "training_seed": identity["training_seed"],
                "host": socket.gethostname(),
                "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "run_stamp": arguments.run_stamp,
            },
            "checkpoint_identity": identity,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_dir / "manifest.json"),
            "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
            "query_ids_sha256": hash_json(list(task.query_ids)),
            "model_ids_sha256": hash_json(list(task.model_ids)),
            "logits_sha256": hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
            "shape": list(logits.shape),
            "target_labels_read": False,
        }
        atomic_json(temporary / "metadata.json", metadata)
        files = {
            "scores.npz": sha256_file(temporary / "scores.npz"),
            "metadata.json": sha256_file(temporary / "metadata.json"),
        }
        atomic_json(temporary / "manifest.json", {
            "protocol": metadata["protocol"], "immutable": True,
            "identity": metadata["identity"], "files": files,
            "logits_sha256": metadata["logits_sha256"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
        })
        (temporary / "DONE").touch()
        os.replace(temporary, destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("routereval", "routerbench"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--leaderboard")
    parser.add_argument("--router-dir")
    parser.add_argument("--routerbench-pickle")
    parser.add_argument("--embedding-archive")
    parser.add_argument("--embedding-manifest")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
