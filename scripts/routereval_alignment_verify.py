"""Verify the RouterEval router_dataset to leaderboard row alignment.

The blocker for Section 12 of PLAN_ROUTERBENCH_AND_VALIDATION_20260903.md is that
`router_dataset/<name>_router_dataset.pkl` carries per-query embeddings and
train/val/test index arrays, while `leaderboard_score/leaderboard_new.pkl`
carries the per-query correctness matrix under FINE-GRAINED sub-task keys. For a
multi-sub-task benchmark the router_dataset indices address a CONCATENATION of
the sub-task correctness matrices, and the release does not state the
concatenation order.

Two facts this script establishes.

First, the index counts match exactly. For ifeval the router_dataset holds 541
indices and the single `ifeval` leaderboard key holds 541 rows, so index i maps
to row i with NO ordering assumption at all. ifeval is therefore usable for a
per-query wrapper immediately.

Second, for a multi-sub-task benchmark the ordering can be DECIDED empirically
rather than assumed. Under a candidate order the concatenated correctness matrix
is built, a nearest-neighbour router is fit on the train partition embeddings and
its top-1 correctness-prediction accuracy is measured on the test partition. A
correct alignment produces a router that beats a deliberately permuted alignment
by a wide margin, because a wrong alignment destroys the association between an
embedding and its own correctness row. The script reports the margin so the
ordering is accepted or rejected on evidence.

Labels and embeddings only. No GPU, no model calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np


def _load_router_dataset(router_dir: Path, name: str) -> dict:
    path = router_dir / f"{name}_router_dataset.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_leaderboard(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _concatenate(leaderboard: dict, ordered_keys: list[str], threshold: float) -> np.ndarray:
    blocks = []
    for key in ordered_keys:
        arr = np.asarray(leaderboard["data"][key]["correctness"], dtype=float)
        blocks.append((arr >= threshold).astype(int))
    return np.concatenate(blocks, axis=0)


def _knn_top1_accuracy(
    train_embed: np.ndarray,
    train_correct: np.ndarray,
    test_embed: np.ndarray,
    test_correct: np.ndarray,
    k: int,
) -> float:
    """Fit a k-nearest-neighbour predictor of per-model correctness from embeddings,
    then measure how often its argmax model is actually correct on the test query.

    A correct alignment means an embedding sits next to embeddings with a similar
    correctness profile, so the neighbourhood average is informative and the argmax
    model is correct more often than the pool's own average. A wrong alignment
    scrambles that association and the argmax degenerates toward the pool average.
    """
    # Cosine similarity on L2-normalised embeddings
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    train_n = _normalise(train_embed.astype(float))
    test_n = _normalise(test_embed.astype(float))
    similarity = test_n @ train_n.T
    k = min(k, train_n.shape[0])
    neighbour_idx = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    hits = 0
    for row in range(test_n.shape[0]):
        neighbours = neighbour_idx[row]
        predicted = train_correct[neighbours].mean(axis=0)
        best_model = int(np.argmax(predicted))
        hits += int(test_correct[row, best_model] == 1)
    return hits / test_n.shape[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--name", required=True, help="router_dataset base name, e.g. gpqa")
    parser.add_argument("--keys", required=True,
                        help="comma-separated leaderboard keys belonging to this benchmark")
    parser.add_argument("--k", type=int, default=40, help="neighbourhood size")
    parser.add_argument("--pool-size", type=int, default=200,
                        help="model columns to subsample for the router probe (speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-permutations", type=int, default=6,
                        help="cap on candidate orderings enumerated")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    router_dir = Path(arguments.router_dir)
    leaderboard = _load_leaderboard(Path(arguments.leaderboard))
    rd = _load_router_dataset(router_dir, arguments.name)
    keys = [k.strip() for k in arguments.keys.split(",") if k.strip()]

    split = rd["split_index"]
    embed = rd["embedding"]
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    train_embed = np.asarray(embed["train_embed"], dtype=float)
    test_embed = np.asarray(embed["test_embed"], dtype=float)

    total_indices = sum(
        len(np.asarray(split[key])) for key in ("train_indices", "val_indices", "test_indices")
    )
    leaderboard_rows = sum(
        np.asarray(leaderboard["data"][key]["correctness"]).shape[0] for key in keys
    )
    print(f"benchmark {arguments.name}")
    print(f"  router_dataset total indices: {total_indices}")
    print(f"  leaderboard rows over {len(keys)} keys: {leaderboard_rows}")
    print(f"  count match: {total_indices == leaderboard_rows}")
    if total_indices != leaderboard_rows:
        print("  REFUSING: index count does not match leaderboard row count, "
              "so no concatenation order can be correct.")
        return 1
    if len(keys) == 1:
        print("  SINGLE KEY: index i maps to row i with no ordering assumption. "
              "Alignment is exact by construction.")

    # Subsample model columns for the probe so the KNN pass is fast
    n_models_full = np.asarray(leaderboard["data"][keys[0]]["correctness"]).shape[1]
    generator = np.random.default_rng(arguments.seed)
    pool = arguments.pool_size
    if pool >= n_models_full:
        columns = np.arange(n_models_full)
    else:
        columns = generator.choice(n_models_full, size=pool, replace=False)
    columns = np.sort(columns)

    candidate_orders: list[list[str]] = []
    if len(keys) == 1:
        candidate_orders = [list(keys)]
    else:
        alphabetical = sorted(keys)
        candidate_orders.append(alphabetical)
        # a few alternative orders, capped
        for permutation in itertools.islice(
            (list(p) for p in itertools.permutations(keys) if list(p) != alphabetical),
            arguments.max_permutations - 1,
        ):
            candidate_orders.append(permutation)

    results = []
    for order in candidate_orders:
        concatenated = _concatenate(leaderboard, order, arguments.threshold)[:, columns]
        train_correct = concatenated[train_idx]
        test_correct = concatenated[test_idx]
        accuracy = _knn_top1_accuracy(
            train_embed, train_correct, test_embed, test_correct, arguments.k
        )
        pool_average = float(test_correct.mean())
        best_single = float(test_correct.mean(axis=0).max())
        results.append({
            "order": order,
            "knn_top1_accuracy": accuracy,
            "pool_average_correctness": pool_average,
            "best_single_model_accuracy": best_single,
            "lift_over_pool_average": accuracy - pool_average,
            "lift_over_best_single": accuracy - best_single,
        })

    # Scrambled control: keep the alphabetical concatenation but permute the ROW order
    scrambled_concatenated = _concatenate(
        leaderboard, sorted(keys), arguments.threshold
    )[:, columns]
    row_permutation = generator.permutation(scrambled_concatenated.shape[0])
    scrambled = scrambled_concatenated[row_permutation]
    scrambled_accuracy = _knn_top1_accuracy(
        train_embed, scrambled[train_idx], test_embed, scrambled[test_idx], arguments.k
    )

    print()
    print(f"{'order':<70s}  {'knn top1':>9s}  {'pool avg':>9s}  {'best single':>11s}  {'lift vs pool':>12s}")
    for entry in sorted(results, key=lambda e: -e["knn_top1_accuracy"]):
        order_label = ", ".join(entry["order"])
        if len(order_label) > 68:
            order_label = order_label[:65] + "..."
        print(
            f"{order_label:<70s}  {entry['knn_top1_accuracy']:>9.4f}  "
            f"{entry['pool_average_correctness']:>9.4f}  "
            f"{entry['best_single_model_accuracy']:>11.4f}  "
            f"{entry['lift_over_pool_average']:>12.4f}"
        )
    print()
    print(f"scrambled-row control knn top1 accuracy: {scrambled_accuracy:.4f}")
    best = max(results, key=lambda e: e["knn_top1_accuracy"])
    margin = best["knn_top1_accuracy"] - scrambled_accuracy
    print(f"best candidate minus scrambled control: {margin:+.4f}")
    if len(keys) == 1:
        verdict = "EXACT_BY_CONSTRUCTION"
    elif margin > 0.02:
        verdict = "ORDER_SUPPORTED"
    else:
        verdict = "ORDER_NOT_DISTINGUISHABLE"
    print(f"verdict: {verdict}")

    if arguments.out:
        out_path = Path(arguments.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "name": arguments.name,
            "keys": keys,
            "total_indices": int(total_indices),
            "leaderboard_rows": int(leaderboard_rows),
            "count_match": bool(total_indices == leaderboard_rows),
            "probe_pool_size": int(len(columns)),
            "knn_k": arguments.k,
            "candidates": results,
            "scrambled_control_accuracy": scrambled_accuracy,
            "best_minus_scrambled": margin,
            "verdict": verdict,
        }, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
