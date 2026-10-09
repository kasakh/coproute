"""Gate A: reproduce the official RACER GSM8K numbers with our implementation.

Targets, read out of the official repo's own results JSON
(results/gsm8k/repeated_racer_alpha0.03_test_ratio0.4_n_splits100):

  knn_gap  : mean_avg_set_size 5.689026, mean_risk 0.028299, abstention 0.006392
  knn_prob : mean_avg_set_size 3.944787, mean_risk 0.028971, abstention 0.000000

Protocol transcribed from their main.py, RACER.py and routers/knn_router.py:
  - pooled eval = gsm8k_cal.json + gsm8k_test.json, n = 1819
  - n_ho = int(n*0.1) = 181, n_cal = int(n*0.5) = 909, n_test = rest = 729
  - split i uses np.random.seed(42 + i) then np.random.permutation(n)
  - router: sklearn NearestNeighbors(n_neighbors=40, metric="cosine") fitted on the
    pooled train embeddings; probs = MEAN of the 40 neighbours' label rows
  - augmentation: null column = 1 - max(probs), appended BEFORE the nonconformity
    map, so for "gap" the row max is taken over the AUGMENTED row
  - nonconformity: gap = rowmax - p + U(0, 1e-6); prob = 1 - p + U(0, 1e-6)
  - ell_i = min score over the augmented correct set (never empty after augmenting)
  - threshold = smallest sorted ell with (n - (i+1) + 1)/(n+1) <= alpha
  - risk on test = fraction of rows whose selected augmented set misses every
    augmented correct model, so a selected null counts as covered when all real
    models fail (their evaluate_racer convention)

Two engines are run on identical probabilities:
  official  : a direct transcription of RACER.py, including its max-ell fallback
  ours      : components.racer_protocol, our own implementation under test

Agreement between the two localizes any divergence; agreement with their
published summary is Gate A itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.racer_protocol import (  # noqa: E402
    ground_truth_critical_scores,
    misrouting_loss,
    nonconformity_scores,
    prediction_sets,
    select_crc_threshold,
)

XI = 1e-6


def official_scores(probs: np.ndarray, method: str, rng: np.random.Generator) -> np.ndarray:
    """Transcription of compute_non_conformity_scores from RACER.py."""

    epsilon = rng.uniform(0.0, XI, size=probs.shape)
    if method == "gap":
        return np.max(probs, axis=1, keepdims=True) - probs + epsilon
    if method == "one_minus_prob":
        return 1.0 - probs + epsilon
    raise ValueError(method)


def augment(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    has_correct = labels.sum(axis=1) > 0
    null_labels = (~has_correct).astype(int).reshape(-1, 1)
    labels_aug = np.hstack([labels, null_labels])
    null_probs = 1.0 - np.max(probs, axis=1, keepdims=True)
    probs_aug = np.hstack([probs, null_probs])
    return probs_aug, labels_aug


def official_threshold(ell: np.ndarray, alpha: float) -> float:
    """Transcription of RACER_Module.calibrate's lambda scan, max-ell fallback."""

    ell_sorted = np.sort(ell)
    n = len(ell_sorted)
    for index, lam in enumerate(ell_sorted):
        risk = (n - (index + 1) + 1) / (n + 1)
        if risk <= alpha:
            return float(lam)
    return float(np.max(ell_sorted))


def evaluate(
    probs_all: np.ndarray,
    labels_all: np.ndarray,
    *,
    alpha: float,
    method: str,
    n_splits: int,
    seed_base: int,
    engine: str,
) -> dict:
    n_all = probs_all.shape[0]
    n_ho = int(n_all * 0.1)
    n_cal = int(n_all * 0.5)
    risks, sizes, abstentions, lambdas, non_null_sizes = [], [], [], [], []
    augmented_sizes = []

    for split in range(n_splits):
        np.random.seed(seed_base + split)
        indices = np.random.permutation(n_all)
        idx_cal = indices[n_ho : n_ho + n_cal]
        idx_test = indices[n_ho + n_cal :]

        if engine == "official":
            rng = np.random.default_rng(seed_base + split)
            probs_aug, labels_aug = augment(probs_all, labels_all)
            scores = official_scores(probs_aug, method, rng)
            ell = np.min(np.where(labels_aug.astype(bool), scores, np.inf), axis=1)
            lam = official_threshold(ell[idx_cal], alpha)
            mask = scores[idx_test] <= lam
            covered = np.any(mask & labels_aug[idx_test].astype(bool), axis=1)
            standard = mask[:, :-1]
            size = standard.sum(axis=1)
        elif engine == "ours":
            variant = "G" if method == "gap" else "P"
            augmented, scores = nonconformity_scores(
                probs_all, variant, seed=seed_base, split_seed=split + 1
            )
            _, labels_aug = augment(probs_all, labels_all)
            critical = ground_truth_critical_scores(scores[idx_cal], labels_aug[idx_cal])
            calibration = select_crc_threshold(critical, alpha)
            lam = calibration.threshold
            routing = prediction_sets(augmented[idx_test], scores[idx_test], lam)
            covered = ~misrouting_loss(routing.augmented_mask, labels_aug[idx_test])
            size = routing.size
            standard = routing.standard_mask
        else:
            raise ValueError(engine)

        risks.append(float(1.0 - covered.mean()))
        sizes.append(float(size.mean()))
        augmented_sizes.append(float(mask.sum(axis=1).mean()) if engine == "official" else float(routing.augmented_mask.sum(axis=1).mean()))
        abstentions.append(float((size == 0).mean()))
        non_null_sizes.append(float(size[size > 0].mean()) if np.any(size > 0) else 0.0)
        lambdas.append(float(lam) if np.isfinite(lam) else float("inf"))

    return {
        "engine": engine,
        "method": method,
        "alpha": alpha,
        "n_splits": n_splits,
        "n_test_per_split": int(n_all - n_ho - n_cal),
        "mean_risk": float(np.mean(risks)),
        "std_risk": float(np.std(risks, ddof=1)),
        "mean_coverage": float(1.0 - np.mean(risks)),
        "mean_avg_set_size": float(np.mean(sizes)),
        "mean_augmented_set_size": float(np.mean(augmented_sizes)),
        "std_avg_set_size": float(np.std(sizes, ddof=1)),
        "mean_abstention_rate": float(np.mean(abstentions)),
        "mean_non_null_set_size": float(np.mean(non_null_sizes)),
        "mean_lambda": float(np.mean([v for v in lambdas if np.isfinite(v)])),
        "infinite_lambda_splits": int(sum(1 for v in lambdas if not np.isfinite(v))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--knearest", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument(
        "--use-softmax",
        default="true",
        choices=("true", "false"),
        help="official default is true (main.py --use_softmax True)",
    )
    arguments = parser.parse_args()
    arguments.use_softmax = arguments.use_softmax == "true"

    out_dir = Path(arguments.out_dir)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)

    parsed = Path(arguments.parsed_dir)
    embeddings = Path(arguments.embeddings_dir)
    meta = json.loads((parsed / "meta.json").read_text())
    train_labels = np.load(parsed / "train_labels.npy").astype(float)
    eval_labels = np.load(parsed / "eval_labels.npy").astype(int)
    train_embed = np.load(embeddings / "train_cls.npy")
    eval_embed = np.load(embeddings / "eval_cls.npy")

    if train_embed.shape[0] != train_labels.shape[0]:
        raise SystemExit("train embedding rows do not match label rows")
    if eval_embed.shape[0] != eval_labels.shape[0]:
        raise SystemExit("eval embedding rows do not match label rows")

    from sklearn.neighbors import NearestNeighbors

    knn = NearestNeighbors(n_neighbors=arguments.knearest, metric="cosine")
    knn.fit(train_embed)
    _, neighbours = knn.kneighbors(eval_embed)
    logits = train_labels[neighbours].mean(axis=1)
    if arguments.use_softmax:
        # main.py::load_dataset_scores_and_labels applies F.softmax(logits, dim=1)
        # to the router output before RACER ever sees it. The KNN "logits" are the
        # mean neighbour label vector, so this is a softmax over the model axis.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probs_all = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    else:
        probs_all = logits
    print(
        f"router probs {probs_all.shape} mean {probs_all.mean():.4f} "
        f"min {probs_all.min():.4f} max {probs_all.max():.4f} "
        f"row_sum {probs_all.sum(axis=1).mean():.4f} softmax={arguments.use_softmax}"
    )

    targets = meta["protocol"]["official_targets"]
    summary = {
        "machine": os.uname().nodename,
        "parsed_dir": str(parsed),
        "embeddings_dir": str(embeddings),
        "models": meta["models"],
        "knearest": arguments.knearest,
        "alpha": arguments.alpha,
        "official_targets": targets,
        "runs": {},
        "gate_a": {},
    }

    for method in ("gap", "one_minus_prob"):
        for engine in ("official", "ours"):
            result = evaluate(
                probs_all,
                eval_labels,
                alpha=arguments.alpha,
                method=method,
                n_splits=arguments.n_splits,
                seed_base=arguments.seed_base,
                engine=engine,
            )
            summary["runs"][f"{method}|{engine}"] = result
            target = targets[method]
            # Their avg_set_size COUNTS the null token: their own per-split records
            # show avg_set_size and non_null_avg_set_size exactly 1.0 apart. Compare
            # like with like using our augmented count.
            size_error = result["mean_augmented_set_size"] - target["mean_avg_set_size"]
            risk_error = result["mean_risk"] - target["mean_risk"]
            relative = abs(size_error) / target["mean_avg_set_size"]
            summary["gate_a"][f"{method}|{engine}"] = {
                "set_size_absolute_error": size_error,
                "set_size_relative_error": relative,
                "risk_absolute_error": risk_error,
                "verdict": (
                    "REPRODUCED_WITHIN_5PCT"
                    if relative <= 0.05
                    else ("CLOSE_WITHIN_15PCT" if relative <= 0.15 else "NOT_REPRODUCED")
                ),
            }
            print(
                f"{method:<15s} {engine:<9s} aug_size={result['mean_augmented_set_size']:.4f} std_size={result['mean_avg_set_size']:.4f} "
                f"(target {target['mean_avg_set_size']:.4f}, rel err {relative:.2%})  "
                f"risk={result['mean_risk']:.4f} (target {target['mean_risk']:.4f})  "
                f"abst={result['mean_abstention_rate']:.4f}  "
                f"{summary['gate_a'][f'{method}|{engine}']['verdict']}"
            )

    (out_dir / "gate_a_reproduction.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "DONE").write_text("")
    print("WROTE", out_dir / "gate_a_reproduction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
