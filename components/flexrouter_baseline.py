"""Faithful FlexRouter core over frozen, precomputed query embeddings.

This module matches the trainable part described in the FlexRouter paper. It
projects frozen query embeddings into a 128 dimensional latent space and learns
one embedding per candidate model. It intentionally does not instantiate or
claim to reproduce the paper's RoBERTa encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LATENT_DIM = 128
AUXILIARY_BCE_WEIGHT = 1.0
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
MAX_SUBSET_SIZE = 10
_SERIALIZATION_VERSION = 1


@dataclass(frozen=True)
class FlexRouterTrainingConfig:
    """Reproducible settings for the published FlexRouter training core."""

    learning_rate: float = LEARNING_RATE
    max_epochs: int = MAX_EPOCHS
    patience: int = EARLY_STOPPING_PATIENCE
    auxiliary_bce_weight: float = AUXILIARY_BCE_WEIGHT
    batch_size: int = 32
    validation_max_size: int = MAX_SUBSET_SIZE

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.auxiliary_bce_weight < 0.0:
            raise ValueError("auxiliary_bce_weight must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validation_max_size <= 0:
            raise ValueError("validation_max_size must be positive")


@dataclass(frozen=True)
class GreedyDPPResult:
    """A complete native greedy path and its thresholded prefix."""

    selected_indices: tuple[int, ...]
    order: tuple[int, ...]
    gains: tuple[float, ...]
    relative_gains: tuple[float, ...]
    scores: tuple[float, ...]
    relative_threshold: float
    candidate_indices: tuple[int, ...]


def _validate_binary_labels(labels: Tensor, expected_shape: tuple[int, int]) -> Tensor:
    if labels.ndim != 2 or tuple(labels.shape) != expected_shape:
        raise ValueError(f"correctness must have shape {expected_shape}")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("correctness must contain only binary values")
    return labels


def _unit_model_embeddings(model_embeddings: Tensor) -> Tensor:
    if model_embeddings.ndim != 2 or model_embeddings.shape[0] == 0:
        raise ValueError("model_embeddings must be a non-empty matrix")
    if not torch.isfinite(model_embeddings).all():
        raise ValueError("model_embeddings must contain only finite values")
    norms = torch.linalg.vector_norm(model_embeddings, dim=1)
    if torch.any(norms == 0):
        raise ValueError("model_embeddings must not contain zero-norm rows")
    return model_embeddings / norms[:, None]


def cosine_gram(model_embeddings: Tensor) -> Tensor:
    """Return the PSD cosine Gram matrix used by FlexRouter."""

    unit = _unit_model_embeddings(model_embeddings)
    return unit @ unit.T


def dpp_kernel(qualities: Tensor, model_embeddings: Tensor) -> Tensor:
    """Construct ``diag(q) K diag(q)`` for one query or a query batch."""

    unit = _unit_model_embeddings(model_embeddings)
    if qualities.shape[-1] != unit.shape[0]:
        raise ValueError("qualities and model_embeddings disagree on model count")
    if not torch.isfinite(qualities).all():
        raise ValueError("qualities must contain only finite values")
    features = qualities[..., :, None] * unit
    return features @ features.transpose(-1, -2)


def low_rank_logdet_i_plus_l(
    qualities: Tensor,
    model_embeddings: Tensor,
    *,
    inclusion_mask: Tensor | None = None,
) -> Tensor:
    """Compute ``log det(I + L_A)`` with the low-rank determinant lemma.

    ``inclusion_mask`` selects the item set A independently for each row. The
    computation forms a latent-dimension matrix, never the full model kernel.
    """

    unit = _unit_model_embeddings(model_embeddings)
    squeeze = qualities.ndim == 1
    if squeeze:
        qualities = qualities.unsqueeze(0)
    if qualities.ndim != 2 or qualities.shape[1] != unit.shape[0]:
        raise ValueError("qualities must have shape (batch, n_models)")
    if not torch.isfinite(qualities).all():
        raise ValueError("qualities must contain only finite values")

    weights = qualities.square()
    if inclusion_mask is not None:
        if inclusion_mask.ndim == 1:
            inclusion_mask = inclusion_mask.unsqueeze(0)
        if tuple(inclusion_mask.shape) != tuple(weights.shape):
            raise ValueError("inclusion_mask must match qualities")
        weights = weights * inclusion_mask.to(dtype=weights.dtype)

    latent_gram = torch.einsum("bm,md,me->bde", weights, unit, unit)
    identity = torch.eye(
        unit.shape[1], dtype=unit.dtype, device=unit.device
    ).expand(latent_gram.shape[0], -1, -1)
    sign, logabsdet = torch.linalg.slogdet(identity + latent_gram)
    if torch.any(sign <= 0):
        raise RuntimeError("I + L must be positive definite")
    return logabsdet.squeeze(0) if squeeze else logabsdet


def coverage_loss(
    qualities: Tensor,
    correctness: Tensor,
    model_embeddings: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Stable negative log DPP probability of selecting a correct model."""

    squeeze = qualities.ndim == 1
    if squeeze:
        qualities = qualities.unsqueeze(0)
        correctness = correctness.unsqueeze(0)
    if qualities.ndim != 2:
        raise ValueError("qualities must have shape (batch, n_models)")
    correctness = _validate_binary_labels(correctness, tuple(qualities.shape))

    log_total = low_rank_logdet_i_plus_l(qualities, model_embeddings)
    log_failure = low_rank_logdet_i_plus_l(
        qualities,
        model_embeddings,
        inclusion_mask=(correctness == 0),
    )
    log_failure_probability = log_failure - log_total

    # In exact arithmetic this value is non-positive. Keeping it one epsilon
    # below zero makes the all-failure edge case finite while preserving the
    # stable log(1 - exp(x)) evaluation via expm1.
    epsilon = torch.finfo(qualities.dtype).eps
    log_failure_probability = torch.clamp(
        log_failure_probability, max=-epsilon
    )
    losses = -torch.log(-torch.expm1(log_failure_probability))
    if squeeze:
        losses = losses.squeeze(0)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def auxiliary_bce_loss(
    qualities: Tensor,
    correctness: Tensor,
    *,
    reduction: str = "mean",
) -> Tensor:
    """Per-query sum of model-wise BCE, as in the paper's objective."""

    squeeze = qualities.ndim == 1
    if squeeze:
        qualities = qualities.unsqueeze(0)
        correctness = correctness.unsqueeze(0)
    correctness = _validate_binary_labels(correctness, tuple(qualities.shape))
    elementwise = F.binary_cross_entropy(
        qualities,
        correctness.to(dtype=qualities.dtype),
        reduction="none",
    )
    losses = elementwise.sum(dim=1)
    if squeeze:
        losses = losses.squeeze(0)
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _validate_greedy_arguments(
    n_models: int,
    max_size: int,
    relative_threshold: float,
    candidate_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size <= 0:
        raise ValueError("max_size must be a positive integer")
    if not 0.0 <= float(relative_threshold) <= 1.0:
        raise ValueError("relative_threshold must be between zero and one")
    if candidate_indices is None:
        return tuple(range(n_models))
    candidates = tuple(int(index) for index in candidate_indices)
    if not candidates:
        raise ValueError("candidate_indices must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must not contain duplicates")
    if min(candidates) < 0 or max(candidates) >= n_models:
        raise ValueError("candidate_indices contains an out-of-range model")
    return tuple(sorted(candidates))


def _threshold_path(
    order: list[int],
    gains: list[float],
    relative_threshold: float,
    candidates: tuple[int, ...],
) -> GreedyDPPResult:
    if not gains:
        return GreedyDPPResult(
            selected_indices=(),
            order=(),
            gains=(),
            relative_gains=(),
            scores=(),
            relative_threshold=float(relative_threshold),
            candidate_indices=candidates,
        )
    first = gains[0]
    relative: list[float] = []
    previous = 1.0
    for gain in gains:
        current = min(previous, max(0.0, min(1.0, gain / first)))
        relative.append(current)
        previous = current
    scores = [1.0 - value for value in relative]
    selected_count = sum(value >= relative_threshold for value in relative)
    return GreedyDPPResult(
        selected_indices=tuple(order[:selected_count]),
        order=tuple(order),
        gains=tuple(gains),
        relative_gains=tuple(relative),
        scores=tuple(scores),
        relative_threshold=float(relative_threshold),
        candidate_indices=candidates,
    )


def greedy_dpp_map(
    kernel: Tensor,
    *,
    max_size: int = MAX_SUBSET_SIZE,
    relative_threshold: float = 0.0,
    candidate_indices: Sequence[int] | None = None,
) -> GreedyDPPResult:
    """Greedy DPP MAP path using pivoted-Cholesky Schur gains.

    Exact gain ties are resolved by the smallest original model index. The
    thresholded selection contains every path item whose gain is at least the
    requested fraction of the first gain.
    """

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    if not torch.isfinite(kernel).all():
        raise ValueError("kernel must contain only finite values")
    if not torch.allclose(kernel, kernel.T, atol=1e-8, rtol=1e-6):
        raise ValueError("kernel must be symmetric")
    n_models = int(kernel.shape[0])
    candidates = _validate_greedy_arguments(
        n_models, max_size, relative_threshold, candidate_indices
    )
    work = kernel[list(candidates)][:, list(candidates)]
    residual = torch.diagonal(work).clone()
    factors = torch.zeros(
        (len(candidates), min(max_size, len(candidates))),
        dtype=kernel.dtype,
        device=kernel.device,
    )
    available = torch.ones(len(candidates), dtype=torch.bool, device=kernel.device)
    order: list[int] = []
    gains: list[float] = []
    tolerance = 32.0 * torch.finfo(kernel.dtype).eps

    for step in range(min(max_size, len(candidates))):
        masked = residual.masked_fill(~available, -torch.inf)
        local_index = int(torch.argmax(masked).item())
        gain = float(masked[local_index].item())
        if gain <= tolerance:
            break
        order.append(candidates[local_index])
        gains.append(gain)
        available[local_index] = False

        numerator = work[:, local_index]
        if step:
            numerator = numerator - factors[:, :step] @ factors[local_index, :step]
        new_factor = numerator / gain**0.5
        factors[:, step] = new_factor
        residual = torch.clamp(residual - new_factor.square(), min=0.0)
        residual[~available] = 0.0

    return _threshold_path(order, gains, relative_threshold, candidates)


def greedy_dpp_map_low_rank(
    qualities: Tensor,
    model_embeddings: Tensor,
    *,
    max_size: int = MAX_SUBSET_SIZE,
    relative_threshold: float = 0.0,
    candidate_indices: Sequence[int] | None = None,
) -> GreedyDPPResult:
    """Greedy DPP MAP without materializing the full model-by-model kernel."""

    if qualities.ndim != 1:
        raise ValueError("qualities must be a one-dimensional vector")
    unit = _unit_model_embeddings(model_embeddings)
    if qualities.shape[0] != unit.shape[0]:
        raise ValueError("qualities and model_embeddings disagree on model count")
    candidates = _validate_greedy_arguments(
        int(unit.shape[0]), max_size, relative_threshold, candidate_indices
    )
    candidate_tensor = torch.tensor(candidates, device=qualities.device)
    features = qualities[candidate_tensor, None] * unit[candidate_tensor]
    residual = features.square().sum(dim=1)
    factors = torch.zeros(
        (len(candidates), min(max_size, len(candidates))),
        dtype=qualities.dtype,
        device=qualities.device,
    )
    available = torch.ones(len(candidates), dtype=torch.bool, device=qualities.device)
    order: list[int] = []
    gains: list[float] = []
    tolerance = 32.0 * torch.finfo(qualities.dtype).eps

    for step in range(min(max_size, len(candidates))):
        masked = residual.masked_fill(~available, -torch.inf)
        local_index = int(torch.argmax(masked).item())
        gain = float(masked[local_index].item())
        if gain <= tolerance:
            break
        order.append(candidates[local_index])
        gains.append(gain)
        available[local_index] = False

        numerator = features @ features[local_index]
        if step:
            numerator = numerator - factors[:, :step] @ factors[local_index, :step]
        new_factor = numerator / gain**0.5
        factors[:, step] = new_factor
        residual = torch.clamp(residual - new_factor.square(), min=0.0)
        residual[~available] = 0.0

    return _threshold_path(order, gains, relative_threshold, candidates)


def _tensor_digest(tensor: Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    return {
        "sha256": digest,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


class FlexRouter(nn.Module):
    """Trainable FlexRouter head for frozen precomputed query embeddings."""

    def __init__(
        self,
        input_dim: int,
        n_models: int,
        *,
        latent_dim: int = LATENT_DIM,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        top_k0: int | None = None,
        embedding_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or n_models <= 0 or latent_dim <= 0:
            raise ValueError("input_dim, n_models, and latent_dim must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if top_k0 is not None and (top_k0 <= 0 or top_k0 > n_models):
            raise ValueError("top_k0 must be between one and n_models")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")
        metadata = dict(embedding_metadata or {})
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding_metadata must be JSON serializable") from exc

        self.input_dim = int(input_dim)
        self.n_models = int(n_models)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.default_top_k0 = top_k0
        self.embedding_metadata = metadata
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.query_projection = nn.Linear(
                self.input_dim, self.latent_dim, bias=True, dtype=dtype
            )
            self.model_embeddings = nn.Parameter(
                torch.empty(self.n_models, self.latent_dim, dtype=dtype)
            )
            nn.init.normal_(self.model_embeddings, mean=0.0, std=0.02)
        self.to(device)

        self.training_config: FlexRouterTrainingConfig | None = None
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None
        self.best_validation_success_at_10: float | None = None
        self.reproducibility_metadata: dict[str, Any] = {}

    @property
    def dtype(self) -> torch.dtype:
        return self.model_embeddings.dtype

    @property
    def device(self) -> torch.device:
        return self.model_embeddings.device

    def _features(self, values: Tensor | Any, name: str) -> Tensor:
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != self.input_dim:
            raise ValueError(f"{name} must have shape (n_samples, {self.input_dim})")
        if tensor.shape[0] == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be non-empty and finite")
        return tensor

    def _labels(self, values: Tensor | Any, rows: int, name: str) -> Tensor:
        tensor = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        return _validate_binary_labels(tensor, (rows, self.n_models))

    def forward(self, frozen_query_embeddings: Tensor) -> Tensor:
        if frozen_query_embeddings.ndim != 2:
            raise ValueError("frozen_query_embeddings must be a matrix")
        projected = self.query_projection(frozen_query_embeddings)
        logits = projected @ self.model_embeddings.T
        return torch.sigmoid(logits)

    def predict_quality(self, frozen_query_embeddings: Tensor | Any) -> Tensor:
        """Return ``q_i(x)`` without modifying the input embeddings."""

        features = self._features(frozen_query_embeddings, "frozen_query_embeddings")
        with torch.no_grad():
            return self(features)

    def kernel(self, frozen_query_embeddings: Tensor | Any) -> Tensor:
        qualities = self.predict_quality(frozen_query_embeddings)
        return dpp_kernel(qualities, self.model_embeddings)

    def _shortlist(self, qualities: Tensor, top_k0: int | None) -> tuple[int, ...] | None:
        if top_k0 is None:
            return None
        if top_k0 <= 0 or top_k0 > self.n_models:
            raise ValueError("top_k0 must be between one and n_models")
        ranking = torch.argsort(qualities, descending=True, stable=True)
        return tuple(int(index) for index in ranking[:top_k0].tolist())

    def select_from_qualities(
        self,
        qualities: Tensor,
        *,
        max_size: int = MAX_SUBSET_SIZE,
        relative_threshold: float = 0.0,
        top_k0: int | None = None,
    ) -> GreedyDPPResult:
        qualities = torch.as_tensor(qualities, dtype=self.dtype, device=self.device)
        if qualities.ndim != 1 or qualities.shape[0] != self.n_models:
            raise ValueError("qualities must have shape (n_models,)")
        effective_top_k0 = self.default_top_k0 if top_k0 is None else top_k0
        candidates = self._shortlist(qualities, effective_top_k0)
        return greedy_dpp_map_low_rank(
            qualities,
            self.model_embeddings,
            max_size=max_size,
            relative_threshold=relative_threshold,
            candidate_indices=candidates,
        )

    def select(
        self,
        frozen_query_embedding: Tensor | Any,
        *,
        max_size: int = MAX_SUBSET_SIZE,
        relative_threshold: float = 0.0,
        top_k0: int | None = None,
    ) -> GreedyDPPResult:
        features = torch.as_tensor(
            frozen_query_embedding, dtype=self.dtype, device=self.device
        )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.shape != (1, self.input_dim):
            raise ValueError("select expects exactly one query embedding")
        with torch.no_grad():
            qualities = self(features)[0]
            return self.select_from_qualities(
                qualities,
                max_size=max_size,
                relative_threshold=relative_threshold,
                top_k0=top_k0,
            )

    def crc_scores(
        self,
        frozen_query_embedding: Tensor | Any,
        *,
        max_size: int = MAX_SUBSET_SIZE,
        top_k0: int | None = None,
    ) -> Tensor:
        """Return nested prefix scores, where lower thresholds select less.

        For path item j the score is one minus its Schur gain relative to the
        first gain. Scores are non-decreasing along the greedy path. Models
        outside the native path receive positive infinity.
        """

        result = self.select(
            frozen_query_embedding,
            max_size=max_size,
            relative_threshold=0.0,
            top_k0=top_k0,
        )
        scores = torch.full(
            (self.n_models,), torch.inf, dtype=self.dtype, device=self.device
        )
        for index, score in zip(result.order, result.scores):
            scores[index] = score
        return scores

    def validation_success_at_10(
        self, embeddings: Tensor, correctness: Tensor
    ) -> float:
        successes = 0
        with torch.no_grad():
            qualities = self(embeddings)
            for row in range(embeddings.shape[0]):
                result = self.select_from_qualities(
                    qualities[row],
                    max_size=min(MAX_SUBSET_SIZE, self.n_models),
                    relative_threshold=0.0,
                )
                if result.selected_indices and torch.any(
                    correctness[row, list(result.selected_indices)] == 1
                ):
                    successes += 1
        return successes / embeddings.shape[0]

    def fit(
        self,
        train_embeddings: Tensor | Any,
        train_correctness: Tensor | Any,
        validation_embeddings: Tensor | Any,
        validation_correctness: Tensor | Any,
        *,
        config: FlexRouterTrainingConfig | None = None,
    ) -> "FlexRouter":
        """Train with Adam and restore the best validation Success@10 state."""

        config = config or FlexRouterTrainingConfig()
        train_x = self._features(train_embeddings, "train_embeddings")
        train_y = self._labels(train_correctness, train_x.shape[0], "train_correctness")
        validation_x = self._features(validation_embeddings, "validation_embeddings")
        validation_y = self._labels(
            validation_correctness,
            validation_x.shape[0],
            "validation_correctness",
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        best_state: dict[str, Tensor] | None = None
        best_score = -1.0
        best_epoch: int | None = None
        epochs_without_improvement = 0
        self.training_history = []
        self.training_config = config

        for epoch in range(config.max_epochs):
            self.train()
            permutation = torch.randperm(train_x.shape[0], generator=generator)
            total_loss = 0.0
            total_coverage = 0.0
            total_bce = 0.0
            seen = 0
            for start in range(0, train_x.shape[0], config.batch_size):
                indices = permutation[start : start + config.batch_size].to(self.device)
                batch_x = train_x[indices]
                batch_y = train_y[indices]
                optimizer.zero_grad(set_to_none=True)
                qualities = self(batch_x)
                hit = coverage_loss(
                    qualities, batch_y, self.model_embeddings, reduction="mean"
                )
                bce = auxiliary_bce_loss(qualities, batch_y, reduction="mean")
                loss = hit + config.auxiliary_bce_weight * bce
                loss.backward()
                optimizer.step()

                count = int(indices.numel())
                total_loss += float(loss.detach()) * count
                total_coverage += float(hit.detach()) * count
                total_bce += float(bce.detach()) * count
                seen += count

            self.eval()
            validation_score = self.validation_success_at_10(
                validation_x, validation_y
            )
            self.training_history.append(
                {
                    "epoch": epoch,
                    "training_loss": total_loss / seen,
                    "coverage_loss": total_coverage / seen,
                    "auxiliary_bce_loss": total_bce / seen,
                    "validation_success_at_10": validation_score,
                }
            )
            if validation_score > best_score + 1e-12:
                best_score = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        assert best_state is not None and best_epoch is not None
        self.load_state_dict(best_state)
        self.eval()
        self.best_epoch = best_epoch
        self.best_validation_success_at_10 = best_score
        self.reproducibility_metadata = {
            "seed": self.seed,
            "optimizer": "Adam",
            "training_config": asdict(config),
            "epochs_completed": len(self.training_history),
            "best_epoch": best_epoch,
            "best_validation_success_at_10": best_score,
            "train_embeddings": _tensor_digest(train_x),
            "train_correctness": _tensor_digest(train_y),
            "validation_embeddings": _tensor_digest(validation_x),
            "validation_correctness": _tensor_digest(validation_y),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device_trained": str(self.device),
        }
        return self

    def metadata(self) -> dict[str, Any]:
        shortlist = (
            {"policy": "full_candidate_pool", "top_k0": None}
            if self.default_top_k0 is None
            else {"policy": "explicit_quality_top_k0", "top_k0": self.default_top_k0}
        )
        return {
            "serialization_version": _SERIALIZATION_VERSION,
            "router_type": "FlexRouter",
            "fidelity": {
                "trainable_flexrouter_core": True,
                "input_representation": "matched_frozen_embeddings",
                "frozen_precomputed_query_embeddings": True,
                "exact_roberta_encoder_replication": False,
            },
            "architecture": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "n_models": self.n_models,
                "quality": "sigmoid(projected_query_dot_model_embedding)",
                "similarity": "cosine_gram",
                "kernel": "diag(q)_K_diag(q)",
            },
            "inference": {
                "algorithm": "greedy_dpp_map_schur_gain",
                "native_max_size": MAX_SUBSET_SIZE,
                "deterministic_tie_break": "smallest_model_index",
                "shortlist": shortlist,
            },
            "runtime_requirements": {
                "framework": "PyTorch",
                "cpu_supported": True,
                "paper_reported_hardware": "NVIDIA A100 80GB GPU",
                "practical_disadvantage": (
                    "The trainable large-pool baseline adds PyTorch and accelerator "
                    "substrate burden versus the NumPy-only baselines in this repository."
                ),
            },
            "embedding_metadata": copy.deepcopy(self.embedding_metadata),
            "dtype": str(self.dtype),
            "training_history": copy.deepcopy(self.training_history),
            "reproducibility": copy.deepcopy(self.reproducibility_metadata),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.metadata(),
                "state_dict": self.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FlexRouter":
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("FlexRouter checkpoint has invalid metadata")
        if metadata.get("serialization_version") != _SERIALIZATION_VERSION:
            raise ValueError("FlexRouter checkpoint uses an unsupported version")
        if metadata.get("router_type") != "FlexRouter":
            raise ValueError("checkpoint does not contain a FlexRouter")
        architecture = metadata["architecture"]
        dtype_name = metadata.get("dtype", "torch.float64")
        dtype = torch.float32 if dtype_name == "torch.float32" else torch.float64
        shortlist = metadata["inference"]["shortlist"]
        model = cls(
            input_dim=int(architecture["input_dim"]),
            n_models=int(architecture["n_models"]),
            latent_dim=int(architecture["latent_dim"]),
            seed=int(metadata.get("reproducibility", {}).get("seed", 0)),
            dtype=dtype,
            device=device,
            top_k0=shortlist.get("top_k0"),
            embedding_metadata=metadata.get("embedding_metadata", {}),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.training_history = list(metadata.get("training_history", []))
        reproducibility = dict(metadata.get("reproducibility", {}))
        model.reproducibility_metadata = reproducibility
        config_values = reproducibility.get("training_config")
        if config_values:
            model.training_config = FlexRouterTrainingConfig(**config_values)
        model.best_epoch = reproducibility.get("best_epoch")
        model.best_validation_success_at_10 = reproducibility.get(
            "best_validation_success_at_10"
        )
        model.eval()
        return model
