"""Exact base router models used by the corrected RACER reproduction.

The module consumes precomputed mDeBERTa embeddings. It deliberately does not
load or download an encoder. KNNR uses cosine similarity with exactly 40
neighbors. MLPR is a one hidden layer network trained with the numerically
stable binary cross entropy with logits objective. RouterDC is only an
interface adapter because no validated RouterDC implementation or configuration
is available in this project.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


MDEBERTA_MODEL_NAME = "microsoft/mdeberta-v3-base"
KNN_NEIGHBORS = 40
MLP_HIDDEN_UNITS = 256
MLP_BATCH_SIZE = 32
MLP_LEARNING_RATE = 1e-4
MLP_EPOCHS = 100
_SERIALIZATION_VERSION = 1


class RouterNotFittedError(RuntimeError):
    """Raised when prediction or persistence is attempted before fitting."""


class RouterDCUnavailableError(RuntimeError):
    """Raised when a validated RouterDC implementation was not supplied."""


def _as_feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _as_label_matrix(values: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 2 or labels.shape[0] != n_rows or labels.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (n_samples, n_models) with the expected rows"
        )
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return labels.astype(np.float64, copy=False)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _normalize_device(device: str) -> str:
    normalized = str(device).lower()
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    return normalized


def _array_module(device: str):
    if device == "cpu":
        return np
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CUDA execution requires CuPy in the active environment; "
            "no CPU fallback is used when device='cuda'"
        ) from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device is visible")
    except Exception as exc:
        raise RuntimeError(
            "device='cuda' was requested but no usable CUDA device is available"
        ) from exc
    return cp


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) or np.isscalar(value):
        return np.asarray(value)
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise RuntimeError("cannot transfer a CUDA array without CuPy") from exc
    return cp.asnumpy(value)


def _write_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _read_metadata(archive: Any, expected_type: str) -> dict[str, Any]:
    try:
        metadata = json.loads(str(archive["metadata"].item()))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("router file has invalid or missing metadata") from exc
    if metadata.get("serialization_version") != _SERIALIZATION_VERSION:
        raise ValueError("router file uses an unsupported serialization version")
    if metadata.get("router_type") != expected_type:
        raise ValueError(f"router file does not contain a {expected_type} model")
    return metadata


class KNNR:
    """RACER KNN router over precomputed mDeBERTa embeddings.

    Each output column is the fraction of the 40 nearest design examples on
    which the corresponding model was correct. Similarity is cosine similarity.
    Stable sorting makes equal-similarity ties deterministic.
    """

    def __init__(
        self,
        *,
        k: int = KNN_NEIGHBORS,
        embedding_model: str = MDEBERTA_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        if k != KNN_NEIGHBORS:
            raise ValueError("exact RACER KNNR requires k=40")
        if embedding_model != MDEBERTA_MODEL_NAME:
            raise ValueError(
                "exact RACER KNNR requires microsoft/mdeberta-v3-base embeddings"
            )
        self.k = KNN_NEIGHBORS
        self.embedding_model = MDEBERTA_MODEL_NAME
        self.device = _normalize_device(device)
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    @property
    def n_models(self) -> int:
        self._require_fitted()
        assert self._labels is not None
        return int(self._labels.shape[1])

    @property
    def embedding_dim(self) -> int:
        self._require_fitted()
        assert self._embeddings is not None
        return int(self._embeddings.shape[1])

    def fit(self, embeddings: np.ndarray, correctness: np.ndarray) -> "KNNR":
        features = _as_feature_matrix(embeddings, "embeddings")
        labels = _as_label_matrix(correctness, features.shape[0], "correctness")
        if features.shape[0] < self.k:
            raise ValueError("exact RACER KNNR requires at least 40 fitted examples")
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("embeddings must not contain zero-norm rows")
        self._embeddings = features / norms[:, None]
        self._labels = labels.copy()
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self._embeddings is not None and self._labels is not None
        features = _as_feature_matrix(embeddings, "embeddings")
        if features.shape[1] != self._embeddings.shape[1]:
            raise ValueError("embedding feature count does not match the fitted model")
        norms = np.linalg.norm(features, axis=1)
        if np.any(norms == 0.0):
            raise ValueError("embeddings must not contain zero-norm rows")
        normalized = features / norms[:, None]

        xp = _array_module(self.device)
        similarities = xp.asarray(normalized) @ xp.asarray(self._embeddings).T
        similarities_cpu = _to_numpy(similarities)
        neighbor_indices = np.argsort(
            -similarities_cpu, axis=1, kind="stable"
        )[:, : self.k]
        return self._labels[neighbor_indices].mean(axis=1)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Alias matching common probabilistic router interfaces."""

        return self.predict(embeddings)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        assert self._embeddings is not None and self._labels is not None
        metadata = {
            "serialization_version": _SERIALIZATION_VERSION,
            "router_type": "KNNR",
            "k": self.k,
            "embedding_model": self.embedding_model,
            "device": self.device,
        }
        _write_npz(
            path,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            embeddings=self._embeddings,
            labels=self._labels,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: Optional[str] = None
    ) -> "KNNR":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = _read_metadata(archive, "KNNR")
            model = cls(
                k=int(metadata["k"]),
                embedding_model=str(metadata["embedding_model"]),
                device=metadata["device"] if device is None else device,
            )
            model._embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
            model._labels = np.asarray(archive["labels"], dtype=np.float64)
        if model._embeddings.ndim != 2 or model._labels.ndim != 2:
            raise ValueError("router file contains invalid fitted arrays")
        if model._embeddings.shape[0] != model._labels.shape[0]:
            raise ValueError("router file contains inconsistent fitted arrays")
        return model

    def _require_fitted(self) -> None:
        if self._embeddings is None or self._labels is None:
            raise RouterNotFittedError("KNNR must be fitted before this operation")


class MLPR:
    """Exact RACER one hidden layer multilabel router.

    The fixed contract is 256 ReLU hidden units, binary cross entropy with
    logits, batch size 32, learning rate 1e-4, and 100 epochs. Adam is used for
    optimization. The checkpoint with minimum validation loss is restored after
    training. A separate validation set is mandatory.
    """

    hidden_units = MLP_HIDDEN_UNITS
    batch_size = MLP_BATCH_SIZE
    learning_rate = MLP_LEARNING_RATE
    epochs = MLP_EPOCHS
    loss_name = "BCEWithLogitsLoss"

    def __init__(self, *, seed: int = 0, device: str = "cpu") -> None:
        self.seed = _validate_seed(seed)
        self.device = _normalize_device(device)
        self.input_dim: Optional[int] = None
        self.n_models: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_validation_loss: Optional[float] = None
        self.training_loss: list[float] = []
        self.validation_loss: list[float] = []
        self._weights: Optional[Tuple[np.ndarray, ...]] = None

    def fit(
        self,
        embeddings: np.ndarray,
        correctness: np.ndarray,
        *,
        validation_data: Tuple[np.ndarray, np.ndarray],
    ) -> "MLPR":
        if validation_data is None:
            raise ValueError("validation_data is required for checkpoint selection")
        features = _as_feature_matrix(embeddings, "embeddings")
        labels = _as_label_matrix(correctness, features.shape[0], "correctness")
        validation_features = _as_feature_matrix(validation_data[0], "validation embeddings")
        validation_labels = _as_label_matrix(
            validation_data[1], validation_features.shape[0], "validation correctness"
        )
        if validation_features.shape[1] != features.shape[1]:
            raise ValueError("training and validation feature counts must match")
        if validation_labels.shape[1] != labels.shape[1]:
            raise ValueError("training and validation model counts must match")

        self.input_dim = int(features.shape[1])
        self.n_models = int(labels.shape[1])
        rng = np.random.default_rng(self.seed)
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / (self.hidden_units + self.n_models))
        initial = (
            rng.standard_normal((self.input_dim, self.hidden_units)) * scale1,
            np.zeros(self.hidden_units, dtype=np.float64),
            rng.standard_normal((self.hidden_units, self.n_models)) * scale2,
            np.zeros(self.n_models, dtype=np.float64),
        )

        xp = _array_module(self.device)
        x_train = xp.asarray(features)
        y_train = xp.asarray(labels)
        x_validation = xp.asarray(validation_features)
        y_validation = xp.asarray(validation_labels)
        parameters = [xp.asarray(value) for value in initial]
        first_moments = [xp.zeros_like(value) for value in parameters]
        second_moments = [xp.zeros_like(value) for value in parameters]

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        step = 0
        best_loss = np.inf
        best_epoch = -1
        best_weights: Optional[Tuple[np.ndarray, ...]] = None
        self.training_loss = []
        self.validation_loss = []

        for epoch in range(self.epochs):
            permutation = rng.permutation(features.shape[0])
            epoch_weighted_loss = 0.0
            seen = 0
            for start in range(0, features.shape[0], self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                logits, hidden = self._forward(batch_x, parameters, xp)
                loss = self._bce_with_logits(logits, batch_y, xp)

                sigmoid = self._sigmoid(logits, xp)
                grad_logits = (sigmoid - batch_y) / batch_y.size
                grad_w2 = hidden.T @ grad_logits
                grad_b2 = xp.sum(grad_logits, axis=0)
                grad_hidden = (grad_logits @ parameters[2].T) * (hidden > 0.0)
                grad_w1 = batch_x.T @ grad_hidden
                grad_b1 = xp.sum(grad_hidden, axis=0)
                gradients = (grad_w1, grad_b1, grad_w2, grad_b2)

                step += 1
                for index, gradient in enumerate(gradients):
                    first_moments[index] = (
                        beta1 * first_moments[index] + (1.0 - beta1) * gradient
                    )
                    second_moments[index] = (
                        beta2 * second_moments[index]
                        + (1.0 - beta2) * gradient * gradient
                    )
                    corrected_first = first_moments[index] / (1.0 - beta1**step)
                    corrected_second = second_moments[index] / (1.0 - beta2**step)
                    parameters[index] -= self.learning_rate * corrected_first / (
                        xp.sqrt(corrected_second) + epsilon
                    )

                batch_count = int(batch_x.shape[0])
                epoch_weighted_loss += float(_to_numpy(loss)) * batch_count
                seen += batch_count

            training_loss = epoch_weighted_loss / seen
            validation_logits, _ = self._forward(x_validation, parameters, xp)
            validation_loss = float(
                _to_numpy(
                    self._bce_with_logits(validation_logits, y_validation, xp)
                )
            )
            self.training_loss.append(training_loss)
            self.validation_loss.append(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = tuple(
                    _to_numpy(value).astype(np.float64, copy=True)
                    for value in parameters
                )

        if best_weights is None:
            raise RuntimeError("MLPR training did not produce a validation checkpoint")
        self._weights = best_weights
        self.best_epoch = best_epoch
        self.best_validation_loss = best_loss
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.input_dim is not None and self._weights is not None
        features = _as_feature_matrix(embeddings, "embeddings")
        if features.shape[1] != self.input_dim:
            raise ValueError("embedding feature count does not match the fitted model")
        xp = _array_module(self.device)
        parameters = [xp.asarray(value) for value in self._weights]
        logits, _ = self._forward(xp.asarray(features), parameters, xp)
        probabilities = self._sigmoid(logits, xp)
        return _to_numpy(probabilities).astype(np.float64, copy=False)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Alias matching common probabilistic router interfaces."""

        return self.predict(embeddings)

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        assert self._weights is not None
        metadata = {
            "serialization_version": _SERIALIZATION_VERSION,
            "router_type": "MLPR",
            "seed": self.seed,
            "device": self.device,
            "input_dim": self.input_dim,
            "n_models": self.n_models,
            "hidden_units": self.hidden_units,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "loss_name": self.loss_name,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
        }
        _write_npz(
            path,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
            weight1=self._weights[0],
            bias1=self._weights[1],
            weight2=self._weights[2],
            bias2=self._weights[3],
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: Optional[str] = None
    ) -> "MLPR":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = _read_metadata(archive, "MLPR")
            required_contract = {
                "hidden_units": MLP_HIDDEN_UNITS,
                "batch_size": MLP_BATCH_SIZE,
                "learning_rate": MLP_LEARNING_RATE,
                "epochs": MLP_EPOCHS,
                "loss_name": "BCEWithLogitsLoss",
            }
            for key, expected in required_contract.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"router file violates the exact MLPR {key} contract")
            model = cls(
                seed=int(metadata["seed"]),
                device=metadata["device"] if device is None else device,
            )
            model.input_dim = int(metadata["input_dim"])
            model.n_models = int(metadata["n_models"])
            model.best_epoch = int(metadata["best_epoch"])
            model.best_validation_loss = float(metadata["best_validation_loss"])
            model.training_loss = [float(value) for value in metadata["training_loss"]]
            model.validation_loss = [
                float(value) for value in metadata["validation_loss"]
            ]
            model._weights = tuple(
                np.asarray(archive[name], dtype=np.float64)
                for name in ("weight1", "bias1", "weight2", "bias2")
            )
        model._validate_loaded_shapes()
        return model

    @staticmethod
    def _sigmoid(logits: Any, xp: Any) -> Any:
        positive = logits >= 0.0
        return xp.where(
            positive,
            1.0 / (1.0 + xp.exp(-logits)),
            xp.exp(logits) / (1.0 + xp.exp(logits)),
        )

    @staticmethod
    def _bce_with_logits(logits: Any, targets: Any, xp: Any) -> Any:
        losses = (
            xp.maximum(logits, 0.0)
            - logits * targets
            + xp.log1p(xp.exp(-xp.abs(logits)))
        )
        return xp.mean(losses)

    @staticmethod
    def _forward(
        features: Any, parameters: list[Any], xp: Any
    ) -> Tuple[Any, Any]:
        hidden = xp.maximum(features @ parameters[0] + parameters[1], 0.0)
        logits = hidden @ parameters[2] + parameters[3]
        return logits, hidden

    def _require_fitted(self) -> None:
        if self._weights is None:
            raise RouterNotFittedError("MLPR must be fitted before this operation")

    def _validate_loaded_shapes(self) -> None:
        assert self.input_dim is not None and self.n_models is not None
        assert self._weights is not None
        expected = (
            (self.input_dim, self.hidden_units),
            (self.hidden_units,),
            (self.hidden_units, self.n_models),
            (self.n_models,),
        )
        if tuple(value.shape for value in self._weights) != expected:
            raise ValueError("router file contains invalid MLPR parameter shapes")


class RouterDC:
    """Validated implementation adapter for RouterDC.

    This project does not contain the published RouterDC implementation or a
    validated configuration. Therefore this class never substitutes a logistic
    model or MLPR. Operational methods fail loudly unless the caller injects an
    implementation together with explicit validated configuration metadata.
    """

    def __init__(
        self,
        implementation: Optional[Any] = None,
        *,
        validated_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.implementation = implementation
        self.validated_config = (
            dict(validated_config) if validated_config is not None else None
        )
        if implementation is not None or validated_config is not None:
            self._validate_supplied()

    def fit(self, *args: Any, **kwargs: Any) -> "RouterDC":
        implementation = self._require_available()
        result = implementation.fit(*args, **kwargs)
        if result is not None and result is not implementation:
            raise RuntimeError("validated RouterDC fit returned an unexpected object")
        return self

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        implementation = self._require_available()
        result = np.asarray(implementation.predict(*args, **kwargs), dtype=np.float64)
        if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
            raise RuntimeError("validated RouterDC predict must return a non-empty matrix")
        if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
            raise RuntimeError(
                "validated RouterDC predict must return finite probabilities in [0, 1]"
            )
        return result

    def predict_proba(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.predict(*args, **kwargs)

    def save(self, path: str | Path) -> None:
        implementation = self._require_available()
        implementation.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        implementation_loader: Optional[Callable[[str | Path], Any]] = None,
        validated_config: Optional[Mapping[str, Any]] = None,
    ) -> "RouterDC":
        if implementation_loader is None or validated_config is None:
            raise RouterDCUnavailableError(
                "RouterDC.load requires a validated implementation loader and config"
            )
        implementation = implementation_loader(path)
        return cls(implementation, validated_config=validated_config)

    def _require_available(self) -> Any:
        if self.implementation is None or self.validated_config is None:
            raise RouterDCUnavailableError(
                "RouterDC is unavailable: supply the validated published "
                "implementation and an explicit validated config; no surrogate is used"
            )
        return self.implementation

    def _validate_supplied(self) -> None:
        if self.implementation is None or self.validated_config is None:
            raise RouterDCUnavailableError(
                "RouterDC requires both implementation and validated_config"
            )
        if self.validated_config.get("validated") is not True:
            raise RouterDCUnavailableError(
                "RouterDC config must explicitly set validated=True"
            )
        for key in ("name", "version"):
            value = self.validated_config.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RouterDCUnavailableError(
                    f"RouterDC validated_config requires a non-empty {key}"
                )
        missing = [
            method
            for method in ("fit", "predict", "save")
            if not callable(getattr(self.implementation, method, None))
        ]
        if missing:
            raise RouterDCUnavailableError(
                "RouterDC implementation is missing required methods: "
                + ", ".join(missing)
            )
