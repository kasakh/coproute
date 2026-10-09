"""Faithful BaRP contextual-bandit baseline over frozen prompt embeddings.

The paper leaves the preference embedding size, decision-head hidden size, and
cost cap unspecified. This implementation makes those choices explicit in
``BaRPConfig`` and stores them in every checkpoint. It never loads or updates a
prompt encoder: callers provide frozen, precomputed prompt embeddings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


DEFAULT_PREFERENCE_DIM = 32
DEFAULT_HIDDEN_DIM = 256
DEFAULT_ENTROPY_BETA = 0.05
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 100
_SERIALIZATION_VERSION = 1


class BaRPNotFittedError(RuntimeError):
    """Raised when inference is attempted before fitting the router."""


@dataclass(frozen=True)
class BaRPConfig:
    """Fully recorded BaRP hyperparameters.

    ``tau`` has no paper default and must be supplied by the caller. The two
    architecture defaults are documented implementation choices rather than
    values claimed by the paper.
    """

    tau: float
    preference_dim: int = DEFAULT_PREFERENCE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    entropy_beta: float = DEFAULT_ENTROPY_BETA
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.tau, bool) or not isinstance(self.tau, (int, float)):
            raise TypeError("tau must be a positive finite number")
        if not torch.isfinite(torch.tensor(float(self.tau))) or self.tau <= 0:
            raise ValueError("tau must be a positive finite number")
        for name in ("preference_dim", "hidden_dim", "batch_size", "max_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_epochs > DEFAULT_MAX_EPOCHS:
            raise ValueError("max_epochs must not exceed the paper's 100 epoch limit")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be a non-negative integer")
        if self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("entropy_beta", "learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be a finite number")
        if self.entropy_beta < 0:
            raise ValueError("entropy_beta must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_metadata(self) -> dict[str, int | float]:
        """Return a JSON-compatible snapshot of every hyperparameter."""

        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "BaRPConfig":
        expected = {field.name for field in fields(cls)}
        if set(metadata) != expected:
            raise ValueError("checkpoint contains incomplete or unknown BaRP config fields")
        return cls(**dict(metadata))


class PreferenceMLP(nn.Module):
    """Map a two-objective preference to its learned embedding."""

    def __init__(self, preference_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, preference_dim), nn.ReLU())

    def forward(self, preferences: Tensor) -> Tensor:
        return self.network(preferences)


class BaRPPolicy(nn.Module):
    """Preference-conditioned categorical policy from the BaRP paper."""

    def __init__(
        self,
        embedding_dim: int,
        n_models: int,
        *,
        preference_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.preference_encoder = PreferenceMLP(preference_dim)
        self.decision_head = nn.Sequential(
            nn.Linear(embedding_dim + preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
        )

    def forward(self, embeddings: Tensor, preferences: Tensor) -> Tensor:
        preference_features = self.preference_encoder(preferences)
        return self.decision_head(torch.cat((embeddings, preference_features), dim=1))


def sample_simplex_preferences(
    count: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample uniformly from the two-objective 1-simplex."""

    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be a positive integer")
    if count <= 0:
        raise ValueError("count must be a positive integer")
    quality_weight = torch.rand(
        count, generator=generator, device=device, dtype=dtype
    )
    return torch.stack((quality_weight, 1.0 - quality_weight), dim=1)


def sample_categorical_actions(
    logits: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Sample one model per row from the softmax policy."""

    _validate_logits(logits)
    probabilities = torch.softmax(logits, dim=1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def chosen_action_feedback(
    qualities: Tensor,
    costs: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Reveal only the quality and cost associated with each chosen action."""

    if qualities.ndim != 2 or qualities.shape[0] == 0 or qualities.shape[1] == 0:
        raise ValueError("qualities must have shape (n_samples, n_models)")
    actions = _validate_actions(actions, qualities.shape[0], qualities.shape[1])
    if costs.ndim == 1:
        if costs.shape[0] != qualities.shape[1]:
            raise ValueError("one-dimensional costs must contain one value per model")
        selected_costs = costs.gather(0, actions)
    elif costs.ndim == 2:
        if costs.shape != qualities.shape:
            raise ValueError("two-dimensional costs must match qualities")
        selected_costs = costs.gather(1, actions[:, None]).squeeze(1)
    else:
        raise ValueError("costs must have shape (n_models,) or (n_samples, n_models)")
    selected_qualities = qualities.gather(1, actions[:, None]).squeeze(1)
    return selected_qualities, selected_costs


def bandit_reward(
    selected_qualities: Tensor,
    selected_costs: Tensor,
    preferences: Tensor,
    *,
    tau: float,
) -> Tensor:
    """Compute the BaRP selected-action reward with capped normalized cost."""

    if selected_qualities.ndim != 1 or selected_costs.ndim != 1:
        raise ValueError("selected qualities and costs must be one-dimensional")
    if selected_qualities.shape != selected_costs.shape:
        raise ValueError("selected qualities and costs must have the same shape")
    _validate_preferences(preferences, selected_qualities.shape[0])
    if tau <= 0:
        raise ValueError("tau must be positive")
    normalized_cost = torch.clamp(selected_costs / float(tau), min=0.0, max=1.0)
    return (
        preferences[:, 0] * selected_qualities
        - preferences[:, 1] * normalized_cost
    )


def reinforce_loss(
    logits: Tensor,
    actions: Tensor,
    rewards: Tensor,
    *,
    entropy_beta: float = DEFAULT_ENTROPY_BETA,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return mean entropy-regularized REINFORCE loss and detached diagnostics."""

    _validate_logits(logits)
    actions = _validate_actions(actions, logits.shape[0], logits.shape[1])
    if rewards.ndim != 1 or rewards.shape[0] != logits.shape[0]:
        raise ValueError("rewards must have one value per policy row")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must contain only finite values")
    if entropy_beta < 0:
        raise ValueError("entropy_beta must be non-negative")

    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    chosen_log_probabilities = log_probabilities.gather(
        1, actions[:, None]
    ).squeeze(1)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    baseline = rewards.mean().detach()
    advantages = rewards.detach() - baseline
    policy_loss = -(advantages * chosen_log_probabilities).mean()
    entropy_mean = entropy.mean()
    loss = policy_loss - float(entropy_beta) * entropy_mean
    diagnostics = {
        "baseline": baseline,
        "policy_loss": policy_loss.detach(),
        "entropy": entropy_mean.detach(),
    }
    return loss, diagnostics


class BaRPBaseline:
    """Train and evaluate one preference-conditioned BaRP routing policy."""

    def __init__(
        self,
        embedding_dim: int,
        n_models: int,
        *,
        config: BaRPConfig,
        model_ids: Sequence[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.n_models = _positive_int(n_models, "n_models")
        if not isinstance(config, BaRPConfig):
            raise TypeError("config must be an explicit BaRPConfig instance")
        self.config = config
        self.device = torch.device(device)
        if model_ids is None:
            self.model_ids = tuple(str(index) for index in range(self.n_models))
        else:
            if len(model_ids) != self.n_models:
                raise ValueError("model_ids must contain one unique id per model")
            self.model_ids = tuple(str(value) for value in model_ids)
            if len(set(self.model_ids)) != self.n_models:
                raise ValueError("model_ids must be unique")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.policy = BaRPPolicy(
                self.embedding_dim,
                self.n_models,
                preference_dim=self.config.preference_dim,
                hidden_dim=self.config.hidden_dim,
            )
        self.policy.to(self.device)
        self.training_history: list[dict[str, float | int]] = []
        self.is_fitted = False

    def fit(
        self,
        embeddings: Any,
        qualities: Any,
        costs: Any,
        *,
        epochs: int | None = None,
    ) -> "BaRPBaseline":
        """Fit with sampled preferences and chosen-action-only feedback."""

        features, quality_matrix, cost_values = self._training_tensors(
            embeddings, qualities, costs
        )
        epoch_count = self.config.max_epochs if epochs is None else _positive_int(
            epochs, "epochs"
        )
        if epoch_count > self.config.max_epochs:
            raise ValueError("epochs must not exceed config.max_epochs")

        generator = torch.Generator(device=self.device.type)
        generator.manual_seed(self.config.seed)
        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.training_history = []
        self.policy.train()

        for epoch in range(epoch_count):
            permutation = torch.randperm(
                features.shape[0], generator=generator, device=self.device
            )
            weighted_loss = 0.0
            weighted_reward = 0.0
            weighted_entropy = 0.0
            seen = 0
            for start in range(0, features.shape[0], self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size]
                batch_features = features.index_select(0, indices)
                batch_qualities = quality_matrix.index_select(0, indices)
                batch_costs = (
                    cost_values
                    if cost_values.ndim == 1
                    else cost_values.index_select(0, indices)
                )
                preferences = sample_simplex_preferences(
                    len(indices),
                    generator=generator,
                    device=self.device,
                    dtype=batch_features.dtype,
                )
                logits = self.policy(batch_features, preferences)
                actions = sample_categorical_actions(logits, generator=generator)
                selected_quality, selected_cost = chosen_action_feedback(
                    batch_qualities, batch_costs, actions
                )
                rewards = bandit_reward(
                    selected_quality,
                    selected_cost,
                    preferences,
                    tau=self.config.tau,
                )
                loss, diagnostics = reinforce_loss(
                    logits,
                    actions,
                    rewards,
                    entropy_beta=self.config.entropy_beta,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = int(indices.numel())
                weighted_loss += float(loss.detach().cpu()) * batch_size
                weighted_reward += float(rewards.mean().detach().cpu()) * batch_size
                weighted_entropy += float(diagnostics["entropy"].cpu()) * batch_size
                seen += batch_size

            self.training_history.append(
                {
                    "epoch": epoch,
                    "loss": weighted_loss / seen,
                    "mean_reward": weighted_reward / seen,
                    "entropy": weighted_entropy / seen,
                }
            )

        self.is_fitted = True
        return self

    def logits(self, embeddings: Any, preferences: Any | None = None) -> Tensor:
        """Return action logits, defaulting to quality-only preference (1, 0)."""

        self._require_fitted()
        features = self._inference_features(embeddings)
        preference_tensor = self._inference_preferences(
            preferences, features.shape[0], features.dtype
        )
        self.policy.eval()
        with torch.no_grad():
            return self.policy(features, preference_tensor).cpu()

    def predict_proba(self, embeddings: Any, preferences: Any | None = None) -> Tensor:
        """Return softmax policy probabilities for one preference per query."""

        return torch.softmax(self.logits(embeddings, preferences), dim=1)

    def predict(self, embeddings: Any, preferences: Any | None = None) -> Tensor:
        """Select deterministic argmax actions, quality-only by default."""

        return torch.argmax(self.logits(embeddings, preferences), dim=1)

    def predict_sweep(self, embeddings: Any, preferences: Any) -> Tensor:
        """Evaluate a native preference sweep without retraining.

        Returns an integer tensor with shape ``(n_preferences, n_queries)``.
        """

        preference_tensor = torch.as_tensor(preferences, dtype=torch.float32)
        if preference_tensor.ndim != 2 or preference_tensor.shape[1] != 2:
            raise ValueError("sweep preferences must have shape (n_preferences, 2)")
        _validate_preferences(preference_tensor, preference_tensor.shape[0])
        return torch.stack(
            [self.predict(embeddings, row) for row in preference_tensor], dim=0
        )

    def rank_models(
        self,
        embeddings: Any,
        preferences: Any | None = None,
        *,
        top_k: int | None = None,
    ) -> Tensor:
        """Return stable descending logit orderings for set-routing adaptation."""

        if top_k is None:
            count = self.n_models
        else:
            count = _positive_int(top_k, "top_k")
            if count > self.n_models:
                raise ValueError("top_k must not exceed n_models")
        return torch.argsort(
            self.logits(embeddings, preferences), dim=1, descending=True, stable=True
        )[:, :count]

    def save(self, path: str | Path) -> None:
        """Save parameters plus complete architecture, training, and ID metadata."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "serialization_version": _SERIALIZATION_VERSION,
                "baseline_type": "BaRPBaseline",
                "embedding_dim": self.embedding_dim,
                "n_models": self.n_models,
                "model_ids": list(self.model_ids),
                "config": self.config.to_metadata(),
                "is_fitted": self.is_fitted,
                "training_history": self.training_history,
                "architecture": {
                    "prompt_features": "frozen_precomputed",
                    "preference_encoder": "Linear(2, preference_dim)+ReLU",
                    "decision_head": "Linear+ReLU+Linear",
                    "policy": "softmax_categorical",
                },
                "training": {
                    "optimizer": "Adam",
                    "baseline": "batch_mean_reward",
                    "feedback": "chosen_action_only",
                },
            },
            "state_dict": self.policy.state_dict(),
        }
        torch.save(payload, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "BaRPBaseline":
        """Restore a checkpoint and validate its recorded contract."""

        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid BaRP checkpoint: {exc}") from exc
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("metadata"), Mapping
        ):
            raise ValueError("checkpoint has invalid or missing metadata")
        metadata = payload["metadata"]
        if metadata.get("serialization_version") != _SERIALIZATION_VERSION:
            raise ValueError("checkpoint uses an unsupported serialization version")
        if metadata.get("baseline_type") != "BaRPBaseline":
            raise ValueError("checkpoint does not contain a BaRPBaseline")
        config_value = metadata.get("config")
        if not isinstance(config_value, Mapping):
            raise ValueError("checkpoint has invalid BaRP config metadata")
        config = BaRPConfig.from_metadata(config_value)
        model = cls(
            int(metadata["embedding_dim"]),
            int(metadata["n_models"]),
            config=config,
            model_ids=metadata["model_ids"],
            device=device,
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint has invalid or missing policy parameters")
        try:
            model.policy.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"checkpoint policy parameters are invalid: {exc}") from exc
        history = metadata.get("training_history")
        if not isinstance(history, list):
            raise ValueError("checkpoint has invalid training history")
        model.training_history = history
        model.is_fitted = bool(metadata.get("is_fitted", False))
        return model

    def _training_tensors(
        self, embeddings: Any, qualities: Any, costs: Any
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = self._inference_features(embeddings)
        quality_matrix = torch.as_tensor(
            qualities, dtype=features.dtype, device=self.device
        ).detach()
        if quality_matrix.shape != (features.shape[0], self.n_models):
            raise ValueError("qualities must have shape (n_samples, n_models)")
        if not torch.isfinite(quality_matrix).all():
            raise ValueError("qualities must contain only finite values")
        if torch.any((quality_matrix < 0) | (quality_matrix > 1)):
            raise ValueError("qualities must lie within [0, 1]")

        cost_values = torch.as_tensor(
            costs, dtype=features.dtype, device=self.device
        ).detach()
        valid_cost_shape = cost_values.shape in {
            (self.n_models,),
            (features.shape[0], self.n_models),
        }
        if not valid_cost_shape:
            raise ValueError(
                "costs must have shape (n_models,) or (n_samples, n_models)"
            )
        if not torch.isfinite(cost_values).all() or torch.any(cost_values < 0):
            raise ValueError("costs must contain finite non-negative values")
        return features, quality_matrix, cost_values

    def _inference_features(self, embeddings: Any) -> Tensor:
        features = torch.as_tensor(
            embeddings, dtype=torch.float32, device=self.device
        ).detach()
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("embeddings must be a nonempty two-dimensional tensor")
        if features.shape[1] != self.embedding_dim:
            raise ValueError("embedding feature count does not match embedding_dim")
        if not torch.isfinite(features).all():
            raise ValueError("embeddings must contain only finite values")
        return features

    def _inference_preferences(
        self, preferences: Any | None, n_rows: int, dtype: torch.dtype
    ) -> Tensor:
        if preferences is None:
            values = torch.tensor([1.0, 0.0], device=self.device, dtype=dtype)
        else:
            values = torch.as_tensor(
                preferences, device=self.device, dtype=dtype
            ).detach()
        if values.ndim == 1:
            if values.shape != (2,):
                raise ValueError("preference must contain quality and cost weights")
            values = values.unsqueeze(0).expand(n_rows, -1)
        elif values.ndim == 2 and values.shape == (1, 2):
            values = values.expand(n_rows, -1)
        elif values.ndim != 2 or values.shape != (n_rows, 2):
            raise ValueError("preferences must have shape (2,) or (n_samples, 2)")
        _validate_preferences(values, n_rows)
        return values

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise BaRPNotFittedError("BaRPBaseline must be fitted before inference")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_logits(logits: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be a nonempty two-dimensional tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")


def _validate_actions(actions: Tensor, n_rows: int, n_models: int) -> Tensor:
    if actions.ndim != 1 or actions.shape[0] != n_rows:
        raise ValueError("actions must contain one index per row")
    if actions.dtype == torch.bool or actions.dtype.is_floating_point:
        raise TypeError("actions must contain integer indices")
    actions = actions.to(dtype=torch.long)
    if torch.any((actions < 0) | (actions >= n_models)):
        raise ValueError("actions contain an out-of-range model index")
    return actions


def _validate_preferences(preferences: Tensor, n_rows: int) -> None:
    if preferences.ndim != 2 or preferences.shape != (n_rows, 2):
        raise ValueError("preferences must have shape (n_samples, 2)")
    if not torch.isfinite(preferences).all():
        raise ValueError("preferences must contain only finite values")
    if torch.any(preferences < 0):
        raise ValueError("preference weights must be non-negative")
    sums = preferences.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("preference weights must lie on the 1-simplex")
