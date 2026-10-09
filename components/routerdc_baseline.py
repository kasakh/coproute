"""Clean-room RouterDC reconstruction from the published equations.

The implementation follows RouterDC's sample-LLM and sample-sample contrastive
objectives. It does not copy the unlicensed reference repository. The encoder
is mDeBERTa-v3-base, model embeddings are learned, and source validation selects
the checkpoint. Clusters are supplied by the caller so their provenance stays
outside the model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoConfig, DebertaV2Model, DebertaV2Tokenizer


OFFICIAL_REPOSITORY = "https://github.com/shuhao02/RouterDC"
OFFICIAL_COMMIT = "168c026d7a6cd7029cb856a57d935bedfb746e7a"
PAPER_ARXIV = "2409.19886"


@dataclass(frozen=True)
class RouterDCConfig:
    backbone_name: str = "microsoft/mdeberta-v3-base"
    hidden_size: int = 768
    top_k: int = 3
    bottom_k: int = 3
    out_group_count: int = 3
    cluster_loss_weight: float = 1.0
    temperature: float = 1.0
    n_clusters: int = 5
    batch_size: int = 64
    micro_batch_size: int = 32
    training_steps: int = 1000
    evaluation_steps: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_length: int = 512
    model_embedding_std: float = 0.78
    seed: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size", "top_k", "bottom_k", "out_group_count", "n_clusters",
            "batch_size", "micro_batch_size", "training_steps", "evaluation_steps", "max_length",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError("batch size must be divisible by microbatch size")
        if self.temperature <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature and learning rate must be positive")
        if self.weight_decay < 0 or self.cluster_loss_weight < 0:
            raise ValueError("loss weight and weight decay must be non-negative")
        if self.model_embedding_std <= 0:
            raise ValueError("model embedding standard deviation must be positive")


def cosine_scores(left: Tensor, right: Tensor, temperature: float = 1.0) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("cosine inputs must be compatible matrices")
    return nn.functional.normalize(left, dim=1) @ nn.functional.normalize(right, dim=1).T / temperature


def sample_llm_contrastive_loss(
    logits: Tensor,
    quality: Tensor,
    *,
    top_k: int,
    bottom_k: int,
) -> Tensor:
    """Equation 4 with invalid positives and nonnegative negatives masked."""

    if logits.shape != quality.shape or logits.ndim != 2:
        raise ValueError("logits and quality must be equal two-dimensional matrices")
    n_models = logits.shape[1]
    if top_k > n_models or bottom_k > n_models:
        raise ValueError("contrastive set size exceeds model count")
    descending = torch.argsort(quality, dim=1, descending=True, stable=True)
    ascending = torch.argsort(quality, dim=1, descending=False, stable=True)
    positive_indices = descending[:, :top_k]
    negative_indices = ascending[:, :bottom_k]
    negative_quality = torch.gather(quality, 1, negative_indices)
    negative_logits = torch.gather(logits, 1, negative_indices)
    negative_logits = negative_logits.masked_fill(negative_quality >= 0.5, float("-inf"))
    valid_negative = torch.any(torch.isfinite(negative_logits), dim=1)
    terms: list[Tensor] = []
    for position in range(top_k):
        index = positive_indices[:, position : position + 1]
        positive_quality = torch.gather(quality, 1, index).squeeze(1)
        positive_logit = torch.gather(logits, 1, index)
        candidates = torch.cat((positive_logit, negative_logits), dim=1)
        loss = -torch.log_softmax(candidates, dim=1)[:, 0]
        valid = (positive_quality > 0) & valid_negative & torch.isfinite(loss)
        if torch.any(valid):
            terms.append(loss[valid].mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).sum()


def sample_sample_contrastive_loss(
    hidden: Tensor,
    cluster_ids: Tensor,
    *,
    out_group_count: int,
    generator: torch.Generator,
    temperature: float = 1.0,
) -> Tensor:
    """Equation 5 using deterministic seeded in-batch sampling."""

    if hidden.ndim != 2 or cluster_ids.ndim != 1 or len(hidden) != len(cluster_ids):
        raise ValueError("hidden states and cluster IDs do not align")
    similarity = cosine_scores(hidden, hidden, temperature)
    losses: list[Tensor] = []
    for row in range(len(hidden)):
        same = torch.nonzero(cluster_ids == cluster_ids[row], as_tuple=False).flatten()
        other = torch.nonzero(cluster_ids != cluster_ids[row], as_tuple=False).flatten()
        if len(same) == 0 or len(other) < out_group_count:
            continue
        positive = same[torch.randint(len(same), (1,), generator=generator, device=same.device)]
        order = torch.randperm(len(other), generator=generator, device=other.device)[:out_group_count]
        negatives = other[order]
        indices = torch.cat((positive, negatives))
        losses.append(-torch.log_softmax(similarity[row, indices], dim=0)[0])
    if not losses:
        return hidden.sum() * 0.0
    return torch.stack(losses).mean()


class RouterDCBaseline(nn.Module):
    """RouterDC with target-held-out source validation checkpoint selection."""

    def __init__(
        self,
        backbone_path: str | Path,
        model_ids: Sequence[str],
        *,
        config: RouterDCConfig = RouterDCConfig(),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        ids = tuple(str(value) for value in model_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("model IDs must be non-empty and unique")
        if config.top_k > len(ids) or config.bottom_k > len(ids):
            raise ValueError("RouterDC contrastive sets exceed the model pool")
        self.config = config
        self.model_ids = ids
        self.backbone_path = str(Path(backbone_path))
        self.device_name = str(device)
        auto_config = AutoConfig.from_pretrained(self.backbone_path, local_files_only=True)
        if int(auto_config.hidden_size) != config.hidden_size:
            raise ValueError("backbone hidden size differs from RouterDC config")
        self.tokenizer = DebertaV2Tokenizer(
            vocab_file=str(Path(self.backbone_path) / "spm.model")
        )
        self.backbone = DebertaV2Model.from_pretrained(self.backbone_path, local_files_only=True)
        self.model_embeddings = nn.Embedding(len(ids), config.hidden_size)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        with torch.no_grad():
            self.model_embeddings.weight.copy_(
                torch.randn(self.model_embeddings.weight.shape, generator=generator)
                * config.model_embedding_std
            )
        self.training_history: list[dict[str, float | int]] = []
        self.best_step: int | None = None
        self.best_validation_quality: float | None = None
        self.to(torch.device(device))

    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(self, prompts: Sequence[str]) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            [str(value) for value in prompts],
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in encoded.items()}

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = output.last_hidden_state[:, 0, :]
        logits = cosine_scores(hidden, self.model_embeddings.weight, self.config.temperature)
        return logits, hidden

    def _validation_quality(self, tokens: dict[str, Tensor], quality: Tensor) -> float:
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(quality), self.config.micro_batch_size):
                stop = min(start + self.config.micro_batch_size, len(quality))
                batch = {name: value[start:stop] for name, value in tokens.items()}
                logits, _ = self(**batch)
                chosen = torch.argmax(logits, dim=1)
                total += float(quality[start:stop].gather(1, chosen[:, None]).sum().cpu())
                count += stop - start
        self.train()
        return total / count

    def fit(
        self,
        prompts: Sequence[str],
        quality: np.ndarray,
        cluster_ids: Sequence[int],
        *,
        validation_prompts: Sequence[str],
        validation_quality: np.ndarray,
    ) -> "RouterDCBaseline":
        scores = np.asarray(quality, dtype=np.float32)
        validation_scores = np.asarray(validation_quality, dtype=np.float32)
        clusters = np.asarray(cluster_ids, dtype=np.int64)
        if scores.shape != (len(prompts), self.n_models):
            raise ValueError("training quality shape differs from prompts and model pool")
        if validation_scores.shape != (len(validation_prompts), self.n_models):
            raise ValueError("validation quality shape differs from prompts and model pool")
        if clusters.shape != (len(prompts),):
            raise ValueError("cluster IDs differ from training prompt count")
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(validation_scores)):
            raise ValueError("RouterDC quality contains non-finite values")
        train_tokens = self.tokenize(prompts)
        validation_tokens = self.tokenize(validation_prompts)
        quality_tensor = torch.as_tensor(scores, device=self.device)
        validation_tensor = torch.as_tensor(validation_scores, device=self.device)
        cluster_tensor = torch.as_tensor(clusters, device=self.device)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(self.config.seed)
        device_generator = torch.Generator(device=self.device.type)
        device_generator.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        order = torch.empty(0, dtype=torch.long)
        cursor = 0
        best_state: dict[str, Tensor] | None = None
        best_quality = -math.inf
        best_step: int | None = None
        self.training_history = []
        self.train()
        for step in range(1, self.config.training_steps + 1):
            if cursor + self.config.batch_size > len(order):
                order = torch.randperm(len(scores), generator=cpu_generator)
                cursor = 0
            rows_cpu = order[cursor : cursor + self.config.batch_size]
            cursor += self.config.batch_size
            optimizer.zero_grad(set_to_none=True)
            micro_count = self.config.batch_size // self.config.micro_batch_size
            loss_values: list[Tensor] = []
            llm_values: list[Tensor] = []
            cluster_values: list[Tensor] = []
            for micro_start in range(0, len(rows_cpu), self.config.micro_batch_size):
                micro_cpu = rows_cpu[micro_start : micro_start + self.config.micro_batch_size]
                rows = micro_cpu.to(self.device)
                batch = {name: value.index_select(0, rows) for name, value in train_tokens.items()}
                logits, hidden = self(**batch)
                llm_loss = sample_llm_contrastive_loss(
                    logits,
                    quality_tensor.index_select(0, rows),
                    top_k=self.config.top_k,
                    bottom_k=self.config.bottom_k,
                )
                cluster_loss = sample_sample_contrastive_loss(
                    hidden,
                    cluster_tensor.index_select(0, rows),
                    out_group_count=self.config.out_group_count,
                    generator=device_generator,
                    temperature=self.config.temperature,
                )
                loss = llm_loss + self.config.cluster_loss_weight * cluster_loss
                if not torch.isfinite(loss):
                    raise ValueError("RouterDC training produced non-finite loss")
                (loss / micro_count).backward()
                loss_values.append(loss.detach())
                llm_values.append(llm_loss.detach())
                cluster_values.append(cluster_loss.detach())
            optimizer.step()
            mean_loss = torch.stack(loss_values).mean()
            mean_llm_loss = torch.stack(llm_values).mean()
            mean_cluster_loss = torch.stack(cluster_values).mean()
            if step % self.config.evaluation_steps == 0 or step == self.config.training_steps:
                value = self._validation_quality(validation_tokens, validation_tensor)
                row = {
                    "step": step,
                    "loss": float(mean_loss.cpu()),
                    "sample_llm_loss": float(mean_llm_loss.cpu()),
                    "sample_sample_loss": float(mean_cluster_loss.cpu()),
                    "validation_selected_quality": value,
                }
                self.training_history.append(row)
                if value > best_quality:
                    best_quality = value
                    best_step = step
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.state_dict().items()
                    }
        if best_state is None or best_step is None:
            raise RuntimeError("RouterDC training did not produce a validation checkpoint")
        self.load_state_dict(best_state)
        self.best_step = best_step
        self.best_validation_quality = best_quality
        self.eval()
        return self

    def predict_logits(
        self,
        prompts: Sequence[str],
        *,
        model_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        indices = np.arange(self.n_models) if model_indices is None else np.asarray(model_indices, dtype=int)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= self.n_models):
            raise ValueError("model indices are invalid")
        tokens = self.tokenize(prompts)
        selected_embeddings = self.model_embeddings.weight[torch.as_tensor(indices, device=self.device)]
        values: list[np.ndarray] = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(prompts), self.config.micro_batch_size):
                stop = min(start + self.config.micro_batch_size, len(prompts))
                batch = {name: value[start:stop] for name, value in tokens.items()}
                output = self.backbone(**batch)
                hidden = output.last_hidden_state[:, 0, :]
                logits = cosine_scores(hidden, selected_embeddings, self.config.temperature)
                values.append(logits.detach().cpu().numpy())
        return np.concatenate(values, axis=0).astype(np.float64, copy=False)

    def predict_proba(
        self,
        prompts: Sequence[str],
        *,
        model_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        logits = self.predict_logits(prompts, model_indices=model_indices)
        shifted = logits - logits.max(axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "RouterDCBaseline",
            "config": asdict(self.config),
            "model_ids": list(self.model_ids),
            "backbone_source": self.config.backbone_name,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "paper_arxiv": PAPER_ARXIV,
            "reconstruction": "clean_room_full_encoder_dual_contrastive",
            "best_step": self.best_step,
            "best_validation_quality": self.best_validation_quality,
            "training_history": self.training_history,
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backbone_path: str | Path,
        device: str = "cpu",
    ) -> "RouterDCBaseline":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("type") != "RouterDCBaseline":
            raise ValueError("checkpoint is not a RouterDCBaseline")
        config = RouterDCConfig(**payload["config"])
        model = cls(backbone_path, payload["model_ids"], config=config, device=device)
        model.load_state_dict(payload["state_dict"])
        model.best_step = payload["best_step"]
        model.best_validation_quality = payload["best_validation_quality"]
        model.training_history = list(payload["training_history"])
        model.eval()
        return model

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "model_ids": list(self.model_ids),
            "n_models": self.n_models,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "paper_arxiv": PAPER_ARXIV,
            "reconstruction": "clean_room_full_encoder_dual_contrastive",
            "best_step": self.best_step,
            "best_validation_quality": self.best_validation_quality,
        }
