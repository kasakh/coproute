from __future__ import annotations

import numpy as np
import pytest
import torch

from components.routerdc_baseline import (
    RouterDCConfig,
    cosine_scores,
    sample_llm_contrastive_loss,
    sample_sample_contrastive_loss,
)


def test_cosine_scores_are_normalized_and_temperature_scaled() -> None:
    left = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    right = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    actual = cosine_scores(left, right, temperature=0.5)
    assert torch.allclose(actual, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_sample_llm_loss_rewards_positive_above_negative() -> None:
    quality = torch.tensor([[1.0, 0.8, 0.0, 0.0]])
    good = torch.tensor([[4.0, 3.0, -2.0, -3.0]], requires_grad=True)
    bad = torch.tensor([[-2.0, -3.0, 4.0, 3.0]], requires_grad=True)
    good_loss = sample_llm_contrastive_loss(good, quality, top_k=2, bottom_k=2)
    bad_loss = sample_llm_contrastive_loss(bad, quality, top_k=2, bottom_k=2)
    assert float(good_loss.detach()) < float(bad_loss.detach())
    good_loss.backward()
    assert good.grad is not None and torch.all(torch.isfinite(good.grad))


def test_sample_llm_loss_skips_rows_without_valid_contrast() -> None:
    logits = torch.ones((2, 3), requires_grad=True)
    quality = torch.ones((2, 3))
    loss = sample_llm_contrastive_loss(logits, quality, top_k=2, bottom_k=2)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None


def test_sample_sample_loss_is_seed_deterministic() -> None:
    hidden = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    clusters = torch.tensor([0, 0, 1, 1])
    first = torch.Generator().manual_seed(19)
    second = torch.Generator().manual_seed(19)
    left = sample_sample_contrastive_loss(
        hidden, clusters, out_group_count=2, generator=first
    )
    right = sample_sample_contrastive_loss(
        hidden, clusters, out_group_count=2, generator=second
    )
    assert torch.allclose(left, right)
    assert np.isfinite(float(left.detach()))


def test_routerdc_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="training_steps"):
        RouterDCConfig(training_steps=0)
    with pytest.raises(ValueError, match="temperature"):
        RouterDCConfig(temperature=0.0)
    with pytest.raises(ValueError, match="divisible"):
        RouterDCConfig(batch_size=64, micro_batch_size=30)
