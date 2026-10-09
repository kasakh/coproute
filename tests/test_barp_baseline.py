"""Focused synthetic tests for the faithful BaRP baseline core."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.barp_baseline import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENTROPY_BETA,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PREFERENCE_DIM,
    BaRPBaseline,
    BaRPConfig,
    bandit_reward,
    chosen_action_feedback,
    reinforce_loss,
    sample_simplex_preferences,
)


def _configured_toy_router() -> BaRPBaseline:
    config = BaRPConfig(
        tau=1.0,
        preference_dim=2,
        hidden_dim=2,
        max_epochs=1,
        seed=7,
    )
    router = BaRPBaseline(
        1,
        2,
        config=config,
        model_ids=("quality-model", "cheap-model"),
    )
    with torch.no_grad():
        preference_layer = router.policy.preference_encoder.network[0]
        preference_layer.weight.copy_(torch.eye(2))
        preference_layer.bias.zero_()

        hidden_layer = router.policy.decision_head[0]
        hidden_layer.weight.zero_()
        hidden_layer.bias.zero_()
        hidden_layer.weight[0, 1] = 1.0
        hidden_layer.weight[1, 2] = 1.0

        output_layer = router.policy.decision_head[2]
        output_layer.weight.copy_(torch.eye(2))
        output_layer.bias.zero_()
    router.is_fitted = True
    return router


def test_config_records_paper_contract_and_explicit_architecture_defaults() -> None:
    config = BaRPConfig(tau=0.25)

    assert config.preference_dim == DEFAULT_PREFERENCE_DIM == 32
    assert config.hidden_dim == DEFAULT_HIDDEN_DIM == 256
    assert config.entropy_beta == DEFAULT_ENTROPY_BETA == 0.05
    assert config.learning_rate == DEFAULT_LEARNING_RATE == 1e-4
    assert config.batch_size == DEFAULT_BATCH_SIZE == 32
    assert config.max_epochs == DEFAULT_MAX_EPOCHS == 100
    assert config.to_metadata()["tau"] == 0.25

    with pytest.raises(TypeError, match="explicit BaRPConfig"):
        BaRPBaseline(3, 2, config=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau"):
        BaRPConfig(tau=0.0)
    with pytest.raises(ValueError, match="100 epoch"):
        BaRPConfig(tau=1.0, max_epochs=101)


def test_reward_matches_quality_minus_capped_normalized_cost() -> None:
    qualities = torch.tensor([0.8, 0.4, 1.0])
    costs = torch.tensor([0.1, 0.5, 0.0])
    preferences = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    rewards = bandit_reward(qualities, costs, preferences, tau=0.5)

    torch.testing.assert_close(rewards, torch.tensor([0.8, -0.3, 0.0]))


def test_cost_penalty_is_capped_at_one() -> None:
    qualities = torch.zeros(3)
    costs = torch.tensor([1.0, 2.0, 100.0])
    preferences = torch.tensor([[0.0, 1.0]]).repeat(3, 1)

    rewards = bandit_reward(qualities, costs, preferences, tau=2.0)

    torch.testing.assert_close(rewards, torch.tensor([-0.5, -1.0, -1.0]))


def test_simplex_sampling_is_uniform_reproducible_and_sums_to_one() -> None:
    first_generator = torch.Generator().manual_seed(19)
    second_generator = torch.Generator().manual_seed(19)

    first = sample_simplex_preferences(20_000, generator=first_generator)
    second = sample_simplex_preferences(20_000, generator=second_generator)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.sum(dim=1), torch.ones(20_000))
    assert torch.all((first >= 0.0) & (first <= 1.0))
    assert abs(float(first[:, 0].mean()) - 0.5) < 0.01
    assert abs(float(first[:, 0].var(unbiased=False)) - 1.0 / 12.0) < 0.003


def test_only_selected_action_feedback_can_affect_reward_or_loss() -> None:
    actions = torch.tensor([0, 1])
    preferences = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    logits = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])

    qualities_a = torch.tensor([[0.9, 0.0], [0.0, 0.8]])
    qualities_b = torch.tensor([[0.9, 1.0], [1.0, 0.8]])
    costs_a = torch.tensor([[0.2, 99.0], [99.0, 0.1]])
    costs_b = torch.tensor([[0.2, 0.0], [0.0, 0.1]])

    selected_a = chosen_action_feedback(qualities_a, costs_a, actions)
    selected_b = chosen_action_feedback(qualities_b, costs_b, actions)
    torch.testing.assert_close(selected_a[0], selected_b[0])
    torch.testing.assert_close(selected_a[1], selected_b[1])

    rewards_a = bandit_reward(*selected_a, preferences, tau=1.0)
    rewards_b = bandit_reward(*selected_b, preferences, tau=1.0)
    loss_a, _ = reinforce_loss(logits, actions, rewards_a)
    loss_b, _ = reinforce_loss(logits, actions, rewards_b)

    torch.testing.assert_close(rewards_a, rewards_b)
    torch.testing.assert_close(loss_a, loss_b)


def test_reinforce_gradient_has_correct_sign_for_positive_and_negative_advantage() -> None:
    logits = torch.zeros((2, 2), requires_grad=True)
    actions = torch.tensor([0, 1])
    rewards = torch.tensor([1.0, 0.0])

    loss, diagnostics = reinforce_loss(
        logits, actions, rewards, entropy_beta=0.0
    )
    loss.backward()

    assert diagnostics["baseline"].item() == 0.5
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0.0
    assert logits.grad[0, 1] > 0.0
    assert logits.grad[1, 1] > 0.0
    assert logits.grad[1, 0] < 0.0


def test_entropy_term_favors_a_high_entropy_policy() -> None:
    actions = torch.tensor([0, 0])
    equal_rewards = torch.ones(2)
    uniform_logits = torch.zeros((2, 2))
    peaked_logits = torch.tensor([[8.0, -8.0], [8.0, -8.0]])

    uniform_loss, uniform_diagnostics = reinforce_loss(
        uniform_logits, actions, equal_rewards, entropy_beta=0.05
    )
    peaked_loss, peaked_diagnostics = reinforce_loss(
        peaked_logits, actions, equal_rewards, entropy_beta=0.05
    )

    assert uniform_diagnostics["policy_loss"].item() == 0.0
    assert peaked_diagnostics["policy_loss"].item() == 0.0
    assert uniform_diagnostics["entropy"] > peaked_diagnostics["entropy"]
    assert uniform_loss < peaked_loss


def test_training_is_deterministic_for_fixed_seed_and_inputs() -> None:
    generator = torch.Generator().manual_seed(123)
    embeddings = torch.randn((64, 4), generator=generator)
    qualities = torch.randint(0, 2, (64, 3), generator=generator).float()
    costs = torch.tensor([0.1, 0.4, 0.9])
    config = BaRPConfig(tau=0.5, max_epochs=3, seed=31)

    first = BaRPBaseline(4, 3, config=config).fit(embeddings, qualities, costs)
    second = BaRPBaseline(4, 3, config=config).fit(embeddings, qualities, costs)

    for name, first_value in first.policy.state_dict().items():
        torch.testing.assert_close(
            first_value,
            second.policy.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )
    assert first.training_history == second.training_history


def test_inference_changes_with_preference_on_toy_policy_and_supports_sweep() -> None:
    router = _configured_toy_router()
    embeddings = torch.tensor([[0.0], [4.0]])

    quality_only = router.predict(embeddings)
    cost_only = router.predict(embeddings, [0.0, 1.0])
    sweep = router.predict_sweep(
        embeddings,
        [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75], [0.0, 1.0]],
    )

    torch.testing.assert_close(quality_only, torch.tensor([0, 0]))
    torch.testing.assert_close(cost_only, torch.tensor([1, 1]))
    torch.testing.assert_close(
        sweep,
        torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]),
    )


def test_top_k_ordering_uses_descending_logits_with_stable_ties() -> None:
    router = _configured_toy_router()
    with torch.no_grad():
        router.policy.decision_head[2].weight.zero_()
        router.policy.decision_head[2].bias.copy_(torch.tensor([0.4, 0.4]))

    ordering = router.rank_models([[0.0]], [0.5, 0.5], top_k=2)

    torch.testing.assert_close(ordering, torch.tensor([[0, 1]]))


def test_save_load_preserves_policy_predictions_and_full_metadata(tmp_path: Path) -> None:
    router = _configured_toy_router()
    router.training_history = [
        {"epoch": 0, "loss": -0.01, "mean_reward": 0.4, "entropy": 0.6}
    ]
    checkpoint = tmp_path / "barp.pt"

    router.save(checkpoint)
    restored = BaRPBaseline.load(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    embeddings = torch.tensor([[0.0], [1.0]])
    preferences = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(
        restored.predict_proba(embeddings, preferences),
        router.predict_proba(embeddings, preferences),
        rtol=0.0,
        atol=0.0,
    )
    assert restored.config == router.config
    assert restored.model_ids == router.model_ids
    assert restored.training_history == router.training_history
    assert payload["metadata"]["config"] == router.config.to_metadata()
    assert payload["metadata"]["architecture"]["prompt_features"] == "frozen_precomputed"
    assert payload["metadata"]["training"] == {
        "optimizer": "Adam",
        "baseline": "batch_mean_reward",
        "feedback": "chosen_action_only",
    }
