"""Focused mathematical and reproducibility tests for the FlexRouter core."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from components.flexrouter_baseline import (  # noqa: E402
    AUXILIARY_BCE_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    LATENT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_SUBSET_SIZE,
    FlexRouter,
    FlexRouterTrainingConfig,
    auxiliary_bce_loss,
    cosine_gram,
    coverage_loss,
    dpp_kernel,
    greedy_dpp_map,
    greedy_dpp_map_low_rank,
    low_rank_logdet_i_plus_l,
)


DTYPE = torch.float64


def _all_subsets(n_items: int):
    for size in range(n_items + 1):
        yield from combinations(range(n_items), size)


def _determinant_of_subset(kernel: torch.Tensor, subset: tuple[int, ...]) -> torch.Tensor:
    if not subset:
        return torch.ones((), dtype=kernel.dtype)
    indices = torch.tensor(subset)
    return torch.linalg.det(kernel[indices][:, indices])


def _tiny_model_embeddings() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.6, 0.8],
            [0.2, -0.4, 0.9],
        ],
        dtype=DTYPE,
    )


def test_published_configuration_defaults_are_pinned() -> None:
    config = FlexRouterTrainingConfig()
    assert LATENT_DIM == 128
    assert AUXILIARY_BCE_WEIGHT == 1.0
    assert LEARNING_RATE == 1e-3
    assert MAX_EPOCHS == 100
    assert EARLY_STOPPING_PATIENCE == 15
    assert MAX_SUBSET_SIZE == 10
    assert config.learning_rate == 1e-3
    assert config.max_epochs == 100
    assert config.patience == 15
    assert config.auxiliary_bce_weight == 1.0
    assert config.validation_max_size == 10


def test_cosine_gram_and_scaled_kernel_are_psd() -> None:
    embeddings = _tiny_model_embeddings().requires_grad_(True)
    qualities = torch.tensor([0.9, 0.7, 0.4, 0.8], dtype=DTYPE)

    gram = cosine_gram(embeddings)
    kernel = dpp_kernel(qualities, embeddings)

    torch.testing.assert_close(torch.diagonal(gram), torch.ones(4, dtype=DTYPE))
    torch.testing.assert_close(gram, gram.T)
    assert torch.linalg.eigvalsh(gram).min() >= -1e-12
    assert torch.linalg.eigvalsh(kernel).min() >= -1e-12
    kernel.sum().backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_low_rank_determinant_lemma_matches_full_kernel() -> None:
    embeddings = _tiny_model_embeddings()
    qualities = torch.tensor(
        [[0.8, 0.3, 0.6, 0.9], [0.2, 0.7, 0.4, 0.5]], dtype=DTYPE
    )
    masks = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)

    low_rank = low_rank_logdet_i_plus_l(
        qualities, embeddings, inclusion_mask=masks
    )
    expected = []
    for quality, mask in zip(qualities, masks):
        subset = torch.nonzero(mask, as_tuple=False).flatten()
        full_kernel = dpp_kernel(quality, embeddings)
        restricted = full_kernel[subset][:, subset]
        expected.append(
            torch.logdet(torch.eye(len(subset), dtype=DTYPE) + restricted)
        )

    torch.testing.assert_close(low_rank, torch.stack(expected), atol=1e-12, rtol=1e-12)


def test_coverage_loss_matches_explicit_subset_enumeration() -> None:
    embeddings = _tiny_model_embeddings()
    qualities = torch.tensor([0.82, 0.61, 0.47, 0.73], dtype=DTYPE)
    correctness = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=DTYPE)
    kernel = dpp_kernel(qualities, embeddings)
    denominator = torch.linalg.det(torch.eye(4, dtype=DTYPE) + kernel)

    success_mass = torch.zeros((), dtype=DTYPE)
    for subset in _all_subsets(4):
        if any(correctness[index] == 1 for index in subset):
            success_mass += _determinant_of_subset(kernel, subset)
    expected = -torch.log(success_mass / denominator)

    actual = coverage_loss(qualities, correctness, embeddings)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_coverage_loss_is_finite_for_all_failure_edge_case() -> None:
    embeddings = _tiny_model_embeddings()
    qualities = torch.tensor([0.8, 0.6, 0.4, 0.7], dtype=DTYPE)
    loss = coverage_loss(qualities, torch.zeros(4, dtype=DTYPE), embeddings)
    assert torch.isfinite(loss)
    assert loss > 30.0


def test_auxiliary_bce_is_sum_over_models_then_batch_mean() -> None:
    qualities = torch.tensor(
        [[0.8, 0.4, 0.7], [0.3, 0.9, 0.2]], dtype=DTYPE
    )
    correctness = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=DTYPE)
    expected_per_query = F.binary_cross_entropy(
        qualities, correctness, reduction="none"
    ).sum(dim=1)

    torch.testing.assert_close(
        auxiliary_bce_loss(qualities, correctness, reduction="none"),
        expected_per_query,
    )
    torch.testing.assert_close(
        auxiliary_bce_loss(qualities, correctness), expected_per_query.mean()
    )


def test_greedy_schur_gains_equal_determinant_ratios() -> None:
    embeddings = _tiny_model_embeddings()
    qualities = torch.tensor([0.92, 0.78, 0.55, 0.63], dtype=DTYPE)
    kernel = dpp_kernel(qualities, embeddings)
    result = greedy_dpp_map(kernel, max_size=4)

    previous_subset: tuple[int, ...] = ()
    previous_determinant = torch.ones((), dtype=DTYPE)
    for model_index, gain in zip(result.order, result.gains):
        current_subset = previous_subset + (model_index,)
        current_determinant = _determinant_of_subset(kernel, current_subset)
        ratio = current_determinant / previous_determinant
        assert gain == pytest.approx(float(ratio), rel=1e-10, abs=1e-12)
        previous_subset = current_subset
        previous_determinant = current_determinant

    low_rank = greedy_dpp_map_low_rank(
        qualities, embeddings, max_size=4
    )
    assert low_rank.order == result.order
    assert low_rank.gains == pytest.approx(result.gains, rel=1e-10, abs=1e-12)


def test_greedy_tie_break_and_relative_stopping_rule() -> None:
    kernel = torch.diag(torch.tensor([4.0, 1.0, 0.2, 1.0], dtype=DTYPE))
    result = greedy_dpp_map(kernel, max_size=4, relative_threshold=0.2)

    assert result.order == (0, 1, 3, 2)
    assert result.gains == pytest.approx((4.0, 1.0, 1.0, 0.2))
    assert result.relative_gains == pytest.approx((1.0, 0.25, 0.25, 0.05))
    assert result.selected_indices == (0, 1, 3)


def test_relative_gain_scores_define_nested_prefix_sets() -> None:
    kernel = torch.diag(torch.tensor([4.0, 1.0, 0.2, 0.8], dtype=DTYPE))
    path = greedy_dpp_map(kernel, max_size=4)
    score_by_model = {model: score for model, score in zip(path.order, path.scores)}

    assert list(path.scores) == sorted(path.scores)
    previous: set[int] = set()
    for threshold in [0.0, 0.5, 0.8, 0.96]:
        selected = {
            model for model, score in score_by_model.items() if score <= threshold
        }
        assert previous <= selected
        assert selected == set(path.order[: len(selected)])
        previous = selected


def _configured_router() -> FlexRouter:
    router = FlexRouter(input_dim=3, n_models=4, seed=17)
    with torch.no_grad():
        router.query_projection.weight.zero_()
        router.query_projection.bias.zero_()
        router.query_projection.bias[:4] = 1.0
        router.model_embeddings.zero_()
        router.model_embeddings[0, 0] = 2.0
        router.model_embeddings[1, 1] = 1.0
        router.model_embeddings[2, 2] = 0.5
        router.model_embeddings[3, 3] = -1.0
    return router


def test_crc_scores_and_explicit_shortlist_behavior() -> None:
    router = _configured_router()
    query = torch.tensor([0.2, -0.4, 1.0], dtype=DTYPE)

    full = router.select(query)
    shortlisted = router.select(query, top_k0=2)
    scores = router.crc_scores(query)

    assert full.candidate_indices == (0, 1, 2, 3)
    assert len(full.order) == 4
    assert shortlisted.candidate_indices == (0, 1)
    assert shortlisted.order == (0, 1)
    ordered_scores = scores[list(full.order)]
    assert torch.all(ordered_scores[1:] >= ordered_scores[:-1])
    assert router.default_top_k0 is None
    assert router.metadata()["inference"]["shortlist"] == {
        "policy": "full_candidate_pool",
        "top_k0": None,
    }


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(991)
    embeddings = torch.randn(32, 5, generator=generator, dtype=DTYPE)
    labels = torch.stack(
        [
            embeddings[:, 0] > 0,
            embeddings[:, 1] + 0.2 * embeddings[:, 2] > 0,
            embeddings[:, 3] < 0,
            embeddings[:, 4] - embeddings[:, 0] > 0,
        ],
        dim=1,
    ).to(DTYPE)
    return embeddings[:24], labels[:24], embeddings[24:], labels[24:]


def test_training_is_deterministic_and_uses_success_at_10_early_stopping() -> None:
    train_x, train_y, validation_x, validation_y = _training_data()
    config = FlexRouterTrainingConfig(max_epochs=8, patience=3, batch_size=8)

    first = FlexRouter(5, 4, seed=123).fit(
        train_x, train_y, validation_x, validation_y, config=config
    )
    second = FlexRouter(5, 4, seed=123).fit(
        train_x, train_y, validation_x, validation_y, config=config
    )

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)
    assert first.training_history == second.training_history
    assert first.best_epoch == second.best_epoch
    assert (
        first.best_validation_success_at_10
        == second.best_validation_success_at_10
    )
    assert 1 <= len(first.training_history) <= config.max_epochs
    assert all(
        "validation_success_at_10" in row for row in first.training_history
    )
    assert first.training_config == config


def test_save_load_preserves_model_selection_and_reproducibility_metadata(
    tmp_path: Path,
) -> None:
    train_x, train_y, validation_x, validation_y = _training_data()
    config = FlexRouterTrainingConfig(max_epochs=5, patience=2, batch_size=8)
    router = FlexRouter(
        5,
        4,
        seed=29,
        embedding_metadata={
            "source": "frozen-test-fixture",
            "encoder_match": "precomputed embeddings supplied by caller",
        },
    ).fit(train_x, train_y, validation_x, validation_y, config=config)
    path = tmp_path / "flexrouter.pt"
    query = validation_x[0]
    expected_quality = router.predict_quality(query.unsqueeze(0))
    expected_selection = router.select(query)

    router.save(path)
    restored = FlexRouter.load(path)

    torch.testing.assert_close(restored.predict_quality(query.unsqueeze(0)), expected_quality)
    assert restored.select(query) == expected_selection
    assert restored.training_history == router.training_history
    assert restored.best_epoch == router.best_epoch
    metadata = restored.metadata()
    assert metadata["fidelity"] == {
        "trainable_flexrouter_core": True,
        "input_representation": "matched_frozen_embeddings",
        "frozen_precomputed_query_embeddings": True,
        "exact_roberta_encoder_replication": False,
    }
    assert metadata["embedding_metadata"]["source"] == "frozen-test-fixture"
    assert metadata["runtime_requirements"] == {
        "framework": "PyTorch",
        "cpu_supported": True,
        "paper_reported_hardware": "NVIDIA A100 80GB GPU",
        "practical_disadvantage": (
            "The trainable large-pool baseline adds PyTorch and accelerator "
            "substrate burden versus the NumPy-only baselines in this repository."
        ),
    }
    assert metadata["reproducibility"]["optimizer"] == "Adam"
    assert metadata["reproducibility"]["training_config"] == {
        "learning_rate": 1e-3,
        "max_epochs": 5,
        "patience": 2,
        "auxiliary_bce_weight": 1.0,
        "batch_size": 8,
        "validation_max_size": 10,
    }
    for split in [
        "train_embeddings",
        "train_correctness",
        "validation_embeddings",
        "validation_correctness",
    ]:
        assert len(metadata["reproducibility"][split]["sha256"]) == 64
