"""Deterministic synthetic tests for exact RACER base routers."""

from pathlib import Path
import sys
import tempfile

import numpy as np


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORRECTED_ROOT))

from components.routers import (  # noqa: E402
    KNN_NEIGHBORS,
    MDEBERTA_MODEL_NAME,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_UNITS,
    MLP_LEARNING_RATE,
    KNNR,
    MLPR,
    RouterDC,
    RouterDCUnavailableError,
    RouterNotFittedError,
)


def _expect_raises(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type as error:
        return error
    raise AssertionError(f"expected {error_type.__name__}")


def _knn_training_data() -> tuple[np.ndarray, np.ndarray]:
    positive = np.column_stack(
        (np.ones(40), np.linspace(-0.02, 0.02, 40), np.zeros(40))
    )
    negative = np.column_stack(
        (-np.ones(40), np.linspace(-0.02, 0.02, 40), np.zeros(40))
    )
    embeddings = np.vstack((positive, negative))
    labels = np.vstack(
        (
            np.tile([1, 0, 1], (40, 1)),
            np.tile([0, 1, 0], (40, 1)),
        )
    )
    return embeddings, labels


def _mlp_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(913)
    features = rng.normal(size=(128, 6))
    labels = np.column_stack(
        (
            features[:, 0] + 0.4 * features[:, 1] > 0.0,
            features[:, 2] - 0.3 * features[:, 3] > 0.0,
            features[:, 0] + features[:, 4] < 0.0,
        )
    ).astype(int)
    return features[:96], labels[:96], features[96:], labels[96:]


def test_knnr_uses_exact_racer_contract_and_per_model_probabilities() -> None:
    embeddings, labels = _knn_training_data()
    router = KNNR().fit(embeddings, labels)

    probabilities = router.predict(
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    )

    assert router.k == KNN_NEIGHBORS == 40
    assert router.embedding_model == MDEBERTA_MODEL_NAME
    assert router.n_models == 3
    np.testing.assert_array_equal(
        probabilities,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_array_equal(router.predict_proba([[1.0, 0.0, 0.0]]), [[1, 0, 1]])


def test_knnr_equal_similarity_ties_are_stable() -> None:
    embeddings = np.tile([1.0, 0.0], (50, 1))
    labels = np.zeros((50, 2), dtype=int)
    labels[:10, 0] = 1
    labels[10:40, 1] = 1
    labels[40:, :] = 1

    probabilities = KNNR().fit(embeddings, labels).predict([[1.0, 0.0]])

    np.testing.assert_allclose(probabilities, [[0.25, 0.75]])


def test_knnr_rejects_nonexact_settings_and_invalid_fit_data() -> None:
    _expect_raises(ValueError, KNNR, k=39)
    _expect_raises(ValueError, KNNR, embedding_model="another-encoder")
    _expect_raises(ValueError, KNNR().fit, np.ones((39, 3)), np.ones((39, 2)))
    _expect_raises(RouterNotFittedError, KNNR().predict, np.ones((1, 3)))


def test_knnr_save_and_load_preserve_predictions() -> None:
    embeddings, labels = _knn_training_data()
    query = np.array([[1.0, 0.01, 0.0], [-1.0, -0.01, 0.0]])
    router = KNNR().fit(embeddings, labels)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "knnr.model"
        router.save(path)
        restored = KNNR.load(path)

    np.testing.assert_array_equal(restored.predict(query), router.predict(query))
    assert restored.k == 40
    assert restored.embedding_model == MDEBERTA_MODEL_NAME


def test_mlpr_exact_architecture_training_contract_and_checkpoint() -> None:
    train_x, train_y, validation_x, validation_y = _mlp_data()
    router = MLPR(seed=29).fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
    )

    probabilities = router.predict(validation_x)

    assert router.hidden_units == MLP_HIDDEN_UNITS == 256
    assert router.batch_size == MLP_BATCH_SIZE == 32
    assert router.learning_rate == MLP_LEARNING_RATE == 1e-4
    assert router.epochs == MLP_EPOCHS == 100
    assert router.loss_name == "BCEWithLogitsLoss"
    assert router.best_epoch is not None
    assert 0 <= router.best_epoch < 100
    assert router.best_validation_loss == min(router.validation_loss)
    assert len(router.training_loss) == 100
    assert len(router.validation_loss) == 100
    assert router._weights is not None
    assert tuple(weight.shape for weight in router._weights) == (
        (6, 256),
        (256,),
        (256, 3),
        (3,),
    )
    assert probabilities.shape == (32, 3)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    predicted = probabilities >= 0.5
    assert np.mean(predicted == validation_y) >= 0.75


def test_mlpr_is_deterministic_for_seed_and_data() -> None:
    train_x, train_y, validation_x, validation_y = _mlp_data()
    first = MLPR(seed=7).fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
    )
    second = MLPR(seed=7).fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
    )

    np.testing.assert_array_equal(first.predict(validation_x), second.predict(validation_x))
    np.testing.assert_array_equal(first.training_loss, second.training_loss)
    np.testing.assert_array_equal(first.validation_loss, second.validation_loss)
    assert first.best_epoch == second.best_epoch


def test_mlpr_validation_is_required_and_shapes_are_checked() -> None:
    train_x, train_y, validation_x, validation_y = _mlp_data()
    _expect_raises(
        ValueError,
        MLPR().fit,
        train_x,
        train_y,
        validation_data=None,
    )
    _expect_raises(
        ValueError,
        MLPR().fit,
        train_x,
        train_y,
        validation_data=(validation_x[:, :-1], validation_y),
    )
    _expect_raises(RouterNotFittedError, MLPR().predict, validation_x)
    _expect_raises(ValueError, MLPR, device="gpu")


def test_mlpr_save_and_load_restore_best_validation_checkpoint() -> None:
    train_x, train_y, validation_x, validation_y = _mlp_data()
    router = MLPR(seed=101).fit(
        train_x,
        train_y,
        validation_data=(validation_x, validation_y),
    )
    expected = router.predict(validation_x)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mlpr.model"
        router.save(path)
        restored = MLPR.load(path)

    np.testing.assert_array_equal(restored.predict(validation_x), expected)
    assert restored.best_epoch == router.best_epoch
    assert restored.best_validation_loss == router.best_validation_loss
    assert restored.validation_loss == router.validation_loss


def test_routerdc_fails_loudly_without_validated_implementation() -> None:
    router = RouterDC()
    error = _expect_raises(RouterDCUnavailableError, router.fit, np.ones((2, 2)))
    assert "no surrogate is used" in str(error)
    _expect_raises(RouterDCUnavailableError, router.predict, np.ones((2, 2)))
    _expect_raises(RouterDCUnavailableError, router.save, "unused.model")
    _expect_raises(RouterDCUnavailableError, RouterDC.load, "unused.model")
    _expect_raises(
        RouterDCUnavailableError,
        RouterDC,
        object(),
        validated_config={"validated": False, "name": "RouterDC", "version": "1"},
    )


def test_routerdc_delegates_only_with_explicit_validated_config() -> None:
    class ValidatedFake:
        def __init__(self) -> None:
            self.fitted = False
            self.saved_path = None

        def fit(self, features, labels):
            self.fitted = features.shape[0] == labels.shape[0]
            return self

        def predict(self, features):
            assert self.fitted
            return np.tile([0.2, 0.8], (features.shape[0], 1))

        def save(self, path):
            self.saved_path = Path(path)

    config = {"validated": True, "name": "published-routerdc", "version": "1.0"}
    implementation = ValidatedFake()
    router = RouterDC(implementation, validated_config=config)
    router.fit(np.ones((3, 2)), np.ones((3, 2)))

    probabilities = router.predict(np.ones((2, 2)))
    router.save("routerdc.model")

    np.testing.assert_allclose(probabilities, [[0.2, 0.8], [0.2, 0.8]])
    assert implementation.saved_path == Path("routerdc.model")


def test_routerdc_load_requires_and_uses_validated_loader() -> None:
    class LoadedFake:
        def fit(self, *args, **kwargs):
            return self

        def predict(self, features):
            return np.full((features.shape[0], 1), 0.6)

        def save(self, path):
            return None

    seen = []

    def loader(path):
        seen.append(Path(path))
        return LoadedFake()

    config = {"validated": True, "name": "published-routerdc", "version": "1.0"}
    router = RouterDC.load(
        "validated.model",
        implementation_loader=loader,
        validated_config=config,
    )

    assert seen == [Path("validated.model")]
    np.testing.assert_allclose(router.predict(np.ones((2, 3))), [[0.6], [0.6]])
