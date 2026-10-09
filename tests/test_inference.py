import numpy as np
import pytest

from components.inference import (
    DEFAULT_ENDPOINT_REGISTRY,
    holm_adjust,
    infer_paired_metrics,
    paired_query_bootstrap,
    validate_primary_endpoint_registry,
)


def _four_metric_arms(n=40):
    baseline = {
        "calls": np.linspace(4.0, 8.0, n),
        "risk": np.linspace(0.10, 0.30, n),
        "accuracy": np.linspace(0.55, 0.75, n),
        "cost": np.linspace(1.0, 3.0, n),
    }
    corrected = {
        "calls": baseline["calls"] - 1.0,
        "risk": baseline["risk"] - 0.05,
        "accuracy": baseline["accuracy"] + 0.10,
        "cost": baseline["cost"] - 0.25,
    }
    return baseline, corrected


def test_all_metrics_have_paired_cis_and_expected_direction():
    baseline, corrected = _four_metric_arms()
    report = infer_paired_metrics(
        baseline,
        corrected,
        n_bootstrap=800,
        seed=17,
    )

    assert report["method"] == "paired_query_bootstrap"
    assert report["resampling_unit"] == "query_pair"
    assert report["n_queries"] == 40

    expected = {
        "calls": (-1.0, "corrected_lower"),
        "risk": (-0.05, "corrected_lower"),
        "accuracy": (0.10, "corrected_higher"),
        "cost": (-0.25, "corrected_lower"),
    }
    for metric, (difference, direction) in expected.items():
        result = report["metrics"][metric]
        assert result["difference"]["estimate"] == pytest.approx(difference)
        assert result["difference"]["ci_low"] == pytest.approx(difference)
        assert result["difference"]["ci_high"] == pytest.approx(difference)
        assert result["observed_direction"] == direction
        assert result["difference_definition"] == "corrected_minus_baseline"
        for arm in ("baseline", "corrected", "difference"):
            assert set(result[arm]) == {"estimate", "ci_low", "ci_high"}


def test_identical_arms_produce_zero_difference_and_unit_p_value():
    values = np.linspace(0.1, 0.9, 25)
    result = paired_query_bootstrap(
        values,
        values.copy(),
        metric="accuracy",
        n_bootstrap=500,
        seed=3,
    )

    assert result["difference"] == {
        "estimate": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
    }
    assert result["p_value"] == 1.0
    assert result["observed_direction"] == "identical"


def test_pairing_keeps_constant_query_difference_exact():
    baseline = np.arange(3.0, 33.0)
    corrected = baseline - 2.0
    result = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        n_bootstrap=600,
        seed=9,
    )

    assert result["baseline"]["ci_low"] < result["baseline"]["ci_high"]
    assert result["corrected"]["ci_low"] < result["corrected"]["ci_high"]
    assert result["difference"]["ci_low"] == pytest.approx(-2.0)
    assert result["difference"]["ci_high"] == pytest.approx(-2.0)


def test_benchmark_task_stratification_fixes_observed_cell_counts():
    benchmark = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    task = np.array(["x", "x", "y", "y", "x", "x", "y", "y"])
    baseline = np.full(8, 5.0)
    corrected = np.array([2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0])

    stratified = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        strata={"benchmark": benchmark, "task": task},
        n_bootstrap=700,
        seed=12,
        alternative="two-sided",
    )
    unstratified = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        n_bootstrap=700,
        seed=12,
        alternative="two-sided",
    )

    assert stratified["stratified"] is True
    assert stratified["n_strata"] == 4
    assert stratified["difference"]["estimate"] == 0.0
    assert stratified["difference"]["ci_low"] == 0.0
    assert stratified["difference"]["ci_high"] == 0.0
    assert unstratified["difference"]["ci_low"] < 0.0
    assert unstratified["difference"]["ci_high"] > 0.0


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(100)
    baseline = rng.normal(5.0, 1.0, 80)
    corrected = baseline + rng.normal(-0.2, 0.8, 80)

    first = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        n_bootstrap=650,
        seed=2026,
    )
    second = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        n_bootstrap=650,
        seed=2026,
    )
    different_seed = paired_query_bootstrap(
        baseline,
        corrected,
        metric="calls",
        n_bootstrap=650,
        seed=2027,
    )

    assert first == second
    assert first["difference"] != different_seed["difference"]


def test_holm_adjustment_is_monotone_and_only_applies_to_secondaries():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})

    baseline, corrected = _four_metric_arms(30)
    report = infer_paired_metrics(
        baseline,
        corrected,
        endpoint_registry=DEFAULT_ENDPOINT_REGISTRY,
        n_bootstrap=400,
        seed=4,
    )
    assert report["metrics"]["calls"]["role"] == "primary"
    assert report["metrics"]["calls"]["p_value_holm"] is None
    for metric in ("risk", "accuracy", "cost"):
        result = report["metrics"][metric]
        assert result["role"] == "secondary"
        assert result["p_value_holm"] >= result["p_value"]


def test_primary_endpoint_registry_normalizes_supported_forms():
    normalized = validate_primary_endpoint_registry(
        {
            "calls": True,
            "risk": "secondary",
            "accuracy": {"primary": False, "alternative": "two-sided"},
            "cost": {"role": "secondary"},
        }
    )
    assert normalized["calls"] == {"role": "primary", "alternative": "less"}
    assert normalized["accuracy"] == {
        "role": "secondary",
        "alternative": "two-sided",
    }

    from_primary_names = validate_primary_endpoint_registry(["calls", "accuracy"])
    assert from_primary_names["calls"]["role"] == "primary"
    assert from_primary_names["risk"]["role"] == "secondary"


@pytest.mark.parametrize(
    "registry,match",
    [
        ({"calls": "primary"}, "missing endpoints"),
        (
            {
                "calls": "secondary",
                "risk": "secondary",
                "accuracy": "secondary",
                "cost": "secondary",
            },
            "at least one primary",
        ),
        (["calls", "calls"], "must be unique"),
        (["not-a-metric"], "unknown primary"),
    ],
)
def test_invalid_endpoint_registries_are_rejected(registry, match):
    with pytest.raises(ValueError, match=match):
        validate_primary_endpoint_registry(registry)


@pytest.mark.parametrize(
    "baseline,corrected,kwargs,error,match",
    [
        ([1.0, 2.0], [1.0], {}, ValueError, "different lengths"),
        ([], [], {}, ValueError, "must not be empty"),
        ([1.0, np.nan], [1.0, 2.0], {}, ValueError, "finite"),
        ([0.5, 1.2], [0.5, 0.8], {"metric": "accuracy"}, ValueError, "between 0 and 1"),
        ([-1.0, 2.0], [1.0, 2.0], {}, ValueError, "nonnegative"),
        ([1.0, 2.0], [1.0, 2.0], {"n_bootstrap": 1}, ValueError, "at least 2"),
        ([1.0, 2.0], [1.0, 2.0], {"seed": -1}, ValueError, "nonnegative"),
    ],
)
def test_invalid_paired_inputs_are_rejected(
    baseline, corrected, kwargs, error, match
):
    options = {"metric": "calls", "n_bootstrap": 20, "seed": 1}
    options.update(kwargs)
    with pytest.raises(error, match=match):
        paired_query_bootstrap(baseline, corrected, **options)


@pytest.mark.parametrize(
    "strata,match",
    [
        (["a"], "match query count"),
        (["a", None], "missing labels"),
        ({}, "must not be empty"),
        ({"benchmark": ["a", "b"], "task": ["x"]}, "must match query count"),
    ],
)
def test_invalid_strata_are_rejected(strata, match):
    with pytest.raises(ValueError, match=match):
        paired_query_bootstrap(
            [1.0, 2.0],
            [1.0, 2.0],
            metric="calls",
            strata=strata,
            n_bootstrap=20,
            seed=1,
        )


def test_no_repeated_split_sign_test_is_reported_as_inference():
    result = paired_query_bootstrap(
        [2.0, 3.0, 4.0],
        [1.0, 3.0, 5.0],
        metric="calls",
        n_bootstrap=100,
        seed=8,
        alternative="two-sided",
    )
    assert result["method"] == "paired_query_bootstrap"
    assert result["resampling_unit"] == "query_pair"
    assert "split" not in result
    assert "sign_test" not in result
