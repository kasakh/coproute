"""Tests for the corrected_v1 immutable primary experiment specification."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
import unittest
from dataclasses import FrozenInstanceError, replace


CORRECTED_ROOT = Path(__file__).resolve().parents[1]
if str(CORRECTED_ROOT) not in sys.path:
    sys.path.insert(0, str(CORRECTED_ROOT))

from components.experiment_spec import (  # noqa: E402
    AggregationMethod,
    Benchmark,
    ComputeSpec,
    CoveragePolicy,
    DependenceEstimatorSpec,
    ExperimentSpec,
    FractionalThresholdPolicy,
    MultiplicityMethod,
    PoolingMethod,
    PRIMARY_EXPERIMENT_SPEC,
    REQUIRED_OUTPUT_ARTIFACTS,
    RouterKind,
    SelectorArm,
    SplitFractions,
    SplitRole,
    canonical_primary_spec,
    validate_disjoint_split_roles,
)


class CanonicalPrimarySpecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = canonical_primary_spec()

    def test_module_constant_is_the_canonical_spec(self) -> None:
        self.assertEqual(PRIMARY_EXPERIMENT_SPEC, self.spec)
        self.assertIsNot(PRIMARY_EXPERIMENT_SPEC, self.spec)
        self.assertEqual(self.spec.version, "corrected_v1")

    def test_entire_schema_is_frozen_and_uses_immutable_collections(self) -> None:
        nested_objects = (
            self.spec,
            *self.spec.primary_endpoints,
            self.spec.coverage,
            self.spec.router,
            *self.spec.router.candidates,
            self.spec.pooling,
            *self.spec.selector_arms,
            *self.spec.dependence_estimators,
            self.spec.aggregation,
            self.spec.splits,
            *self.spec.splits.fractions_by_task,
            self.spec.bootstrap,
            self.spec.multiplicity,
            self.spec.fractional_thresholds,
            self.spec.compute,
            self.spec.outputs,
        )
        for item in nested_objects:
            self.assertTrue(dataclasses.is_dataclass(item))
            self.assertTrue(item.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            self.spec.shot_setting = 5
        with self.assertRaises(FrozenInstanceError):
            self.spec.router.include_null_model = False
        self.assertIsInstance(self.spec.selector_arms, tuple)
        self.assertIsInstance(self.spec.outputs.required_artifacts, tuple)

    def test_primary_endpoints_are_calls_risk_and_official_task_score(self) -> None:
        endpoints = {endpoint.name: endpoint for endpoint in self.spec.primary_endpoints}
        self.assertEqual(
            set(endpoints),
            {
                "mean_selected_standard_models",
                "all_models_miss_rate",
                "official_task_score",
            },
        )
        self.assertEqual(endpoints["mean_selected_standard_models"].direction.value, "minimize")
        self.assertEqual(endpoints["all_models_miss_rate"].direction.value, "control_at_alpha")
        self.assertEqual(endpoints["official_task_score"].direction.value, "maximize")

    def test_benchmark_coverage_and_alpha_grid_are_preregistered(self) -> None:
        self.assertEqual(
            self.spec.coverage.primary_headroom_tasks,
            (Benchmark.MMLU, Benchmark.WINOGRANDE),
        )
        self.assertEqual(
            self.spec.coverage.secondary_tasks,
            (
                Benchmark.ARC,
                Benchmark.HELLASWAG,
                Benchmark.GSM8K,
                Benchmark.MBPP,
            ),
        )
        self.assertEqual(self.spec.alpha_values, (0.05, 0.10))
        self.assertTrue(self.spec.coverage.run_every_alpha_on_primary)
        self.assertTrue(self.spec.coverage.run_every_registered_arm_on_primary)
        self.assertTrue(self.spec.coverage.run_controls_on_secondary)
        self.assertEqual(self.spec.shot_setting, 0)

    def test_exact_racer_router_and_pooling_ablation_are_fixed(self) -> None:
        router = self.spec.router
        self.assertEqual(router.encoder, "microsoft/mdeberta-v3-base")
        self.assertEqual({candidate.kind for candidate in router.candidates}, set(RouterKind))
        knnr = next(candidate for candidate in router.candidates if candidate.kind is RouterKind.KNNR)
        mlpr = next(candidate for candidate in router.candidates if candidate.kind is RouterKind.MLPR)
        self.assertEqual(knnr.k_neighbors, 40)
        self.assertEqual(mlpr.hidden_units, (256,))
        self.assertTrue(router.include_null_model)
        self.assertTrue(router.randomized_smoothing)
        self.assertEqual(router.smoothing_epsilon, 1e-6)
        self.assertEqual(router.selection_split, SplitRole.VALIDATION)
        self.assertEqual(set(self.spec.pooling.candidates), set(PoolingMethod))
        self.assertTrue(self.spec.pooling.freeze_after_selection)

    def test_selector_registry_contains_every_required_primary_and_control_arm(self) -> None:
        arms = {arm.name: arm for arm in self.spec.selector_arms}
        self.assertEqual(
            set(arms),
            {
                "exact_racer_p",
                "exact_racer_g",
                "depth_independence",
                "residual_gaussian_dacr",
                "pooled_gaussian_ablation",
                "empirical_all_miss_control",
                "shrinkage_empirical_control",
                "pairwise_double_fault_control",
                "pairwise_error_correlation_control",
                "mi_style_control",
                "complementary_coverage_control",
            },
        )
        self.assertEqual(arms["exact_racer_p"].exact_racer_variant, "P")
        self.assertEqual(arms["exact_racer_g"].exact_racer_variant, "G")
        self.assertEqual(arms["depth_independence"].selector_implementation, "marginal_depth")
        self.assertEqual(
            arms["residual_gaussian_dacr"].dependence_estimator,
            "query_clustered_cross_fitted_residual_correlation",
        )
        self.assertEqual(
            arms["pooled_gaussian_ablation"].dependence_estimator,
            "pooled_tetrachoric",
        )
        self.assertTrue(all(arm.deployable_after_independent_crc for arm in arms.values()))

    def test_dependence_estimators_are_complete_and_design_only(self) -> None:
        estimators = {item.name: item for item in self.spec.dependence_estimators}
        self.assertEqual(
            set(estimators),
            {
                "none",
                "independence",
                "query_clustered_cross_fitted_residual_correlation",
                "pooled_tetrachoric",
                "empirical_all_miss",
                "shrinkage_empirical",
                "pairwise_double_fault",
                "pairwise_error_correlation",
                "mutual_information",
                "complementary_coverage",
            },
        )
        self.assertTrue(all(item.design_split_only for item in estimators.values()))
        self.assertTrue(
            estimators[
                "query_clustered_cross_fitted_residual_correlation"
            ].conditional_on_router_marginals
        )
        self.assertFalse(estimators["pooled_tetrachoric"].conditional_on_router_marginals)

    def test_all_aggregation_candidates_are_selected_then_frozen(self) -> None:
        self.assertEqual(set(self.spec.aggregation.candidates), set(AggregationMethod))
        self.assertEqual(self.spec.aggregation.selection_split, SplitRole.VALIDATION)
        self.assertTrue(self.spec.aggregation.freeze_after_selection)

    def test_task_specific_split_fractions_match_corrected_protocol(self) -> None:
        fractions = {item.task: item for item in self.spec.splits.fractions_by_task}
        self.assertEqual(self.spec.splits.seed, 20260903)
        self.assertEqual(
            fractions[Benchmark.MMLU].as_role_pairs(),
            (
                (SplitRole.TRAIN, 0.20),
                (SplitRole.DESIGN, 0.20),
                (SplitRole.CALIBRATION, 0.30),
                (SplitRole.VALIDATION, 0.06),
                (SplitRole.TEST, 0.24),
            ),
        )
        self.assertEqual(
            fractions[Benchmark.ARC].as_role_pairs(),
            (
                (SplitRole.TRAIN, 0.20),
                (SplitRole.DESIGN, 0.20),
                (SplitRole.CALIBRATION, 0.24),
                (SplitRole.VALIDATION, 0.12),
                (SplitRole.TEST, 0.24),
            ),
        )
        self.assertEqual(
            fractions[Benchmark.GSM8K].as_role_pairs(),
            (
                (SplitRole.TRAIN, 0.25),
                (SplitRole.DESIGN, 0.25),
                (SplitRole.CALIBRATION, 0.25),
                (SplitRole.VALIDATION, 0.05),
                (SplitRole.TEST, 0.20),
            ),
        )
        for item in fractions.values():
            self.assertAlmostEqual(sum(value for _, value in item.as_role_pairs()), 1.0)

    def test_bootstrap_and_multiplicity_family_are_fixed(self) -> None:
        self.assertEqual(self.spec.bootstrap.method.value, "paired_query_bootstrap")
        self.assertEqual(self.spec.bootstrap.seed, 20260904)
        self.assertEqual(self.spec.bootstrap.repetitions, 10_000)
        self.assertEqual(self.spec.bootstrap.resampling_unit, "query_id")
        self.assertEqual(self.spec.multiplicity.method, MultiplicityMethod.HOLM)
        self.assertEqual(self.spec.multiplicity.family_name, "primary_dacr_headroom")
        self.assertEqual(len(self.spec.multiplicity.members), 4)
        self.assertTrue(all("mean_selected_standard_models" in member for member in self.spec.multiplicity.members))

    def test_gpu_and_output_requirements_are_exact(self) -> None:
        self.assertEqual(self.spec.compute.gpu_ids, (2, 3))
        self.assertTrue(REQUIRED_OUTPUT_ARTIFACTS <= set(self.spec.outputs.required_artifacts))
        self.assertTrue(self.spec.outputs.immutable_run_directory)
        self.assertTrue(self.spec.outputs.versioned_run_directory)
        self.assertTrue(self.spec.outputs.atomic_manifest)


class ValidationFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = canonical_primary_spec()

    def test_silent_fractional_depth_threshold_is_rejected(self) -> None:
        changed_arm = replace(
            self.spec.selector_arms[2], uses_fractional_depth_threshold=True
        )
        changed_arms = (
            self.spec.selector_arms[:2]
            + (changed_arm,)
            + self.spec.selector_arms[3:]
        )
        with self.assertRaisesRegex(ValueError, "silent fractional arms"):
            replace(self.spec, selector_arms=changed_arms)

    def test_fractional_depth_threshold_can_only_be_explicitly_preregistered(self) -> None:
        changed_arm = replace(
            self.spec.selector_arms[2], uses_fractional_depth_threshold=True
        )
        changed_arms = (
            self.spec.selector_arms[:2]
            + (changed_arm,)
            + self.spec.selector_arms[3:]
        )
        explicit = FractionalThresholdPolicy(
            enabled=True, registered_arms=(changed_arm.name,)
        )
        registered = replace(
            self.spec,
            selector_arms=changed_arms,
            fractional_thresholds=explicit,
        )
        self.assertEqual(
            registered.fractional_thresholds.registered_arms,
            ("depth_independence",),
        )

    def test_fractional_registry_cannot_name_a_nonfractional_or_unknown_arm(self) -> None:
        policy = FractionalThresholdPolicy(
            enabled=True, registered_arms=("not_an_arm",)
        )
        with self.assertRaisesRegex(ValueError, "stale fractional registrations"):
            replace(self.spec, fractional_thresholds=policy)

    def test_overlapping_split_roles_are_rejected(self) -> None:
        assignments = {
            SplitRole.TRAIN: ("q1", "q2"),
            SplitRole.DESIGN: ("q3",),
            SplitRole.CALIBRATION: ("q4",),
            SplitRole.VALIDATION: ("q2", "q5"),
            SplitRole.TEST: ("q6",),
        }
        with self.assertRaisesRegex(ValueError, "split roles overlap"):
            validate_disjoint_split_roles(assignments)
        with self.assertRaisesRegex(ValueError, "split roles overlap"):
            self.spec.splits.validate_assignments(assignments)

    def test_missing_split_role_and_duplicate_within_role_are_rejected(self) -> None:
        missing = {
            SplitRole.TRAIN: ("q1",),
            SplitRole.DESIGN: ("q2",),
            SplitRole.CALIBRATION: ("q3",),
            SplitRole.TEST: ("q4",),
        }
        with self.assertRaisesRegex(ValueError, "exactly five roles"):
            validate_disjoint_split_roles(missing)
        duplicate = dict(missing)
        duplicate[SplitRole.VALIDATION] = ("q5", "q5")
        with self.assertRaisesRegex(ValueError, "duplicate query id"):
            validate_disjoint_split_roles(duplicate)

    def test_valid_split_assignments_are_accepted(self) -> None:
        assignments = {
            role: ("query_" + role.value,) for role in SplitRole
        }
        self.assertIsNone(validate_disjoint_split_roles(assignments))

    def test_unregistered_selector_implementation_is_rejected_at_arm_creation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered selector implementation"):
            SelectorArm("mystery", "not_registered", "none")

    def test_unregistered_requested_experiment_arm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered experiment arm"):
            self.spec.validate_requested_arms(
                ("depth_independence", "invented_arm")
            )

    def test_unregistered_dependence_estimator_is_rejected(self) -> None:
        bad_arm = replace(
            self.spec.selector_arms[2], dependence_estimator="not_registered"
        )
        changed = (
            self.spec.selector_arms[:2]
            + (bad_arm,)
            + self.spec.selector_arms[3:]
        )
        with self.assertRaisesRegex(ValueError, "unregistered dependence estimator"):
            replace(self.spec, selector_arms=changed)

    def test_forbidden_gpu_is_rejected(self) -> None:
        for gpu_ids in ((0,), (1, 2), (2, 3, 4)):
            with self.subTest(gpu_ids=gpu_ids):
                with self.assertRaisesRegex(ValueError, "forbidden GPU allocation"):
                    ComputeSpec(gpu_ids=gpu_ids)
        with self.assertRaises(TypeError):
            ComputeSpec(gpu_ids=(True,))

    def test_invalid_split_fractions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to one"):
            SplitFractions(Benchmark.MMLU, 0.2, 0.2, 0.2, 0.1, 0.1)
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            SplitFractions(Benchmark.MMLU, 0.0, 0.2, 0.3, 0.1, 0.4)

    def test_primary_and_secondary_coverage_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage overlaps"):
            CoveragePolicy(
                primary_headroom_tasks=(Benchmark.MMLU,),
                secondary_tasks=(Benchmark.MMLU, Benchmark.ARC),
            )

    def test_every_arm_must_have_exactly_one_registered_estimator(self) -> None:
        estimators = self.spec.dependence_estimators + (
            DependenceEstimatorSpec(
                "none", "duplicate", False
            ),
        )
        with self.assertRaisesRegex(ValueError, "dependence estimator names"):
            replace(self.spec, dependence_estimators=estimators)

    def test_split_policy_must_cover_every_task(self) -> None:
        incomplete = replace(
            self.spec.splits,
            fractions_by_task=self.spec.splits.fractions_by_task[:-1],
        )
        with self.assertRaisesRegex(ValueError, "every preregistered task"):
            replace(self.spec, splits=incomplete)


if __name__ == "__main__":
    unittest.main()
