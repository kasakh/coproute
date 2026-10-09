"""Immutable primary experiment specification for corrected DACR experiments.

The objects in this module are the preregistration contract. They contain only
immutable values and reject ambiguous protocol changes when constructed. Runtime
artifacts may record realized thresholds, but random mixtures between adjacent
integer depths must be declared here rather than introduced silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Iterable, Mapping, Optional, Sequence, Tuple


class _ValueEnum(str, Enum):
    """String enum whose values serialize without a custom registry."""

    def __str__(self) -> str:
        return self.value


class Benchmark(_ValueEnum):
    MMLU = "mmlu"
    WINOGRANDE = "winogrande"
    ARC = "arc-challenge"
    HELLASWAG = "hellaswag"
    GSM8K = "gsm8k"
    MBPP = "mbpp"


class SplitRole(_ValueEnum):
    TRAIN = "train"
    DESIGN = "design"
    CALIBRATION = "calibration"
    VALIDATION = "validation"
    TEST = "test"


class EndpointDirection(_ValueEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CONTROL_AT_ALPHA = "control_at_alpha"


class EndpointRole(_ValueEnum):
    EFFICACY = "efficacy"
    VALIDITY = "validity"
    SAFETY = "safety"


class PoolingMethod(_ValueEnum):
    FIRST_TOKEN = "first_token_cls_style"
    ATTENTION_MASK_MEAN = "attention_mask_mean"


class RouterKind(_ValueEnum):
    KNNR = "knnr"
    MLPR = "mlpr"


class AggregationMethod(_ValueEnum):
    MAJORITY = "majority"
    SCORE_WEIGHTED = "score_weighted"
    TOP_ROUTER = "top_router"
    FULL_ENSEMBLE = "full_ensemble"


class MultiplicityMethod(_ValueEnum):
    HOLM = "holm"


class BootstrapMethod(_ValueEnum):
    PAIRED_QUERY = "paired_query_bootstrap"


REGISTERED_SELECTOR_IMPLEMENTATIONS = frozenset(
    {
        "racer_p",
        "racer_g",
        "marginal_depth",
        "empirical_all_miss_greedy",
        "shrinkage_empirical_greedy",
        "pairwise_double_fault",
        "pairwise_error_correlation",
        "mi_style_greedy",
        "complementary_incremental_coverage",
        "gaussian_copula_greedy",
    }
)

REQUIRED_OUTPUT_ARTIFACTS = frozenset(
    {
        "config",
        "environment",
        "gpu",
        "split_assignments",
        "selected_sets",
        "calibrated_thresholds",
        "realized_risks",
        "model_calls",
        "task_scores",
        "costs",
        "dependence_correlations",
        "random_seeds",
        "raw_predictions",
        "bootstrap_results",
        "manifest_hashes",
    }
)

_ALLOWED_GPU_IDS = frozenset({2, 3})
_REQUIRED_SPLIT_ROLES = tuple(SplitRole)


def _require_nonempty_unique(values: Sequence[object], name: str) -> None:
    if not values:
        raise ValueError(name + " must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(name + " must not contain duplicates")


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(name + " must be an integer")
    if value < 0:
        raise ValueError(name + " must be non-negative")
    return value


def _require_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name + " must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(name + " must be strictly between zero and one")
    return result


@dataclass(frozen=True)
class PrimaryEndpoint:
    name: str
    direction: EndpointDirection
    role: EndpointRole
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("endpoint name and unit must not be empty")


@dataclass(frozen=True)
class CoveragePolicy:
    primary_headroom_tasks: Tuple[Benchmark, ...]
    secondary_tasks: Tuple[Benchmark, ...]
    run_every_alpha_on_primary: bool = True
    run_every_registered_arm_on_primary: bool = True
    run_controls_on_secondary: bool = True

    def __post_init__(self) -> None:
        _require_nonempty_unique(self.primary_headroom_tasks, "primary_headroom_tasks")
        _require_nonempty_unique(self.secondary_tasks, "secondary_tasks")
        overlap = set(self.primary_headroom_tasks) & set(self.secondary_tasks)
        if overlap:
            names = ", ".join(sorted(task.value for task in overlap))
            raise ValueError("primary and secondary task coverage overlaps: " + names)

    @property
    def all_tasks(self) -> Tuple[Benchmark, ...]:
        return self.primary_headroom_tasks + self.secondary_tasks


@dataclass(frozen=True)
class RouterCandidate:
    kind: RouterKind
    k_neighbors: Optional[int] = None
    hidden_units: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is RouterKind.KNNR:
            if self.k_neighbors != 40 or self.hidden_units:
                raise ValueError("exact RACER KNNR must use K=40 and no hidden layers")
        elif self.kind is RouterKind.MLPR:
            if self.k_neighbors is not None or self.hidden_units != (256,):
                raise ValueError("exact RACER MLPR must use one 256-unit hidden layer")


@dataclass(frozen=True)
class RouterSpec:
    encoder: str
    candidates: Tuple[RouterCandidate, ...]
    include_null_model: bool
    randomized_smoothing: bool
    smoothing_epsilon: float
    selection_split: SplitRole

    def __post_init__(self) -> None:
        if self.encoder != "microsoft/mdeberta-v3-base":
            raise ValueError("primary router encoder must be microsoft/mdeberta-v3-base")
        _require_nonempty_unique(tuple(candidate.kind for candidate in self.candidates), "router candidates")
        required = {RouterKind.KNNR, RouterKind.MLPR}
        if set(candidate.kind for candidate in self.candidates) != required:
            raise ValueError("exact RACER requires both KNNR and MLPR router candidates")
        if not self.include_null_model:
            raise ValueError("exact RACER requires null-model abstention")
        if not self.randomized_smoothing:
            raise ValueError("exact RACER requires randomized smoothing")
        if not math.isfinite(self.smoothing_epsilon) or self.smoothing_epsilon <= 0.0:
            raise ValueError("smoothing_epsilon must be finite and positive")
        if self.selection_split is not SplitRole.VALIDATION:
            raise ValueError("router selection must use only the validation split")


@dataclass(frozen=True)
class PoolingSpec:
    candidates: Tuple[PoolingMethod, ...]
    selection_split: SplitRole
    freeze_after_selection: bool

    def __post_init__(self) -> None:
        _require_nonempty_unique(self.candidates, "pooling candidates")
        if set(self.candidates) != set(PoolingMethod):
            raise ValueError("pooling ablation must include first-token and mean pooling")
        if self.selection_split is not SplitRole.VALIDATION:
            raise ValueError("pooling selection must use only the validation split")
        if not self.freeze_after_selection:
            raise ValueError("pooling choice must be frozen before calibration and test")


@dataclass(frozen=True)
class DependenceEstimatorSpec:
    name: str
    family: str
    conditional_on_router_marginals: bool
    design_split_only: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("dependence estimator name and family must not be empty")
        if not self.design_split_only:
            raise ValueError("dependence estimators must be fitted on the design split only")


@dataclass(frozen=True)
class SelectorArm:
    name: str
    selector_implementation: str
    dependence_estimator: str
    exact_racer_variant: Optional[str] = None
    uses_fractional_depth_threshold: bool = False
    deployable_after_independent_crc: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("selector arm name must not be empty")
        if self.selector_implementation not in REGISTERED_SELECTOR_IMPLEMENTATIONS:
            raise ValueError(
                "unregistered selector implementation: " + self.selector_implementation
            )
        if self.exact_racer_variant is not None:
            if self.exact_racer_variant not in {"P", "G"}:
                raise ValueError("exact_racer_variant must be P or G")
            expected = "racer_" + self.exact_racer_variant.lower()
            if self.selector_implementation != expected:
                raise ValueError("RACER variant does not match selector implementation")


@dataclass(frozen=True)
class FractionalThresholdPolicy:
    """Explicit registry for random mixtures between adjacent integer depths."""

    enabled: bool
    registered_arms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.registered_arms)) != len(self.registered_arms):
            raise ValueError("registered fractional-threshold arms contain duplicates")
        if self.enabled != bool(self.registered_arms):
            raise ValueError(
                "fractional depth thresholds require an explicit nonempty arm registry"
            )


@dataclass(frozen=True)
class AggregationSpec:
    candidates: Tuple[AggregationMethod, ...]
    selection_split: SplitRole
    freeze_after_selection: bool

    def __post_init__(self) -> None:
        _require_nonempty_unique(self.candidates, "aggregation candidates")
        if set(self.candidates) != set(AggregationMethod):
            raise ValueError("all four preregistered aggregation methods are required")
        if self.selection_split is not SplitRole.VALIDATION:
            raise ValueError("aggregation selection must use only the validation split")
        if not self.freeze_after_selection:
            raise ValueError("aggregation must be frozen before calibration and test")


@dataclass(frozen=True)
class SplitFractions:
    task: Benchmark
    train: float
    design: float
    calibration: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.design, self.calibration, self.validation, self.test)
        for role, value in zip(_REQUIRED_SPLIT_ROLES, values):
            _require_probability(value, self.task.value + " " + role.value + " fraction")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("split fractions for " + self.task.value + " must sum to one")

    def as_role_pairs(self) -> Tuple[Tuple[SplitRole, float], ...]:
        return tuple(
            zip(
                _REQUIRED_SPLIT_ROLES,
                (self.train, self.design, self.calibration, self.validation, self.test),
            )
        )


@dataclass(frozen=True)
class SplitPolicy:
    seed: int
    fractions_by_task: Tuple[SplitFractions, ...]
    assignment_unit: str = "stable_query_id"
    task_stratified: bool = True

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "split seed")
        _require_nonempty_unique(
            tuple(item.task for item in self.fractions_by_task), "split task policies"
        )
        if self.assignment_unit != "stable_query_id" or not self.task_stratified:
            raise ValueError("splits must be task-stratified by stable query id")

    def validate_assignments(
        self, assignments: Mapping[SplitRole, Iterable[str]]
    ) -> None:
        validate_disjoint_split_roles(assignments)


@dataclass(frozen=True)
class BootstrapSpec:
    method: BootstrapMethod
    seed: int
    repetitions: int
    confidence_level: float
    resampling_unit: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "bootstrap seed")
        _require_nonnegative_int(self.repetitions, "bootstrap repetitions")
        if self.repetitions < 2:
            raise ValueError("bootstrap repetitions must be at least 2")
        _require_probability(self.confidence_level, "bootstrap confidence level")
        if self.method is not BootstrapMethod.PAIRED_QUERY:
            raise ValueError("primary inference requires paired query bootstrap")
        if self.resampling_unit != "query_id":
            raise ValueError("bootstrap resampling unit must be query_id")


@dataclass(frozen=True)
class MultiplicitySpec:
    family_name: str
    method: MultiplicityMethod
    members: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.family_name:
            raise ValueError("multiplicity family name must not be empty")
        _require_nonempty_unique(self.members, "multiplicity family members")


@dataclass(frozen=True)
class ComputeSpec:
    gpu_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_unique(self.gpu_ids, "gpu_ids")
        for gpu_id in self.gpu_ids:
            _require_nonnegative_int(gpu_id, "gpu id")
        forbidden = set(self.gpu_ids) - _ALLOWED_GPU_IDS
        if forbidden:
            names = ", ".join(map(str, sorted(forbidden)))
            raise ValueError("forbidden GPU allocation: " + names)


@dataclass(frozen=True)
class OutputRequirements:
    required_artifacts: Tuple[str, ...]
    immutable_run_directory: bool
    versioned_run_directory: bool
    atomic_manifest: bool

    def __post_init__(self) -> None:
        _require_nonempty_unique(self.required_artifacts, "required output artifacts")
        missing = REQUIRED_OUTPUT_ARTIFACTS - set(self.required_artifacts)
        if missing:
            raise ValueError("missing required output artifacts: " + ", ".join(sorted(missing)))
        if not (self.immutable_run_directory and self.versioned_run_directory and self.atomic_manifest):
            raise ValueError("outputs must be immutable, versioned, and atomically manifested")


@dataclass(frozen=True)
class ExperimentSpec:
    version: str
    primary_endpoints: Tuple[PrimaryEndpoint, ...]
    coverage: CoveragePolicy
    alpha_values: Tuple[float, ...]
    shot_setting: int
    router: RouterSpec
    pooling: PoolingSpec
    selector_arms: Tuple[SelectorArm, ...]
    dependence_estimators: Tuple[DependenceEstimatorSpec, ...]
    aggregation: AggregationSpec
    splits: SplitPolicy
    bootstrap: BootstrapSpec
    multiplicity: MultiplicitySpec
    fractional_thresholds: FractionalThresholdPolicy
    compute: ComputeSpec
    outputs: OutputRequirements

    def __post_init__(self) -> None:
        if self.version != "corrected_v1":
            raise ValueError("primary experiment version must be corrected_v1")
        _require_nonempty_unique(
            tuple(endpoint.name for endpoint in self.primary_endpoints), "primary endpoints"
        )
        _require_nonempty_unique(self.alpha_values, "alpha_values")
        for index, alpha in enumerate(self.alpha_values):
            _require_probability(alpha, "alpha_values[" + str(index) + "]")
        _require_nonnegative_int(self.shot_setting, "shot_setting")
        _require_nonempty_unique(tuple(arm.name for arm in self.selector_arms), "selector arm names")
        _require_nonempty_unique(
            tuple(item.name for item in self.dependence_estimators),
            "dependence estimator names",
        )
        estimator_names = {item.name for item in self.dependence_estimators}
        for arm in self.selector_arms:
            if arm.dependence_estimator not in estimator_names:
                raise ValueError(
                    "selector arm " + arm.name + " uses unregistered dependence estimator "
                    + arm.dependence_estimator
                )
        fractional_arms = {
            arm.name for arm in self.selector_arms if arm.uses_fractional_depth_threshold
        }
        registered_fractional = set(self.fractional_thresholds.registered_arms)
        if fractional_arms != registered_fractional:
            missing = sorted(fractional_arms - registered_fractional)
            stale = sorted(registered_fractional - fractional_arms)
            details = []
            if missing:
                details.append("silent fractional arms=" + ",".join(missing))
            if stale:
                details.append("stale fractional registrations=" + ",".join(stale))
            raise ValueError("fractional threshold registry mismatch: " + "; ".join(details))
        task_policies = {item.task for item in self.splits.fractions_by_task}
        if task_policies != set(self.coverage.all_tasks):
            raise ValueError("split policies must cover every preregistered task exactly once")
        arm_names = {arm.name for arm in self.selector_arms}
        fractional_unknown = registered_fractional - arm_names
        if fractional_unknown:
            raise ValueError("fractional threshold registry contains an unknown arm")

    def validate_requested_arms(self, arm_names: Iterable[str]) -> None:
        requested = tuple(arm_names)
        _require_nonempty_unique(requested, "requested arm names")
        registered = {arm.name for arm in self.selector_arms}
        unknown = set(requested) - registered
        if unknown:
            raise ValueError("unregistered experiment arm: " + ", ".join(sorted(unknown)))


def validate_disjoint_split_roles(
    assignments: Mapping[SplitRole, Iterable[str]]
) -> None:
    """Reject missing roles, duplicate IDs, and overlap between split roles."""

    normalized = {}
    supplied_roles = set(assignments)
    required_roles = set(_REQUIRED_SPLIT_ROLES)
    if supplied_roles != required_roles:
        missing = required_roles - supplied_roles
        extra = supplied_roles - required_roles
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(role.value for role in missing)))
        if extra:
            details.append("extra=" + ",".join(sorted(str(role) for role in extra)))
        raise ValueError("split assignments must contain exactly five roles: " + "; ".join(details))
    for role in _REQUIRED_SPLIT_ROLES:
        values = tuple(str(value) for value in assignments[role])
        if len(values) != len(set(values)):
            raise ValueError("duplicate query id within " + role.value + " split")
        normalized[role] = frozenset(values)
    for left, right in combinations(_REQUIRED_SPLIT_ROLES, 2):
        overlap = normalized[left] & normalized[right]
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                "split roles overlap: " + left.value + " and " + right.value
                + " share query " + example
            )


def _canonical_split_fractions() -> Tuple[SplitFractions, ...]:
    standard = (0.20, 0.20, 0.30, 0.06, 0.24)
    return (
        SplitFractions(Benchmark.MMLU, *standard),
        SplitFractions(Benchmark.WINOGRANDE, *standard),
        SplitFractions(Benchmark.ARC, 0.20, 0.20, 0.24, 0.12, 0.24),
        SplitFractions(Benchmark.HELLASWAG, *standard),
        SplitFractions(Benchmark.GSM8K, 0.25, 0.25, 0.25, 0.05, 0.20),
        SplitFractions(Benchmark.MBPP, *standard),
    )


def _primary_multiplicity_members() -> Tuple[str, ...]:
    return tuple(
        "residual_gaussian_dacr_vs_depth_independence:"
        + task.value
        + ":alpha="
        + format(alpha, ".2f")
        + ":mean_selected_standard_models"
        for task in (Benchmark.MMLU, Benchmark.WINOGRANDE)
        for alpha in (0.05, 0.10)
    )


def canonical_primary_spec() -> ExperimentSpec:
    """Return the canonical corrected_v1 primary preregistration."""

    estimators = (
        DependenceEstimatorSpec("none", "none", False),
        DependenceEstimatorSpec("independence", "factorized_marginals", True),
        DependenceEstimatorSpec(
            "query_clustered_cross_fitted_residual_correlation",
            "residual_gaussian_copula",
            True,
        ),
        DependenceEstimatorSpec("pooled_tetrachoric", "pooled_gaussian_copula", False),
        DependenceEstimatorSpec("empirical_all_miss", "empirical_joint", False),
        DependenceEstimatorSpec("shrinkage_empirical", "shrinkage_joint", False),
        DependenceEstimatorSpec("pairwise_double_fault", "pairwise_diversity", False),
        DependenceEstimatorSpec("pairwise_error_correlation", "pairwise_diversity", False),
        DependenceEstimatorSpec("mutual_information", "information_selector", False),
        DependenceEstimatorSpec("complementary_coverage", "coverage_selector", False),
    )
    arms = (
        SelectorArm("exact_racer_p", "racer_p", "none", exact_racer_variant="P"),
        SelectorArm("exact_racer_g", "racer_g", "none", exact_racer_variant="G"),
        SelectorArm("depth_independence", "marginal_depth", "independence"),
        SelectorArm(
            "residual_gaussian_dacr",
            "gaussian_copula_greedy",
            "query_clustered_cross_fitted_residual_correlation",
        ),
        SelectorArm(
            "pooled_gaussian_ablation",
            "gaussian_copula_greedy",
            "pooled_tetrachoric",
        ),
        SelectorArm(
            "empirical_all_miss_control",
            "empirical_all_miss_greedy",
            "empirical_all_miss",
        ),
        SelectorArm(
            "shrinkage_empirical_control",
            "shrinkage_empirical_greedy",
            "shrinkage_empirical",
        ),
        SelectorArm(
            "pairwise_double_fault_control",
            "pairwise_double_fault",
            "pairwise_double_fault",
        ),
        SelectorArm(
            "pairwise_error_correlation_control",
            "pairwise_error_correlation",
            "pairwise_error_correlation",
        ),
        SelectorArm("mi_style_control", "mi_style_greedy", "mutual_information"),
        SelectorArm(
            "complementary_coverage_control",
            "complementary_incremental_coverage",
            "complementary_coverage",
        ),
    )
    return ExperimentSpec(
        version="corrected_v1",
        primary_endpoints=(
            PrimaryEndpoint(
                "mean_selected_standard_models",
                EndpointDirection.MINIMIZE,
                EndpointRole.EFFICACY,
                "models_per_query",
            ),
            PrimaryEndpoint(
                "all_models_miss_rate",
                EndpointDirection.CONTROL_AT_ALPHA,
                EndpointRole.VALIDITY,
                "query_fraction",
            ),
            PrimaryEndpoint(
                "official_task_score",
                EndpointDirection.MAXIMIZE,
                EndpointRole.SAFETY,
                "task_specific_score",
            ),
        ),
        coverage=CoveragePolicy(
            primary_headroom_tasks=(Benchmark.MMLU, Benchmark.WINOGRANDE),
            secondary_tasks=(
                Benchmark.ARC,
                Benchmark.HELLASWAG,
                Benchmark.GSM8K,
                Benchmark.MBPP,
            ),
        ),
        alpha_values=(0.05, 0.10),
        shot_setting=0,
        router=RouterSpec(
            encoder="microsoft/mdeberta-v3-base",
            candidates=(
                RouterCandidate(RouterKind.KNNR, k_neighbors=40),
                RouterCandidate(RouterKind.MLPR, hidden_units=(256,)),
            ),
            include_null_model=True,
            randomized_smoothing=True,
            smoothing_epsilon=1e-6,
            selection_split=SplitRole.VALIDATION,
        ),
        pooling=PoolingSpec(
            candidates=(PoolingMethod.FIRST_TOKEN, PoolingMethod.ATTENTION_MASK_MEAN),
            selection_split=SplitRole.VALIDATION,
            freeze_after_selection=True,
        ),
        selector_arms=arms,
        dependence_estimators=estimators,
        aggregation=AggregationSpec(
            candidates=tuple(AggregationMethod),
            selection_split=SplitRole.VALIDATION,
            freeze_after_selection=True,
        ),
        splits=SplitPolicy(seed=20260903, fractions_by_task=_canonical_split_fractions()),
        bootstrap=BootstrapSpec(
            method=BootstrapMethod.PAIRED_QUERY,
            seed=20260904,
            repetitions=10_000,
            confidence_level=0.95,
            resampling_unit="query_id",
        ),
        multiplicity=MultiplicitySpec(
            family_name="primary_dacr_headroom",
            method=MultiplicityMethod.HOLM,
            members=_primary_multiplicity_members(),
        ),
        fractional_thresholds=FractionalThresholdPolicy(enabled=False),
        compute=ComputeSpec(gpu_ids=(2, 3)),
        outputs=OutputRequirements(
            required_artifacts=tuple(sorted(REQUIRED_OUTPUT_ARTIFACTS)),
            immutable_run_directory=True,
            versioned_run_directory=True,
            atomic_manifest=True,
        ),
    )


PRIMARY_EXPERIMENT_SPEC = canonical_primary_spec()
PRIMARY_SPEC = PRIMARY_EXPERIMENT_SPEC


__all__ = [
    "AggregationMethod",
    "AggregationSpec",
    "Benchmark",
    "BootstrapMethod",
    "BootstrapSpec",
    "ComputeSpec",
    "CoveragePolicy",
    "DependenceEstimatorSpec",
    "EndpointDirection",
    "EndpointRole",
    "ExperimentSpec",
    "FractionalThresholdPolicy",
    "MultiplicityMethod",
    "MultiplicitySpec",
    "OutputRequirements",
    "PoolingMethod",
    "PoolingSpec",
    "PrimaryEndpoint",
    "PRIMARY_EXPERIMENT_SPEC",
    "PRIMARY_SPEC",
    "REGISTERED_SELECTOR_IMPLEMENTATIONS",
    "REQUIRED_OUTPUT_ARTIFACTS",
    "RouterCandidate",
    "RouterKind",
    "RouterSpec",
    "SelectorArm",
    "SplitFractions",
    "SplitPolicy",
    "SplitRole",
    "canonical_primary_spec",
    "validate_disjoint_split_roles",
]
