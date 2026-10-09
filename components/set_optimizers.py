"""Set-order optimizers for the ISSUE-65 min-sum ordering ablation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from components.scale_selectors import _as_binary, empirical_tally_order


def _failure_bitsets(design: np.ndarray) -> tuple[list[int], int]:
    labels = _as_binary(design)
    n_rows, n_models = labels.shape
    bits = []
    for model in range(n_models):
        value = 0
        for row in np.flatnonzero(~labels[:, model]):
            value |= 1 << int(row)
        bits.append(value)
    return bits, (1 << n_rows) - 1



def _popcount(value: int) -> int:
    """Count set bits without requiring Python 3.8 int.bit_count."""

    return bin(int(value)).count("1")

def _objective_from_bitsets(failures: Sequence[int], all_rows: int, order: Sequence[int]) -> int:
    alive = int(all_rows)
    total = 0
    for model in order:
        alive &= failures[int(model)]
        total += _popcount(alive)
    return int(total)


def min_sum_objective(design: np.ndarray, order: Sequence[int]) -> int:
    """Sum design all-miss counts over every prefix in an order."""

    labels = _as_binary(design)
    permutation = tuple(int(index) for index in order)
    if len(permutation) != labels.shape[1] or set(permutation) != set(range(labels.shape[1])):
        raise ValueError("order must be a full-pool permutation")
    failures, all_rows = _failure_bitsets(labels)
    return _objective_from_bitsets(failures, all_rows, permutation)


def prefix_miss_counts(design: np.ndarray, order: Sequence[int]) -> tuple[int, ...]:
    """Return all-miss counts for every fixed prefix."""

    labels = _as_binary(design)
    permutation = tuple(int(index) for index in order)
    failures, alive = _failure_bitsets(labels)
    output = []
    for model in permutation:
        alive &= failures[model]
        output.append(int(_popcount(alive)))
    return tuple(output)


@dataclass(frozen=True)
class ExactOptimization:
    order: tuple[int, ...]
    objective: int
    greedy_order: tuple[int, ...]
    greedy_objective: int
    exact_depth_misses: tuple[int, ...]
    exact_depth_subsets: tuple[tuple[int, ...], ...]
    state_count: int
    elapsed_seconds: float


def exact_min_sum_order(design: np.ndarray, *, maximum_models: int = 20) -> ExactOptimization:
    """Solve the optimal nested min-sum permutation by subset dynamic programming."""

    labels = _as_binary(design)
    n_models = labels.shape[1]
    if n_models > maximum_models:
        raise ValueError(f"exact optimization is capped at {maximum_models} models")
    started = perf_counter()
    failures, all_rows = _failure_bitsets(labels)
    size = 1 << n_models
    alive = [0] * size
    miss = [0] * size
    alive[0] = all_rows
    exact_depth_misses = [labels.shape[0] + 1] * n_models
    exact_depth_subsets: list[tuple[int, ...] | None] = [None] * n_models
    for mask in range(1, size):
        bit = mask & -mask
        model = bit.bit_length() - 1
        previous = mask ^ bit
        alive[mask] = alive[previous] & failures[model]
        miss[mask] = _popcount(alive[mask])
        depth = _popcount(mask) - 1
        subset = tuple(index for index in range(n_models) if mask & (1 << index))
        if miss[mask] < exact_depth_misses[depth] or (
            miss[mask] == exact_depth_misses[depth]
            and (exact_depth_subsets[depth] is None or subset < exact_depth_subsets[depth])
        ):
            exact_depth_misses[depth] = miss[mask]
            exact_depth_subsets[depth] = subset

    infinity = sum(miss) + labels.shape[0] * n_models + 1
    cost = [infinity] * size
    sequence: list[tuple[int, ...] | None] = [None] * size
    cost[0] = 0
    sequence[0] = ()
    for mask in range(1, size):
        candidates = []
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            model = bit.bit_length() - 1
            previous = mask ^ bit
            candidates.append((cost[previous] + miss[mask], sequence[previous] + (model,)))
            remaining ^= bit
        cost[mask], sequence[mask] = min(candidates, key=lambda item: (item[0], item[1]))
    order = sequence[-1]
    if order is None:
        raise RuntimeError("exact dynamic program produced no order")
    greedy = empirical_tally_order(labels, max_depth=n_models)
    return ExactOptimization(
        order=order,
        objective=int(cost[-1]),
        greedy_order=greedy,
        greedy_objective=min_sum_objective(labels, greedy),
        exact_depth_misses=tuple(int(value) for value in exact_depth_misses),
        exact_depth_subsets=tuple(value or () for value in exact_depth_subsets),
        state_count=size,
        elapsed_seconds=float(perf_counter() - started),
    )


def beam_min_sum_order(
    design: np.ndarray,
    *,
    beam_width: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Approximate the nested min-sum order with deterministic subset beam search."""

    labels = _as_binary(design)
    if beam_width < 1:
        raise ValueError("beam width must be positive")
    started = perf_counter()
    n_models = labels.shape[1]
    failures, all_rows = _failure_bitsets(labels)
    accuracy = labels.mean(axis=0)
    beam = [(0, (), 0, all_rows)]
    expanded = 0
    peak = 1
    for _depth in range(n_models):
        by_mask: dict[int, tuple[int, tuple[int, ...], int, int]] = {}
        for cost, order, mask, alive in beam:
            for model in range(n_models):
                bit = 1 << model
                if mask & bit:
                    continue
                new_alive = alive & failures[model]
                candidate = (cost + _popcount(new_alive), order + (model,), mask | bit, new_alive)
                existing = by_mask.get(mask | bit)
                if existing is None or (candidate[0], candidate[1]) < (existing[0], existing[1]):
                    by_mask[mask | bit] = candidate
                expanded += 1
        beam = sorted(
            by_mask.values(),
            key=lambda item: (
                item[0], _popcount(item[3]),
                tuple((-float(accuracy[index]), index) for index in item[1]),
            ),
        )[: int(beam_width)]
        peak = max(peak, len(beam))
        if beam and all(state[3] == 0 for state in beam):
            break
    best = min(beam, key=lambda item: (item[0], item[1]))
    order = list(best[1])
    remaining = [index for index in range(n_models) if index not in set(order)]
    remaining.sort(key=lambda index: (-float(accuracy[index]), index))
    order.extend(remaining)
    result = tuple(order)
    return result, {
        "optimizer": "beam_min_sum",
        "beam_width": int(beam_width),
        "expanded_states": int(expanded),
        "peak_beam_size": int(peak),
        "optimized_prefix_depth": int(len(best[1])),
        "objective": min_sum_objective(labels, result),
        "elapsed_seconds": float(perf_counter() - started),
    }


def local_swap_min_sum_order(
    design: np.ndarray,
    *,
    maximum_positions: int,
    passes: int = 2,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Refine greedy tally with deterministic pair swaps in its active prefix."""

    labels = _as_binary(design)
    if maximum_positions < 2 or passes < 1:
        raise ValueError("local-swap limits must be positive")
    started = perf_counter()
    failures, all_rows = _failure_bitsets(labels)
    order = list(empirical_tally_order(labels, max_depth=labels.shape[1]))
    initial = _objective_from_bitsets(failures, all_rows, order)
    current = initial
    evaluated = accepted = 0
    search_depth = min(int(maximum_positions), len(order))
    for _ in range(int(passes)):
        best_objective = current
        best_pair = None
        for left in range(search_depth - 1):
            for right in range(left + 1, search_depth):
                candidate = order.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                objective = _objective_from_bitsets(failures, all_rows, candidate)
                evaluated += 1
                if objective < best_objective or (
                    objective == best_objective and best_pair is not None and (left, right) < best_pair
                ):
                    best_objective = objective
                    best_pair = (left, right)
        if best_pair is None or best_objective >= current:
            break
        left, right = best_pair
        order[left], order[right] = order[right], order[left]
        current = best_objective
        accepted += 1
    return tuple(order), {
        "optimizer": "local_swap_min_sum",
        "maximum_positions": int(maximum_positions),
        "passes": int(passes),
        "evaluated_swaps": int(evaluated),
        "accepted_swaps": int(accepted),
        "initial_objective": int(initial),
        "objective": int(current),
        "objective_improvement": int(initial - current),
        "elapsed_seconds": float(perf_counter() - started),
    }


__all__ = [
    "ExactOptimization",
    "beam_min_sum_order",
    "exact_min_sum_order",
    "local_swap_min_sum_order",
    "min_sum_objective",
    "prefix_miss_counts",
]
