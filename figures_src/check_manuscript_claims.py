"""Check displayed manuscript claims against authoritative generated artifacts.

The checker applies the manuscript's presentation rounding, validates every
claim-bearing table supplied on the command line, and emits sorted diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Iterable


PRIMARY_ALPHA = Decimal("0.05")
LOOSE_ALPHA = Decimal("0.10")
TIGHT_ALPHA = Decimal("0.02")
POOLS = (100, 200, 300, 500)
QUERY_METHODS = (
    "racer_g",
    "exact_independence",
    "fixed_tally_factorized",
    "local_empirical",
)
FACTORIAL_METHODS = (
    ("racer_g", "RACER-G"),
    ("racer_p", "RACER-P"),
    ("exact_independence", "Exact independence"),
    (
        "fixed_design_marginal_factorized",
        "Fixed design marginal + factorized sizing",
    ),
    ("fixed_tally_factorized", "Fixed tally + factorized sizing"),
    ("local_empirical", "Local ordering + local empirical sizing"),
    ("local_order_factorized", "Local ordering + factorized sizing"),
)

REQUIRED_QUERY_COLUMNS = {
    "pool_size",
    "target_alpha",
    "method",
    "mean_calls",
    "standard_all_miss_risk_mean",
    "certification_rate",
}
REQUIRED_E3_COLUMNS = {
    "pool_size",
    "target_alpha",
    "redraw_count",
    "selected_mean_calls",
    "selected_standard_all_miss_risk_mean",
    "selected_certification_count",
    "selected_certification_rate",
    "racer_g_mean_calls",
    "racer_g_standard_all_miss_risk_mean",
    "racer_g_certification_count",
    "racer_g_certification_rate",
    "class_win_counts",
    "class_penalty_enabled",
}
REQUIRED_TAXONOMY_COLUMNS = {
    "target_alpha",
    "method_count",
    "risk_bar_pass_count",
    "information_destruction_sensitivity_pass_count",
}
REQUIRED_FACTORIAL_COLUMNS = REQUIRED_QUERY_COLUMNS | {
    "method_label",
    "diagnostic_query_count",
    "distinct_finite_score_count_mean",
    "largest_tied_block_mean",
    "score_one_tail_size_mean",
    "full_pool_call_rate",
    "factorized_underflow_count",
    "factorized_saturation_count",
}
REQUIRED_FIXED_DEPTH_COLUMNS = {
    "availability_status",
    "method",
    "target_task",
    "pool_size",
    "depth",
    "depth_cap",
    "success_mean",
    "success_se",
}
REQUIRED_CALIBRATED_COLUMNS = {
    "availability_status",
    "method",
    "target_task",
    "pool_size",
    "target_alpha",
    "mean_calls",
    "standard_all_miss_risk_mean",
    "certification_rate",
    "certification_count",
    "certification_denominator",
}
REQUIRED_NATIVE_FLEX_COLUMNS = {
    "availability_status",
    "method",
    "target_task",
    "pool_size",
    "tau",
    "depth_cap",
    "success_mean",
    "mean_calls",
    "risk_certificate_claimed",
}
REQUIRED_NATIVE_BARP_COLUMNS = {
    "availability_status",
    "method",
    "target_task",
    "w_quality",
    "w_cost",
    "mean_official_quality",
    "mean_cost",
    "training_seed_count",
}


def _decimal(value: Any, context: str) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except Exception as error:
        raise ValueError(f"{context}: expected a decimal, found {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"{context}: expected a finite decimal, found {value!r}")
    return number


def _optional_decimal(value: Any, context: str) -> Decimal | None:
    if str(value).strip() == "":
        return None
    return _decimal(value, context)


def _integer(value: Any, context: str) -> int:
    number = _decimal(value, context)
    if number != number.to_integral_value():
        raise ValueError(f"{context}: expected an integer, found {value!r}")
    return int(number)


def _boolean(value: Any, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{context}: expected a boolean, found {value!r}")


def _read_csv(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path}: missing columns {missing!r}")
        return list(reader)


def _indexed_rows(
    path: str | Path,
    required: set[str],
    key: Callable[[dict[str, str], str], tuple[Any, ...] | None],
) -> dict[tuple[Any, ...], dict[str, str]]:
    indexed: dict[tuple[Any, ...], dict[str, str]] = {}
    for index, row in enumerate(_read_csv(path, required), start=2):
        context = f"{path}: row {index}"
        row_key = key(row, context)
        if row_key is None:
            continue
        if row_key in indexed:
            raise ValueError(f"{context}: duplicate artifact row {row_key!r}")
        indexed[row_key] = row
    return indexed


def _fixed(value: Decimal, places: int) -> str:
    quantum = Decimal(1).scaleb(-places)
    return f"{value.quantize(quantum, rounding=ROUND_HALF_UP):.{places}f}"



def _learned_fixed(value: Decimal, places: int) -> str:
    """Format learned artifacts exactly as their float-based table generator does."""
    return f"{float(value):.{places}f}"

def _calls(value: Decimal) -> str:
    return _fixed(value, 1)


def _risk(value: Decimal, leading_zero: bool = True) -> str:
    displayed = _fixed(value, 3)
    if not leading_zero and displayed.startswith("0."):
        return displayed[1:]
    return displayed


def _percent(value: Decimal) -> int:
    tenths = (value * Decimal(100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return int(tenths.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _percent_one(value: Decimal) -> str:
    return f"{_fixed(value * Decimal(100), 1)}%"


def _reduction(reference: Decimal, candidate: Decimal) -> int:
    if reference <= 0:
        raise ValueError("call-reduction reference must be positive")
    return _percent((reference - candidate) / reference)


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def _plain_tex(tex: str) -> str:
    text = _strip_comments(tex)
    text = text.replace("\\alpha", " alpha ").replace("\\%", " percent ")
    text = text.replace("\\texttt", " ")
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?", " ", text)
    text = text.replace(chr(36), " ").replace("{", " ").replace("}", " ")
    text = text.replace("~", " ")
    text = re.sub(r"-\s+", "-", text)
    text = re.sub(r"\s*=\s*", "=", text)
    text = re.sub(r"\s+([,;:])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_cell(cell: str) -> str:
    value = cell.strip()
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"\\(?:textbf|emph)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\%", "%").replace("\\pm", "±")
    value = value.replace(chr(36), "").replace("\\(", "").replace("\\)", "")
    value = value.replace("{", "").replace("}", "").replace("~", " ")
    return re.sub(r"\s+", " ", value).strip()


def _extract_table_rows(tex: str, label: str) -> list[list[str]]:
    tables = re.findall(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, flags=re.DOTALL
    )
    matching = [table for table in tables if f"\\label{{{label}}}" in table]
    if len(matching) != 1:
        raise ValueError(f"manuscript must contain exactly one table labeled {label}")
    table = matching[0]
    begin = re.search(r"\\begin\{tabular\}\{", table)
    if begin is None:
        raise ValueError(f"{label} has no parseable tabular environment")
    position = begin.end()
    depth = 1
    while position < len(table) and depth:
        if table[position] == "{":
            depth += 1
        elif table[position] == "}":
            depth -= 1
        position += 1
    end = table.find(r"\end{tabular}", position)
    if depth or end < 0:
        raise ValueError(f"{label} has an unterminated tabular environment")
    rows: list[list[str]] = []
    for line in table[position:end].splitlines():
        stripped = line.strip()
        if "&" not in stripped or not stripped.endswith(r"\\"):
            continue
        rows.append([_normalize_cell(cell) for cell in stripped[:-2].split("&")])
    return rows


def _check_table(
    tex: str,
    label: str,
    expected: list[list[str]],
    include: Callable[[list[str]], bool],
) -> list[str]:
    actual = [row for row in _extract_table_rows(tex, label) if include(row)]
    if actual == expected:
        return []
    for index, (wanted, found) in enumerate(zip(expected, actual), start=1):
        if wanted != found:
            return [
                f"{label} row {index} mismatch: expected {wanted!r}, found {found!r}"
            ]
    return [
        f"{label} row count mismatch: expected {len(expected)}, found {len(actual)}"
    ]


def _require_pattern(
    text: str, pattern: str, description: str, errors: list[str]
) -> None:
    if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is None:
        errors.append(f"missing or stale manuscript claim: {description}")


def _joined(values: Iterable[int]) -> str:
    items = [str(value) for value in values]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _load_query(path: str | Path) -> dict[tuple[int, Decimal, str], dict[str, str]]:
    return _indexed_rows(
        path,
        REQUIRED_QUERY_COLUMNS,
        lambda row, context: (
            _integer(row["pool_size"], f"{context} pool_size"),
            _decimal(row["target_alpha"], f"{context} target_alpha"),
            row["method"].strip(),
        ),
    )


def _query_value(
    cells: dict[tuple[int, Decimal, str], dict[str, str]],
    pool: int,
    alpha: Decimal,
    method: str,
    column: str,
) -> Decimal:
    key = (pool, alpha, method)
    if key not in cells:
        raise ValueError(f"query table is missing pool={pool}, alpha={alpha}, method={method}")
    return _decimal(cells[key][column], f"query table {key} {column}")


def _check_query_claims(
    experiments_plain: str,
    abstract_plain: str,
    discussion_plain: str,
    cells: dict[tuple[int, Decimal, str], dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    for alpha in (TIGHT_ALPHA, PRIMARY_ALPHA, LOOSE_ALPHA):
        for pool in POOLS:
            rates = {
                _query_value(cells, pool, alpha, method, "certification_rate")
                for method in QUERY_METHODS
            }
            if len(rates) != 1:
                raise ValueError(
                    f"query table pool={pool}, alpha={alpha}: primary methods do not share certification"
                )

    reductions: dict[Decimal, tuple[list[int], list[int]]] = {}
    for alpha in (PRIMARY_ALPHA, LOOSE_ALPHA):
        versus_racer: list[int] = []
        versus_exact: list[int] = []
        for pool in POOLS:
            fixed = _query_value(cells, pool, alpha, "fixed_tally_factorized", "mean_calls")
            versus_racer.append(
                _reduction(_query_value(cells, pool, alpha, "racer_g", "mean_calls"), fixed)
            )
            versus_exact.append(
                _reduction(
                    _query_value(cells, pool, alpha, "exact_independence", "mean_calls"),
                    fixed,
                )
            )
        reductions[alpha] = versus_racer, versus_exact

    primary_racer, primary_exact = reductions[PRIMARY_ALPHA]
    for text, section in ((abstract_plain, "abstract"),):
        _require_pattern(
            text,
            rf"{min(primary_racer)}\s+to\s+{max(primary_racer)}\s+percent\s+versus\s+RACER-G",
            f"{section} alpha 0.05 RACER-G reduction range",
            errors,
        )
        _require_pattern(
            text,
            rf"{min(primary_exact)}\s+to\s+{max(primary_exact)}\s+percent\s+(?:beyond|versus).*?exact[- ]independence",
            f"{section} alpha 0.05 exact-independence reduction range",
            errors,
        )

    loose_racer, loose_exact = reductions[LOOSE_ALPHA]
    _require_pattern(
        experiments_plain,
        rf"{min(loose_racer)}\s+to\s+{max(loose_racer)}\s+percent\s+versus\s+RACER-G.*?{min(loose_exact)}\s+to\s+{max(loose_exact)}\s+percent\s+versus\s+exact independence",
        "alpha 0.10 call-reduction ranges",
        errors,
    )
    tight_rates = [
        _percent(_query_value(cells, pool, TIGHT_ALPHA, "racer_g", "certification_rate"))
        for pool in POOLS
    ]
    _require_pattern(
        experiments_plain,
        rf"certification is\s+{re.escape(_joined(tight_rates))}\s+percent",
        "alpha 0.02 per-pool certification rates",
        errors,
    )
    _require_pattern(
        discussion_plain,
        rf"certification rates of\s+{min(tight_rates)}\s+to\s+{max(tight_rates)}\s+percent",
        "discussion alpha 0.02 certification range",
        errors,
    )
    return errors


def _load_e3(path: str | Path) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(_read_csv(path, REQUIRED_E3_COLUMNS), start=2):
        context = f"{path}: row {index}"
        if _decimal(row["target_alpha"], f"{context} target_alpha") != PRIMARY_ALPHA:
            continue
        if _boolean(row["class_penalty_enabled"], f"{context} class_penalty_enabled"):
            continue
        pool = _integer(row["pool_size"], f"{context} pool_size")
        if pool in selected:
            raise ValueError(f"{context}: duplicate unpenalized E3 row for pool {pool}")
        try:
            class_counts = json.loads(row["class_win_counts"])
        except json.JSONDecodeError as error:
            raise ValueError(f"{context}: invalid class_win_counts JSON") from error
        if not isinstance(class_counts, dict):
            raise ValueError(f"{context}: class_win_counts must be an object")
        selected[pool] = {**row, "_class_counts": class_counts}
    if set(selected) != {100, 200}:
        raise ValueError(
            f"{path}: unpenalized E3 alpha 0.05 pools must be [100, 200], found {sorted(selected)!r}"
        )
    return selected


def _check_e3(experiments_tex: str, appendix_tex: str, cells: dict[int, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_table: list[list[str]] = []
    expected_picks: dict[int, tuple[int, int]] = {}
    for pool in (100, 200):
        row = cells[pool]
        redraws = _integer(row["redraw_count"], f"E3 pool {pool} redraw_count")
        selected_count = _integer(
            row["selected_certification_count"], f"E3 pool {pool} selected count"
        )
        racer_count = _integer(
            row["racer_g_certification_count"], f"E3 pool {pool} RACER count"
        )
        selected_rate = _decimal(
            row["selected_certification_rate"], f"E3 pool {pool} selected rate"
        )
        racer_rate = _decimal(row["racer_g_certification_rate"], f"E3 pool {pool} RACER rate")
        if selected_rate != Decimal(selected_count) / Decimal(redraws):
            errors.append(f"E3 pool {pool}: selected certification count and rate disagree")
        if racer_rate != Decimal(racer_count) / Decimal(redraws):
            errors.append(f"E3 pool {pool}: RACER-G certification count and rate disagree")
        if selected_count != redraws or racer_count != redraws:
            errors.append(f"E3 pool {pool}: both policies must certify all redraws")

        selected_calls = _decimal(row["selected_mean_calls"], f"E3 pool {pool} selected calls")
        selected_risk = _decimal(
            row["selected_standard_all_miss_risk_mean"], f"E3 pool {pool} selected risk"
        )
        racer_calls = _decimal(row["racer_g_mean_calls"], f"E3 pool {pool} RACER calls")
        racer_risk = _decimal(
            row["racer_g_standard_all_miss_risk_mean"], f"E3 pool {pool} RACER risk"
        )
        change = _percent(abs(selected_calls - racer_calls) / racer_calls)
        direction = "more" if selected_calls > racer_calls else "fewer"
        expected_table.append(
            [
                str(pool),
                f"{_calls(selected_calls)} ({_risk(selected_risk, False)})",
                f"{_calls(racer_calls)} ({_risk(racer_risk, False)})",
                f"{change}% {direction}",
            ]
        )
        counts = row["_class_counts"]
        try:
            fixed = _integer(counts["fixed_prefix"], f"E3 pool {pool} fixed_prefix")
            wrapper = _integer(
                counts["per_query_wrapper"], f"E3 pool {pool} per_query_wrapper"
            )
        except KeyError as error:
            raise ValueError(f"E3 pool {pool}: missing class pick {error}") from error
        if fixed + wrapper != redraws:
            errors.append(f"E3 pool {pool}: class pick counts do not sum to redraw count")
        expected_picks[pool] = (fixed, wrapper)

    errors.extend(
        _check_table(
            appendix_tex,
            "tab:e3-calibrated",
            expected_table,
            lambda row: bool(row and re.fullmatch(r"(?:100|200)", row[0])),
        )
    )
    plain_exp = _plain_tex(experiments_tex)
    p100 = cells[100]
    p200 = cells[200]
    _require_pattern(
        plain_exp,
        rf"unpenalized selected policy.*?pool 100.*?{re.escape(_calls(_decimal(p100['selected_mean_calls'], 'E3')))} calls versus {re.escape(_calls(_decimal(p100['racer_g_mean_calls'], 'E3')))}.*?pool 200.*?{re.escape(_calls(_decimal(p200['selected_mean_calls'], 'E3')))} calls versus {re.escape(_calls(_decimal(p200['racer_g_mean_calls'], 'E3')))}",
        "unpenalized E3 calls versus RACER-G",
        errors,
    )
    appendix_plain = _plain_tex(appendix_tex)
    fixed_100, wrapper_100 = expected_picks[100]
    fixed_200, wrapper_200 = expected_picks[200]
    _require_pattern(
        appendix_plain,
        rf"fixed-prefix family in {fixed_100} of 20 redraws and a per-query wrapper in {wrapper_100} of 20 at pool 100.*?At pool 200, the corresponding counts are {fixed_200} and {wrapper_200}",
        "unpenalized E3 class pick counts",
        errors,
    )
    return errors


def _load_factorial(path: str | Path) -> dict[tuple[int, str], dict[str, str]]:
    wanted = {method for method, _ in FACTORIAL_METHODS}
    rows = _indexed_rows(
        path,
        REQUIRED_FACTORIAL_COLUMNS,
        lambda row, context: (
            (
                _integer(row["pool_size"], f"{context} pool_size"),
                row["method"].strip(),
            )
            if _decimal(row["target_alpha"], f"{context} target_alpha") == PRIMARY_ALPHA
            and row["method"].strip() in wanted
            else None
        ),
    )
    expected = {(pool, method) for pool in POOLS for method in wanted}
    if set(rows) != expected:
        raise ValueError(
            f"{path}: factorial alpha 0.05 keys mismatch; missing {sorted(expected - set(rows))!r}, extra {sorted(set(rows) - expected)!r}"
        )
    return rows


def _factorial_value(
    rows: dict[tuple[int, str], dict[str, str]], pool: int, method: str, column: str
) -> Decimal:
    return _decimal(rows[(pool, method)][column], f"factorial pool {pool} {method} {column}")


def _check_factorial(experiments_tex: str, rows: dict[tuple[int, str], dict[str, str]]) -> list[str]:
    errors: list[str] = []
    expected_main: list[list[str]] = []
    for method, label in FACTORIAL_METHODS:
        expected_main.append(
            [
                label,
                *[
                    f"{_calls(_factorial_value(rows, pool, method, 'mean_calls'))} ({_risk(_factorial_value(rows, pool, method, 'standard_all_miss_risk_mean'))})"
                    for pool in POOLS
                ],
            ]
        )
    certifications = [
        _factorial_value(rows, pool, "racer_g", "certification_rate") for pool in POOLS
    ]
    for pool, rate in zip(POOLS, certifications):
        method_rates = {
            _factorial_value(rows, pool, method, "certification_rate")
            for method, _ in FACTORIAL_METHODS
        }
        if method_rates != {rate}:
            errors.append(f"factorial pool {pool}: methods do not share certification rate")
    expected_main.append(
        ["Certification rate", *[f"{_percent(rate)}%" for rate in certifications]]
    )
    labels = {label for _, label in FACTORIAL_METHODS} | {"Certification rate"}
    errors.extend(
        _check_table(
            experiments_tex,
            "tab:query-factorial",
            expected_main,
            lambda row: bool(row and row[0] in labels),
        )
    )

    expected_mechanism: list[list[str]] = []
    sizing_gains: list[int] = []
    tally_gains: list[int] = []
    marginal_gains: list[int] = []
    for pool in POOLS:
        local = _factorial_value(rows, pool, "local_empirical", "mean_calls")
        local_factorized = _factorial_value(
            rows, pool, "local_order_factorized", "mean_calls"
        )
        tally = _factorial_value(rows, pool, "fixed_tally_factorized", "mean_calls")
        marginal = _factorial_value(
            rows, pool, "fixed_design_marginal_factorized", "mean_calls"
        )
        sizing = _reduction(local, local_factorized)
        tally_gain = _reduction(local_factorized, tally)
        marginal_gain = _reduction(marginal, tally)
        sizing_gains.append(sizing)
        tally_gains.append(tally_gain)
        marginal_gains.append(marginal_gain)
        expected_mechanism.append(
            [
                str(pool),
                _calls(local),
                _calls(local_factorized),
                f"{sizing}%",
                _calls(tally),
                f"{tally_gain}%",
                _calls(marginal),
                f"{marginal_gain}%",
            ]
        )
    errors.extend(
        _check_table(
            experiments_tex,
            "tab:mechanism-decomposition",
            expected_mechanism,
            lambda row: bool(row and row[0] in {str(pool) for pool in POOLS}),
        )
    )

    expected_saturation: list[list[str]] = []
    for pool in POOLS:
        local = rows[(pool, "local_empirical")]
        factorized = rows[(pool, "local_order_factorized")]
        underflow = _integer(
            factorized["factorized_underflow_count"],
            f"factorial pool {pool} factorized_underflow_count",
        )
        if underflow != 0:
            errors.append(f"factorial pool {pool}: factorized underflow count is {underflow}")
        saturation = _integer(
            factorized["factorized_saturation_count"],
            f"factorial pool {pool} factorized_saturation_count",
        )
        if saturation < 0:
            errors.append(f"factorial pool {pool}: negative factorized saturation count")
        expected_saturation.append(
            [
                str(pool),
                _fixed(_decimal(local["distinct_finite_score_count_mean"], "distinct"), 1),
                _fixed(_decimal(local["largest_tied_block_mean"], "largest tie"), 1),
                _fixed(_decimal(local["score_one_tail_size_mean"], "score-one tail"), 1),
                _percent_one(_decimal(local["full_pool_call_rate"], "local full pool")),
                _percent_one(
                    _decimal(factorized["full_pool_call_rate"], "factorized full pool")
                ),
            ]
        )
    errors.extend(
        _check_table(
            experiments_tex,
            "tab:local-saturation",
            expected_saturation,
            lambda row: bool(row and row[0] in {str(pool) for pool in POOLS}),
        )
    )

    plain = _plain_tex(experiments_tex)
    _require_pattern(
        plain,
        rf"reduces calls by {re.escape(_joined(sizing_gains))} percent",
        "factorial sizing-gain percentages",
        errors,
    )
    _require_pattern(
        plain,
        rf"better than local ordering.*?by {re.escape(_joined(tally_gains))} percent",
        "factorial tally-gain percentages",
        errors,
    )
    _require_pattern(
        plain,
        rf"fixed tally reduces its calls by {re.escape(_joined(marginal_gains))} percent",
        "factorial fixed-marginal control percentages",
        errors,
    )
    return errors


def _load_fixed_depth(path: str | Path) -> dict[tuple[str, int, int, str], dict[str, str]]:
    rows = _indexed_rows(
        path,
        REQUIRED_FIXED_DEPTH_COLUMNS,
        lambda row, context: (
            (
                row["target_task"].strip(),
                _integer(row["pool_size"], f"{context} pool_size"),
                _integer(row["depth"], f"{context} depth"),
                row["method"].strip(),
            )
            if row["availability_status"].strip() == "available"
            and row["method"].strip() in {"flexrouter", "barp", "fixed_tally"}
            else None
        ),
    )
    for task in ("ifeval", "musr"):
        for pool in POOLS:
            for depth in range(1, 11):
                for method in ("flexrouter", "barp"):
                    key = (task, pool, depth, method)
                    if key not in rows:
                        raise ValueError(f"{path}: missing learned fixed-depth row {key!r}")
                    cap = _integer(rows[key]["depth_cap"], f"{path}: {key} depth_cap")
                    if cap != 10:
                        raise ValueError(f"{path}: {key} depth cap must be 10")
    return rows


def _success_cell(row: dict[str, str]) -> str:
    mean = _decimal(row["success_mean"], "success_mean")
    se = _optional_decimal(row["success_se"], "success_se")
    return (
        _learned_fixed(mean, 3)
        if se is None
        else f"{_learned_fixed(mean, 3)} ± {_learned_fixed(se, 3)}"
    )


def _check_fixed_depth(
    experiments_tex: str,
    appendix_tex: str,
    rows: dict[tuple[str, int, int, str], dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    expected_main: list[list[str]] = []
    for task in ("ifeval", "musr"):
        for pool in POOLS:
            fixed = rows.get((task, pool, 10, "fixed_tally"))
            expected_main.append(
                [
                    task.upper(),
                    str(pool),
                    _success_cell(rows[(task, pool, 10, "flexrouter")]),
                    _success_cell(rows[(task, pool, 10, "barp")]),
                    _fixed(_decimal(fixed["success_mean"], "fixed tally success"), 3)
                    if fixed is not None
                    else "not available",
                ]
            )
    errors.extend(
        _check_table(
            experiments_tex,
            "tab:learned-depth10",
            expected_main,
            lambda row: bool(row and row[0] in {"IFEVAL", "MUSR"}),
        )
    )

    table_specs = (
        ("ifeval", "flexrouter", "tab:flex-ifeval-depth"),
        ("ifeval", "barp", "tab:barp-ifeval-depth"),
        ("musr", "flexrouter", "tab:flex-musr-depth"),
        ("musr", "barp", "tab:barp-musr-depth"),
    )
    for task, method, label in table_specs:
        expected = [
            [
                str(depth),
                *[_success_cell(rows[(task, pool, depth, method)]) for pool in POOLS],
            ]
            for depth in range(1, 11)
        ]
        errors.extend(
            _check_table(
                appendix_tex,
                label,
                expected,
                lambda row: bool(row and re.fullmatch(r"(?:[1-9]|10)", row[0])),
            )
        )
    return errors


def _load_calibrated(path: str | Path) -> dict[tuple[str, int, Decimal, str], dict[str, str]]:
    return _indexed_rows(
        path,
        REQUIRED_CALIBRATED_COLUMNS,
        lambda row, context: (
            (
                row["target_task"].strip(),
                _integer(row["pool_size"], f"{context} pool_size"),
                _decimal(row["target_alpha"], f"{context} target_alpha"),
                row["method"].strip(),
            )
            if row["availability_status"].strip() == "available"
            and row["method"].strip()
            in {"flexrouter", "barp", "fixed_tally_factorized"}
            else None
        ),
    )


def _calibrated_cell(row: dict[str, str]) -> str:
    count = _integer(row["certification_count"], "certification_count")
    denominator = _integer(row["certification_denominator"], "certification_denominator")
    rate = _decimal(row["certification_rate"], "certification_rate")
    exact_rate = Decimal(count) / Decimal(denominator) if denominator > 0 else Decimal(-1)
    if denominator <= 0 or abs(rate - exact_rate) > Decimal("1e-12"):
        raise ValueError("calibrated certification count, denominator, and rate disagree")
    calls = _optional_decimal(row["mean_calls"], "mean_calls")
    risk = _optional_decimal(row["standard_all_miss_risk_mean"], "standard risk")
    if calls is None or risk is None:
        if count != 0:
            raise ValueError("certified calibrated row has no calls or risk")
        return "non-certified (0%)"
    return f"{_fixed(calls, 2)} / {_fixed(risk, 3)} / {_percent_one(exact_rate)}"


def _check_calibrated(
    experiments_tex: str,
    appendix_tex: str,
    rows: dict[tuple[str, int, Decimal, str], dict[str, str]],
) -> list[str]:
    errors: list[str] = []
    expected_ifeval: list[list[str]] = []
    for alpha in (PRIMARY_ALPHA, LOOSE_ALPHA):
        for pool in POOLS:
            expected_ifeval.append(
                [
                    "0.05" if alpha == PRIMARY_ALPHA else "0.1",
                    str(pool),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "fixed_tally_factorized")]),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "flexrouter")]),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "barp")]),
                ]
            )
    errors.extend(
        _check_table(
            experiments_tex,
            "tab:learned-calibrated",
            expected_ifeval,
            lambda row: bool(row and row[0] in {"0.05", "0.1", "0.10"}),
        )
    )

    expected_musr: list[list[str]] = []
    for alpha in (PRIMARY_ALPHA, LOOSE_ALPHA):
        for pool in POOLS:
            expected_musr.append(
                [
                    "0.05" if alpha == PRIMARY_ALPHA else "0.1",
                    str(pool),
                    _calibrated_cell(rows[("musr", pool, alpha, "flexrouter")]),
                    _calibrated_cell(rows[("musr", pool, alpha, "barp")]),
                ]
            )
    errors.extend(
        _check_table(
            appendix_tex,
            "tab:learned-musr-calibrated",
            expected_musr,
            lambda row: bool(row and row[0] in {"0.05", "0.1", "0.10"}),
        )
    )
    return errors


def _load_native_flex(path: str | Path) -> dict[tuple[str, int, Decimal], dict[str, str]]:
    rows = _indexed_rows(
        path,
        REQUIRED_NATIVE_FLEX_COLUMNS,
        lambda row, context: (
            (
                row["target_task"].strip(),
                _integer(row["pool_size"], f"{context} pool_size"),
                _decimal(row["tau"], f"{context} tau"),
            )
            if row["availability_status"].strip() == "available"
            and row["method"].strip() == "flexrouter"
            else None
        ),
    )
    expected = {
        (task, pool, tau)
        for task in ("ifeval", "musr")
        for pool in POOLS
        for tau in map(Decimal, ("0.0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5"))
    }
    if set(rows) != expected:
        raise ValueError(f"{path}: native FlexRouter grid is incomplete")
    for key, row in rows.items():
        if _integer(row["depth_cap"], f"{path}: {key} depth_cap") != 10:
            raise ValueError(f"{path}: {key} depth cap must be 10")
        if _boolean(row["risk_certificate_claimed"], f"{path}: {key} certificate"):
            raise ValueError(f"{path}: native FlexRouter row {key} claims a certificate")
    return rows


def _check_native_flex(
    appendix_tex: str, rows: dict[tuple[str, int, Decimal], dict[str, str]]
) -> list[str]:
    errors: list[str] = []
    taus = tuple(map(Decimal, ("0.0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5")))
    for task, label in (
        ("ifeval", "tab:flex-native-ifeval"),
        ("musr", "tab:flex-native-musr"),
    ):
        expected = [
            [
                str(tau),
                *[
                    f"{_fixed(_decimal(rows[(task, pool, tau)]['mean_calls'], 'native calls'), 2)} / {_fixed(_decimal(rows[(task, pool, tau)]['success_mean'], 'native success'), 3)}"
                    for pool in POOLS
                ],
            ]
            for tau in taus
        ]
        errors.extend(
            _check_table(
                appendix_tex,
                label,
                expected,
                lambda row: bool(row and re.fullmatch(r"0(?:\.0?1|\.0?5|\.1|\.2|\.3|\.5|\.0)", row[0])),
            )
        )
    return errors


def _load_native_barp(path: str | Path) -> dict[tuple[str, Decimal], dict[str, str]]:
    rows = _indexed_rows(
        path,
        REQUIRED_NATIVE_BARP_COLUMNS,
        lambda row, context: (
            (
                row["target_task"].strip(),
                _decimal(row["w_quality"], f"{context} w_quality").quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                ),
            )
            if row["availability_status"].strip() == "available"
            and row["method"].strip() == "barp"
            else None
        ),
    )
    expected = {
        (task, Decimal(index) / Decimal(10))
        for task in ("hellaswag", "mbpp", "macro_average")
        for index in range(11)
    }
    if set(rows) != expected:
        raise ValueError(f"{path}: native BaRP preference grid is incomplete")
    for key, row in rows.items():
        if _integer(row["training_seed_count"], f"{path}: {key} training_seed_count") != 5:
            raise ValueError(f"{path}: native BaRP row {key} does not aggregate five seeds")
        quality = key[1]
        cost_weight = _decimal(row["w_cost"], f"{path}: {key} w_cost").quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        if quality + cost_weight != Decimal(1):
            raise ValueError(f"{path}: native BaRP weights do not sum to one for {key}")
    return rows


def _check_native_barp(
    appendix_tex: str, rows: dict[tuple[str, Decimal], dict[str, str]]
) -> list[str]:
    expected: list[list[str]] = []
    for index in range(10, -1, -1):
        quality_weight = Decimal(index) / Decimal(10)
        cost_weight = Decimal(1) - quality_weight
        cells = []
        for task in ("hellaswag", "mbpp", "macro_average"):
            row = rows[(task, quality_weight)]
            quality = _decimal(row["mean_official_quality"], "native BaRP quality")
            cost = _decimal(row["mean_cost"], "native BaRP cost") * Decimal(1000)
            cells.append(f"{_fixed(quality, 3)} / {_fixed(cost, 3)}")
        expected.append(
            [_fixed(quality_weight, 1), _fixed(cost_weight, 1), *cells]
        )
    return _check_table(
        appendix_tex,
        "tab:barp-native",
        expected,
        lambda row: bool(row and re.fullmatch(r"(?:0|1)\.\d", row[0])),
    )


def _load_taxonomy(path: str | Path) -> dict[Decimal, tuple[int, int, int]]:
    values: dict[Decimal, tuple[int, int, int]] = {}
    for index, row in enumerate(_read_csv(path, REQUIRED_TAXONOMY_COLUMNS), start=2):
        context = f"{path}: row {index}"
        alpha = _decimal(row["target_alpha"], f"{context} target_alpha")
        if alpha in values:
            raise ValueError(f"{context}: duplicate taxonomy alpha {alpha}")
        values[alpha] = (
            _integer(row["method_count"], f"{context} method_count"),
            _integer(row["risk_bar_pass_count"], f"{context} risk count"),
            _integer(
                row["information_destruction_sensitivity_pass_count"],
                f"{context} information count",
            ),
        )
    return values


def _load_permutation_mean(path: str | Path) -> Decimal:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if _integer(payload.get("pool_size"), f"{path}: pool_size") != 100:
        raise ValueError(f"{path}: expected pool 100")
    control = payload.get("information_destruction_control", {})
    if control.get("difference_definition") != "real_calls_test_minus_shuffled_calls_test":
        raise ValueError(f"{path}: unsupported permutation difference definition")
    try:
        record = control["summary"][str(PRIMARY_ALPHA)]["fixed_tally_factorized"]
        count = _integer(record["record_count"], f"{path}: record_count")
        mean = _decimal(
            record["real_minus_shuffled_calls_mean"], f"{path}: mean difference"
        )
    except KeyError as error:
        raise ValueError(f"{path}: missing fixed-tally alpha 0.05 summary: {error}") from error
    if count <= 0:
        raise ValueError(f"{path}: fixed-tally alpha 0.05 has no permutation records")
    return mean


def _check_old_auxiliary_claims(
    experiments_tex: str,
    taxonomy: dict[Decimal, tuple[int, int, int]],
    permutation_mean: Decimal,
) -> list[str]:
    errors: list[str] = []
    plain = _plain_tex(experiments_tex)
    for alpha in (PRIMARY_ALPHA, LOOSE_ALPHA):
        if alpha not in taxonomy:
            errors.append(f"taxonomy artifact is missing alpha {alpha}")
            continue
        total, risk_count, information_count = taxonomy[alpha]
        if alpha == PRIMARY_ALPHA:
            pattern = rf"At alpha=0\.05, {risk_count} of {total} pass.*?and {information_count} of {total} pass"
        else:
            pattern = rf"At alpha=0\.10,.*?counts rise to {risk_count} of {total} and {information_count} of {total}"
        _require_pattern(plain, pattern, f"taxonomy counts at alpha {alpha}", errors)
    magnitude = abs(permutation_mean)
    direction = "fewer" if permutation_mean < 0 else "more"
    _require_pattern(
        plain,
        rf"fixed tally uses about {_calls(magnitude)} {direction} calls.*?permut",
        "pool-100 fixed-tally permutation mean",
        errors,
    )
    return errors


def _check_scientific_wording(named_tex: dict[str, str]) -> list[str]:
    errors: list[str] = []
    plain = {name: _plain_tex(value) for name, value in named_tex.items()}
    for section in ("abstract", "introduction", "experiments", "discussion"):
        _require_pattern(
            plain[section],
            r"among shared certified redraws",
            f"{section} shared-certified qualification",
            errors,
        )
    for section in ("experiments", "appendix"):
        _require_pattern(
            plain[section],
            r"cross-protocol.*?descriptive and unpaired",
            f"{section} descriptive unpaired cross-protocol label",
            errors,
        )
    for section in ("experiments", "introduction", "appendix"):
        _require_pattern(
            plain[section],
            r"(?:registered maximum (?:set size|depth) of 10|register a common maximum set size of 10)",
            f"{section} registered maximum set size 10",
            errors,
        )
        _require_pattern(
            plain[section],
            r"BaRP.*?top-k (?:ordering )?adaptation",
            f"{section} BaRP top-k adaptation label",
            errors,
        )
    for section in ("abstract", "introduction", "experiments", "discussion"):
        _require_pattern(
            plain[section],
            r"At alpha=0\.10, fixed tally certif(?:y|ies).*?every redraw.*?uses fewer calls.*?(?:either learned method|either learned baseline|both learned)",
            f"{section} alpha 0.10 learned-baseline claim",
            errors,
        )
    for section in ("abstract", "introduction", "discussion"):
        _require_pattern(
            plain[section],
            r"available pool-100 descriptive, unpaired overlay",
            f"{section} available pool-100 overlay scope",
            errors,
        )

    forbidden = re.compile(
        r"(?:ISSUE-46|descriptive_nonpaired|\brepaired\b|\blegacy\b)", re.IGNORECASE
    )
    native_barp_cap = re.compile(
        r"(?:native\s+BaRP|BaRP(?:'s)?\s+native).{0,80}(?:depth\s+cap|maximum\s+(?:depth|set size))",
        re.IGNORECASE,
    )
    stale = re.compile(r"\b57\s+(?:to|through)\s+83\b", re.IGNORECASE)
    for name, text in plain.items():
        if forbidden.search(text):
            errors.append(f"{name}: forbidden internal or stale manuscript language")
        if native_barp_cap.search(text):
            errors.append(f"{name}: native BaRP must not be described as having a set-depth cap")
        if stale.search(text):
            errors.append(f"{name}: stale 57 to 83 claim remains")
    return errors


def run_checks(
    *,
    query_table: str | Path,
    e3_summary: str | Path,
    taxonomy_path: str | Path,
    permutation_json: str | Path,
    factorial_summary: str | Path,
    learned_fixed_depth: str | Path,
    calibrated_primary: str | Path,
    native_flex_adaptive: str | Path,
    native_barp_sensitivity: str | Path,
    experiments_tex: str | Path,
    abstract_tex: str | Path,
    discussion_tex: str | Path,
    appendix_tex: str | Path,
    introduction_tex: str | Path,
) -> list[str]:
    """Return sorted deterministic manuscript claim errors."""

    try:
        query = _load_query(query_table)
        e3 = _load_e3(e3_summary)
        taxonomy = _load_taxonomy(taxonomy_path)
        permutation_mean = _load_permutation_mean(permutation_json)
        factorial = _load_factorial(factorial_summary)
        fixed_depth = _load_fixed_depth(learned_fixed_depth)
        calibrated = _load_calibrated(calibrated_primary)
        native_flex = _load_native_flex(native_flex_adaptive)
        native_barp = _load_native_barp(native_barp_sensitivity)
        tex = {
            "experiments": Path(experiments_tex).read_text(encoding="utf-8"),
            "abstract": Path(abstract_tex).read_text(encoding="utf-8"),
            "discussion": Path(discussion_tex).read_text(encoding="utf-8"),
            "appendix": Path(appendix_tex).read_text(encoding="utf-8"),
            "introduction": Path(introduction_tex).read_text(encoding="utf-8"),
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [str(error)]

    try:
        errors = _check_query_claims(
            _plain_tex(tex["experiments"]),
            _plain_tex(tex["abstract"]),
            _plain_tex(tex["discussion"]),
            query,
        )
        errors.extend(_check_e3(tex["experiments"], tex["appendix"], e3))
        errors.extend(_check_factorial(tex["experiments"], factorial))
        errors.extend(_check_fixed_depth(tex["experiments"], tex["appendix"], fixed_depth))
        errors.extend(_check_calibrated(tex["experiments"], tex["appendix"], calibrated))
        errors.extend(_check_native_flex(tex["appendix"], native_flex))
        errors.extend(_check_native_barp(tex["appendix"], native_barp))
        errors.extend(
            _check_old_auxiliary_claims(tex["experiments"], taxonomy, permutation_mean)
        )
        errors.extend(_check_scientific_wording(tex))
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-table", required=True)
    parser.add_argument("--e3-summary", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--permutation-json", required=True)
    parser.add_argument("--factorial-summary", required=True)
    parser.add_argument("--learned-fixed-depth", required=True)
    parser.add_argument("--calibrated-primary", required=True)
    parser.add_argument("--native-flex-adaptive", required=True)
    parser.add_argument("--native-barp-sensitivity", required=True)
    parser.add_argument("--experiments-tex", required=True)
    parser.add_argument("--abstract-tex", required=True)
    parser.add_argument("--discussion-tex", required=True)
    parser.add_argument("--appendix-tex", required=True)
    parser.add_argument("--introduction-tex", required=True)
    arguments = parser.parse_args()

    errors = run_checks(
        query_table=arguments.query_table,
        e3_summary=arguments.e3_summary,
        taxonomy_path=arguments.taxonomy,
        permutation_json=arguments.permutation_json,
        factorial_summary=arguments.factorial_summary,
        learned_fixed_depth=arguments.learned_fixed_depth,
        calibrated_primary=arguments.calibrated_primary,
        native_flex_adaptive=arguments.native_flex_adaptive,
        native_barp_sensitivity=arguments.native_barp_sensitivity,
        experiments_tex=arguments.experiments_tex,
        abstract_tex=arguments.abstract_tex,
        discussion_tex=arguments.discussion_tex,
        appendix_tex=arguments.appendix_tex,
        introduction_tex=arguments.introduction_tex,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"manuscript claim check failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("manuscript claim check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
