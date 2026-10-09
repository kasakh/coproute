"""Integration-style fixtures for deterministic manuscript claim checking."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures_src"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import check_manuscript_claims as checker  # noqa: E402


POOLS = (100, 200, 300, 500)
ALPHAS = (Decimal("0.05"), Decimal("0.10"))
FACTORIAL_METHODS = (
    ("racer_g", "RACER-G"),
    ("racer_p", "RACER-P"),
    ("exact_independence", "Exact independence"),
    ("fixed_design_marginal_factorized", "Fixed design marginal + factorized sizing"),
    ("fixed_tally_factorized", "Fixed tally + factorized sizing"),
    ("local_empirical", "Local ordering + local empirical sizing"),
    ("local_order_factorized", "Local ordering + factorized sizing"),
)
SOURCES = {
    "query_table": ROOT
    / "results/query_level_calibrated/repaired_iclr_review_v4_20260918T132931Z/query_table.csv",
    "e3_summary": ROOT
    / "results/query_level_calibrated/repaired_iclr_review_v2_20260918T124310Z/e3_summary.csv",
    "taxonomy_path": ROOT
    / "results/benchmark_level_calibrated/repaired_iclr_review_v1_20260918T121110Z/taxonomy_survival.csv",
    "permutation_json": ROOT
    / "results/query_level_calibrated/repaired_iclr_review_v4_20260918T132931Z/permutation_merged/e2_pool100_permutations.json",
    "factorial_summary": ROOT / "results/revised_review_controls/v1/factorial_summary.csv",
    "learned_fixed_depth": ROOT
    / "results/learned_baselines/comparison_v2/fixed_depth_success.csv",
    "calibrated_primary": ROOT
    / "results/learned_baselines/comparison_v2/calibrated_primary.csv",
    "native_flex_adaptive": ROOT
    / "results/learned_baselines/comparison_v2/native_flex_adaptive.csv",
    "native_barp_sensitivity": ROOT
    / "results/learned_baselines/comparison_v2/native_barp_sensitivity.csv",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _decimal(value: str) -> Decimal:
    return Decimal(value)


def _fixed(value: str | Decimal, places: int) -> str:
    number = value if isinstance(value, Decimal) else _decimal(value)
    quantum = Decimal(1).scaleb(-places)
    return f"{number.quantize(quantum, rounding=ROUND_HALF_UP):.{places}f}"


def _learned(value: str, places: int) -> str:
    return f"{float(value):.{places}f}"


def _whole_percent(value: Decimal) -> int:
    tenths = (value * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return int(tenths.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _table(label: str, header: list[str], rows: list[list[str]]) -> str:
    body = [
        r"\begin{table}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llllllll}",
        " & ".join(header) + r" \\",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend((r"\end{tabular}", r"\end{table}"))
    return "\n".join(body)


def _factorial_tex(path: Path) -> str:
    selected = {
        (int(row["pool_size"]), row["method"]): row
        for row in _rows(path)
        if _decimal(row["target_alpha"]) == Decimal("0.05")
        and row["method"] in {method for method, _ in FACTORIAL_METHODS}
    }
    main = []
    for method, label in FACTORIAL_METHODS:
        main.append(
            [
                label,
                *[
                    f"{_fixed(selected[(pool, method)]['mean_calls'], 1)} "
                    f"({_fixed(selected[(pool, method)]['standard_all_miss_risk_mean'], 3)})"
                    for pool in POOLS
                ],
            ]
        )
    main.append(
        [
            "Certification rate",
            *[
                f"{_whole_percent(_decimal(selected[(pool, 'racer_g')]['certification_rate']))}%"
                for pool in POOLS
            ],
        ]
    )

    mechanism = []
    sizing_gains = []
    tally_gains = []
    marginal_gains = []
    for pool in POOLS:
        local = _decimal(selected[(pool, "local_empirical")]["mean_calls"])
        local_factorized = _decimal(
            selected[(pool, "local_order_factorized")]["mean_calls"]
        )
        tally = _decimal(selected[(pool, "fixed_tally_factorized")]["mean_calls"])
        marginal = _decimal(
            selected[(pool, "fixed_design_marginal_factorized")]["mean_calls"]
        )
        sizing = _whole_percent((local - local_factorized) / local)
        tally_gain = _whole_percent((local_factorized - tally) / local_factorized)
        marginal_gain = _whole_percent((marginal - tally) / marginal)
        sizing_gains.append(sizing)
        tally_gains.append(tally_gain)
        marginal_gains.append(marginal_gain)
        mechanism.append(
            [
                str(pool),
                _fixed(local, 1),
                _fixed(local_factorized, 1),
                f"{sizing}%",
                _fixed(tally, 1),
                f"{tally_gain}%",
                _fixed(marginal, 1),
                f"{marginal_gain}%",
            ]
        )

    saturation = []
    for pool in POOLS:
        local = selected[(pool, "local_empirical")]
        factorized = selected[(pool, "local_order_factorized")]
        saturation.append(
            [
                str(pool),
                _fixed(local["distinct_finite_score_count_mean"], 1),
                _fixed(local["largest_tied_block_mean"], 1),
                _fixed(local["score_one_tail_size_mean"], 1),
                f"{_fixed(_decimal(local['full_pool_call_rate']) * 100, 1)}%",
                f"{_fixed(_decimal(factorized['full_pool_call_rate']) * 100, 1)}%",
            ]
        )
    prose = (
        "Among shared certified redraws at calibration target alpha=0.05, "
        f"changing sizing reduces calls by {', '.join(map(str, sizing_gains[:-1]))}, "
        f"and {sizing_gains[-1]} percent. Fixed tally is better than local ordering "
        f"with the same factorized sizing by {', '.join(map(str, tally_gains[:-1]))}, "
        f"and {tally_gains[-1]} percent. The fixed design-marginal control uses the "
        f"same rule, yet fixed tally reduces its calls by {', '.join(map(str, marginal_gains[:-1]))}, "
        f"and {marginal_gains[-1]} percent."
    )
    return "\n".join(
        (
            _table(
                "tab:query-factorial",
                ["method", "pool 100", "pool 200", "pool 300", "pool 500"],
                main,
            ),
            prose,
            _table(
                "tab:mechanism-decomposition",
                ["pool", "local/local", "local/fact.", "size gain", "tally", "order gain", "design marg.", "order gain"],
                mechanism,
            ),
            _table(
                "tab:local-saturation",
                ["pool", "distinct finite", "largest tie", "score-1 tail", "local full pool", "fact. full pool"],
                saturation,
            ),
        )
    )


def _success_cell(row: dict[str, str]) -> str:
    return f"{_learned(row['success_mean'], 3)} ± {_learned(row['success_se'], 3)}"


def _fixed_depth_tex(path: Path) -> tuple[str, str]:
    available = {
        (row["target_task"], int(row["pool_size"]), int(row["depth"]), row["method"]): row
        for row in _rows(path)
        if row["availability_status"] == "available"
        and row["method"] in {"flexrouter", "barp", "fixed_tally"}
    }
    main = []
    for task in ("ifeval", "musr"):
        for pool in POOLS:
            fixed = available.get((task, pool, 10, "fixed_tally"))
            main.append(
                [
                    task.upper(),
                    str(pool),
                    _success_cell(available[(task, pool, 10, "flexrouter")]),
                    _success_cell(available[(task, pool, 10, "barp")]),
                    _learned(fixed["success_mean"], 3) if fixed else "not available",
                ]
            )
    main_tex = _table(
        "tab:learned-depth10",
        ["task", "pool", "FlexRouter", "BaRP", "fixed tally"],
        main,
    )
    appendix = []
    for task, method, label in (
        ("ifeval", "flexrouter", "tab:flex-ifeval-depth"),
        ("ifeval", "barp", "tab:barp-ifeval-depth"),
        ("musr", "flexrouter", "tab:flex-musr-depth"),
        ("musr", "barp", "tab:barp-musr-depth"),
    ):
        table_rows = [
            [
                str(depth),
                *[
                    _success_cell(available[(task, pool, depth, method)])
                    for pool in POOLS
                ],
            ]
            for depth in range(1, 11)
        ]
        appendix.append(
            _table(label, ["depth", "pool 100", "pool 200", "pool 300", "pool 500"], table_rows)
        )
    return main_tex, "\n".join(appendix)


def _calibrated_cell(row: dict[str, str]) -> str:
    count = int(row["certification_count"])
    denominator = int(row["certification_denominator"])
    if not row["mean_calls"] or not row["standard_all_miss_risk_mean"]:
        return "non-certified (0%)"
    return (
        f"{_fixed(row['mean_calls'], 2)} / "
        f"{_fixed(row['standard_all_miss_risk_mean'], 3)} / "
        f"{_fixed(Decimal(count) / Decimal(denominator) * 100, 1)}%"
    )


def _calibrated_tex(path: Path) -> tuple[str, str]:
    rows = {
        (row["target_task"], int(row["pool_size"]), _decimal(row["target_alpha"]), row["method"]): row
        for row in _rows(path)
        if row["availability_status"] == "available"
        and row["method"] in {"flexrouter", "barp", "fixed_tally_factorized"}
    }
    ifeval = []
    musr = []
    for alpha in ALPHAS:
        alpha_text = "0.05" if alpha == Decimal("0.05") else "0.1"
        for pool in POOLS:
            ifeval.append(
                [
                    alpha_text,
                    str(pool),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "fixed_tally_factorized")]),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "flexrouter")]),
                    _calibrated_cell(rows[("ifeval", pool, alpha, "barp")]),
                ]
            )
            musr.append(
                [
                    alpha_text,
                    str(pool),
                    _calibrated_cell(rows[("musr", pool, alpha, "flexrouter")]),
                    _calibrated_cell(rows[("musr", pool, alpha, "barp")]),
                ]
            )
    return (
        _table(
            "tab:learned-calibrated",
            ["alpha", "pool", "fixed tally", "FlexRouter", "BaRP"],
            ifeval,
        ),
        _table(
            "tab:learned-musr-calibrated",
            ["alpha", "pool", "FlexRouter", "BaRP"],
            musr,
        ),
    )


def _native_flex_tex(path: Path) -> str:
    rows = {
        (row["target_task"], int(row["pool_size"]), _decimal(row["tau"])): row
        for row in _rows(path)
    }
    tables = []
    for task, label in (
        ("ifeval", "tab:flex-native-ifeval"),
        ("musr", "tab:flex-native-musr"),
    ):
        table_rows = []
        for tau in map(Decimal, ("0.0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5")):
            table_rows.append(
                [
                    str(tau),
                    *[
                        f"{_fixed(rows[(task, pool, tau)]['mean_calls'], 2)} / "
                        f"{_fixed(rows[(task, pool, tau)]['success_mean'], 3)}"
                        for pool in POOLS
                    ],
                ]
            )
        tables.append(
            _table(label, ["tau", "pool 100", "pool 200", "pool 300", "pool 500"], table_rows)
        )
    return "\n".join(tables)


def _native_barp_tex(path: Path) -> str:
    rows = {
        (
            row["target_task"],
            _decimal(row["w_quality"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
        ): row
        for row in _rows(path)
    }
    table_rows = []
    for index in range(10, -1, -1):
        quality_weight = Decimal(index) / 10
        cells = []
        for task in ("hellaswag", "mbpp", "macro_average"):
            row = rows[(task, quality_weight)]
            cells.append(
                f"{_fixed(row['mean_official_quality'], 3)} / "
                f"{_fixed(_decimal(row['mean_cost']) * 1000, 3)}"
            )
        table_rows.append(
            [_fixed(quality_weight, 1), _fixed(1 - quality_weight, 1), *cells]
        )
    return _table(
        "tab:barp-native",
        ["quality wt.", "cost wt.", "HellaSwag", "MBPP", "macro"],
        table_rows,
    )


def _e3_tex(path: Path) -> str:
    rows = {
        int(row["pool_size"]): row
        for row in _rows(path)
        if _decimal(row["target_alpha"]) == Decimal("0.05")
        and row["class_penalty_enabled"].lower() == "false"
    }
    table_rows = []
    for pool in (100, 200):
        row = rows[pool]
        selected = _decimal(row["selected_mean_calls"])
        racer = _decimal(row["racer_g_mean_calls"])
        change = _whole_percent(abs(selected - racer) / racer)
        direction = "more" if selected > racer else "fewer"
        table_rows.append(
            [
                str(pool),
                f"{_fixed(selected, 1)} ({_fixed(row['selected_standard_all_miss_risk_mean'], 3)[1:]})",
                f"{_fixed(racer, 1)} ({_fixed(row['racer_g_standard_all_miss_risk_mean'], 3)[1:]})",
                f"{change}% {direction}",
            ]
        )
    counts_100 = json.loads(rows[100]["class_win_counts"])
    counts_200 = json.loads(rows[200]["class_win_counts"])
    return "\n".join(
        (
            _table(
                "tab:e3-calibrated",
                ["pool", "validation-selected", "RACER-G", "call change"],
                table_rows,
            ),
            f"For the unpenalized selector, validation chooses a fixed-prefix family in "
            f"{counts_100['fixed_prefix']} of 20 redraws and a per-query wrapper in "
            f"{counts_100['per_query_wrapper']} of 20 at pool 100. At pool 200, the "
            f"corresponding counts are {counts_200['fixed_prefix']} and "
            f"{counts_200['per_query_wrapper']}.",
        )
    )


def _write_manuscript(paths: dict[str, Path]) -> None:
    fixed_main, fixed_appendix = _fixed_depth_tex(paths["learned_fixed_depth"])
    calibrated_main, calibrated_appendix = _calibrated_tex(paths["calibrated_primary"])
    experiments = "\n".join(
        (
            _factorial_tex(paths["factorial_summary"]),
            fixed_main,
            calibrated_main,
            "Among shared certified redraws, the comparison is qualified.",
            "Cross-protocol comparisons are descriptive and unpaired.",
            "The registered maximum set size of 10 applies to learned evaluations, including the BaRP top-k adaptation.",
            "At alpha=0.10, fixed tally certifies every redraw and uses fewer calls than either learned method in all four pools.",
            "At alpha=0.10, fixed tally reduces calls by 23 to 40 percent versus RACER-G and by 9 to 21 percent versus exact independence.",
            "At alpha=0.02, certification is 15, 40, 70, and 65 percent.",
            "At alpha=0.05, 10 of 11 pass the prespecified risk bar and 7 of 11 pass the information-destruction sensitivity bar.",
            "At alpha=0.10, the counts rise to 11 of 11 and 10 of 11.",
            "At pool 100 and alpha=0.05, fixed tally uses about 3.1 fewer calls than after independently permuting the design columns.",
            "The unpenalized selected policy stays near RACER-G at pool 100, using 31.5 calls versus 30.5, and at pool 200 uses 17.5 calls versus 31.6.",
        )
    )
    appendix = "\n".join(
        (
            _e3_tex(paths["e3_summary"]),
            fixed_appendix,
            calibrated_appendix,
            _native_flex_tex(paths["native_flex_adaptive"]),
            _native_barp_tex(paths["native_barp_sensitivity"]),
            "Cross-protocol contrasts are descriptive and unpaired.",
            "Learned methods use a registered maximum depth of 10. BaRP uses a top-k adaptation for set evaluation.",
        )
    )
    common = (
        "Among shared certified redraws at calibration target alpha=0.05, fixed tally reduces calls. "
        "In the available pool-100 descriptive, unpaired overlay, BaRP matches fixed tally. "
        "At alpha=0.10, fixed tally certifies every redraw and uses fewer calls than either learned method in all four pools."
    )
    abstract = (
        common
        + " Fixed tally reduces calls by 13 to 51 percent versus RACER-G and by 12 to 37 percent versus exact-independence coverage."
    )
    introduction = (
        common
        + " We register a common maximum set size of 10, and BaRP uses an explicit top-k adaptation."
    )
    discussion = common + " At alpha=0.02, certification rates of 15 to 70 percent identify a shared boundary."
    for name, text in (
        ("experiments_tex", experiments),
        ("appendix_tex", appendix),
        ("abstract_tex", abstract),
        ("introduction_tex", introduction),
        ("discussion_tex", discussion),
    ):
        paths[name].write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, source in SOURCES.items():
        destination = tmp_path / source.name
        shutil.copyfile(source, destination)
        paths[name] = destination
    for name in (
        "experiments_tex",
        "appendix_tex",
        "abstract_tex",
        "introduction_tex",
        "discussion_tex",
    ):
        paths[name] = tmp_path / f"{name}.tex"
    _write_manuscript(paths)
    return paths


def _mutate_csv(
    path: Path,
    predicate: Callable[[dict[str, str]], bool],
    field: str,
    value: str,
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    matches = [row for row in rows if predicate(row)]
    assert len(matches) == 1
    matches[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_current_artifact_fixture_passes(tmp_path: Path) -> None:
    assert checker.run_checks(**_fixture(tmp_path)) == []


def test_unpenalized_e3_row_is_selected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["e3_summary"],
        lambda row: row["pool_size"] == "100"
        and row["target_alpha"] == "0.05"
        and row["class_penalty_enabled"].lower() == "true",
        "selected_mean_calls",
        "999",
    )
    assert checker.run_checks(**paths) == []

    _mutate_csv(
        paths["e3_summary"],
        lambda row: row["pool_size"] == "100"
        and row["target_alpha"] == "0.05"
        and row["class_penalty_enabled"].lower() == "false",
        "selected_mean_calls",
        "32.522950819672133",
    )
    errors = checker.run_checks(**paths)
    assert any("tab:e3-calibrated" in error for error in errors)


def test_stale_factorial_number_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["factorial_summary"],
        lambda row: row["pool_size"] == "100"
        and row["target_alpha"] == "0.05"
        and row["method"] == "racer_g",
        "mean_calls",
        "26.19576379974326",
    )
    assert any("tab:query-factorial" in error for error in checker.run_checks(**paths))


def test_stale_fixed_depth_number_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["learned_fixed_depth"],
        lambda row: row["target_task"] == "ifeval"
        and row["pool_size"] == "100"
        and row["depth"] == "10"
        and row["method"] == "flexrouter",
        "success_mean",
        "0.8344512195121952",
    )
    assert any("tab:learned-depth10" in error for error in checker.run_checks(**paths))


def test_stale_calibrated_number_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["calibrated_primary"],
        lambda row: row["target_task"] == "ifeval"
        and row["pool_size"] == "100"
        and row["target_alpha"] == "0.05"
        and row["method"] == "flexrouter",
        "mean_calls",
        "9.948170731707316",
    )
    assert any("tab:learned-calibrated" in error for error in checker.run_checks(**paths))


def test_stale_native_flex_number_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["native_flex_adaptive"],
        lambda row: row["target_task"] == "ifeval"
        and row["pool_size"] == "100"
        and row["tau"] == "0.5",
        "mean_calls",
        "6.832256097560975",
    )
    assert any("tab:flex-native-ifeval" in error for error in checker.run_checks(**paths))


def test_stale_native_barp_number_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _mutate_csv(
        paths["native_barp_sensitivity"],
        lambda row: row["target_task"] == "hellaswag"
        and row["w_quality"] == "1.0",
        "mean_official_quality",
        "0.6991701244813278",
    )
    assert any("tab:barp-native" in error for error in checker.run_checks(**paths))


def test_forbidden_internal_wording_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["introduction_tex"].write_text(
        paths["introduction_tex"].read_text(encoding="utf-8")
        + " ISSUE-46 used a repaired legacy descriptive_nonpaired label.",
        encoding="utf-8",
    )
    errors = checker.run_checks(**paths)
    assert any("forbidden internal or stale manuscript language" in error for error in errors)
