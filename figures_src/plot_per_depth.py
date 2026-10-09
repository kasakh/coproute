"""Regenerate the per-depth all-miss figures of the paper from packaged records.

Reads the long-format CSVs under results/per_depth/ (columns: source, dataset,
method, phase, depth, mean_miss_rate[, sd_miss_rate]) and renders one nine-panel
figure per requested method comparison, matching the paper's layout: rows one
and two are the five RouterEval benchmarks at pool 100, row three the four
RouterBench task families at pool 11; solid lines are held-out test curves,
dashed lines the design-split curves.

Examples:
  python figures_src/plot_per_depth.py --csv results/per_depth/e2_query_level.csv \
      --methods pairwise_count error_correlation gaussian_copula --out figures_out/e2
  python figures_src/plot_per_depth.py --csv results/per_depth/e1_benchmark_level.csv \
      --methods empirical_tally error_correlation gaussian_copula --out figures_out/e1
  python figures_src/plot_per_depth.py --csv results/per_depth/e3_unified_selection.csv \
      --unified --out figures_out/e3
"""
from __future__ import annotations
import argparse
import csv
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANEL_ORDER = [
    ("RE", "ifeval"), ("RE", "bbh"), ("RE", "gpqa"), ("RE", "math"), ("RE", "musr"),
    ("RB", "mmlu"), ("RB", "hellaswag"), ("RB", "winogrande"), ("RB", "arc-challenge"),
]
TITLES = {
    ("RE", "ifeval"): "instruction following (ifeval), pool 100 / 3811",
    ("RE", "bbh"): "BIG-Bench-Hard (bbh), pool 100 / 3811",
    ("RE", "gpqa"): "graduate science (gpqa), pool 100 / 3811",
    ("RE", "math"): "math hard (math), pool 100 / 3811",
    ("RE", "musr"): "soft reasoning (musr), pool 100 / 3811",
    ("RB", "mmlu"): "mmlu (RouterBench), pool 11",
    ("RB", "hellaswag"): "hellaswag (RouterBench), pool 11",
    ("RB", "winogrande"): "winogrande (RouterBench), pool 11",
    ("RB", "arc-challenge"): "arc-challenge (RouterBench), pool 11",
}
UNIFIED_STYLE = {
    "marginal": ("tab:red", "marginal"),
    "best_E1": ("tab:orange", "best benchmark-level"),
    "best_E2": ("tab:purple", "best query-level"),
    "E3_combined": ("tab:green", "unified selection"),
}


def load(csv_path: Path):
    table = collections.defaultdict(dict)
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["dataset"], row["method"], row["phase"])
            table[key][int(row["depth"])] = float(row["mean_miss_rate"])
    return table


def panel(ax, table, source, dataset, series):
    for method, color, label in series:
        for phase, style in (("test", "-"), ("design", "--")):
            curve = table.get((source, dataset, method, phase))
            if not curve:
                continue
            depths = sorted(curve)
            ax.plot(depths, [curve[d] for d in depths], style, color=color,
                    label=f"{label} on {phase}", linewidth=1.6)
    ax.set_title(TITLES[(source, dataset)], fontsize=9)
    ax.set_xlabel("calls (depth)", fontsize=8)
    ax.set_ylabel("all-models-miss rate", fontsize=8)
    ax.tick_params(labelsize=7)


def render(table, series, out_pdf: Path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (source, dataset) in zip(axes.flat, PANEL_ORDER):
        panel(ax, table, source, dataset, series)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--methods", nargs="*", default=[],
                        help="dependence-aware methods to plot, one figure each, against 'marginal'")
    parser.add_argument("--unified", action="store_true",
                        help="render the four-curve unified-selection figure instead")
    parser.add_argument("--out", required=True, help="output path prefix")
    args = parser.parse_args()

    table = load(Path(args.csv))
    if args.unified:
        series = [(m, c, l) for m, (c, l) in UNIFIED_STYLE.items()]
        render(table, series, Path(f"{args.out}_unified.pdf"))
        print(f"wrote {args.out}_unified.pdf/.png")
        return
    for method in args.methods:
        series = [("marginal", "tab:red", "marginal"), (method, "tab:blue", method)]
        render(table, series, Path(f"{args.out}_{method}.pdf"))
        print(f"wrote {args.out}_{method}.pdf/.png")


if __name__ == "__main__":
    main()
