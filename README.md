# CopRoute: Dependence-Aware Conformal Routing for Language Models

Code, run records, and figure sources for the paper *CopRoute: Dependence-Aware Conformal Routing for Language Models* (Siva Rajesh Kasa, Santhosh Kumar Kasa, Karan Gupta), arXiv:2610.13276, https://arxiv.org/abs/2610.13276.

Risk-controlled set routing selects, per query, a subset of models from a pool and controls the probability that every selected real model answers incorrectly, while calling as few models as possible. This repository contains the estimators and runners, the run records behind tables and figures, and scripts that regenerate and check each reported artifact.

All experiments are labels-only computations on public releases and run on CPU. No model inference is performed. One repetition of the largest cell takes a few core-minutes. Every reported cell completes within one hour on a commodity multicore machine.

## Layout

```
components/    estimators, data protocols, calibration, and scoring
scripts/       one runner per experiment family, plus data preparation utilities
tests/         pytest unit and regression tests
results/       run records behind reported numbers
figures_src/   table, figure, taxonomy, and deterministic checking scripts
```

## Regenerating the paper's tables and figures (seconds)

The rollup records behind every table and figure in the paper are packaged under
`results/paper_rollups/` (integrity: `SHA256SUMS` in that directory). Each script in
`figures_src/paper/` reads those records by relative path and writes its outputs to
`figures_out/paper/`:

```
python figures_src/paper/build_main_table.py           # main certified-calls table
python figures_src/paper/plot_sweep_calls_vs_target.py # risk-sweep figure
python figures_src/paper/plot_radar_call_efficiency.py # relative-efficiency radar
python figures_src/paper/plot_transfer_heatmap.py      # transfer heatmaps
python figures_src/paper/plot_transfer_all_policies.py # per-policy transfer appendix
python figures_src/paper/build_estimator_tables.py     # estimator-family tables
python figures_src/paper/build_cost_tables.py          # cost and cascade tables
python figures_src/paper/build_mechanism_table.py      # synthetic mechanism map
```

Requirements: Python 3.10+ with numpy and matplotlib. No network access and no model
inference; each script completes in seconds.

## Reproducing and checking paper artifacts

Every table and figure regenerates from `results/` without model inference:

| Paper artifact | Command |
|---|---|
| Query-level table and calls-versus-risk figure | `python figures_src/plot_calls_vs_risk.py --records results/query_level_calibrated --table-out out/query_level_table.csv --fig-out out/calls_vs_risk` |
| Deterministic query-level number and range check | `python figures_src/check_paper_numbers.py --records results/query_level_calibrated --table out/query_level_table.csv --figure-metadata out/calls_vs_risk.json` |
| Artifact-to-manuscript claim check | `python figures_src/check_manuscript_claims.py --help` |
| Benchmark-level calibrated sweep figures | `python figures_src/plot_benchmark_level_sweep.py --csv results/benchmark_level_calibrated/bprime_summary.csv --out out/bprime` |
| Repaired E1 benchmark run summary | `python figures_src/summarize_benchmark_runs.py --runs results/benchmark_level_calibrated/runs --csv-out out/repaired_e1.csv --json-out out/repaired_e1.json` |
| Repaired E3 pool and penalty summary | `python figures_src/summarize_e3.py --records out/e3_runs --csv-out out/repaired_e3.csv --json-out out/repaired_e3.json` |
| Taxonomy survival table | `python figures_src/taxonomy_survival.py --runs results/benchmark_level_calibrated/runs --out out/taxonomy_survival.csv` |
| Per-depth figures, query-level variants | `python figures_src/plot_per_depth.py --csv results/per_depth/e2_query_level.csv --methods pairwise_count error_correlation gaussian_copula --out out/query_depth` |
| Per-depth figures, benchmark-level estimators | `python figures_src/plot_per_depth.py --csv results/per_depth/e1_benchmark_level.csv --methods empirical_tally error_correlation gaussian_copula --out out/bench_depth` |
| Unified-selection figure | `python figures_src/plot_per_depth.py --csv results/per_depth/e3_unified_selection.csv --unified --out out/unified` |
| Unified-selection pick-count table | `results/per_depth/unified_selection_picks.json` |

The query-level plot command writes PDF, PNG, and JSON figure metadata. The JSON records every plotted point and the exact axis range. The checker reconstructs the expected table from source JSON, compares every CSV cell, recomputes every risk flag, checks event counts, and verifies that the figure range contains every RACER baseline point. It exits nonzero on any mismatch.

Run the artifact-to-manuscript checker with the generated summaries and the three claim-bearing TeX sections:

```
python figures_src/check_manuscript_claims.py \
  --query-table out/query_table.csv \
  --e3-summary out/e3_summary.csv \
  --taxonomy out/taxonomy_survival.csv \
  --permutation-json out/e2_pool100_permutations.json \
  --experiments-tex paper/sections/experiments.tex \
  --abstract-tex paper/sections/abstract.tex \
  --discussion-tex paper/sections/discussion.tex
```

This check derives manuscript values from the supplied artifacts. It validates the primary pool rows, calls, standard all-miss risks, certification rates, call-reduction ranges, tight-target certification range, enabled E3 pool results, E1 taxonomy counts, and pool-100 permutation sensitivity. It also rejects stale call-reduction text and unsupported positive exact-null or causal-decomposition interpretations. Manuscript display rules are fixed at one decimal for calls, three decimals for risks, and whole percentages after rounding to one decimal percentage point.

## Canonical reporting contract

### Method names

New output always uses these public names and keeps the two RACER controls separate:

1. `RACER-G`
2. `RACER-P`
3. `Exact independence`
4. `Local empirical`
5. `Fixed-tally factorized sizing`
6. `Pairwise heuristic`
7. `Correlation heuristic`
8. `Regularized correlation`

The readers also accept legacy record aliases such as `racer_marginal_G`, `racer_marginal_P`, `pairwise_count`, `error_correlation`, and `gaussian_copula`. Legacy aliases are input compatibility only. They are never emitted as public query-level method names.

### Canonical query-level columns

The generated table uses fixed columns including:

- `pool_size`, `target_alpha`, `method`, `method_label`, and `mean_calls`
- `reported_risk_name`, `reported_risk_mean`, and `reported_risk_se`
- `reported_event_count`, `reported_trial_count`, and `event_count_status`
- `risk_classification` and `above_target_2se`
- separate `safe_action_risk_mean`, `standard_all_miss_risk_mean`, and `augmented_loss_mean`
- separate event-count columns for those three losses
- `certification_rate`, `non_certification_count`, and `redraw_count`
- `source_schema`, which is `legacy` or `repaired`

The losses are not interchangeable:

- **Safe-action risk** is a legacy or explicitly configured fallback diagnostic. The current Stage 1 standard-risk pipeline does not deploy a fallback after non-certification; it reports no operating point.
- **Standard all-miss risk** evaluates whether all selected real models are wrong.
- **Augmented loss** evaluates RACER's augmented label space, including its null label.
- **Non-certification** means calibration did not issue a certificate. It is an outcome to report, not another name for structural complete-pool infeasibility.

Legacy records do not contain all four concepts. Their historical risk is retained as `legacy_reported_risk`, and missing semantics are left blank. Reporting code never silently converts an old risk into safe-action risk, standard all-miss risk, or augmented loss.

### Risk flags and event counts

There is one risk classification rule in the reporting code:

`mean risk > target alpha + 2 * standard error`

The shared implementation is `risk_above_target` in `figures_src/plot_calls_vs_risk.py`. No old verdict field, rounded display value, or separate plotting rule can override it. When repetition-level risk values are missing, the standard error and flag are marked unavailable.

Event counts are reconstructed only when a source record gives an exact denominator and the aggregate mean maps to an integer count. Otherwise the table says `unavailable_in_source`. No fractional or guessed event count is printed.

### Figure ranges

Calls-versus-risk panels include both RACER controls and every available repaired arm. Their x limits are computed from all plotted points, not only the proposed methods. The metadata checker verifies that every RACER-G and RACER-P point lies inside the displayed range.

Benchmark-level calibrated panels are never silently dropped because they have fewer than two certified baseline points. Sparse panels remain visible and state whether they contain one or no certified baseline point. Their JSON companions record every baseline and method point.

### Information-destruction sensitivity

The benchmark-level label-shuffle output is called an **information-destruction sensitivity** analysis. It destroys co-failure information in the design labels while leaving the deployment distribution unchanged. The legacy `dr_bp_4_pass` field is accepted for old records, but the report does not call the bar a causal decomposition, a significance test, or an automatic artifact detector. Its eligible-cell denominator is always printed.

## Record provenance

- `results/query_level_calibrated/ifeval_pool{100,200,300,500}_r20.json`: legacy output of `scripts/design_bprime_e2_run.py`, with twenty repetitions and fifteen target levels. Repaired reruns add separate RACER-G and RACER-P controls, exact independence, local empirical coverage, fixed-tally factorized sizing, repaired heuristic names, repetition-level loss semantics, certification, and non-certification fields. The reporting scripts accept both generations.
- `results/query_level_calibrated/null_shuffle/*.json`: legacy information-destruction sensitivity runs. Each design column is independently permuted and the complete fitting pipeline is rerun.
- `results/benchmark_level_calibrated/runs/*.json`: one benchmark-level record per protocol, benchmark, and design-budget cell. Records contain seed-level values and aggregate method summaries.
- `results/benchmark_level_calibrated/bprime_summary.csv`: flat benchmark-level summary. The plotting reader accepts its legacy columns and repaired canonical aliases.
- `results/per_depth/*.csv`: long-format fixed-depth curves with source, dataset, method, phase, depth, mean miss rate, and uncertainty.
- `results/per_depth/unified_selection_picks.json`: validation-selected candidate counts by benchmark and seed.

## Recomputing records from raw data

### Data

- RouterBench: place the public per-query correctness release under `data/routerbench_5shot.pkl`, with `_0shot` where used.
- RouterEval: place leaderboard correctness and router features under `data/routereval/`.
- Produce nearest-neighbor router embeddings with `python scripts/embed_mdeberta.py`. This is the only step that loads an off-the-shelf sentence encoder.
- Before query-level RouterEval experiments, run `python scripts/routereval_alignment_verify.py`. Query-level runners refuse benchmarks without an alignment certificate.

### Runners

```
# Query-level calibrated comparison, one pool size per call
python scripts/design_bprime_e2_run.py --router-dir data/routereval/router_dataset \
  --leaderboard data/routereval/leaderboard.pkl --benchmark ifeval --pool-size 100 \
  --alphas "0.02,0.025,0.03,0.04,0.05,0.06,0.075,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.18" \
  --redraws 20 --knn-k 40 --design-fraction 0.30 --router-fraction 0.20 \
  --cal-fraction 0.20 --val-fraction 0.15 --seed 7001 --max-depth 25 \
  --out out/pool100.json

# Benchmark-level per-method tuning
python scripts/design_bprime_run.py --help

# Per-depth curves
python scripts/frontier_e1_generalization_v2.py --help
python scripts/frontier_e2_generalization.py --help
python scripts/frontier_e2_routerbench.py --help

# Pooled-family validation selection
python scripts/design_e3_unified.py --help
```

Seeds, split fractions, hyperparameter grids, and target-level grids are frozen in source. Runners refuse to overwrite an existing result path.

## Protocol notes

- Repaired calibrated runs use design data for fitting, validation data for candidate selection, untouched final calibration data once for the frozen candidate's cutoff, and test data once for reporting.
- RACER-G and RACER-P are predeclared controls. They are reported separately and are not selected against each other using final calibration outcomes.
- A non-certified final calibration emits no deployed operating point. Reports separately state certification, non-certification reason, and the structural complete-pool risk floor; any explicitly configured fallback must be evaluated under its own separately named loss.
- A complete-pool miss rate above a target indicates structural infeasibility for that evaluated pool and label population. A particular calibration draw failing to certify is non-certification. Reports keep them separate.
- The information-destruction sensitivity analysis permutes design-label columns. It measures sensitivity to removing that information under an unchanged test distribution. It is not an exact conditional-independence intervention.

## Synthetic conditional-independence null

The standalone null experiment generates an observed query difficulty `X`, heterogeneous true model marginals `p_j(X)`, and correctness labels that are independent Bernoulli draws conditional on `X`. Design, validation, final-calibration, and test samples use independent deterministic random streams. It compares exact factorized independence with a pooled raw-correlation heuristic and the same heuristic applied to oracle residual correlations `Y - p_j(X)`.

Run the frozen 128-seed CPU experiment from the repository root:

```
python scripts/synthetic_conditional_independence.py \
  --output-dir results/synthetic_conditional_independence/v1
```

The runner writes deterministic `results.json` and `summary.csv`, refuses to overwrite them by default, and accepts `--force` for an explicit deterministic regeneration. The JSON contains frozen settings, seed-level records, metric definitions, fixed matched-depth comparisons, and calibrated standard all-miss loss results. The CSV gives means, seed-level standard errors, and 95 percent intervals. Frequency intervals use Wilson intervals.

The raw pooled correlation is expected to be positive because shared query difficulty induces marginal association. The residual off-diagonal correlation should center near zero because model outcomes are independent after conditioning on `X`. At each fixed matched depth, exact factorization minimizes conditional population all-miss risk pointwise. A `false_dependence_win` records a finite-test sample where a correlation heuristic has lower empirical risk despite having no lower conditional population risk. Individual seeds need not tie. The null invariant is that residual correlation centers near zero and neither dependence heuristic has a systematic population advantage over exact factorization.

Run its focused tests with:

```
python -m pytest -q tests/test_synthetic_conditional_independence.py
```

## Tests

Run the focused reporting tests:

```
python -m pytest -q tests/test_reporting_repair.py
```

Run the complete suite:

```
python -m pytest -q tests/
```

## License

MIT (see LICENSE).

## Citation

If you use this code, the run records, or the results in your work, please cite the paper
(https://arxiv.org/abs/2610.13276).

```bibtex
@article{kasa2026coproute,
  title         = {CopRoute: Dependence-Aware Conformal Routing for Language Models},
  author        = {Kasa, Siva Rajesh and Kasa, Santhosh Kumar and Gupta, Karan},
  journal       = {arXiv preprint arXiv:2610.13276},
  year          = {2026},
  eprint        = {2610.13276},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.13276}
}
```
