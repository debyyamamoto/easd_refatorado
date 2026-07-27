# Score Metric Benchmark Summary

Date: 2026-07-24

Command used for the main scalability run:

```bash
uv run python experiments/artificial_datsets/score_metric_benchmark.py \
  --subjects 10000 50000 100000 \
  --metrics legacy_logrank fast_logrank km_cvm km_abc \
  --generations 5 \
  --population 80 \
  --ksize 5 \
  --km-time-bins 512 \
  --output-dir /tmp/mease_score_benchmark_final \
  --dataset-dir /tmp/mease_score_benchmark_final/datasets
```

The synthetic `subgroup` label was passed through `label_col`, so it was used for
F1 evaluation and removed from the covariates before MEASE ran.

## Main Results

| subjects | score_metric | algorithm_time_s | speedup_vs_legacy | max_f1_score |
| ---: | --- | ---: | ---: | ---: |
| 10,000 | legacy_logrank | 1.355 | 1.00x | 0.1050 |
| 10,000 | fast_logrank | 0.344 | 3.94x | 0.1050 |
| 10,000 | km_cvm | 0.379 | 3.57x | 0.0948 |
| 10,000 | km_abc | 0.365 | 3.71x | 0.1905 |
| 50,000 | legacy_logrank | 3.343 | 1.00x | 0.1009 |
| 50,000 | fast_logrank | 0.654 | 5.11x | 0.1009 |
| 50,000 | km_cvm | 0.665 | 5.03x | 0.0814 |
| 50,000 | km_abc | 0.667 | 5.01x | 0.0910 |
| 100,000 | legacy_logrank | 7.207 | 1.00x | 0.1154 |
| 100,000 | fast_logrank | 1.229 | 5.86x | 0.1263 |
| 100,000 | km_cvm | 1.264 | 5.70x | 0.1188 |
| 100,000 | km_abc | 1.238 | 5.82x | 0.1263 |

## Isolated Fitness Evaluation

On the 100,000-subject synthetic dataset with the same 80-individual population:

| score_metric | total_s | ms_per_individual | nonzero_fitness |
| --- | ---: | ---: | ---: |
| legacy_logrank | 0.7948 | 9.93 | 7 |
| fast_logrank | 0.0315 | 0.39 | 7 |
| km_cvm | 0.0354 | 0.44 | 7 |
| km_abc | 0.0398 | 0.50 | 7 |

This isolates the score calculation itself. The full MEASE runtime still includes
rule coverage masks, genetic operators, Top-K redundancy checks, final metrics,
and output handling.

## Rule Quality Notes

- `fast_logrank` preserved the ranking behavior of the legacy log-rank most
  closely and obtained similar F1 in the short runs.
- `km_abc` gave the best F1 in the 10,000-subject short run and tied
  `fast_logrank` in the 100,000-subject run.
- `km_cvm` was fast, but its squared-distance scale produced smaller numerical
  rule scores and did not improve F1 in these short runs.
- In a longer 10,000-subject run with 20 generations, population 200, and Top-K
  10, `km_abc` reached the best observed F1 (`0.2678`), followed by `km_cvm`
  (`0.1467`) and both log-rank variants (`0.1299`).

The low F1 values are not only a score-metric issue. The mined Top-K rules were
mostly univariate, while the planted rule used two features. Several high-scoring
rules selected complementary regions that are also survival-discrepant but do not
match the planted `subgroup == 1` label. Recovering the planted subgroup more
reliably likely requires search/representation changes in addition to the faster
fitness metrics.
