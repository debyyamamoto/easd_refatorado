# Experiments

This directory contains aggregation, statistical tests, and generated outputs
for the article. The fixed article protocol lives in `main_stats.sh`, which
calls the generic `main.py` command with the paper datasets and parameters.

## Run Article Experiments

```bash
bash ./experiments/main_stats.sh
```

The shell script is the fixed article protocol. It runs all article datasets
with both `complement` and `population` baselines, 30 executions each, writes
results to `experiments/results/<dataset>/<baseline>/`, and then compiles
`experiments/resultados_complement.csv` and
`experiments/resultados_population.csv`.

For focused debugging runs, call `main.py` directly with fewer executions:

```bash
uv run python main.py datasets/files/cancer.parquet -time time -event status -comp population -exe 3
```

## Compile Metrics

```bash
uv run python experiments/collect_results.py --baseline complement
uv run python experiments/collect_results.py --baseline population
```

These commands generate `experiments/resultados_complement.csv` and
`experiments/resultados_population.csv`.

## Hypothesis Tests

```bash
uv run python experiments/test_wilcoxon.py
```

By default this compares `experiments/resultados_complement.csv` against
`experiments/results_esmam/metrics_baseline-complement.csv` and saves
`experiments/wilcoxon.csv`.
