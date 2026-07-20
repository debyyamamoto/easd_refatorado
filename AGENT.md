# Agent Guide

## Project Intent

MEASE is the reusable algorithm implementation. The article experiments are a
separate reproducibility layer. Keep these two concerns apart:

- `easd/`: library code, metrics, algorithm internals, and reusable runners.
- `main.py`: generic CLI for running MEASE on any parquet dataset.
- `main_stats.sh`: fixed article protocol, calling `main.py` with the exact
  datasets, columns, and parameters used in the paper.
- `experiments/`: aggregation scripts, statistical tests, and generated
  experiment outputs.
- `datasets/files/`: local parquet datasets and metadata used by examples and
  article runs.

## Reproducible Workflow

Use the generic CLI for ad hoc datasets:

```bash
uv run python main.py datasets/files/cancer.parquet -time time -event status
```

Use the experiment protocol for the paper:

```bash
./main_stats.sh
uv run python experiments/test_wilcoxon.py
```

Article outputs should stay under `experiments/results/` or named aggregate
files in `experiments/`. Generic runs should default to `results/`.

## Change Rules

- Do not add article-only dataset lists, Wilcoxon tests, or baseline comparison
  tables to `main.py`.
- Prefer adding reusable execution behavior to `easd/runner.py`, then call it
  from `main.py`.
- Keep the fixed article protocol in `main_stats.sh`; it should call `main.py`
  directly with article parameters.
- Preserve result file names unless a migration plan is documented, because
  aggregation scripts depend on them.
- Keep generated result CSVs out of algorithm changes unless the task is
  explicitly about reproducing or updating article artifacts.
- When touching metrics, verify both a one-run CLI path and the aggregate
  experiment path.
- Keep quantitative-methods experiments as a separate reproducibility layer
  under `experiments/`. Factorial designs, diagnostic plots, and hypothesis
  tests should call `main.py` with explicit parameters instead of changing the
  default article protocol.

## Quantitative Methods Roadmap

The planned experiments are documented in
`experiments/quantitative_methods_plan.md`.

- Runtime design: full `2^2` factorial with `population_size` and `dataset`
  (`veteran` as low sample size, `mgus2` as high sample size), response
  `total_time`.
- Exceptionality design: full `2^3` factorial with `population_size`,
  `generations`, and `rate_policy`, response `exceptionality`.
- Algorithm comparison: MEASE versus EsmamDS for `total_time` and
  `exceptionality`, using paired tests when run alignment allows it.
- Required assumption checks: residual additivity/interaction review,
  homoscedasticity, residuals-vs-fitted cloud, Q-Q plot normality check, and
  independence through fresh randomized executions without cache reuse.

## Current Article Protocol

The fixed paper script uses:

- Datasets: `breast-cancer`, `cancer`, `carcinoma`, `lung`, `mgus2`, `veteran`.
- Baselines: `complement`, `population`.
- Executions: `30`.
- Parameters: `generations=500`, `population=500`, `restart_gen=5`,
  `restart_pop=5`, `restart_pct=10`, `alpha=0.10`, `ksize=10`,
  `threshold=0.9`.

Update this section whenever the article protocol changes.
