# Quantitative Methods Experiment Plan

This document maps the additional experiments needed to evaluate MEASE with
quantitative algorithm experimentation methods. The search/mining mechanism
itself should remain stable while these experiments are implemented, except for
the explicit configuration needed to enable fixed or adaptive crossover and
mutation rates.

## Target Questions

1. Do adaptive crossover and mutation rates influence exceptionality?
2. Which factors have the strongest impact on algorithm runtime?
3. Is there a statistically significant difference in runtime and
   exceptionality between EsmamDS and MEASE?

See `assumption_audit.md` for the current residual diagnostics, transformation
comparison, and follow-up designs proposed after the first 2^k r analyses.

## Experiment 1: 2^k Factorial Design For Runtime

Response variable:

- `total_time`, collected from each run's `_Info.csv` file.

Factors:

| Factor | Low (-1) | High (+1) | Notes |
| --- | --- | --- | --- |
| `population_size` | 50 | 1000 | Main controllable algorithm factor. |
| `dataset` | `veteran` | `mgus2` | Proxy for sample size. `veteran` has 137 rows; `mgus2` has 1338 rows. |

Fixed controls:

- `generations=500`
- `rate_policy=adaptive`
- `baseline=complement`
- Same `alpha`, `ksize`, `threshold`, restart settings, and hardware.

Design size:

- 2 factors, full factorial: `2^2 = 4` treatment combinations.
- Use repeated independent executions per treatment. The article protocol uses
  30 executions; keep 30 as the confirmatory default unless runtime becomes
  prohibitive.

Primary analysis:

- Build the coded `2^k r` table with levels `-1` and `+1`.
- Compute each contrast manually as `q_i`, then derive the effect estimate,
  coefficient, sum of squares, mean square, `F0`, and p-value.
- Report the explained variability for each factor and interaction as
  `SS_i / SS_total`, plus the experimental error fraction as `SSE / SS_total`.
- Treat the main-effects model as the additive model and use interaction
  contrasts as the additivity diagnostic.
- Runtime often has right skew; evaluate both raw runtime and `log(runtime)` if
  residual diagnostics show non-normality, heteroscedasticity, or significant
  interaction effects.

## Experiment 2: 2^k Factorial Design For Exceptionality

Response variable:

- `exceptionality`, collected from each run's `_RulesMetricsResult.csv` file.

Factors:

| Factor | Low (-1) | High (+1) | Notes |
| --- | --- | --- | --- |
| `population_size` | 50 | 1000 | Search breadth. |
| `generations` | 50 | 500 | Search budget. |
| `rate_policy` | `fixed` | `adaptive` | Fixed means crossover=60% and mutation=40%. |

Fixed controls:

- Primary dataset should be fixed to avoid turning this into a `2^4` design.
  Use `mgus2` for the confirmatory run because it is the larger selected
  dataset. A smaller pilot on `veteran` can be useful before the final run.
- `baseline=complement`
- Same `alpha`, `ksize`, `threshold`, restart settings, and hardware.

Design size:

- 3 factors, full factorial: `2^3 = 8` treatment combinations.
- Use repeated independent executions per treatment; use 30 executions as the
  confirmatory default.

Primary analysis:

- Build the coded `2^k r` table with levels `-1` and `+1`.
- Compute each contrast manually as `q_i`, then derive the effect estimate,
  coefficient, sum of squares, mean square, `F0`, and p-value.
- Report the explained variability for each factor and interaction as
  `SS_i / SS_total`, plus the experimental error fraction as `SSE / SS_total`.
- Report whether `rate_policy` has a meaningful main effect and whether its
  interaction contrasts with `population_size` or `generations` indicate
  non-additivity.
- If assumptions are weak, supplement the factorial method with bootstrap
  confidence intervals or a transformed response.

## Experiment 2b: 2^k Factorial Design For Top-K Rule Quality

Response variable:

- `mean_rule_score`, computed as the mean `Rule_Score` over the Top-K rules in
  each run's `_DetailedRules.csv` file.

Motivation:

- `exceptionality` is a coarse proportion of significant Top-K rules. With
  `ksize=10`, one execution can only move in steps of 0.1, so a large part of
  the variability may remain as experimental error.
- `mean_rule_score` is continuous and directly tied to the log-rank based rule
  quality optimized by MEASE, making it a better response for explaining Top-K
  quality variability.

Factors:

| Factor | Low (-1) | High (+1) | Notes |
| --- | --- | --- | --- |
| `population_size` | 25 | 500 | Search breadth. |
| `generations` | 50 | 500 | Search budget. |
| `rate_policy` | `fixed` | `adaptive` | Fixed means crossover=60% and mutation=40%. |

Fixed controls:

- `dataset=cancer` for the first confirmatory run, matching the current
  exceptionality design and allowing direct comparison against the coarse
  response.
- `ksize=10`
- `baseline=complement`
- Same `alpha`, threshold, restart settings, and hardware.

Primary analysis:

- Use the same manual `2^k r` pipeline: `SSY`, `SS0`, `SS_i`, `SSE`, `SST`,
  `q_i`, `s_qi`, F test, and explained variance `SS_i / SST`.
- Compare the explained-variance profile against the `exceptionality` design.
  If `mean_rule_score` is well explained while `exceptionality` is not, report
  that the selected factors affect the quality score more clearly than the
  binary significance proportion.
- If this design still leaves most variability as `SSE / SST`, add a second
  quality experiment with `ksize` as a factor, for example `ksize=5` versus
  `ksize=20`, while keeping `rate_policy=adaptive` fixed.

## Experiment 3: MEASE Versus EsmamDS

Responses:

- `exceptionality`
- `total_time`

Design:

- Compare MEASE and EsmamDS on the same datasets and, when possible, with the
  same number of independent executions.
- Prefer paired comparisons when runs can be aligned by dataset and execution
  index. If the EsmamDS artifact is not paired by seed/run, document the
  limitation and use an unpaired alternative.

Primary analysis:

- For paired samples: Wilcoxon signed-rank test, plus effect size and median
  difference.
- For unpaired samples: Mann-Whitney U or a permutation test.
- Report practical significance, not only p-values.

Required data improvements:

- Ensure EsmamDS runtime is available in the baseline result table, not only
  quality metrics.
- Normalize column names for both algorithms to include:
  `algorithm`, `dataset`, `execution`, `exceptionality`, `total_time`.

## Assumptions To Check

The factorial analyses should only be treated as valid after checking these
diagnostics:

| Assumption | Diagnostic | Expected Pattern |
| --- | --- | --- |
| Additivity of factor effects | Main effects and interaction plots | Interactions should be explicit; if strong, interpret interactions instead of isolated main effects. |
| Homoscedasticity | Residuals vs fitted values and scale-location plot | Residual spread should be roughly constant across fitted values. |
| Residual cloud around zero | Residuals vs fitted values | No clear curve, funnel, or structured pattern. |
| Normal residuals | Q-Q plot and optional Shapiro-Wilk test | Points close to the Q-Q line; tests are secondary because repeated runs can make tiny deviations significant. |
| Independence | Randomized execution order, separate processes, no reuse of generated outputs | Each treatment execution should be generated fresh, without cache or warm-start artifacts. |

If assumptions fail:

- Try response transformation first, especially `log(runtime)`.
- Use robust standard errors or bootstrap confidence intervals.
- Keep the diagnostic plots in the experiment artifact directory.

## Implementation Improvements Needed

1. Add a public `rate_policy` option to the generic CLI. Implemented:
   - `adaptive` preserves the current behavior.
   - `fixed` uses crossover=60% and mutation=40%.
   - The policy and concrete initial/final rates are saved in `_Info.csv`.

2. Add factorial experiment shell scripts under `experiments/`:
   - `main_factorial_runtime.sh`
   - `main_factorial_exceptionality.sh`
   - Both should call `main.py` directly with fixed treatment parameters.

3. Add a factorial result collector. Implemented in `experiments/analyze_factorial.py`:
   - Read per-run `_Info.csv` and `_RulesMetricsResult.csv` files.
   - Add factor columns such as `population_size`, `dataset_level`,
     `generations`, `rate_policy`, `execution`, and `seed`.
   - Save tidy CSVs suitable for statistical analysis.

4. Add an analysis script. Implemented in `experiments/analyze_factorial.py`:
   - Build the classroom-style `2^k r` method table.
   - Export a treatment-level table with factor signs, all replications,
     treatment totals, treatment means, `SSY_i`, and `SSE_i`.
   - Export a step-by-step table with `SSY`, `SS0`, `SS_i`, `SSE`, `SST`,
     `q_i`, `s_qi`, and explained variance.
   - Compute `q_i`, effects, sums of squares, ANOVA, and p-values manually.
   - Export `SS_i / SS_total` for each factor and interaction, plus
     `SSE / SS_total` for experimental error.
   - Use the additive main-effects model for residual diagnostics.
   - Run an automatic log-scale analysis when raw-scale diagnostics indicate
     multiplicative behavior.
   - Save diagnostic plots: residuals vs fitted, scale-location, Q-Q plot, and
     main effects/interaction plots.

5. Add run hygiene controls:
   - Randomize treatment execution order.
   - Use fresh output directories per experiment batch.
   - Avoid reading aggregate files as inputs for later treatments.
   - Record machine-independent metadata where possible: Python version,
     package versions, OS, CPU count, and timestamp.

6. Extend the MEASE versus EsmamDS comparison:
   - Include runtime in addition to exceptionality.
   - Produce one tidy comparison table for hypothesis tests.
   - Keep the existing Wilcoxon script for paper metric comparisons, but add a
     focused script for runtime/exceptionality comparison.
