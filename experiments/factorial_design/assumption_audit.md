# Factorial Assumption Audit

This note summarizes the current 2^k r assumption checks and the follow-up
designs to run when a response does not satisfy the classroom factorial model
assumptions.

## Current Results

| Experiment | Response scale | Status | Main evidence |
| --- | --- | --- | --- |
| `runtime` | `log(total_time)` | Valid for the 2^k r analysis | Normality, Breusch-Pagan, Levene, and additivity checks pass. `SSE/SST = 3.63%`; population explains `91.22%` and dataset level explains `5.14%`. |
| `runtime` | raw `total_time` | Not valid | Raw runtime fails normality, homoscedasticity, and additivity. Keep the log scale. |
| `score` | `log(exceptionality)` | Partially invalid | Homoscedasticity and additivity pass, but residual normality fails. The selected factors explain very little: `SSE/SST = 91.97%`. |
| `score` | Yeo-Johnson `exceptionality` | Statistically cleaner, weak explanatory value | Normality, homoscedasticity, and additivity pass, but `SSE/SST = 92.68%`. The transform helps the residual assumptions, not the substantive explanation. |
| `topk_quality` | raw/log/arcsin/logit `mean_rule_score` | Not valid | Normality, homoscedasticity, and additivity fail across the interpretable transforms. |
| `topk_quality` | Yeo-Johnson `mean_rule_score` | Not recommended | Normality passes, but homoscedasticity and additivity still fail. The fitted lambda is very large (`~25`), making the scale hard to interpret. |
| `topk_quality_controls` | `logit(mean_rule_score)` | Valid for the complete 2^k r analysis | Normality (`p=0.704`), Breusch-Pagan (`p=0.136`), and Levene (`p=0.309`) pass. `alpha` explains `99.58%` of the transformed variance. The `alpha:restart_gen` interaction is significant, so main effects should not be interpreted alone. |

The consolidated comparison table is saved at:

```text
experiments/factorial_design/factorial_analysis_transform_checks/assumption_summary.csv
```

The transform comparison for the new Top-K controls follow-up is saved at:

```text
experiments/factorial_design/factorial_analysis/topk_quality_controls_transform_comparison.csv
```

## Code And Execution Checks

The analysis pipeline was checked for the main and transformed outputs:

- The full-factorial fitted value matches each treatment mean:
  `max(abs(y_hat_full_factorial - treatment_mean))` is near machine precision.
- The decomposition closes:
  `SST = sum(SS_i) + SSE`, with only numerical floating-point error.
- The current designs are balanced (`r=15` for the generated outputs).
- `y_raw` is now preserved separately from transformed `y`, so the response
  scale is explicit in the output CSVs.
- Residual checks now separate `residuals_vs_fitted` from
  `residuals_vs_observed_response`. The fitted plot is the primary
  homoscedasticity diagnostic; the observed-response plot can show mechanical
  structure because observed response equals fitted response plus residual.

No evidence was found that the current failures are caused by missing runs or a
broken full-factorial reconstruction. The problematic Top-K quality response
appears to have genuinely different residual variance by configuration,
especially between low and high population settings.

## Transformation Guidance

- Use `log(total_time)` for runtime.
- For `exceptionality`, use Yeo-Johnson only if the goal is a cleaner residual
  model. It does not make the chosen factors explain the response well.
- Use `logit(mean_rule_score)` for the `topk_quality_controls` follow-up.
  `mean_rule_score` is a fractional response bounded in `[0, 1]`, so logit is
  more appropriate than a plain log transform. In the current runs, logit
  preserves the full-factorial residual assumptions.
- Do not force Yeo-Johnson on `mean_rule_score` as the main article result. It
  improves the Q-Q plot but leaves heteroscedasticity and interaction problems,
  and the transformed scale is difficult to interpret.
- For bounded quality metrics, also report the untransformed means by
  treatment. If the variance remains treatment-dependent, prefer a revised
  design or a model that explicitly handles unequal variances.

## Top-K Controls Result

The follow-up design
`experiments/factorial_design/designs/factorial_topk_quality_controls.csv`
was run with 15 replications per treatment. The raw `mean_rule_score` scale is
not appropriate for the classroom normal-error factorial assumptions because
the response is bounded and the treatment variance changes with the treatment
mean.

The logit scale fixes the formal residual assumptions for the complete
factorial model:

| Check | Result |
| --- | --- |
| Balanced design | pass, same `r=15` in all treatment cells |
| Full factorial reconstruction | pass, max `|y_hat - treatment_mean| = 4.44e-16` |
| Normal residuals | pass, Shapiro `p=0.704` |
| Homoscedastic residuals | pass, Breusch-Pagan `p=0.136`; Levene `p=0.309` |
| Sum-of-squares decomposition | pass, `sum(SS_i) + SSE - SST = 1.42e-14` |

Variance explained on the `logit(mean_rule_score)` scale:

| Term | Interpretation | Variance explained |
| --- | --- | --- |
| `A` | `alpha` | `99.5814%` |
| `AC` | `alpha:restart_gen` | `0.0333%` |
| `B` | `threshold` | `0.0075%` |
| `ABC` | `alpha:threshold:restart_gen` | `0.0046%` |
| `SSE` | experimental error | `0.3729%` |

The significant `AC` interaction (`p=0.0020`) is an interpretation diagnostic:
it means the effect of `alpha` changes slightly with `restart_gen`. It does not
invalidate the complete 2^k model, because the complete model explicitly
includes the interaction term as an additive coded component on the logit scale.

## New Designs To Run

Two executable follow-up designs were added.

1. `experiments/factorial_design/designs/factorial_runtime_scalability.csv`

   Purpose: confirm runtime effects with `generations` included as a factor,
   instead of keeping it fixed.

   Factors:

   | Factor | Low (-1) | High (+1) |
   | --- | --- | --- |
   | `population` | 50 | 1000 |
   | `dataset_level` | `veteran` | `mgus2` |
   | `generations` | 50 | 500 |

   Analyze this response on log scale.

2. `experiments/factorial_design/designs/factorial_topk_quality_controls.csv`

   Purpose: explain Top-K rule quality with factors that directly affect the
   search objective and rule filtering, while holding the computational budget
   fixed.

   Factors:

   | Factor | Low (-1) | High (+1) |
   | --- | --- | --- |
   | `alpha` | 0.10 | 0.70 |
   | `threshold` | 0.70 | 0.95 |
   | `restart_gen` | 2 | 10 |

   Fixed controls:

   - `population=500`
   - `generations=500`
   - `rate_policy=adaptive`
   - `ksize=10`
   - dataset `cancer`

## Suggested Commands

Run one follow-up design at a time:

```powershell
uv run python factorial.py --designs experiments\factorial_design\designs\factorial_runtime_scalability.csv
uv run python factorial.py --designs experiments\factorial_design\designs\factorial_topk_quality_controls.csv
```

Analyze the runtime follow-up on log scale:

```powershell
uv run python experiments\factorial_design\factorial_analysis.py --designs experiments\factorial_design\designs\factorial_runtime_scalability.csv --log_responses total_time
```

Analyze the Top-K follow-up first on the original score scale:

```powershell
uv run python experiments\factorial_design\factorial_analysis.py --designs experiments\factorial_design\designs\factorial_topk_quality_controls.csv --response_transforms mean_rule_score=identity
```

If assumptions still fail for Top-K quality, compare:

```powershell
uv run python experiments\factorial_design\factorial_analysis.py --designs experiments\factorial_design\designs\factorial_topk_quality_controls.csv --response_transforms mean_rule_score=logit
uv run python experiments\factorial_design\factorial_analysis.py --designs experiments\factorial_design\designs\factorial_topk_quality_controls.csv --response_transforms mean_rule_score=arcsin_sqrt
```

For the current `topk_quality_controls` results, use the logit output as the
main factorial analysis:

```powershell
uv run python experiments\factorial_design\factorial_analysis.py --designs experiments\factorial_design\designs\factorial_topk_quality_controls.csv --output_dir experiments\factorial_design\factorial_analysis --response_transforms mean_rule_score=logit
```

## Next Design If Heteroscedasticity Persists

If `mean_rule_score` remains heteroscedastic, run separate blocked designs by
population level or switch to controlled synthetic datasets. A good synthetic
2^3 design would vary:

| Factor | Low (-1) | High (+1) |
| --- | --- | --- |
| sample size `n` | small | large |
| feature count `p` | few | many |
| signal strength | weak | strong |

Responses should include `top1_rule_score`, `top5_mean_rule_score`,
`mean_rule_score`, and rule recovery if the planted rule is known.
