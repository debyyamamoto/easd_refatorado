"""Fase 5 da vetorização do log-rank: valida a implementação vetorizada.

Dois baselines, duas referências diferentes:

- "complement": o comportamento não mudou -- compara o `fitness()` vetorizado
  contra o gabarito da Fase 1 (`golden_reference_logrank.csv`), gerado com o
  `fitness()` antigo (statsmodels), sobre o mesmo dataset sintético congelado.

- "population": o comportamento MUDOU de propósito -- a versão vetorizada
  restaura a semântica original do ESMAM (subgrupo vs. população inteira, que
  já contém o subgrupo -- contado duas vezes no risco combinado), que o
  `fitness()` antigo NÃO reproduzia (ele tratava "population" == "complement",
  ver anotações_adaptação.md). Por isso esse branch é comparado contra uma
  referência calculada aqui mesmo, via `statsmodels`, replicando literalmente
  o `set_fitness` original do ESMAM (concatenação do subgrupo com a população
  inteira).

Uso:
    PYTHONPATH=. uv run python experiments/logrank_vectorization/validate_vectorized_logrank.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from build_golden_reference import ALPHA, DEFAULT_OUTPUT, build_rule_cases

from easd.dataset import Dataset
from easd.evaluation import RuleEvaluator

DATASET_FILE = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_dataset.parquet")
TOLERANCE = 1e-6


def esmam_original_population_p_value(dataset_obj: Dataset, rule_group_indices: list[int]) -> float:
    "Réplica literal do set_fitness original do ESMAM (baseline 'population')."
    times_all = dataset_obj._original_data[dataset_obj.surv_name].to_numpy(dtype=np.float64)
    events_all = dataset_obj._original_data[dataset_obj._event_name].to_numpy(dtype=np.float64)

    sg_idx = np.asarray(rule_group_indices, dtype=int)
    times = np.concatenate([times_all[sg_idx], times_all])
    events = np.concatenate([events_all[sg_idx], events_all])
    group_id = ["sg"] * len(sg_idx) + ["pop"] * len(times_all)

    try:
        _, p_value = sm.duration.survdiff(time=times, status=events, group=group_id)
        if pd.isna(p_value):
            p_value = 1.0
    except Exception:
        p_value = 1.0
    return float(p_value)


def validate() -> pd.DataFrame:
    golden = pd.read_csv(DEFAULT_OUTPUT)
    data = pd.read_parquet(DATASET_FILE)
    dataset_obj = Dataset(data, "time", "status")
    rule_cases = dict(build_rule_cases(dataset_obj, data))

    rows = []
    for comparacao in ("population", "complement"):
        evaluator = RuleEvaluator(dataset_obj, comparacao, ALPHA)
        for case_id, rule in rule_cases.items():
            rule_group_indices = evaluator.get_covered_indices(rule, dataset_obj)
            new_fitness = evaluator.fitness(rule, dataset_obj)
            support = len(rule_group_indices) / dataset_obj.size if dataset_obj.size else 0.0
            in_support_range = 0.05 <= support <= 0.55

            if comparacao == "complement":
                golden_row = golden[
                    (golden["case_id"] == case_id) & (golden["comparacao"] == "complement")
                ].iloc[0]
                reference_fitness = float(golden_row["fitness"])
                reference_label = "gabarito_fase1 (statsmodels/complement)"
            else:
                if len(rule_group_indices) < 1 or not in_support_range:
                    reference_fitness = 0.0
                else:
                    p_ref = esmam_original_population_p_value(dataset_obj, rule_group_indices)
                    reference_fitness = (1 - p_ref) * (support**ALPHA)
                reference_label = "esmam_original (statsmodels/population duplicado)"

            diff = abs(new_fitness - reference_fitness)
            rows.append(
                {
                    "case_id": case_id,
                    "comparacao": comparacao,
                    "reference": reference_label,
                    "fitness_new": new_fitness,
                    "fitness_reference": reference_fitness,
                    "abs_diff": diff,
                    "pass": diff < TOLERANCE,
                }
            )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_fail = int((~df["pass"]).sum())
    if n_fail:
        print(f"\n{n_fail} caso(s) fora da tolerância ({TOLERANCE}).")
    else:
        print(f"\nTodos os {len(df)} casos dentro da tolerância ({TOLERANCE}).")
    return df


if __name__ == "__main__":
    validate()
