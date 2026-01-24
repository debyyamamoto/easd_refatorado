from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import itertools
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from autorank import autorank
from .dataset import Dataset
from rich.console import Console
console = Console()

ALPHA = 0.05
STAT_ORDER = "ascending"
Rule = Tuple[Sequence[Union[int, str]], Sequence[Any]]


@dataclass(frozen=True)
class RunMetrics:
    """Container de métricas de um run (uma seed)."""

    exceptionality: float
    n_sg: int
    length: float
    sgCov: float
    setCov: float
    description_redundancy: float
    coverage_redundancy: float
    cr: float
    model_redundancy: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exceptionality": float(self.exceptionality),
            "#sg": int(self.n_sg),
            "length": float(self.length),
            "sgCov": float(self.sgCov),
            "setCov": float(self.setCov),
            "description redundancy": float(self.description_redundancy),
            "coverage redundancy": float(self.coverage_redundancy),
            "CR": float(self.cr),
            "model redundancy": float(self.model_redundancy),
        }


# -----------------------------
# Métricas principais
# -----------------------------


def compute_run_metrics(
    df: pd.DataFrame,
    rules: Sequence[Rule],
    time_col: str,
    event_col: str,
    *,
    alpha: float = 0.05,
    dataset_obj: Optional[Dataset] = None,
    baseline: str = "population",
) -> RunMetrics:
    """
    Calcula métricas do artigo-base para um run.

    - exceptionality: proporção de regras com p(SG vs Complemento) <= alpha
    - #sg: números de subgrupos (igual ao tamanho do Top-K)
    - length: tamanho médio da descrição (nº atributos)
    - sgCov: cobertura média |SG|/|D|
    - setCov: |SG|/|D|
    - description redundancy: média sim_D par-a-par (|I∩|/min)
    - cover redundancy: média sim_C par-a-par (|G∩|/min)
    - model redundancy: média sim_M par-a-par, onde sim_M = 1 se p(A vs B) > alpha
      (para evitar artefatos por sobreposição, comparamos A/B vs B/A)
    - CR: redundância por instância (média do excesso de cobertura normalizado)

    dataset_obj é necessário se as regras usam índices inteiros.
    """
    if df is None or df.empty or not rules:
        return RunMetrics(
            exceptionality=0.0,
            n_sg=0,
            length=0.0,
            sgCov=0.0,
            setCov=0.0,
            description_redundancy=0.0,
            coverage_redundancy=0.0,
            cr=0.0,
            model_redundancy=0.0,
        )

    if time_col not in df.columns or event_col not in df.columns:
        raise ValueError(f"df deve conter '{time_col}' e '{event_col}'. Colunas: {list(df.columns)}")

    num_samples = df.shape[0]
    n_sg = int(len(rules))

    cover_sets: List[set] = []
    desc_item_sets: List[set] = []
    rule_lengths: List[int] = []
    pvals_vs_comp: List[float] = []

    for rule in rules:
        try:
            idx = covered_indices(rule, df, dataset_obj=dataset_obj)
        except Exception:
            idx = []

        cov = set(idx)
        cover_sets.append(cov)

        col_names, cons = _rule_to_column_names(rule, dataset_obj=dataset_obj)
        items = {(col, _normalize_constraint(c)) for col, c in zip(col_names, cons)}
        desc_item_sets.append(items)

        rule_lengths.append(len(col_names))
        if baseline == "population":
            pvals_vs_comp.append(_pvalue_subgroup_vs_population(df, idx, time_col, event_col))
        else:
            pvals_vs_comp.append(_pvalue_subgroup_vs_complement(df, idx, time_col, event_col))

    # exceptionality: proporção de subgrupos "significativamente diferentes" do complemento
    exceptionality = float(np.mean([1.0 if p <= alpha else 0.0 for p in pvals_vs_comp])) if pvals_vs_comp else 0.0

    length = float(np.mean(rule_lengths)) if rule_lengths else 0.0

    sgCov = float(np.mean([len(c) / num_samples for c in cover_sets])) if cover_sets else 0.0
    union_cov = set().union(*cover_sets) if cover_sets else set()
    setCov = float(len(union_cov) / num_samples) if num_samples else 0.0

    # redundâncias par-a-par
    if n_sg < 2:
        desc_red = 0.0
        covarage_redundancy = 0.0
        model_red = 0.0
    else:
        simD_vals = []
        simC_vals = []
        simM_vals = []

        for i, j in itertools.combinations(range(n_sg), 2):
            simD_vals.append(_sim_min_intersection(desc_item_sets[i], desc_item_sets[j]))
            simC_vals.append(_sim_min_intersection(cover_sets[i], cover_sets[j]))

            # sim_M = 1 se p > alpha (modelos "similares" => redundância)
            a = list(cover_sets[i] - cover_sets[j])
            b = list(cover_sets[j] - cover_sets[i])
            p_ab = _pvalue_between_groups(df, a, b, time_col, event_col)
            simM_vals.append(1.0 if p_ab > alpha else 0.0)

        desc_red = _pairwise_average(simD_vals)
        covarage_redundancy = _pairwise_average(simC_vals)
        model_red = _pairwise_average(simM_vals)

    # CR: redundância por instância (excesso de cobertura normalizado)
    if n_sg <= 1:
        cover_redundancy = 0.0
    else:
        counts = np.zeros(num_samples, dtype=int)
        for cov in cover_sets:
            for k in cov:
                k = int(k)
                if 0 <= k < num_samples:
                    counts[k] += 1
        cover_redundancy = float(np.mean(np.clip((counts - 1) / max(1, n_sg - 1), 0.0, 1.0)))

    return RunMetrics(
        exceptionality=exceptionality,
        n_sg=n_sg,
        length=length,
        sgCov=sgCov,
        setCov=setCov,
        description_redundancy=float(desc_red),
        coverage_redundancy=float(covarage_redundancy),
        cr=float(cover_redundancy),
        model_redundancy=float(model_red),
    )


def output_metrics(p_metrics_list: list, p_file_name: str):
    """
    Exporta métricas para CSV.
    
    Args:
        p_metrics_list: Lista de dicionários com métricas (um dict por run/seed)
        p_file_name: Caminho do arquivo CSV de saída
        
    Comportamento:
    - 0 runs: apenas aviso, não gera arquivo
    - 1 run: exporta valores diretos
    - 2+ runs: calcula mean±std, tenta autorank, fallback para stats básicas
    """
    # Valida entrada
    if not p_metrics_list:
        console.log("Nenhuma métrica para exportar", style="yellow")
        return
    
    # Converte para DataFrame
    metrics_df = pd.DataFrame(p_metrics_list)
    n_runs = len(metrics_df)
    n_metrics = len(metrics_df.columns)
    
    console.log(f"📊 Exportando {n_runs} run(s) com {n_metrics} métrica(s)", style="blue")
    
    # CASO 1: Apenas 1 run - exporta valores diretos
    if n_runs == 1:
        metrics_df.to_csv(p_file_name, index=False, float_format='%.4f')
        console.log(f"✅ Métricas (1 run) → {p_file_name}", style="green")
        return
    
    # CASO 2: Múltiplos runs - tenta usar autorank
    output_df = None
    
    try:
        results = autorank(
            metrics_df,
            alpha=ALPHA,
            verbose=False,
            order=STAT_ORDER
        )
        
        if results is not None and hasattr(results, 'rankdf'):
            ranked_df = results.rankdf
            
            # Verifica se tem colunas esperadas
            required_cols = ['mean', 'std']
            if all(col in ranked_df.columns for col in required_cols):
                # Formata: mean±std
                output_df = pd.DataFrame()
                output_df['mean±std'] = (
                    ranked_df['mean'].round(4).astype(str) + 
                    '±' + 
                    ranked_df['std'].round(4).astype(str)
                )
                
                # Adiciona meanrank se disponível
                if 'meanrank' in ranked_df.columns:
                    output_df['meanrank'] = ranked_df['meanrank'].round(2)
                
                output_df.to_csv(p_file_name)
                console.log(f"Métricas (autorank) → {p_file_name}", style="green")
                return
    
    except Exception as e:
        console.log(f"Autorank falhou: {str(e)[:50]}...", style="yellow")
    
    # CASO 3: Fallback - estatísticas básicas
    output_df = _compute_basic_statistics(metrics_df)
    output_df.to_csv(p_file_name, float_format='%.4f')
    console.log(f"Métricas (básicas) → {p_file_name}", style="green")


def _compute_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas quando autorank não está disponível.
    
    Args:
        df: DataFrame com runs nas linhas e métricas nas colunas
        
    Returns:
        DataFrame com estatísticas agregadas (uma linha por métrica)
    """
    stats_list = []
    
    for col in df.columns:
        values = df[col]
        
        mean_val = values.mean()
        std_val = values.std()
        
        stats_list.append({
            'metric': col,
            'mean': mean_val,
            'std': std_val,
            'mean±std': f"{mean_val:.4f}±{std_val:.4f}",
            'min': values.min(),
            'max': values.max(),
            'median': values.median(),
        })
    
    result_df = pd.DataFrame(stats_list)
    result_df.set_index('metric', inplace=True)
    
    return result_df


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _normalize_constraint(constraint: Any) -> Any:
    """
    Normaliza constraint para ser comparável/hashable em similaridade de descrição.
    """
    if isinstance(constraint, (list, tuple, set)):
        if len(constraint) == 0:
            return tuple()

        # intervalo numérico (low, high)
        if len(constraint) == 2 and _is_number(list(constraint)[0]) and _is_number(list(constraint)[1]):
            low = float(list(constraint)[0])
            high = float(list(constraint)[1])
            return (round(low, 10), round(high, 10))

        # categórico: ordenar para forma canônica
        try:
            return tuple(sorted(constraint))
        except Exception:
            return tuple(constraint)

    return constraint


# -----------------------------
# Cobertura e p-values
# -----------------------------


def _rule_to_column_names(rule: Rule, dataset_obj: Optional[Dataset] = None) -> Tuple[List[str], List[Any]]:
    attrs, cons = rule
    col_names: List[str] = []
    for a in attrs:
        if isinstance(a, int):
            if dataset_obj is None:
                raise ValueError("Regra contém índices inteiros, mas dataset_obj=None; não é possível mapear colunas.")
            else:
                col_names.append(dataset_obj.get_col_name(a))
        else:
            col_names.append(str(a))
    return col_names, list(cons)


def covered_indices(rule: Rule, df: pd.DataFrame, dataset_obj: Optional[Dataset] = None) -> List[int]:
    """
    Retorna os índices cobertos pela regra no df.

    Assumimos df com index compatível com seleção por .loc (idealmente 0..num_samples-1).
    """
    if rule is None:
        return []
    attrs, cons = rule
    if not attrs or not cons or len(attrs) != len(cons):
        return []

    col_names, cons = _rule_to_column_names(rule, dataset_obj=dataset_obj)

    mask = pd.Series(True, index=df.index)
    for col, c in zip(col_names, cons):
        if col not in df.columns:
            return []

        s = df[col]

        # categórico (no seu evaluation.py: isinstance(constraint[0], str))
        if isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(list(c)[0], str):
            if len(c) > 1:
                mask &= s.isin(list(c))
            else:
                mask &= s == list(c)[0]
        else:
            # numérico: (low, high)
            if isinstance(c, (list, tuple)) and len(c) == 2 and _is_number(c[0]) and _is_number(c[1]):
                low = float(c[0])
                high = float(c[1])
                mask &= (s.astype(float) >= low) & (s.astype(float) <= high)
            else:
                # fallback
                mask &= s == c

    return df.index[mask].to_list()


def _survdiff_pvalue(times: pd.Series, events: pd.Series, group: pd.Series) -> float:
    """
    p-valor via statsmodels.duration.survdiff. Em falhas, retorna 1.0.
    """
    try:
        res = sm.duration.survdiff(times, events, group=group)
        p = float(res[1])
        if math.isnan(p) or p < 0 or p > 1:
            return 1.0
        return p
    except Exception:
        return 1.0


def _pvalue_between_groups(
    df: pd.DataFrame, idx_a: Sequence[int], idx_b: Sequence[int], time_col: str, event_col: str
) -> float:
    """
    Teste A vs B (para model redundancy).
    """
    a = list(dict.fromkeys(idx_a))
    b = list(dict.fromkeys(idx_b))
    if len(a) == 0 or len(b) == 0:
        return 1.0

    ga = pd.Series("A", index=a)
    gb = pd.Series("B", index=b)
    group = pd.concat([ga, gb], axis=0, ignore_index=False).sort_index()

    times = df.loc[group.index, time_col]
    events = df.loc[group.index, event_col]

    return _survdiff_pvalue(times, events, group)


def _pvalue_subgroup_vs_population(df, sg_idx, time_col, event_col):
    num_samples = df.shape[0]
    if num_samples == 0:
        return 1.0

    sg_set = set(sg_idx)
    if len(sg_set) == 0 or len(sg_set) == num_samples:
        return 1.0

    # rotula dataset inteiro: sg vs pop (resto)
    group = pd.Series("pop", index=df.index)
    group.loc[list(sg_set)] = "sg"

    times = df.loc[group.index, time_col]
    events = df.loc[group.index, event_col]

    return _survdiff_pvalue(times, events, group)


def _pvalue_subgroup_vs_complement(df: pd.DataFrame, sg_idx: Sequence[int], time_col: str, event_col: str) -> float:
    """
    Teste SG vs Complemento, seguindo a lógica do seu evaluation.py.
    """
    num_samples = df.shape[0]
    if num_samples == 0:
        return 1.0

    sg_set = set(sg_idx)
    if len(sg_set) == 0 or len(sg_set) == num_samples:
        return 1.0

    comp_idx = [i for i in df.index if i not in sg_set]

    sg = pd.Series("sub_group", index=list(sg_set))
    cpm = pd.Series("complement", index=comp_idx)
    group = pd.concat([sg, cpm], axis=0, ignore_index=False).sort_index()

    times = df.loc[group.index, time_col]
    events = df.loc[group.index, event_col]

    return _survdiff_pvalue(times, events, group)


# -----------------------------
# Similaridades e redundâncias
# -----------------------------


def _sim_min_intersection(a: set, b: set) -> float:
    """
    Similaridade como na Tabela 2: |A ∩ B| / min(|A|, |B|)
    """
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return inter / denom


def _pairwise_average(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(np.mean(vals))
