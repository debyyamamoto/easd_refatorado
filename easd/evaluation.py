from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .dataset import Dataset

ScoreMetric = Literal["legacy_logrank", "fast_logrank", "km_cvm", "km_abc"]
SCORE_METRICS: tuple[str, ...] = ("legacy_logrank", "fast_logrank", "km_cvm", "km_abc")

MIN_RELATIVE_SUPPORT = 0.05
MAX_RELATIVE_SUPPORT = 0.55
EPSILON = 1e-12


class RuleEvaluator:
    def __init__(
        self,
        dataset_obj: Dataset,
        comparacao,
        alpha,
        score_metric: ScoreMetric = "legacy_logrank",
        km_time_bins: int | None = 512,
    ):
        if comparacao not in ("complement", "population"):
            raise ValueError("comparacao must be either 'complement' or 'population'.")
        if score_metric not in SCORE_METRICS:
            raise ValueError(f"score_metric must be one of {SCORE_METRICS}.")

        self.dataset_obj = dataset_obj
        self.comparacao = comparacao
        self.score_metric = score_metric
        self.km_time_bins = None if km_time_bins is None or km_time_bins <= 0 else int(km_time_bins)
        self.sub_group_cases = dataset_obj.get_instances()
        self.p_value = None
        self._fitness = 0.0
        self._n = self.dataset_obj.size
        self._all_indices = set(range(self._n))
        self.alpha = alpha
        self._survival_times = self.dataset_obj._original_data[self.dataset_obj.surv_name]
        self._events = self.dataset_obj._original_data[self.dataset_obj._event_name]
        self._time_values = self._survival_times.to_numpy(dtype=float, copy=False)
        self._event_values = self._events.to_numpy(dtype=int, copy=False)
        self._event_observed = self._event_values.astype(bool)

        self._logrank_gamma = np.zeros(self._n, dtype=float)
        self._logrank_residual = self._event_values.astype(float)
        self._logrank_expected_total = 0.0
        if score_metric == "fast_logrank":
            self._precompute_fast_logrank()

        self._km_grid_times = np.array([], dtype=float)
        self._km_widths = np.array([], dtype=float)
        self._km_weights = np.array([], dtype=float)
        self._km_risk_bin = np.full(self._n, -1, dtype=int)
        self._km_event_bin = np.full(self._n, -1, dtype=int)
        self._km_pooled_subject_counts = np.array([], dtype=float)
        self._km_pooled_event_counts = np.array([], dtype=float)
        self._km_pooled_risk_counts = np.array([], dtype=float)
        self._km_pooled_survival = np.array([], dtype=float)
        if score_metric in ("km_cvm", "km_abc"):
            self._precompute_km_grid()

    def get_covered_mask(self, rule, dataset):
        if not rule or len(rule) < 2 or len(rule[0]) != len(rule[1]):
            return np.zeros(dataset.shape[0], dtype=bool)

        indices, values = rule[0], rule[1]
        n_rows = dataset.shape[0]
        mask = np.ones(n_rows, dtype=bool)

        for idx, val in zip(indices, values):
            if not val:
                return np.zeros(n_rows, dtype=bool)

            col_data = dataset[:, idx]
            if isinstance(val[0], (str, np.str_)):
                if len(val) > 1:
                    row_mask = np.isin(col_data, val)
                else:
                    row_mask = col_data == val[0]
            else:
                row_mask = (col_data >= val[0]) & (col_data <= val[1])
            mask &= row_mask

        return mask

    def get_covered_indices(self, rule, dataset):
        return np.flatnonzero(self.get_covered_mask(rule, dataset)).tolist()

    def fitness(self, rule, dataset_x):
        """Receives a rule and computes its fitness."""
        rule_mask = self.get_covered_mask(rule, dataset_x)
        covered_count = int(rule_mask.sum())
        if covered_count < 1:
            return 0.0

        relative_support = covered_count / self._n
        if relative_support > MAX_RELATIVE_SUPPORT or relative_support < MIN_RELATIVE_SUPPORT:
            return 0.0

        if covered_count >= self._n:
            return 0.0

        if self.score_metric == "legacy_logrank":
            discrepancy = self._legacy_logrank_score(rule_mask)
        elif self.score_metric == "fast_logrank":
            discrepancy = self._fast_logrank_score(rule_mask)
        elif self.score_metric in ("km_cvm", "km_abc"):
            discrepancy = self._km_distance_score(rule_mask)
        else:
            discrepancy = 0.0

        if not np.isfinite(discrepancy) or discrepancy <= 0.0:
            return 0.0
        return float(discrepancy * (relative_support**self.alpha))

    def get_fitness(self, population, dataset_x):
        fitness_list = []
        for i in range(len(population)):
            fitness_list.append(self.fitness(population[i], dataset_x))
        fitness_list = np.array(fitness_list)
        return list(fitness_list)

    def _legacy_logrank_score(self, rule_mask: np.ndarray) -> float:
        try:
            group_id = np.full(self._n, "pop", dtype=object)
            group_id[rule_mask] = "sg"
            if self.comparacao == "complement":
                group_id[~rule_mask] = "complement"

            p_value_result = sm.duration.survdiff(
                time=self._time_values,
                status=self._event_values,
                group=group_id,
            )
            p_value = float(p_value_result[1])
            if pd.isna(p_value) or p_value < 0.0 or p_value > 1.0:
                return 0.0
            return 1.0 - p_value
        except (ValueError, ZeroDivisionError, Exception):
            return 0.0

    def _precompute_fast_logrank(self) -> None:
        event_times = np.sort(np.unique(self._time_values[self._event_observed]))
        if event_times.size == 0:
            return

        event_positions = np.searchsorted(event_times, self._time_values[self._event_observed], side="left")
        event_counts = np.bincount(event_positions, minlength=event_times.size).astype(float)

        last_risk_positions = np.searchsorted(event_times, self._time_values, side="right") - 1
        valid_risk = last_risk_positions >= 0
        subject_counts = np.bincount(last_risk_positions[valid_risk], minlength=event_times.size).astype(float)
        risk_counts = np.cumsum(subject_counts[::-1])[::-1]

        alpha_t = np.divide(event_counts, risk_counts, out=np.zeros_like(event_counts), where=risk_counts > 0.0)
        cumulative_alpha = np.cumsum(alpha_t)

        self._logrank_gamma = np.zeros(self._n, dtype=float)
        self._logrank_gamma[valid_risk] = cumulative_alpha[last_risk_positions[valid_risk]]
        self._logrank_residual = self._event_values.astype(float) - self._logrank_gamma
        self._logrank_expected_total = float(self._logrank_gamma.sum())

    def _fast_logrank_score(self, rule_mask: np.ndarray) -> float:
        expected_sg = float(self._logrank_gamma[rule_mask].sum())
        if expected_sg <= EPSILON:
            return 0.0

        numerator = float(self._logrank_residual[rule_mask].sum())
        if self.comparacao == "population":
            statistic = (numerator * numerator) / expected_sg
        else:
            expected_ref = self._logrank_expected_total - expected_sg
            if expected_ref <= EPSILON:
                return 0.0
            statistic = (numerator * numerator) * ((1.0 / expected_sg) + (1.0 / expected_ref))

        if statistic <= 0.0:
            return 0.0
        return float(statistic / (1.0 + statistic))

    def _precompute_km_grid(self) -> None:
        event_times = np.sort(np.unique(self._time_values[self._event_observed]))
        if event_times.size == 0:
            return

        exact_grid = self.km_time_bins is None or event_times.size <= self.km_time_bins
        if exact_grid:
            grid_times = event_times
        elif self.km_time_bins is not None:
            quantiles = np.linspace(0.0, 1.0, self.km_time_bins)
            grid_times = np.unique(np.quantile(event_times, quantiles))
        else:
            grid_times = event_times

        if grid_times.size == 0:
            return

        self._km_grid_times = grid_times.astype(float, copy=False)
        grid_size = self._km_grid_times.size
        max_time = max(float(np.max(self._time_values)), float(self._km_grid_times[-1]))
        self._km_widths = np.diff(np.r_[self._km_grid_times, max_time])
        self._km_widths = np.maximum(self._km_widths, 0.0)

        if exact_grid:
            risk_bin = np.searchsorted(self._km_grid_times, self._time_values, side="right") - 1
            event_bin = np.full(self._n, -1, dtype=int)
            event_bin[self._event_observed] = np.searchsorted(
                self._km_grid_times,
                self._time_values[self._event_observed],
                side="left",
            )
        else:
            risk_bin = np.searchsorted(self._km_grid_times, self._time_values, side="left")
            risk_bin = np.clip(risk_bin, 0, grid_size - 1)
            risk_bin[self._time_values < self._km_grid_times[0]] = -1
            event_bin = np.full(self._n, -1, dtype=int)
            event_bin[self._event_observed] = risk_bin[self._event_observed]

        self._km_risk_bin = risk_bin.astype(int, copy=False)
        self._km_event_bin = event_bin.astype(int, copy=False)

        valid_risk = self._km_risk_bin >= 0
        self._km_pooled_subject_counts = np.bincount(self._km_risk_bin[valid_risk], minlength=grid_size).astype(float)
        valid_events = self._km_event_bin >= 0
        self._km_pooled_event_counts = np.bincount(self._km_event_bin[valid_events], minlength=grid_size).astype(float)
        self._km_pooled_risk_counts = np.cumsum(self._km_pooled_subject_counts[::-1])[::-1]
        self._km_pooled_survival = self._kaplan_meier_from_counts(
            self._km_pooled_event_counts,
            self._km_pooled_risk_counts,
        )
        self._km_weights = np.divide(
            self._km_pooled_risk_counts,
            float(self._n),
            out=np.zeros_like(self._km_pooled_risk_counts),
            where=self._km_pooled_risk_counts > 0.0,
        )

    def _km_distance_score(self, rule_mask: np.ndarray) -> float:
        if self._km_grid_times.size == 0:
            return 0.0

        group_subject_counts, group_event_counts = self._group_km_counts(rule_mask)
        group_risk_counts = np.cumsum(group_subject_counts[::-1])[::-1]
        group_survival = self._kaplan_meier_from_counts(group_event_counts, group_risk_counts)

        if self.comparacao == "population":
            ref_risk_counts = self._km_pooled_risk_counts
            ref_survival = self._km_pooled_survival
        else:
            ref_subject_counts = self._km_pooled_subject_counts - group_subject_counts
            ref_event_counts = self._km_pooled_event_counts - group_event_counts
            ref_risk_counts = np.cumsum(ref_subject_counts[::-1])[::-1]
            ref_survival = self._kaplan_meier_from_counts(ref_event_counts, ref_risk_counts)

        valid = (
            (group_risk_counts > 0.0) & (ref_risk_counts > 0.0) & (self._km_widths > 0.0) & (self._km_weights > 0.0)
        )
        if not np.any(valid):
            return 0.0

        weighted_widths = self._km_widths[valid] * self._km_weights[valid]
        normalizer = float(weighted_widths.sum())
        if normalizer <= EPSILON:
            return 0.0

        diff = group_survival[valid] - ref_survival[valid]
        if self.score_metric == "km_abc":
            distance = float(np.sum(weighted_widths * np.abs(diff)) / normalizer)
        else:
            distance = float(np.sum(weighted_widths * diff * diff) / normalizer)

        return min(max(distance, 0.0), 1.0)

    def _group_km_counts(self, rule_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grid_size = self._km_grid_times.size
        rule_indices = np.flatnonzero(rule_mask)

        rule_risk_bins = self._km_risk_bin[rule_indices]
        valid_risk = rule_risk_bins >= 0
        subject_counts = np.bincount(rule_risk_bins[valid_risk], minlength=grid_size).astype(float)

        rule_event_bins = self._km_event_bin[rule_indices]
        valid_events = rule_event_bins >= 0
        event_counts = np.bincount(rule_event_bins[valid_events], minlength=grid_size).astype(float)

        return subject_counts, event_counts

    def _kaplan_meier_from_counts(self, event_counts: np.ndarray, risk_counts: np.ndarray) -> np.ndarray:
        hazards = np.divide(event_counts, risk_counts, out=np.zeros_like(event_counts), where=risk_counts > 0.0)
        hazards = np.clip(hazards, 0.0, 1.0)
        return np.cumprod(1.0 - hazards)
