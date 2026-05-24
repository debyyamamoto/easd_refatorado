import numpy as np
import pandas as pd
from .dataset import Dataset
import statsmodels.api as sm


class RuleEvaluator:
    def __init__(self, dataset_obj: Dataset, comparacao, alpha):
        self.dataset_obj = dataset_obj
        self.comparacao = comparacao
        self.sub_group_cases = dataset_obj.get_instances()
        self.p_value = None
        self._fitness = 0.0
        self._all_indices = set(range(self.dataset_obj.size))
        self.alpha = alpha
        # Precompute survival times and events.
        self._survival_times = self.dataset_obj._original_data[self.dataset_obj.surv_name]
        self._events = self.dataset_obj._original_data[self.dataset_obj._event_name]
        # Precompute the complementary population.
        if comparacao == "complement":
            all_indices = set(range(len(self._survival_times)))
            self._complement_cases = list(all_indices - set(self.sub_group_cases))

    def get_covered_indices(self, rule, dataset):
        # Validate whether the rule follows the expected format: indices and values.
        if not rule or len(rule) < 2 or len(rule[0]) != len(rule[1]):
            return []
        indices, values = rule[0], rule[1]
        n_rows = dataset.shape[0]

        mask = np.ones(n_rows, dtype=bool)

        for idx, val in zip(indices, values):
            col_data = dataset[:, idx]
            if isinstance(val[0], str):
                if len(val) > 1:
                    row_mask = np.isin(col_data, val)
                else:
                    row_mask = col_data == val[0]
            else:
                row_mask = (col_data >= val[0]) & (col_data <= val[1])
            mask &= row_mask
        return np.where(mask)[0].tolist()

    def fitness(self, rule, dataset_x):
        "Receives a rule and computes its fitness."
        # Split the group covered by the rule and compare it with the selected baseline group.
        rule_group_indices = []
        rule_group_indices = self.get_covered_indices(rule, dataset_x)
        p_value = 1.0

        if len(rule_group_indices) < 1:
            return 0.0

        rule_complement_indices = list(self._all_indices - set(rule_group_indices))

        if len(rule_complement_indices) < 1:
            return 0.0
        if self.comparacao == "population":
            try:
                times = self._survival_times.to_list()
                events = self._events.to_list()
                group_id = ["sg" if i in set(rule_group_indices) else "pop" for i in range(len(times))]

                p_value_result = sm.duration.survdiff(time=times, status=events, group=group_id)
                p_value = p_value_result[1]
                if pd.isna(p_value):
                    p_value = 1.0
            except (ValueError, ZeroDivisionError, Exception) as e:
                p_value = 1.0
        elif self.comparacao == "complement":
            try:
                sg = pd.Series("sub_group", index=rule_group_indices)
                cpm = pd.Series("complement", index=rule_complement_indices)
                group = pd.concat([sg, cpm], axis=0, ignore_index=False).sort_index()
                # Ensure group, survival times, and events share the same indices.
                filtered_times = self._survival_times.loc[group.index]
                filtered_events = self._events.loc[group.index]

                p_value_result = sm.duration.survdiff(filtered_times, filtered_events, group=group)
                p_value = p_value_result[1]
                if pd.isna(p_value):
                    p_value = 1.0
            except (ValueError, ZeroDivisionError, Exception) as e:
                p_value = 1.0
        relative_support = len(rule_group_indices) / len(self._all_indices)
        if relative_support > 0.55 or relative_support < 0.05:
            return 0.0  # p=1 => fitness 0
        return (1 - p_value) * (relative_support**self.alpha)

    def get_fitness(self, population, dataset_x):
        fitness_list = []
        for i in range(len(population)):
            fitness_list.append(self.fitness(population[i], dataset_x))
        fitness_list = np.array(fitness_list)
        return list(fitness_list)
