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
        # pré-computar os dados de tempo e eventos
        self._survival_times = self.dataset_obj._original_data[self.dataset_obj.surv_name]
        self._events = self.dataset_obj._original_data[self.dataset_obj._event_name]
        # pré-computar a população complementar
        if comparacao == "complement":
            all_indices = set(range(len(self._survival_times)))
            self._complement_cases = list(all_indices - set(self.sub_group_cases))

    def get_covered_indices(self, rule, dataset):
        # verificação se rule obedece o formato esperado: indices e valores
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
        "Recebe uma regra e calcula o seu fitness"
        # Pensar em como adaptar para uma regra só
        # separar o grupo que possui uma determinada regra e comparar com o basegroup escolhido
        indices_group_regra = []
        indices_group_regra = self.get_covered_indices(rule, dataset_x)
        p_value = 1.0

        if len(indices_group_regra) < 1:
            return 0.0

        indices_complemento_regra = list(self._all_indices - set(indices_group_regra))

        if len(indices_complemento_regra) < 1:
            return 0.0
        # passo para garantir que
        if self.comparacao == "population":
            try:
                times = self._survival_times.to_list()
                events = self._events.to_list()
                group_id = ["sg" if i in set(indices_group_regra) else "pop" for i in range(len(times))]

                resultado_p_valor = sm.duration.survdiff(time=times, status=events, group=group_id)
                p_value = resultado_p_valor[1]
                if pd.isna(p_value):
                    p_value = 1.0
            except (ValueError, ZeroDivisionError, Exception) as e:
                p_value = 1.0
        elif self.comparacao == "complement":
            try:
                sg = pd.Series("sub_group", index=indices_group_regra)
                cpm = pd.Series("complement", index=indices_complemento_regra)
                group = pd.concat([sg, cpm], axis=0, ignore_index=False).sort_index()
                # para ter certeza que group e tempos e eventos compartilham os mesmos indices
                tempos_filtrados = self._survival_times.loc[group.index]
                eventos_filtrados = self._events.loc[group.index]

                resultado_p_valor = sm.duration.survdiff(tempos_filtrados, eventos_filtrados, group=group)
                p_value = resultado_p_valor[1]
                if pd.isna(p_value):
                    p_value = 1.0
            except (ValueError, ZeroDivisionError, Exception) as e:
                p_value = 1.0
        suporte_relativo = len(indices_group_regra) / len(self._all_indices)
        if suporte_relativo > 0.55 or suporte_relativo < 0.05:
            return 0.0  # p=1 => fitness 0
        return (1 - p_value) * (suporte_relativo**self.alpha)

    def get_fitness(self, population, dataset_x):
        fitness_list = []
        for i in range(len(population)):
            fitness_list.append(self.fitness(population[i], dataset_x))
        fitness_list = np.array(fitness_list)
        return list(fitness_list)
