import numpy as np
from .dataset import Dataset
from scipy.stats import chi2
import hashlib

class RuleEvaluator:
    def __init__(self, dataset_obj: Dataset, comparacao, alpha):
        self.dataset_obj = dataset_obj
        self.comparacao = comparacao
        self.sub_group_cases = dataset_obj.get_instances()
        self.p_value = None
        self._fitness = 0.0
        self.alpha = alpha

        times = dataset_obj._original_data[dataset_obj.surv_name].to_numpy(dtype=np.float64)
        events = dataset_obj._original_data[dataset_obj._event_name].to_numpy(dtype=np.float64)
        self._survival_times = times
        self._events = events
        self._fitness_cache: dict[bytes, float] = {}

        ## Pre compute the log-rank values 
        # chronological ordering
        order = np.argsort(times, kind="mergesort")
        self._sort_order = order 
        sorted_times = times[order]
        sorted_events = events[order]
        self._event_mask_sorted = sorted_events == 1

        # total death count
        event_times_sorted = sorted_times[self._event_mask_sorted]
        unique_times, d_j = np.unique(event_times_sorted, return_counts=True)
        self._unique_times = unique_times
        self._d_j = d_j.astype(np.float64)

        # total number of people alive/ at risk 
        risk_start_idx = np.searchsorted(sorted_times, unique_times, side="left")
        self._risk_start_idx = risk_start_idx
        self._n_j = (self.dataset_obj.size - risk_start_idx).astype(np.float64)

        # to which unique event time each event row (ordered by time) belongs.
        self._event_time_bin = np.searchsorted(unique_times, event_times_sorted)

    def get_covered_indices(self, rule, dataset):
        # Validate whether the rule follows the expected format: indices and values.
        if not rule or len(rule) < 2 or len(rule[0]) != len(rule[1]):
            return []
        mask = self.dataset_obj.get_rule_mask(rule)
        return np.where(mask)[0].tolist()

    def _subgroup_risk_components(self, mask: np.ndarray):
        "n_1j e d_1j: em risco e eventos do subgrupo, por tempo de evento único."
        mask_sorted = mask[self._sort_order]

        suffix_count = np.cumsum(mask_sorted[::-1])[::-1]
        n_1j = suffix_count[self._risk_start_idx].astype(np.float64)

        event_mask_in_group = mask_sorted[self._event_mask_sorted]
        d_1j = np.bincount(
            self._event_time_bin,
            weights=event_mask_in_group.astype(np.float64),
            minlength=len(self._unique_times),
        )
        return n_1j, d_1j

    def _chi2_p_value(self, n_1j, d_1j, n_j, d_j) -> float:
        e_1j = n_1j * d_j / n_j

        with np.errstate(divide="ignore", invalid="ignore"):
            v_j = np.where(
                n_j > 1,
                (n_1j * (n_j - n_1j) * d_j * (n_j - d_j)) / (n_j**2 * (n_j - 1)),
                0.0,
            )

        O1, E1, V = d_1j.sum(), e_1j.sum(), v_j.sum()
        if V <= 0:
            return 1.0

        stat = (O1 - E1) ** 2 / V
        return float(chi2.sf(stat, df=1))


    def fitness(self, rule, dataset_x):
        "Receives a rule and computes its fitness."
        # Split the group covered by the rule and compare it with the selected baseline group.
        rule_group_indices = self.get_covered_indices(rule, dataset_x)

        if len(rule_group_indices) < 1:
            return 0.0

        relative_support = len(rule_group_indices) / self.dataset_obj.size
        if relative_support > 0.55 or relative_support < 0.05:
            return 0.0  # p=1 => fitness 0

        mask = np.zeros(self.dataset_obj.size, dtype=bool)
        mask[rule_group_indices] = True
        cache_key = hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).digest()
        cached = self._fitness_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            n_1j, d_1j = self._subgroup_risk_components(mask)

            if self.comparacao == "population":
                n_j = self._n_j + n_1j
                d_j = self._d_j + d_1j
            else:  # "complement"
                n_j = self._n_j
                d_j = self._d_j

            p_value = self._chi2_p_value(n_1j, d_1j, n_j, d_j)
            if np.isnan(p_value):
                p_value = 1.0
        except Exception:
            p_value = 1.0

        fitness_value = (1 - p_value) * (relative_support**self.alpha)
        self._fitness_cache[cache_key] = fitness_value
        return fitness_value

    def get_fitness(self, population, dataset_x):
        fitness_list = []
        for i in range(len(population)):
            fitness_list.append(self.fitness(population[i], dataset_x))
        fitness_list = np.array(fitness_list)
        return list(fitness_list)
