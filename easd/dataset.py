import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, data, attr_survival_name, attr_event_name):
        data = self._binarize_events(data, attr_event_name)
        self.survival_times = ()
        self.average_survival = None
        self.events = ()
        self.attr_values = {}
        self.data = None

        self._col_index = {}
        self._uncovered_cases = [True] * data.shape[0]
        self._original_data = data.copy()
        self._surv_name = attr_survival_name
        self._event_name = attr_event_name
        self._count = [0] * data.shape[0]

        self._constructor(attr_survival_name, attr_event_name)

        self._map_rules_columns = {}
        for key, value in self._col_index.items():
            self._map_rules_columns[value] = key

    def _constructor(self, attr_survival_name, attr_event_name):

        data = self._original_data.copy()

        self.survival_times = (attr_survival_name, data[attr_survival_name])
        self.average_survival = data[attr_survival_name].mean()
        self.events = (attr_event_name, data[attr_event_name])

        to_drop = [attr_survival_name, attr_event_name]
        data.drop(columns=to_drop, inplace=True)

        col_names = list(data.columns.values)
        self.attr_values = dict.fromkeys(col_names)
        for name in col_names:
            self.attr_values[name] = list(set(pd.unique(data[name])))

        self._col_index = dict.fromkeys(col_names)
        for name in col_names:
            self._col_index[name] = data.columns.get_loc(name)

        self.data = np.array(data.values)
        return

    def _binarize_events(self, data: pd.DataFrame, event_col: str):
        data[event_col] = data[event_col].replace(min(data[event_col]), 0)
        data[event_col] = data[event_col].replace(max(data[event_col]), 1)

        return data

    @property
    def size(self):
        """ """
        return self._original_data.shape[0]

    @property
    def surv_name(self):
        return self._surv_name

    def remove_covered_cases(self, cases):
        for case in cases:
            if self._count[case] <= 1:  # if covered only once > becomes uncovered
                self._count[case] = 0
                self._uncovered_cases[case] = True
            else:  # if covered more than once > decrements cover count
                self._count[case] -= 1
        return

    def update_covered_cases(self, covered_cases):
        # set flag for rule-covered cases
        for case in covered_cases:
            self._uncovered_cases[case] = False
            self._count[case] += 1
        return

    def get_case_count(self):
        return self._count

    def get_col_index(self, col_name):
        return self._col_index[col_name]

    def get_col_name(self, col_index) -> str:
        return self._map_rules_columns[col_index]

    def get_data(self) -> pd.DataFrame:
        return self._original_data.copy()

    def get_cases_coverage(self):  # returns a bool list with True for covered cases
        return [covered == False for covered in self._uncovered_cases]

    def get_no_of_uncovered_cases(self):
        return sum(self._uncovered_cases)

    def get_uncovered_cases(self):
        return list(self._original_data[self._uncovered_cases].index)

    def get_instances(self) -> list:
        """ """
        if self.data is not None:
            return list(range(len(self.data)))
        else:
            return []
