from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from lifelines import KaplanMeierFitter

POPULATION = "Population"
COMPLEMENT = "Complement"


@dataclass
class RulesPlotter:
    dataset: pd.DataFrame
    rules: list[list]
    events_column: str
    time_column: str

    def kaplan_meier(self, num_top: int) -> list[Figure]:
        """
        Realiza o plot de Kaplan-Meier das regras registradas no Top-K

        :param num_top: Números de regras que serão mostradas no plot de top-n melhores subgrupos registrados no top-k (Ex: num_top=3, compara as 3 melhores regras com a população/complemento)
        :type num_top: int
        """
        if num_top > 0:
            figures_list = []

            figures_list.append(self._multiple_rules_curves(num_top))
            individual_rules_plots = self._individual_rules_curves()
            figures_list.extend(individual_rules_plots)

            return figures_list
        else:
            return []

    def _multiple_rules_curves(self, p_num_top: int):
        fig, ax = plt.subplots(figsize=(12, 10))

        fitter = KaplanMeierFitter(label=POPULATION)
        fitter.fit(self.dataset[self.time_column], self.dataset[self.events_column])
        fitter.plot_survival_function(ax=ax)

        for rule in self.rules[:p_num_top]:
            self._plot_rules_curves(rule, ax)
        ax.set_title(f"Top-{p_num_top} Kaplan-Meier Survival Curves")
        ax.set_xlabel("Time (e.g., months, days)")
        ax.set_ylabel("Survival Probability")
        ax.grid()
        ax.legend()
        ax.set_ylim(0, 1.1)

        return fig

    def _individual_rules_curves(self):
        figures_list = []
        for i, rule in enumerate(self.rules):
            fig, ax = plt.subplots(figsize=(12, 10))

            self._plot_rule_and_complement(rule, ax)
            ax.set_title(f"Top-{i+1} Kaplan-Meier Survival Curves")
            ax.set_xlabel("Time (e.g., months, days)")
            ax.set_ylabel("Survival Probability")
            ax.grid()
            ax.legend()
            ax.set_ylim(0, 1.1)

            figures_list.append(fig)

        return figures_list

    def _plot_rules_curves(self, p_rule: list[list], p_ax: Axes):
        rule_string = ""
        rule_df = self.dataset.copy()
        atributes_list, constraints_list = p_rule
        for atribute, constraint in zip(atributes_list, constraints_list):
            rule_df = rule_df[(rule_df[atribute] >= constraint[0]) & (rule_df[atribute] <= constraint[1])]
            rule_string = f"{rule_string} ({constraint[0]}≤{atribute}≤{constraint[1]})"

        fitter = KaplanMeierFitter(label=rule_string)
        fitter.fit(rule_df[self.time_column], rule_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax)

    def _plot_rule_and_complement(self, p_rule: list[list], p_ax: Axes):
        rule_string = ""
        rule_df = self.dataset.copy()
        rule_complement_df = self.dataset.copy()
        atributes_list, constraints_list = p_rule
        for atribute, constraint in zip(atributes_list, constraints_list):
            rule_df = rule_df[(rule_df[atribute] >= constraint[0]) & (rule_df[atribute] <= constraint[1])]
            rule_complement_df = rule_complement_df[
                ~((rule_complement_df[atribute] >= constraint[0]) & (rule_complement_df[atribute] <= constraint[1]))
            ]
            rule_string = f"{rule_string} ({constraint[0]}≤{atribute}≤{constraint[1]})"

        if rule_complement_df.empty:
            rule_complement_df = self.dataset
        fitter = KaplanMeierFitter(label=COMPLEMENT)
        fitter.fit(rule_complement_df[self.time_column], rule_complement_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax)

        fitter = KaplanMeierFitter(label=rule_string)
        fitter.fit(rule_df[self.time_column], rule_df[self.events_column])
        fitter.plot_survival_function(ax=p_ax)


if __name__ == "__main__":
    df = pd.read_csv("datasets/cancer.csv")
    rules = [
        [["ph-ecog"], [[np.float64(0.0), np.float64(3.0)]]],
        [["age", "sex"], [[np.float64(54.0), np.float64(82.0)], [np.float64(1.0), np.float64(1.0)]]],
    ]

    plotter = RulesPlotter(df, rules, events_column="status", time_column="time")
    plot_list = plotter.kaplan_meier(2)
    for i, plot in enumerate(plot_list):
        plot.show()
        plot.savefig(f"image{i}")
