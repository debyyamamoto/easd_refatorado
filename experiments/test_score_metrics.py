from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import main
from easd.dataset import Dataset
from easd.evaluation import SCORE_METRICS, RuleEvaluator


class ScoreMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n_subgroup = 40
        n_population = 160

        subgroup_feature = rng.uniform(0.10, 0.20, size=n_subgroup)
        population_feature = rng.uniform(0.65, 0.95, size=n_population)
        feature = np.r_[subgroup_feature, population_feature]
        noise = rng.uniform(0.0, 1.0, size=n_subgroup + n_population)

        subgroup_time = rng.weibull(1.5, size=n_subgroup) * 1.0
        population_time = rng.weibull(1.5, size=n_population) * 5.0
        time = np.maximum(np.r_[subgroup_time, population_time], np.finfo(float).tiny)
        event = np.ones(n_subgroup + n_population, dtype=int)

        self.df = pd.DataFrame(
            {
                "feature_0": feature,
                "feature_1": noise,
                "time": time,
                "event": event,
            }
        )
        self.dataset = Dataset(self.df.copy(), "time", "event")
        self.rule = [[0], [[0.0, 0.3]]]

    def test_all_score_metrics_return_finite_positive_fitness(self) -> None:
        for score_metric in SCORE_METRICS:
            with self.subTest(score_metric=score_metric):
                evaluator = RuleEvaluator(
                    self.dataset,
                    "complement",
                    alpha=0.5,
                    score_metric=score_metric,
                    km_time_bins=64,
                )

                fitness = evaluator.fitness(self.rule, self.dataset.data)

                self.assertTrue(np.isfinite(fitness))
                self.assertGreater(fitness, 0.0)

    def test_support_filter_runs_before_score_metric(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset,
            "complement",
            alpha=0.5,
            score_metric="km_cvm",
            km_time_bins=64,
        )
        tiny_rule = [[0], [[0.10, 0.101]]]

        self.assertEqual(evaluator.fitness(tiny_rule, self.dataset.data), 0.0)

    def test_cli_accepts_score_metric_arguments(self) -> None:
        args = main.build_parser().parse_args(
            [
                "datasets/files/cancer.parquet",
                "-time",
                "time",
                "-event",
                "status",
                "--score_metric",
                "km_abc",
                "--km_time_bins",
                "128",
            ]
        )
        config = main.config_from_args(args)

        self.assertEqual(config.score_metric, "km_abc")
        self.assertEqual(config.km_time_bins, 128)


if __name__ == "__main__":
    unittest.main()
