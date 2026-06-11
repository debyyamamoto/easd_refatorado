from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_generator_module():
    module_path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("synthetic_survival_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = _load_generator_module()


class SyntheticSurvivalGeneratorSmokeTest(unittest.TestCase):
    def test_small_dataset_invariants(self) -> None:
        config = generator.SyntheticSurvivalConfig(n=100, p=5, k=2, seed=0)
        rng = np.random.default_rng(config.seed)

        df, metadata = generator.make_survival_data(config, rng)

        expected_columns = [f"feature_{i}" for i in range(config.p)] + ["time", "event", "subgroup"]
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(df.shape, (100, 8))
        self.assertTrue((df["time"] > 0).all())
        self.assertEqual(set(df["event"].unique()).difference({0, 1}), set())
        self.assertEqual(set(df["subgroup"].unique()).difference({0, 1}), set())

        subgroup_df = df[df["subgroup"] == 1]
        for interval in metadata["true_rule"]["intervals"]:
            values = subgroup_df[interval["feature"]]
            self.assertTrue(((values >= interval["lower"]) & (values <= interval["upper"])).all())

    def test_cli_censoring_variation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status = generator.main(
                [
                    "--output-dir",
                    temp_dir,
                    "--dataset-name",
                    "smoke",
                    "--n",
                    "100",
                    "--p",
                    "5",
                    "--k",
                    "2",
                    "--seed",
                    "0",
                    "--vary",
                    "censoring",
                    "--values",
                    "0.0",
                    "0.5",
                    "--repeats",
                    "2",
                ]
            )

            self.assertEqual(status, 0)
            output_dir = Path(temp_dir)
            parquet_files = sorted(output_dir.glob("*.parquet"))
            metadata_files = sorted(output_dir.glob("*_metadata.json"))
            self.assertEqual(len(parquet_files), 4)
            self.assertEqual(len(metadata_files), 4)

            metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
            self.assertEqual({item["variation"]["factor"] for item in metadata}, {"censoring"})
            self.assertEqual({item["parameters"]["censoring_ratio"] for item in metadata}, {0.0, 0.5})

            stable_fields = [
                "n",
                "p",
                "k",
                "subgroup_ratio",
                "population_scale",
                "subgroup_scale",
                "population_shape",
                "subgroup_shape",
            ]
            stable_values = {
                tuple(item["parameters"][field] for field in stable_fields)
                for item in metadata
            }
            self.assertEqual(len(stable_values), 1)


if __name__ == "__main__":
    unittest.main()
