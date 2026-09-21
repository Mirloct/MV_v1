"""Focused tests for post-training sensitivity contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.sensitivity import run_post_training_sensitivity
from src.reporting.report import _build_html, _build_markdown


class _Preprocessor:
    def transform(self, frame):
        return frame[["x1", "x2", "x3"]].fillna(0).to_numpy(dtype=float)


class _Detector:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def score_samples(self, matrix):
        return np.asarray(matrix, dtype=float) @ self.weights


class SensitivityTests(unittest.TestCase):
    def test_generates_rankings_scenarios_and_high_zero_recommendation(self):
        rows = []
        for entity in range(4):
            for period in range(3):
                rows.append({
                    "entity_id": f"E{entity}", "period": f"2026-0{period+1}",
                    "x1": float(entity + period + 1),
                    "x2": float(2 * entity + period + 1),
                    "x3": float(period + 1),
                })
        df = pd.DataFrame(rows)
        # One test observation is deliberately 100% zero information.
        df.loc[(df["entity_id"] == "E3") & (df["period"] == "2026-03"), ["x1", "x2", "x3"]] = 0.0
        train_mask = df["period"].isin(["2026-01", "2026-02"]).to_numpy()
        test_mask = df["period"].eq("2026-03").to_numpy()
        preprocessor = _Preprocessor()
        matrix = preprocessor.transform(df)
        models = {"iforest": _Detector([3.0, 0.2, 0.0]), "vae": _Detector([1.0, 1.0, 0.1])}
        base = {name: model.score_samples(matrix) for name, model in models.items()}
        labels = np.asarray([int(i % 3 == 2 and i >= 8) for i in range(len(df))])

        with tempfile.TemporaryDirectory() as tmp:
            result = run_post_training_sensitivity(
                df=df,
                schema=PanelSchema(time_col="period", entity_col="entity_id", target_col=None),
                preprocessor=preprocessor,
                feature_names=["num__x1", "num__x2", "num__x3"],
                models=models,
                baseline_scores=base,
                thresholds={"iforest": 7.0, "vae": 5.0},
                train_mask=train_mask,
                test_mask=test_mask,
                labels=labels,
                max_test_rows=10,
                combination_top_k=2,
                random_subsets_per_level=1,
                missing_levels=(0.5, 0.9),
                out_dir=tmp,
            )

            for path in result["artifacts"].values():
                self.assertTrue(Path(path).is_file(), path)
            scenarios = pd.read_csv(result["artifacts"]["scenarios_csv"])
            variables = pd.read_csv(result["artifacts"]["variables_csv"])
            high_zero = pd.read_csv(result["artifacts"]["high_zero_csv"])
            self.assertTrue({"ablation", "null_replacement", "zero", "information_loss", "pruning_path"}.issubset(
                set(scenarios["scenario_type"])
            ))
            self.assertEqual(set(variables["variable"]), {"x1", "x2", "x3"})
            self.assertGreaterEqual(len(high_zero), 2)  # one record, once per model
            summary = json.loads(Path(result["artifacts"]["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["method"], "post_training_no_refit")
            self.assertIn("iforest", summary["required_variables"])
            context = {
                "title": "test", "dataset": {}, "models": {},
                "sensitivity_analysis": result,
            }
            self.assertIn("Sensibilidad post-entrenamiento", _build_html(context, None, tmp))
            self.assertIn("Sensibilidad post-entrenamiento", _build_markdown(context, tmp))


if __name__ == "__main__":
    unittest.main()
