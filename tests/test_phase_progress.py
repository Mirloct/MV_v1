"""Behavior: the heavy work inside the pipeline's phases is followed, not just
the phases themselves.

A phase-level timer says "Phase 8 is running"; it cannot say whether it is in
the silhouette score, the UMAP reducer or a bootstrap loop, nor how far along.
These tests run the real functions behind several phases (evaluation metrics,
preprocessing diagnostics, the post-training sensitivity study, the report
build) and assert that each one publishes named steps with durations and a bar
per loop -- the events the dashboard and both flow HTML pages read.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import PanelSchema  # noqa: E402
from src.utils import logging_config, observability, progress  # noqa: E402
from test_sensitivity import _Detector, _Preprocessor  # noqa: E402


class _Watch:
    def __enter__(self):
        self.phases: list[tuple[str, str]] = []
        self.bars: list[dict] = []
        logging_config.add_phase_observer(self._phase)
        observability.add_progress_observer(self.bars.append)
        progress.configure(min_interval_s=0.0)
        return self

    def __exit__(self, *exc):
        logging_config.remove_phase_observer(self._phase)
        observability.remove_progress_observer(self.bars.append)
        progress.configure(min_interval_s=progress.DEFAULT_MIN_INTERVAL_S)

    def _phase(self, name, event, duration_s):
        self.phases.append((name, event))

    def started(self) -> list[str]:
        return [n for n, e in self.phases if e == "phase_started"]

    def finished(self, desc: str) -> dict:
        ends = [b for b in self.bars if b["desc"] == desc and b["state"] == "end"]
        self.assertion_target.assertEqual(len(ends), 1, f"bar {desc!r} must end exactly once")
        return ends[0]


class EvaluationPhaseTests(unittest.TestCase):
    def test_unsupervised_metrics_names_its_slow_steps_and_its_bootstrap_bar(self):
        from src.evaluation import unsupervised_metrics

        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 5))
        scores = X[:, 0] ** 2 + rng.normal(scale=0.1, size=300)
        with _Watch() as watch:
            watch.assertion_target = self
            unsupervised_metrics(X, scores, contamination=0.1)
            bar = watch.finished("rank_stability[bootstraps]")
        for step in ("evaluation.silhouette_score", "evaluation.calinski_harabasz_score",
                     "evaluation.rank_stability"):
            self.assertIn(step, watch.started())
        self.assertEqual((bar["n"], bar["total"], bar["unit"]), (10, 10, "bootstrap"))
        self.assertLess(watch.started().index("evaluation.silhouette_score"),
                        watch.started().index("evaluation.rank_stability"))


class PreprocessingPhaseTests(unittest.TestCase):
    def test_transform_diagnostics_reports_one_tick_per_numeric_feature(self):
        from src.preprocessing import compute_transform_diagnostics

        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "entity_id": np.repeat(np.arange(50), 4), "period": np.tile(np.arange(4), 50),
            "income": rng.lognormal(size=200), "balance": rng.lognormal(size=200),
            "age": rng.integers(18, 80, size=200).astype(float),
        })
        schema = PanelSchema(time_col="period", entity_col="entity_id", target_col=None)
        with _Watch() as watch:
            watch.assertion_target = self
            compute_transform_diagnostics(df, schema, random_state=0)
            bar = watch.finished("transform_diagnostics[features]")
        self.assertEqual((bar["n"], bar["total"]), (3, 3))
        self.assertIn("preprocessing.compute_transform_diagnostics", watch.started())
        self.assertTrue({b["current"] for b in watch.bars
                         if b["desc"] == "transform_diagnostics[features]"} >= {"income", "balance"})


class SensitivityPhaseTests(unittest.TestCase):
    def test_every_scenario_block_is_a_bar_that_finishes_and_outputs_are_named_steps(self):
        from src.evaluation.sensitivity import run_post_training_sensitivity

        rows = [{"entity_id": f"E{e}", "period": f"2026-0{p + 1}", "x1": float(e + p + 1),
                 "x2": float(2 * e + p + 1), "x3": float(p + 1)}
                for e in range(4) for p in range(3)]
        df = pd.DataFrame(rows)
        train_mask = df["period"].isin(["2026-01", "2026-02"]).to_numpy()
        test_mask = df["period"].eq("2026-03").to_numpy()
        pre = _Preprocessor()
        matrix = pre.transform(df)
        models = {"iforest": _Detector([3.0, 0.2, 0.0]), "vae": _Detector([1.0, 1.0, 0.1])}
        base = {n: m.score_samples(matrix) for n, m in models.items()}
        with tempfile.TemporaryDirectory() as tmp, _Watch() as watch:
            watch.assertion_target = self
            run_post_training_sensitivity(
                df=df, schema=PanelSchema(time_col="period", entity_col="entity_id", target_col=None),
                preprocessor=pre, feature_names=["num__x1", "num__x2", "num__x3"], models=models,
                baseline_scores=base, thresholds={"iforest": 7.0, "vae": 5.0},
                train_mask=train_mask, test_mask=test_mask, max_test_rows=10,
                combination_top_k=2, random_subsets_per_level=2, missing_levels=(0.5, 0.9),
                out_dir=tmp,
            )
            singles = watch.finished("sensitivity[single-variable]")
            loss = watch.finished("sensitivity[information-loss]")
            pruning = watch.finished("sensitivity[pruning-path]")
        self.assertEqual((singles["n"], singles["total"]), (3, 3))     # x1, x2, x3
        self.assertEqual((loss["n"], loss["total"]), (4, 4))           # 2 levels x 2 replicas
        self.assertEqual((pruning["n"], pruning["total"]), (3, 3))     # remove 1..3 variables
        for step in ("sensitivity.high_zero_analysis", "sensitivity.write_tables",
                     "sensitivity.write_workbook", "sensitivity.write_html_report"):
            self.assertIn(step, watch.started())


class ReportPhaseTests(unittest.TestCase):
    def test_build_report_ticks_once_per_format_and_names_each_writer(self):
        from src.reporting.report import build_report

        context = {"title": "t", "dataset": {}, "models": {}}
        with tempfile.TemporaryDirectory() as tmp, _Watch() as watch:
            watch.assertion_target = self
            written = build_report(context, out_dir=tmp, basename="r", formats=("md", "html"))
            bar = watch.finished("build_report[formats]")
            self.assertTrue(os.path.isfile(written["md"]) and os.path.isfile(written["html"]))
        self.assertEqual((bar["n"], bar["total"]), (2, 2))
        self.assertIn("reporting.build_markdown", watch.started())
        self.assertIn("reporting.build_html", watch.started())


if __name__ == "__main__":
    unittest.main()
