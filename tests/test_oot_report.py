"""Contract tests for the P95+ OOT control export and P99 selection.

Data sources / inputs: synthetic scored panel fixtures and temporary Excel files.
Created: 2026-11-22
Last modified: 2026-11-26
Changelog:
- 2026-11-26: Verify P95 defaults, full-population cutoffs, and inclusive ties.
- 2026-11-25: Added adversarial input drift and percentile boundary coverage.
- 2026-11-22: Added P90 control default plus explicit P99 population/tie tests.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.oot_report import (
    _resolve_out_path,
    export_oot_top_anomalies,
)
from src.utils import paths


class OOTExportTests(unittest.TestCase):
    schema = PanelSchema(time_col="period", entity_col="entity_id", target_col=None)

    def frame(self):
        return pd.DataFrame({
            "entity_id": ["a", "b", "c"],
            "period": ["2026-10"] * 3,
            "anomaly_score": [1.0, 2.0, 3.0],
            "feature": [10, 20, 30],
        })

    def assert_invalid_export(self, frame, message, **kwargs):
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "new", "oot.xlsx")
            with patch.object(pd.DataFrame, "to_excel") as writer:
                with self.assertRaisesRegex(ValueError, message):
                    export_oot_top_anomalies(
                        frame, self.schema, out_path=destination, **kwargs,
                    )
                writer.assert_not_called()
            self.assertFalse(os.path.exists(os.path.dirname(destination)))
        pd.testing.assert_frame_equal(frame, original)

    def test_empty_oot_and_missing_columns_fail_before_workbook(self):
        self.assert_invalid_export(self.frame().iloc[:0], "No rows found for OOT")
        frame = self.frame()
        frame["period"] = pd.NaT
        self.assert_invalid_export(frame, "No rows found for OOT")
        for column in ("entity_id", "period", "anomaly_score"):
            with self.subTest(column=column):
                self.assert_invalid_export(
                    self.frame().drop(columns=column), f"missing required columns.*{column}",
                )

    def test_invalid_scores_and_null_entities_fail_before_dedup(self):
        for value in (np.nan, np.inf, -np.inf, pd.NA, "broken", 1 + 2j):
            with self.subTest(score=value):
                frame = self.frame().astype({"anomaly_score": object})
                frame.loc[0, "anomaly_score"] = value
                frame.loc[0, "entity_id"] = "c"
                self.assert_invalid_export(frame, "anomaly_score.*finite numeric scores")
        for value in (None, np.nan, pd.NA):
            with self.subTest(entity=value):
                frame = self.frame()
                frame.loc[0, "entity_id"] = value
                self.assert_invalid_export(frame, "entity_id.*null entity")

    def test_invalid_percentiles_fail_even_with_default_model_path(self):
        for percentile in (-1, 101, np.nan, np.inf, -np.inf, "bad", [90], True):
            with self.subTest(percentile=percentile):
                self.assert_invalid_export(
                    self.frame(), "min_percentile.*finite number.*0, 100",
                    min_percentile=percentile, model_name="iforest",
                )
                with patch.object(pd.DataFrame, "to_excel") as writer:
                    with self.assertRaisesRegex(ValueError, "min_percentile"):
                        export_oot_top_anomalies(
                            self.frame(), self.schema, min_percentile=percentile,
                            model_name="iforest",
                        )
                    writer.assert_not_called()

    def test_numeric_text_scores_sort_numerically_without_mutating_input(self):
        frame = self.frame()
        frame["anomaly_score"] = ["2", "10", "9"]
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            path, table = export_oot_top_anomalies(
                frame, self.schema, out_path=os.path.join(tmp, "oot.xlsx"),
                min_percentile=0,
            )
            reread = pd.read_excel(path)
        self.assertEqual(list(table["entity_id"]), ["b", "c", "a"])
        self.assertEqual(list(reread["anomaly_score"]), [10, 9, 2])
        pd.testing.assert_frame_equal(frame, original)

    def test_percentile_endpoints_singleton_and_all_ties(self):
        for frame, percentile, expected in (
            (self.frame(), 0, ["c", "b", "a"]),
            (self.frame(), 100, ["c"]),
            (self.frame().iloc[:1], 95, ["a"]),
            (self.frame().assign(anomaly_score=5.0), 95, ["a", "b", "c"]),
        ):
            with self.subTest(percentile=percentile, rows=len(frame)):
                with tempfile.TemporaryDirectory() as tmp:
                    _, table = export_oot_top_anomalies(
                        frame, self.schema, out_path=os.path.join(tmp, "oot.xlsx"),
                        min_percentile=percentile,
                    )
                self.assertEqual(list(table["entity_id"]), expected)

    def test_p90_includes_exact_cutoff_and_keeps_max_month_features(self):
        frame = pd.DataFrame({
            "entity_id": [f"e{i:02d}" for i in range(11)] + ["e10"] * 20,
            "period": ["2026-10"] * 11 + ["2026-09"] * 20,
            "anomaly_score": list(range(11)) + [-100] * 20,
            "feature": list(range(11)) + [-1] * 20,
        })
        with tempfile.TemporaryDirectory() as tmp:
            _, table = export_oot_top_anomalies(
                frame, self.schema, out_path=os.path.join(tmp, "oot.xlsx"),
                n_oot_periods=2,
                min_percentile=90.0,
            )
        self.assertEqual(list(table["entity_id"]), ["e10", "e09"])
        self.assertEqual(list(table["anomaly_score"]), [10, 9])
        self.assertEqual(list(table["feature"]), [10, 9])
        self.assertEqual(list(table["period"]), ["2026-10"] * 2)

    def test_duplicate_columns_fail_with_clear_error(self):
        frame = self.frame()
        frame.columns = ["entity_id", "period", "anomaly_score", "anomaly_score"]
        self.assert_invalid_export(frame, "unique column names")

    def test_invalid_export_preserves_existing_workbook_then_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = os.path.join(tmp, "oot.xlsx")
            export_oot_top_anomalies(self.frame(), self.schema, out_path=destination)
            with open(destination, "rb") as stream:
                original = stream.read()
            invalid = self.frame().assign(anomaly_score=np.inf)
            with self.assertRaisesRegex(ValueError, "finite numeric scores"):
                export_oot_top_anomalies(invalid, self.schema, out_path=destination)
            with open(destination, "rb") as stream:
                self.assertEqual(stream.read(), original)
            _, table = export_oot_top_anomalies(
                self.frame(), self.schema, out_path=destination, min_percentile=None,
                top_n=2,
            )
            self.assertEqual(list(pd.read_excel(destination)["entity_id"]), ["c", "b"])
            self.assertEqual(list(table["entity_id"]), ["c", "b"])

    def test_custom_columns_nullable_scores_and_oot_boundary(self):
        frame = self.frame()
        frame["anomaly_score"] = pd.Series([1, 2, 3], dtype="Float64")
        historical = pd.DataFrame({
            "entity_id": [None], "period": ["2026-09"],
            "anomaly_score": [np.inf], "feature": [-1],
        })
        frame = pd.concat([historical, frame], ignore_index=True).rename(columns={
            "entity_id": "customer", "period": "month", "anomaly_score": "risk",
        })
        schema = PanelSchema(time_col="month", entity_col="customer", target_col=None)
        with tempfile.TemporaryDirectory() as tmp:
            _, table = export_oot_top_anomalies(
                frame, schema, score_col="risk", out_path=os.path.join(tmp, "oot.xlsx"),
            )
        self.assertEqual(list(table["customer"]), ["c"])
        self.assertEqual(list(table["risk"]), [3])
        self.assertEqual(list(table.columns[:4]), ["customer", "month", "risk", "percentil"])

    def test_default_path_and_selection_are_p95_control_base(self):
        self.assertTrue(
            _resolve_out_path(paths.OOT_REPORT_DEFAULT, 0.10, "iforest", None).endswith(
                os.path.join("reports", "oot_p95_iforest.xlsx")
            )
        )
        frame = pd.DataFrame({
            "entity_id": [f"e{i:03d}" for i in range(100)],
            "period": ["2026-10"] * 100,
            "anomaly_score": np.arange(100, dtype=float),
            "feature": np.arange(100),
        })
        with tempfile.TemporaryDirectory() as tmp:
            _, table = export_oot_top_anomalies(
                frame, self.schema, out_path=os.path.join(tmp, "oot.xlsx")
            )
        self.assertEqual(len(table), 5)
        self.assertEqual(list(table["anomaly_score"]), [99, 98, 97, 96, 95])
        self.assertEqual(list(table["percentil"]), ["p99"] + ["p95"] * 4)

    def test_p95_uses_full_deduplicated_oot_population_and_keeps_cutoff_ties(self):
        frame = pd.DataFrame({
            "entity_id": [f"e{i:03d}" for i in range(101)],
            "period": ["2026-10"] * 101,
            "anomaly_score": list(range(94)) + [95, 95, 95, 97, 98, 99, 100],
            "feature": list(range(101)),
        })
        duplicates = pd.DataFrame({
            "entity_id": ["e100"] * 40,
            "period": ["2026-09"] * 40,
            "anomaly_score": [-100] * 40,
            "feature": [-1] * 40,
        })
        historical = pd.DataFrame({
            "entity_id": ["historical"], "period": ["2026-08"],
            "anomaly_score": [10000], "feature": [-2],
        })
        frame = pd.concat([frame, duplicates, historical], ignore_index=True)
        original = frame.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            path, table = export_oot_top_anomalies(
                frame, self.schema, out_path=os.path.join(tmp, "oot.xlsx"),
                n_oot_periods=2, min_percentile=95.0,
            )
            reread = pd.read_excel(path)
        self.assertEqual(list(table["entity_id"]), [
            "e100", "e099", "e098", "e097", "e094", "e095", "e096",
        ])
        self.assertEqual(list(table["anomaly_score"]), [100, 99, 98, 97, 95, 95, 95])
        self.assertEqual(list(table["percentil"]), ["p99", "p99"] + ["p95"] * 5)
        self.assertEqual(list(table["feature"]), [100, 99, 98, 97, 94, 95, 96])
        self.assertEqual(list(table["period"]), ["2026-10"] * 7)
        pd.testing.assert_frame_equal(reread, table, check_dtype=False)
        pd.testing.assert_frame_equal(frame, original)

    def test_p99_uses_full_deduplicated_population_and_keeps_cutoff_ties(self):
        scores = list(range(98)) + [100, 100]
        rows = [
            {"entity_id": f"e{i:03d}", "period": "2026-10", "anomaly_score": score,
             "feature": i}
            for i, score in enumerate(scores)
        ]
        # Same entity in an earlier OOT month: the max-score month must win,
        # while the percentile denominator remains one row per entity.
        rows.append({"entity_id": "e099", "period": "2026-09", "anomaly_score": 1,
                     "feature": -1})
        with tempfile.TemporaryDirectory() as tmp:
            _, table = export_oot_top_anomalies(
                pd.DataFrame(rows), self.schema, out_path=os.path.join(tmp, "oot.xlsx"),
                n_oot_periods=2, min_percentile=99.0,
            )
        self.assertEqual(set(table["entity_id"]), {"e098", "e099"})
        self.assertEqual(set(table["period"]), {"2026-10"})
        self.assertEqual(set(table["percentil"]), {"p99"})


if __name__ == "__main__":
    unittest.main()
