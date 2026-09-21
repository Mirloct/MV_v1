"""Exact-zero row filter: rows with >= cutoff of input columns == 0 are dropped."""

import logging
import unittest

import numpy as np
import pandas as pd

from src.data import PanelSchema, drop_exact_zero_rows

_COLS = [f"f{i}" for i in range(1, 11)]


def _schema() -> PanelSchema:
    return PanelSchema(time_col="period", entity_col="entity_id", target_col=None)


def _frame() -> pd.DataFrame:
    rows = {
        "A": [0] * 10,                          # 10/10 zeros -> dropped
        "B": [0] * 9 + [1],                     # 9/10 = exactly 90% -> dropped
        "C": [5, 3] + [0] * 8,                  # 8/10 -> kept
        "D": [0] * 8 + [np.nan, 4],             # 8 zeros + 1 NaN -> 80%, kept
        "E": [0] * 10,                          # 10/10 zeros -> dropped
    }
    frame = pd.DataFrame(list(rows.values()), columns=_COLS)
    frame.insert(0, "entity_id", list(rows))
    frame.insert(1, "period", pd.to_datetime(["2026-01-01"] * len(rows)))
    # A text category can never be "exactly 0"; it must not dilute the share.
    frame["region"] = ["north", "south", "east", "west", "north"]
    return frame


class ExactZeroRowFilterTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_zero_row_filter")

    def test_drops_rows_at_or_above_cutoff_and_reports_counts(self):
        out, stats = drop_exact_zero_rows(_frame(), _schema(), self.logger, cutoff=0.90)
        self.assertEqual(out["entity_id"].tolist(), ["C", "D"])
        self.assertEqual(stats["n_rows_before"], 5)
        self.assertEqual(stats["n_rows_dropped"], 3)
        self.assertEqual(stats["n_rows_after"], 2)

    def test_missing_value_is_not_counted_as_zero(self):
        # D would reach 9/10 (90%) if its NaN counted as zero, and be dropped.
        out, _ = drop_exact_zero_rows(_frame(), _schema(), self.logger, cutoff=0.90)
        self.assertIn("D", out["entity_id"].tolist())

    def test_key_datetime_and_categorical_columns_are_not_checked(self):
        _, stats = drop_exact_zero_rows(_frame(), _schema(), self.logger, cutoff=0.90)
        self.assertEqual(stats["n_columns_checked"], 10)

    def test_no_match_drops_nothing(self):
        frame = _frame().loc[[2, 3]].reset_index(drop=True)
        out, stats = drop_exact_zero_rows(frame, _schema(), self.logger, cutoff=0.90)
        self.assertEqual(len(out), 2)
        self.assertEqual(stats["n_rows_dropped"], 0)

    def test_index_is_reset_after_dropping(self):
        out, _ = drop_exact_zero_rows(_frame(), _schema(), self.logger, cutoff=0.90)
        self.assertEqual(out.index.tolist(), [0, 1])

    def test_logs_counts_when_rows_are_dropped(self):
        with self.assertLogs(self.logger, level="WARNING") as captured:
            drop_exact_zero_rows(_frame(), _schema(), self.logger, cutoff=0.90)
        message = " ".join(captured.output)
        self.assertIn("3/5", message)
        self.assertIn("quedan 2 filas", message)


if __name__ == "__main__":
    unittest.main()
