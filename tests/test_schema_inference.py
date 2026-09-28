"""Panel-schema column-name inference: hints for the real-data column names this project's
own banking panels use, on top of the generic ones (entity/individual/id, period/date/time).

Data sources / inputs: small in-memory DataFrames.
Created: 2026-09-27
"""
from __future__ import annotations

import logging
import unittest

import pandas as pd

from src.data.loader import _infer_entity_col, _infer_time_col

LOG = logging.getLogger("test_schema_inference")


class RealDataColumnNameHintsTests(unittest.TestCase):
    """`codclavepartycli` (entity) and `codmes` (time) are real column names this project's
    own banking data uses; neither contains a generic hint substring ("id", "period", ...),
    so without an explicit hint they were not recognised as the entity/time column."""

    def test_codclavepartycli_is_recognised_as_the_entity_column(self):
        df = pd.DataFrame({
            "codclavepartycli": ["a", "a", "b", "b"],
            "period": pd.to_datetime(["2024-01-01", "2024-02-01"] * 2),
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        self.assertEqual(_infer_entity_col(df, time_col="period", logger=LOG), "codclavepartycli")

    def test_codmes_is_recognised_as_the_time_column(self):
        # An integer codmes (202401-style, dtype int64) has neither a datetime dtype nor an
        # object dtype, so the structural fallbacks in `_infer_time_col` cannot catch it --
        # only the name hint can.
        df = pd.DataFrame({
            "codclavepartycli": ["a", "a", "b", "b"],
            "codmes": [202401, 202402, 202401, 202402],
            "x": [1.0, 2.0, 3.0, 4.0],
        })
        self.assertEqual(_infer_time_col(df, logger=LOG), "codmes")

    def test_both_together_resolve_a_real_style_panel(self):
        df = pd.DataFrame({
            "codclavepartycli": ["a", "a", "b", "b"],
            "codmes": [202401, 202402, 202401, 202402],
            "monto": [10.0, 20.0, 30.0, 40.0],
        })
        time_col = _infer_time_col(df, logger=LOG)
        entity_col = _infer_entity_col(df, time_col=time_col, logger=LOG)
        self.assertEqual((entity_col, time_col), ("codclavepartycli", "codmes"))

    def test_hints_are_substring_case_insensitive_like_the_existing_ones(self):
        # Case-insensitive and matches as a substring of a longer name -- same rule the
        # existing hints already follow (e.g. "entity_id" matches on "entity").
        df = pd.DataFrame({
            "CODCLAVEPARTYCLI": ["a", "b"], "CODMESREPORTE": [202401, 202402], "x": [1.0, 2.0],
        })
        time_col = _infer_time_col(df, logger=LOG)
        self.assertEqual(time_col, "CODMESREPORTE")
        self.assertEqual(_infer_entity_col(df, time_col=time_col, logger=LOG), "CODCLAVEPARTYCLI")


if __name__ == "__main__":
    unittest.main()
