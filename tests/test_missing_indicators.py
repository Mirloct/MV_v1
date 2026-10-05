"""`missing__<col>` features (`MissingnessIndicator`, on by default):
explicit user question ("why do these appear, and I don't want them")
resolved into an explicit opt-out, `add_missing_indicators=False`
(`--no-add-missing-indicators` on the CLI), rather than editing code.

Data sources / inputs: small synthetic DataFrames built in-test.
Created: 2026-10-05
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402
from src.data.loader import PanelSchema  # noqa: E402
from src.preprocessing.pipeline import build_preprocessing_pipeline  # noqa: E402


def _schema() -> PanelSchema:
    return PanelSchema(time_col="period", entity_col="entity_id", target_col=None)


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "entity_id": range(6),
        "period": pd.to_datetime(["2024-01-01"] * 6),
        "balance": [100.0, np.nan, 300.0, np.nan, 500.0, 0.0],   # 2 real NaNs
        "complete": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],              # never missing
    })


def _feature_names(**pipeline_kwargs) -> list:
    pipe = build_preprocessing_pipeline(_schema(), add_panel_features=False, **pipeline_kwargs)
    pipe.fit(_panel())
    return list(pipe.named_steps["column_transform"].get_feature_names_out())


class MissingIndicatorDefaultOnTests(unittest.TestCase):
    def test_a_column_with_a_nan_gets_a_missing_flag(self):
        names = _feature_names()
        self.assertIn("missing__balance__missing", names)

    def test_a_column_with_no_nan_gets_no_flag(self):
        names = _feature_names()
        self.assertNotIn("missing__complete__missing", names)

    def test_the_flag_values_mark_exactly_the_originally_missing_rows(self):
        pipe = build_preprocessing_pipeline(_schema(), add_panel_features=False)
        pipe.fit(_panel())
        names = list(pipe.named_steps["column_transform"].get_feature_names_out())
        X = pipe.transform(_panel())
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        flag_col = X[:, names.index("missing__balance__missing")]
        self.assertEqual(list(flag_col), [0.0, 1.0, 0.0, 1.0, 0.0, 0.0])


class MissingIndicatorOptOutTests(unittest.TestCase):
    def test_add_missing_indicators_false_creates_no_flag_at_all(self):
        names = _feature_names(add_missing_indicators=False)
        self.assertFalse(any(n.startswith("missing__") for n in names))
        # The underlying column is still there, just imputed -- disabling the
        # flag must not also drop the feature itself.
        self.assertIn("num__balance", names)

    def test_cli_flag_defaults_to_true_and_can_be_turned_off(self):
        parser = main.build_arg_parser()
        cfg_on = main.config_from_args(parser.parse_args([]))
        self.assertTrue(cfg_on.add_missing_indicators)
        cfg_off = main.config_from_args(parser.parse_args(["--no-add-missing-indicators"]))
        self.assertFalse(cfg_off.add_missing_indicators)


if __name__ == "__main__":
    unittest.main()
