"""Identification-only columns (job title, name, area, an internal reference number, ...) declared via
``PanelSchema.identification_columns`` must never reach any modelling phase, from the ONE place that set
is assembled: ``src.data.loader.key_columns``. This file pins that single source of truth and the two
call sites not already covered by their own test file (``PanelFeatureEngineer``/``build_preprocessing_pipeline``,
and ``infer_numeric_features``); ``categorical_sources`` (tests/test_mixed_vae.py), ``drop_exact_zero_rows``
(tests/test_zero_row_filter.py) and post-training sensitivity (tests/test_sensitivity.py) pin their own call site.
Display contexts (the analyst dashboard, the OOT Excel "VARIABLES" columns) are deliberately NOT covered here:
identification is exactly what those are for, and they keep seeing these columns.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema, key_columns
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.preprocessing.statistics import infer_numeric_features


def _schema(**ident) -> PanelSchema:
    return PanelSchema(time_col="period", entity_col="entity_id", target_col=None, **ident)


class KeyColumnsTests(unittest.TestCase):
    def test_gathers_structural_keys_target_and_identification_columns(self):
        schema = _schema(identification_columns=("puesto", "area"))
        schema.target_col = "label"
        self.assertEqual(key_columns(schema),
                         {"entity_id", "period", "label", "puesto", "area"})

    def test_none_and_empty_are_dropped_not_kept_as_literal_values(self):
        schema = PanelSchema(time_col=None, entity_col="entity_id", target_col=None,
                             identification_columns=())
        self.assertEqual(key_columns(schema), {"entity_id"})

    def test_a_schema_without_the_field_still_works(self):
        # An older PanelSchema instance (pickled/cached before this field existed) has no
        # `identification_columns` attribute at all; key_columns must not raise on it.
        class _OldSchema:
            entity_col, time_col, target_col = "entity_id", "period", None

        self.assertEqual(key_columns(_OldSchema()), {"entity_id", "period"})


class FeatureMatrixExclusionTests(unittest.TestCase):
    """`build_preprocessing_pipeline` -> `PanelFeatureEngineer`: the actual model input."""

    def _panel(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": ["e1", "e1", "e2", "e2"],
            "period": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
            "puesto": ["analista", "analista", "gerente", "gerente"],   # identification only
            "employee_ref": [1001.0, 1001.0, 1002.0, 1002.0],           # identification only, numeric
            "x": [1.0, 2.0, 3.0, 4.0],
        })

    def _feature_names(self, schema: PanelSchema) -> list:
        df = self._panel()
        pipe = build_preprocessing_pipeline(schema, add_panel_features=False)
        pipe.fit(df)
        return list(pipe.named_steps["column_transform"].get_feature_names_out())

    def test_identification_columns_never_become_a_feature(self):
        schema = _schema(identification_columns=("puesto", "employee_ref"))
        names = self._feature_names(schema)
        self.assertEqual(names, ["num__x"])

    def test_without_identification_columns_they_would_have_been_features(self):
        # Same panel, nothing declared: the historical (pre-fix) behaviour, so the test above
        # is proven to depend on the declaration and not on some other exclusion.
        names = self._feature_names(_schema())
        self.assertIn("num__employee_ref", names)
        self.assertTrue(any(n.startswith("cat__puesto") for n in names))


class NumericTransformDiagnosticExclusionTests(unittest.TestCase):
    def test_identification_columns_are_not_diagnosed_for_a_transform(self):
        df = pd.DataFrame({
            "entity_id": range(30), "period": pd.to_datetime(["2024-01-01"] * 30),
            "employee_ref": np.arange(1000.0, 1030.0),   # >20 distinct values: would qualify otherwise
            "x": np.random.default_rng(0).normal(size=30),
        })
        schema = _schema(identification_columns=("employee_ref",))
        cols = infer_numeric_features(df, schema)
        self.assertEqual(cols, ["x"])


if __name__ == "__main__":
    unittest.main()
