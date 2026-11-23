"""Contract tests for the broad P90+ OOT control export and P99 selection.

Data sources / inputs: synthetic scored panel fixtures and temporary Excel files.
Created: 2026-11-22
Last modified: 2026-11-22
Changelog:
- 2026-11-22: Added P90 control default plus explicit P99 population/tie tests.
"""

from __future__ import annotations

import os
import tempfile
import unittest

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

    def test_default_path_and_selection_are_p90_control_base(self):
        self.assertTrue(
            _resolve_out_path(paths.OOT_REPORT_DEFAULT, 0.10, "iforest", None).endswith(
                os.path.join("reports", "oot_p90_iforest.xlsx")
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
        self.assertEqual(len(table), 10)
        self.assertEqual(set(table["percentil"]), {"p90", "p95", "p99"})

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
