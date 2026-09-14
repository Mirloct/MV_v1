"""Focused contract tests for the analyst dashboard's review tabs/download."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.loader import PanelSchema
from src.reporting.analyst_dashboard import build_analyst_dashboard


class AnalystDashboardTests(unittest.TestCase):
    def _build(self) -> str:
        schema = PanelSchema(time_col="period", entity_col="customer_id", target_col=None)
        if_table = pd.DataFrame({
            "customer_id": ["only-if", "both"],
            "period": ["2026-08", "2026-09"],
            "anomaly_score": [0.9, 0.8],
            "percentil": ["p99", "p95"],
            "top_5_variables": ["balance, transfers", "income"],
        })
        vae_table = pd.DataFrame({
            "customer_id": ["only-vae", "both"],
            "period": ["2026-09", "2026-09"],
            "anomaly_score": [8.0, 7.0],
            "percentil": ["p99", "p95"],
            "top_5_variables": ["age, products", "transactions"],
        })
        raw = pd.DataFrame({
            "customer_id": ["only-if", "only-if", "only-vae", "both"],
            "period": ["2026-08", "2026-09", "2026-09", "2026-09"],
            "balance": [100, 150, 200, 300],
            "free_text": ["first", "second", "third", "kept verbatim"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.html"
            build_analyst_dashboard(
                vae_table, schema, "vae", ["2026-08", "2026-09"],
                {"only-if": 98.0, "only-vae": 50.0, "both": 97.0},
                {"only-if": 40.0, "only-vae": 99.0, "both": 98.0},
                {"only-if": 0.9, "only-vae": 0.2, "both": 0.8},
                {"only-if": 1.0, "only-vae": 8.0, "both": 7.0},
                {"only-vae": ["2026-09"], "both": ["2026-09"]},
                n_total_oot=3,
                out_path=str(out),
                model_tables={"iforest": if_table, "vae": vae_table},
                months_present_by_model={
                    "iforest": {"only-if": ["2026-08", "2026-09"], "both": ["2026-09"]},
                    "vae": {"only-vae": ["2026-09"], "both": ["2026-09"]},
                },
                oot_records=raw,
            )
            return out.read_text(encoding="utf-8")

    def test_three_tabs_partition_p95_union(self):
        html = self._build()
        self.assertIn("Solo IF <span>1</span>", html)
        self.assertIn("Solo IF+VAE <span>1</span>", html)
        self.assertIn("Intersección <span>1</span>", html)
        self.assertEqual(html.count('data-tab="if_only"'), 1)
        self.assertEqual(html.count('data-tab="vae_only"'), 1)
        self.assertEqual(html.count('data-tab="intersection"'), 1)

    def test_profile_payload_keeps_every_oot_row_and_source_column(self):
        html = self._build()
        columns_match = re.search(r"var OOT_COLUMNS = (.*?);", html)
        self.assertIsNotNone(columns_match)
        self.assertEqual(
            json.loads(columns_match.group(1)),
            ["customer_id", "period", "balance", "free_text"],
        )
        self.assertIn('"free_text": "first"', html)
        self.assertIn('"free_text": "second"', html)
        self.assertIn("Descargar todas las variables (.csv)", html)
        self.assertIn("downloadObservation()", html)

    def test_detector_specific_explanations_and_months_are_embedded(self):
        html = self._build()
        self.assertIn('"if_top5": "income"', html)
        self.assertIn('"vae_top5": "transactions"', html)
        self.assertIn('"if_months": ["2026-08", "2026-09"]', html)
        self.assertIn('"vae_months": ["2026-09"]', html)


if __name__ == "__main__":
    unittest.main()
