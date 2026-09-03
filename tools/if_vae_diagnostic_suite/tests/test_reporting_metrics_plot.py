"""Behavior: a chart visualizing per-detector operational performance
(precision@K by score candidate) so a reader can monitor "each layer" (IF,
VAE, ensembles) at a glance instead of only reading `metrics.csv` as a
table -- requested after the IF-VAE Diagnostic Suite integration produced
a report with no visual comparison of detector performance, only the
disagreement-percentile scatter (which shows agreement, not performance).
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ifvae_diag.reporting import metrics_bar_plot


class MetricsBarPlotTests(unittest.TestCase):
    def test_writes_a_png_when_metrics_present(self):
        metrics = pd.DataFrame([
            {"score_name": "if_percentile", "k": 10, "precision_at_k": 0.1,
             "recall_at_k": 0.03, "average_precision": 0.06},
            {"score_name": "vae_percentile", "k": 10, "precision_at_k": 0.8,
             "recall_at_k": 0.24, "average_precision": 0.28},
            {"score_name": "if_percentile", "k": 25, "precision_at_k": 0.12,
             "recall_at_k": 0.09, "average_precision": 0.06},
            {"score_name": "vae_percentile", "k": 25, "precision_at_k": 0.4,
             "recall_at_k": 0.30, "average_precision": 0.28},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.png"
            written = metrics_bar_plot(metrics, path)
            self.assertTrue(written)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_skips_without_writing_when_metrics_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.png"
            written = metrics_bar_plot(pd.DataFrame(), path)
            self.assertFalse(written)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
