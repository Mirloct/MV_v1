"""The two chart additions requested for the anomaly report: Isolation Forest
split-count (clearest/noisiest variables) and per-seed stability (Jaccard heatmap).

Data sources / inputs: hand-built ``chart_data`` payloads, the same shape
``main.py`` assembles into ``chart_static``.
Created: 2026-09-28
"""
from __future__ import annotations

import unittest

from src.reporting.report_content import build_plotly_figures

_BASE_MODELS = {
    "iforest": {"oot_scores": [0.1, 0.9, 0.3], "metrics": {}},
    "vae": {"oot_scores": [0.2, 0.8, 0.4], "metrics": {}},
}


def _figures(static: dict) -> dict:
    chart_data = {"anomaly_rate": 0.02, "models": _BASE_MODELS, "static": static}
    return {f["id"]: f for f in build_plotly_figures(chart_data)}


class SplitCountChartsTests(unittest.TestCase):
    def test_both_clear_and_noisy_charts_render_when_both_lists_are_present(self):
        figs = _figures({"iforest_splits": {
            "top_clear": [("f0", 3.2), ("f1", 3.8)],
            "top_noisy": [("f9", 9.1), ("f8", 8.7)],
        }})
        self.assertIn("fig-splits-clear", figs)
        self.assertIn("fig-splits-noisy", figs)
        self.assertEqual(figs["fig-splits-clear"]["group"], "modelo")
        self.assertIn("f0", figs["fig-splits-clear"]["html"])
        self.assertIn("f9", figs["fig-splits-noisy"]["html"])

    def test_only_the_populated_side_renders(self):
        figs = _figures({"iforest_splits": {"top_clear": [("f0", 3.2)], "top_noisy": []}})
        self.assertIn("fig-splits-clear", figs)
        self.assertNotIn("fig-splits-noisy", figs)

    def test_absent_payload_renders_neither_chart(self):
        figs = _figures({})
        self.assertNotIn("fig-splits-clear", figs)
        self.assertNotIn("fig-splits-noisy", figs)

    def test_malformed_payload_is_skipped_not_raised(self):
        figs = _figures({"iforest_splits": "not-a-dict"})
        self.assertNotIn("fig-splits-clear", figs)
        self.assertNotIn("fig-splits-noisy", figs)


class StabilitySeedsChartsTests(unittest.TestCase):
    _MATRIX = [[1.0, 0.6, 0.5], [0.6, 1.0, 0.55], [0.5, 0.55, 1.0]]

    def test_one_heatmap_per_detector_that_ran(self):
        figs = _figures({"stability_seeds": {
            "iforest": {"seeds": [1001, 2001, 3001], "pairwise_jaccard": self._MATRIX},
            "vae": {"seeds": [1001, 2001, 3001], "pairwise_jaccard": self._MATRIX},
        }})
        self.assertIn("fig-stability-seeds-if", figs)
        self.assertIn("fig-stability-seeds-vae", figs)

    def test_a_detector_with_no_pairwise_matrix_gets_no_heatmap(self):
        figs = _figures({"stability_seeds": {
            "iforest": {"seeds": [1001, 2001, 3001], "pairwise_jaccard": self._MATRIX},
            "vae": {"status": "UNAVAILABLE"},
        }})
        self.assertIn("fig-stability-seeds-if", figs)
        self.assertNotIn("fig-stability-seeds-vae", figs)

    def test_the_seed_labels_appear_in_the_rendered_chart(self):
        figs = _figures({"stability_seeds": {
            "iforest": {"seeds": [1001, 2001, 3001], "pairwise_jaccard": self._MATRIX},
        }})
        self.assertIn("1001", figs["fig-stability-seeds-if"]["html"])


if __name__ == "__main__":
    unittest.main()
