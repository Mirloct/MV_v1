"""Isolation Forest split-count analysis: which features isolate the flagged rows with
the fewest splits ("cortes"), and which need the most.

Data sources / inputs: seeded in-memory matrices, temporary figure directory.
Created: 2026-09-28
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402
from src.interpretability import split_count_analysis  # noqa: E402
from src.models import IsolationForestDetector  # noqa: E402


def _panel(n=400, seed=0, shift=20.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(float)
    outlier_idx = np.arange(10)
    X[outlier_idx, 0] += shift          # f0 is the ONLY separating feature
    return X, outlier_idx


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name

    def _detector(self, X):
        return IsolationForestDetector(
            n_estimators=100, max_samples=128, max_features=1.0,
            bootstrap=False, random_state=1,
        ).fit(X)


class TestSplitCountAnalysis(Base):
    def test_the_true_discriminative_feature_ranks_clearest(self):
        X, outlier_idx = _panel()
        det = self._detector(X)
        mask = np.zeros(len(X), dtype=bool)
        mask[outlier_idx] = True
        result = split_count_analysis(
            det, X, row_mask=mask, feature_names=[f"f{i}" for i in range(5)],
            out_dir=self.d, top_n=5,
        )
        self.assertEqual(result["top_clear"][0][0], "f0")
        self.assertEqual(result["top_noisy"][-1][0], "f0")          # f0 is never the noisiest
        self.assertLess(result["per_feature"]["f0"]["avg_cuts_when_used"],
                        min(v["avg_cuts_when_used"] for k, v in result["per_feature"].items() if k != "f0"))
        self.assertTrue(os.path.isfile(result["figure_path"]))

    def test_row_mask_restricts_to_exactly_those_rows(self):
        X, outlier_idx = _panel()
        det = self._detector(X)
        mask = np.zeros(len(X), dtype=bool)
        mask[outlier_idx] = True
        result = split_count_analysis(det, X, row_mask=mask, out_dir=self.d)
        self.assertEqual(result["n_rows_analyzed"], len(outlier_idx))

    def test_no_row_mask_uses_every_row(self):
        X, _ = _panel()
        det = self._detector(X)
        result = split_count_analysis(det, X, out_dir=self.d)
        self.assertEqual(result["n_rows_analyzed"], len(X))

    def test_an_all_false_mask_returns_an_empty_result_without_raising(self):
        X, _ = _panel()
        det = self._detector(X)
        result = split_count_analysis(det, X, row_mask=np.zeros(len(X), dtype=bool), out_dir=self.d)
        self.assertEqual(result["n_rows_analyzed"], 0)
        self.assertEqual(result["top_clear"], [])
        self.assertIsNone(result["figure_path"])

    def test_a_feature_never_used_to_isolate_is_left_out_of_the_ranking(self):
        # Two rows, three features, but only feature 0 ever separates -- with max_features
        # capped the forest may never even consider feature 2 while isolating these rows.
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 3)).astype(float)
        X[:5, 0] += 25.0
        det = IsolationForestDetector(
            n_estimators=60, max_samples=64, max_features=1.0, bootstrap=False, random_state=3,
        ).fit(X)
        mask = np.zeros(len(X), dtype=bool)
        mask[:5] = True
        result = split_count_analysis(det, X, row_mask=mask, feature_names=["f0", "f1", "f2"], out_dir=self.d)
        # Whatever appears in per_feature must have been used at least once -- usage_count > 0.
        for stats in result["per_feature"].values():
            self.assertGreater(stats["usage_count"], 0)

    def test_max_features_below_one_translates_the_per_tree_feature_subset_correctly(self):
        # Regression check for the estimators_features_ local->global index translation:
        # with max_features < 1 each tree only sees a random SUBSET of columns, and a bug
        # there would silently attribute splits to the wrong (or an out-of-range) feature.
        X, outlier_idx = _panel(seed=5)
        det = IsolationForestDetector(
            n_estimators=150, max_samples=128, max_features=0.6, bootstrap=False, random_state=4,
        ).fit(X)
        mask = np.zeros(len(X), dtype=bool)
        mask[outlier_idx] = True
        result = split_count_analysis(
            det, X, row_mask=mask, feature_names=[f"f{i}" for i in range(5)], out_dir=self.d,
        )
        for name in result["per_feature"]:
            self.assertIn(name, {"f0", "f1", "f2", "f3", "f4"})   # no out-of-range/garbage name
        self.assertEqual(result["top_clear"][0][0], "f0")


class FlaggedRowsForSplitCountTests(unittest.TestCase):
    """`main._flagged_rows_for_split_count`: normally the calibrated-threshold alert set
    within OOT, but a strict business threshold can legitimately flag zero (or very few)
    rows in a small OOT window -- real, not a bug -- which must not leave the chart empty."""

    def _panel(self, n=1500, n_oot=300, seed=0):
        rng = np.random.default_rng(seed)
        oot_mask = np.zeros(n, dtype=bool)
        if n_oot:                     # `arr[-0:]` is the WHOLE array, not "the last 0"
            oot_mask[-n_oot:] = True
        scores = rng.normal(size=n)
        return oot_mask, scores

    def test_uses_the_calibrated_threshold_when_enough_rows_clear_it(self):
        oot_mask, scores = self._panel()
        # Set the threshold so ~half the OOT rows clear it -- comfortably above min_rows.
        threshold = float(np.median(scores[oot_mask]))
        result = main._flagged_rows_for_split_count(oot_mask, scores, threshold)
        expected = oot_mask & (scores >= threshold)
        np.testing.assert_array_equal(result, expected)
        self.assertGreaterEqual(result.sum(), 10)

    def test_falls_back_to_top_oot_scores_when_the_threshold_flags_almost_nothing(self):
        oot_mask, scores = self._panel()
        threshold = float(np.max(scores)) + 10.0     # flags literally zero rows
        result = main._flagged_rows_for_split_count(oot_mask, scores, threshold, min_rows=10, fallback_n=50)
        self.assertEqual(result.sum(), 50)
        self.assertTrue(np.all(oot_mask[result]))                     # fallback stays within OOT
        # It is genuinely the top-scoring OOT rows, not an arbitrary subset.
        oot_scores_sorted = np.sort(scores[oot_mask])[::-1]
        self.assertAlmostEqual(scores[result].min(), oot_scores_sorted[49], places=9)

    def test_a_non_finite_threshold_falls_back_too(self):
        oot_mask, scores = self._panel()
        result = main._flagged_rows_for_split_count(oot_mask, scores, float("nan"))
        self.assertEqual(result.sum(), 50)
        self.assertTrue(np.all(oot_mask[result]))

    def test_no_oot_rows_returns_an_all_false_mask_without_raising(self):
        oot_mask, scores = self._panel(n_oot=0)
        result = main._flagged_rows_for_split_count(oot_mask, scores, 1.0)
        self.assertEqual(result.sum(), 0)

    def test_fallback_never_exceeds_the_oot_population(self):
        oot_mask, scores = self._panel(n_oot=5)      # fewer OOT rows than fallback_n
        threshold = float(np.max(scores)) + 10.0
        result = main._flagged_rows_for_split_count(oot_mask, scores, threshold, fallback_n=50)
        self.assertEqual(result.sum(), 5)             # capped at however many OOT rows exist

    def test_a_missing_logger_is_accepted_silently(self):
        oot_mask, scores = self._panel()
        threshold = float(np.max(scores)) + 10.0
        # Must not raise just because no logger was passed.
        result = main._flagged_rows_for_split_count(oot_mask, scores, threshold, logger=None)
        self.assertEqual(result.sum(), 50)

    def test_logs_the_fallback_when_a_logger_is_given(self):
        oot_mask, scores = self._panel()
        threshold = float(np.max(scores)) + 10.0
        log = logging.getLogger("test_flagged_rows_for_split_count")
        with self.assertLogs(log, level="INFO") as captured:
            main._flagged_rows_for_split_count(oot_mask, scores, threshold, logger=log)
        self.assertTrue(any("split_count_analysis" in m for m in captured.output))


if __name__ == "__main__":
    unittest.main()
