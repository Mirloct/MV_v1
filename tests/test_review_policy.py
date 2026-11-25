"""Adversarial numeric-policy tests independent of percentile ranks.

Data sources / inputs: deterministic entity-score maps.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — boundary, ties, missingness and union contracts.
"""
import unittest

from src.evaluation.review_policy import review_ids, review_union, split_cohort_mode


class ReviewPolicyTests(unittest.TestCase):
    def test_critical_collapse_blocks_operational_publication(self):
        from src.evaluation.model_acceptance import require_acceptable_health
        with self.assertRaisesRegex(RuntimeError, "Official OOT publication is blocked"):
            require_acceptable_health({"collapsed": True, "severity": "critical"})
        require_acceptable_health({"collapsed": False, "severity": "info"})
        require_acceptable_health({"collapsed": True, "severity": "warning"})

    def test_small_threshold_cohort_discloses_fallback(self):
        self.assertEqual(split_cohort_mode([True, True], [1, 2], 1), "top_score_fallback")
        self.assertEqual(split_cohort_mode([True] * 10, [2] * 10, 1), "calibrated_threshold")

    def test_report_discloses_fallback_without_calibrated_cohort_claim(self):
        from src.reporting.report import _iforest_splits_lead_sentence
        text = _iforest_splits_lead_sentence({"selection_mode": "top_score_fallback"})
        self.assertIn("fallback de mayor score OOT", text)
        self.assertNotIn("marcadas por el umbral calibrado", text)

    def test_chart_counts_numeric_ties_instead_of_rank(self):
        from src.reporting.report_content import build_plotly_figures
        data = {"models": {
            "iforest": {"true_oot_entity_scores": {str(i): 7 for i in range(20)}},
            "vae": {"true_oot_entity_scores": {str(i): 8 for i in range(20)}},
        }}
        charts = build_plotly_figures(data)
        agreement = next(c for c in charts if c["id"] == "fig-agreement")
        self.assertIn("20 individuos", agreement["html"])

    def test_500_distinct_scores_select_five_not_six(self):
        self.assertEqual(review_ids({str(i): i for i in range(500)}),
                         {"495", "496", "497", "498", "499"})

    def test_constant_population_retains_every_tie(self):
        self.assertEqual(review_ids({"a": 7, "b": 7}), {"a", "b"})

    def test_union_counts_entities_once_across_detectors(self):
        self.assertEqual(review_union({"a": 1, "b": 9}, {"a": 9, "b": 1}), {"a", "b"})

    def test_nonfinite_source_drift_fails_with_diagnostic(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "scores must be finite"):
                review_ids({"a": score})

    def test_invalid_percentiles_fail_even_when_population_empty(self):
        for percentile in (0, 100, float("nan")):
            with self.assertRaisesRegex(ValueError, "percentile must be finite"):
                review_ids({}, percentile)
