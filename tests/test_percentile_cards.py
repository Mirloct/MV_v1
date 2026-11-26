"""Independent OOT person-count and rendered-card contracts.

Data sources / inputs: full synthetic per-detector entity-score populations.
Created: 2026-11-26
Last modified: 2026-11-26
Changelog: 2026-11-26 — cumulative union, overlap, ties and rendering checks.
"""
import unittest

from src.reporting.percentile_cards import (
    percentile_summary, percentile_cards_html, percentile_cards_markdown,
)


class PercentileCardTests(unittest.TestCase):
    def test_pipeline_defaults_keep_control_p95_and_dashboard_p99(self):
        from main import PipelineConfig, build_arg_parser
        self.assertEqual(PipelineConfig().oot_min_percentile, 95.0)
        self.assertEqual(build_arg_parser().parse_args([]).oot_min_percentile, 95.0)
        self.assertEqual(PipelineConfig().analyst_review_percentile, 99.0)

    def context(self, forest, vae):
        return {"chart_data": {"models": {
            "iforest": {"true_oot_entity_scores": forest},
            "vae": {"true_oot_entity_scores": vae},
        }}}

    def test_disjoint_detector_tails_count_unique_people_cumulatively(self):
        context = self.context({str(i): i for i in range(1000)},
                               {str(i): 999 - i for i in range(1000)})
        summary = percentile_summary(context)
        self.assertEqual([r["count"] for r in summary], [100, 60, 20])
        self.assertEqual([r["if_count"] for r in summary], [50, 30, 10])
        self.assertEqual([r["intersection"] for r in summary], [0, 0, 0])
        self.assertEqual([r["total"] for r in summary], [1000] * 3)
        rendered = percentile_cards_html(context)
        self.assertIn("<div class='value'>100</div>", rendered)
        self.assertIn("<div class='value'>60</div>", rendered)
        self.assertIn("<div class='value'>20</div>", rendered)
        self.assertIn("| P97+ | 60 | 30 | 30 | 0 | 1000 |", percentile_cards_markdown(context))

    def test_same_people_are_not_double_counted_and_ties_are_retained(self):
        summary = percentile_summary(self.context({"a": 7, "b": 7}, {"a": 8, "b": 8}))
        self.assertEqual([r["count"] for r in summary], [2, 2, 2])
        self.assertEqual([r["intersection"] for r in summary], [2, 2, 2])

    def test_cards_ignore_filtered_excel_rows_and_use_full_oot_population(self):
        context = self.context({str(i): i for i in range(1000)}, {})
        context["oot_excel"] = "filtered-control.xlsx"
        self.assertEqual([r["count"] for r in percentile_summary(context)], [50, 30, 10])

    def test_no_data_omits_cards_instead_of_inventing_zero_people(self):
        self.assertEqual(percentile_cards_html({}), "")
        self.assertEqual(percentile_cards_markdown({}), "")

    def test_invalid_detector_scores_fail_with_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "scores must be finite"):
            percentile_summary(self.context({"a": float("nan")}, {}))
