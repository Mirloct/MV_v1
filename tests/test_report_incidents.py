"""Report incidents must contain failures, never routine warnings."""

from __future__ import annotations

import unittest

from src.reporting.report import _incidents_section_html, _incidents_section_md


class ReportIncidentTests(unittest.TestCase):
    def test_warning_only_is_omitted_from_both_formats(self):
        context = {"incidents": [{"time": "10:00", "level": "WARNING", "message": "noise"}]}
        self.assertEqual(_incidents_section_md(context), "")
        self.assertEqual(_incidents_section_html(context), "")

    def test_errors_remain_but_neighboring_warning_is_filtered(self):
        context = {"incidents": [
            {"time": "10:00", "level": "WARNING", "message": "noise"},
            {"time": "10:01", "level": "ERROR", "message": "real failure"},
        ]}
        for rendered in (_incidents_section_md(context), _incidents_section_html(context)):
            self.assertIn("real failure", rendered)
            self.assertNotIn("noise", rendered)


if __name__ == "__main__":
    unittest.main()


class RowFilterSectionTests(unittest.TestCase):
    STATS = {"n_rows_before": 6000, "n_rows_dropped": 60, "n_rows_after": 5940,
             "n_columns_checked": 13, "cutoff": 0.9}

    def test_markdown_and_html_show_counts(self):
        from src.reporting.report import _row_filter_section_html, _row_filter_section_md
        ctx = {"row_filter": self.STATS}
        for out in (_row_filter_section_md(ctx), _row_filter_section_html(ctx)):
            self.assertIn("6,000", out)
            self.assertIn("60", out)
            self.assertIn("5,940", out)

    def test_absent_stats_render_nothing(self):
        from src.reporting.report import _row_filter_section_md
        self.assertEqual(_row_filter_section_md({}), "")

    def test_an_explicit_sentence_states_the_count_and_percentage(self):
        # Explicit user request: state plainly how many observations (with >= 90% of
        # their variables exactly 0) were removed, not only leave it in a table cell.
        from src.reporting.report import _row_filter_section_html, _row_filter_section_md

        ctx = {"row_filter": self.STATS}
        for out in (_row_filter_section_md(ctx), _row_filter_section_html(ctx)):
            self.assertIn("Se eliminaron", out)
            self.assertIn("60", out)
            self.assertIn("6,000", out)
            self.assertIn("1.0%", out)          # 60 / 6000 = 1.0%
            self.assertIn("90%", out)
            self.assertIn("13", out)            # n_columns_checked


class IForestSplitsRecommendationSectionTests(unittest.TestCase):
    """The feature-removal recommendation built from `split_count_analysis`'s
    output (`chart_data.static.iforest_splits`) -- explicit user request:
    beyond the existing "fewest/most cuts" charts, state a recommendation on
    which variables contribute least to telling outliers apart."""

    def _ctx(self, **overrides):
        block = {
            "top_clear": [("balance", 3.2), ("age", 4.1)],
            "top_noisy": [("region_code", 11.7), ("channel", 10.9)],
            "n_rows_analyzed": 50,
            "n_trees": 300,
            "mean_path_length": 8.4,
            "n_features_total": 5,
            "unused_features": ["legacy_flag"],
        }
        block.update(overrides)
        return {"chart_data": {"static": {"iforest_splits": block}}}

    def test_absent_block_renders_nothing(self):
        from src.reporting.report import (
            _iforest_splits_section_html, _iforest_splits_section_md,
        )
        self.assertEqual(_iforest_splits_section_md({}), "")
        self.assertEqual(_iforest_splits_section_html({}), "")

    def test_empty_top_lists_render_nothing(self):
        # split_count_analysis found no rows/features to analyse -- nothing to
        # recommend from an analysis that never ran, not "everything is unused".
        from src.reporting.report import (
            _iforest_splits_section_html, _iforest_splits_section_md,
        )
        ctx = self._ctx(top_clear=[], top_noisy=[], unused_features=[])
        self.assertEqual(_iforest_splits_section_md(ctx), "")
        self.assertEqual(_iforest_splits_section_html(ctx), "")

    def test_zero_usage_variable_is_recommended_for_removal(self):
        from src.reporting.report import (
            _iforest_splits_section_html, _iforest_splits_section_md,
        )
        ctx = self._ctx()
        for out in (_iforest_splits_section_md(ctx), _iforest_splits_section_html(ctx)):
            self.assertIn("legacy_flag", out)
            self.assertIn("Candidata a eliminar", out)
            self.assertIn("0% de uso", out)
            self.assertIn("1 de las 5", out)   # 1 unused_feature out of n_features_total

    def test_noisiest_used_variables_are_flagged_to_review_not_remove(self):
        from src.reporting.report import (
            _iforest_splits_section_html, _iforest_splits_section_md,
        )
        ctx = self._ctx()
        for out in (_iforest_splits_section_md(ctx), _iforest_splits_section_html(ctx)):
            self.assertIn("region_code", out)
            self.assertIn("Candidata a revisar", out)
            # A weak-signal variable must never be worded as a removal
            # recommendation -- only the 0%-usage tier earns that wording.
            self.assertNotIn("region_code</td><td>0% de uso", out)

    def test_no_unused_variables_states_that_explicitly(self):
        from src.reporting.report import (
            _iforest_splits_section_html, _iforest_splits_section_md,
        )
        ctx = self._ctx(unused_features=[])
        for out in (_iforest_splits_section_md(ctx), _iforest_splits_section_html(ctx)):
            self.assertIn("Ninguna de las 5 variables", out)
            self.assertIn("0% de participación", out)
            self.assertNotIn("Candidata a eliminar", out)

    def test_a_missing_feature_count_renders_a_placeholder_not_the_word_none(self):
        # Adversarial: _iforest_splits_block only requires top_clear/top_noisy
        # to be non-empty, so a payload missing n_features_total is possible
        # in principle -- the prose must not silently print "None".
        from src.reporting.report import _iforest_splits_section_md

        ctx = self._ctx(n_features_total=None)
        out = _iforest_splits_section_md(ctx)
        self.assertNotIn("None", out)
        self.assertIn("1 de las ? variables", out)

    def test_a_pipe_character_in_a_feature_name_does_not_corrupt_the_table(self):
        # Adversarial: _md_table joined cells with a literal "|" and no
        # escaping; a feature name carrying one would split into extra columns.
        from src.reporting.report import _iforest_splits_section_md

        ctx = self._ctx(unused_features=["weird|column"])
        out = _iforest_splits_section_md(ctx)
        self.assertIn("weird\\|column", out)
        # The escaped row must still have exactly 3 columns (Variable,
        # Evidencia, Recomendación) -- 4 unescaped "|" separators per row.
        row_line = next(l for l in out.splitlines() if "weird" in l)
        self.assertEqual(row_line.count("|") - row_line.count("\\|"), 4)
