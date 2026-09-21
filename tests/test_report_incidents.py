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
