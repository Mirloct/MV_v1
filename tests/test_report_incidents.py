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
