"""VAE score-view behavior from synthetic score payloads.

Data sources / inputs: synthetic scores and report chart payloads.
Created: 2026-11-26
Last modified: 2026-11-26
Changelog: 2026-11-26 — verify stacked detail, full tails and unchanged IF views.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.reporting.report_content import build_plotly_figures


def _capture(model, name="vae"):
    figures = {}

    def render(fig, **kwargs):
        figures[kwargs["div_id"]] = fig
        return "rendered"

    with patch("plotly.io.to_html", side_effect=render):
        records = build_plotly_figures({"models": {name: model}})
    return figures, records


class ScienceContractTests(unittest.TestCase):
    def test_long_tail_has_resolved_center_and_conserves_full_count(self):
        scores = list(range(0, 3000, 3)) + [150000]
        figures, records = _capture({"oot_scores": scores, "threshold": 3000})
        fig = figures["fig-scores-vae"]
        detail, full = fig.data
        self.assertEqual(sum(detail.y), 951)
        self.assertGreater(np.count_nonzero(detail.y), 50)
        self.assertLess(max(detail.x), 3000)
        self.assertEqual(sum(full.y), 1001)
        self.assertEqual(full.y[-1], 1)
        self.assertAlmostEqual(full.customdata[-1][1], 150000)
        self.assertIn("50 filas fuera del detalle", records[0]["note"])
        self.assertIn("1001 filas finitas de 1001", records[0]["note"])
        self.assertIn("log1p", fig.layout.xaxis2.title.text)
        self.assertEqual(fig.layout.yaxis2.title.text, "filas (conteo)")
        self.assertEqual(fig.layout.title.text, "VAE: distribución de puntajes")
        self.assertEqual(fig.layout.height, 700)
        self.assertEqual(list(fig.layout.xaxis.domain), [0, 1])
        self.assertEqual(list(fig.layout.xaxis2.domain), [0, 1])
        self.assertLess(fig.layout.yaxis2.domain[1], fig.layout.yaxis.domain[0])

    def test_p95_ties_are_included(self):
        figures, records = _capture({"oot_scores": [0] * 99 + [150000]})
        self.assertEqual(sum(figures["fig-scores-vae"].data[0].y), 99)
        self.assertEqual(sum(figures["fig-scores-vae"].data[1].y), 100)
        self.assertIn("1 filas fuera del detalle", records[0]["note"])

    def test_threshold_counts_ties_in_original_space_even_outside_detail(self):
        figures, _ = _capture({"oot_scores": [0] * 98 + [150000] * 2,
                               "threshold": 150000})
        fig = figures["fig-scores-vae"]
        self.assertEqual(len(fig.layout.shapes), 1)
        self.assertEqual(fig.layout.shapes[0].xref, "x2")
        self.assertEqual(fig.layout.shapes[0].yref, "y2 domain")
        self.assertAlmostEqual(fig.layout.shapes[0].x0, np.log1p(150000))
        annotations = " ".join(a.text for a in fig.layout.annotations)
        self.assertIn("2 de 100 filas", annotations)
        self.assertIn("fuera del detalle", annotations)


class IntegrationContractTests(unittest.TestCase):
    def test_label_groups_share_edges_and_remain_aligned_after_filtering(self):
        figures, records = _capture({"oot_scores": [0, np.nan, 10, 150000],
                                    "oot_labels": [0, 1, 1, 1], "threshold": 10})
        fig = figures["fig-scores-vae"]
        self.assertEqual(len(fig.data), 4)
        self.assertEqual(list(fig.data[0].x), list(fig.data[1].x))
        self.assertEqual([sum(t.y) for t in fig.data], [1, 1, 1, 2])
        self.assertEqual(records[0]["series"], ["__neutral__", "vae"] * 2)
        self.assertIn("2 de 3 filas", " ".join(a.text for a in fig.layout.annotations))

    def test_if_histogram_and_threshold_are_unchanged(self):
        figures, records = _capture({"oot_scores": [0.1, 0.5, 0.5, 0.9],
                                    "threshold": 0.5}, name="iforest")
        fig = figures["fig-scores-iforest"]
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "histogram")
        self.assertEqual(fig.data[0].nbinsx, 60)
        self.assertEqual(list(fig.data[0].x), [0.1, 0.5, 0.5, 0.9])
        self.assertEqual(fig.layout.shapes[0].x0, 0.5)
        self.assertIn("3 de 4 filas", fig.layout.annotations[0].text)
        self.assertEqual(records[0]["series"], ["iforest"])

    def test_if_supervised_density_is_unchanged(self):
        figures, _ = _capture({"oot_scores": [0.1, 0.2, 0.8, 0.9],
                               "oot_labels": [0, 0, 1, 1]}, name="iforest")
        fig = figures["fig-scores-iforest"]
        self.assertEqual([t.histnorm for t in fig.data], ["probability density"] * 2)
        self.assertEqual(list(fig.data[1].x), [0.8, 0.9])


class FailureContractTests(unittest.TestCase):
    def test_empty_and_all_nonfinite_inputs_do_not_emit_a_distribution(self):
        for scores in ([], [np.nan, np.inf, -np.inf], None):
            with self.subTest(scores=scores):
                figures, records = _capture({"oot_scores": scores})
                self.assertNotIn("fig-scores-vae", figures)
                self.assertEqual(records, [])

    def test_nonfinite_scores_are_counted_as_excluded(self):
        figures, records = _capture({"oot_scores": [0, 1, np.nan, np.inf, -np.inf]})
        self.assertEqual(sum(figures["fig-scores-vae"].data[1].y), 2)
        self.assertIn("no finitos excluidos: 3", records[0]["note"])
        self.assertIn("2 filas finitas de 5", records[0]["note"])

    def test_zero_and_constant_scores_have_finite_nonzero_width_bins(self):
        for value in (0, 7):
            with self.subTest(value=value):
                figures, _ = _capture({"oot_scores": [value] * 20})
                fig = figures["fig-scores-vae"]
                self.assertEqual([sum(t.y) for t in fig.data], [20, 20])
                self.assertTrue(np.all(np.isfinite(fig.data[1].x)))
                self.assertTrue(np.all(np.asarray(fig.data[0].width) > 0))

    def test_negative_scores_use_explicit_signed_transform_without_loss(self):
        figures, _ = _capture({"oot_scores": [-150000, -1, 0, 10]})
        fig = figures["fig-scores-vae"]
        self.assertIn("signo(puntaje)", fig.layout.xaxis2.title.text)
        self.assertEqual(sum(fig.data[1].y), 4)
        self.assertAlmostEqual(fig.data[1].customdata[0][0], -150000)


class ReleaseFlowTests(unittest.TestCase):
    def test_public_html_has_both_views_without_a_hidden_modebar_dependency(self):
        records = build_plotly_figures({"models": {"vae": {
            "oot_scores": [0] * 99 + [150000], "threshold": 10000}}})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "fig-scores-vae")
        self.assertEqual(records[0]["group"], "resultados")
        self.assertIn("Cola completa", records[0]["html"])
        self.assertIn("Detalle", records[0]["html"])
        self.assertIn('"xaxis2"', records[0]["html"])
        self.assertIn("umbral de alerta", records[0]["html"])

    def test_replay_with_numpy_input_is_deterministic_and_does_not_mutate_scores(self):
        scores = np.array([0., 1., 2., 150000.])
        first, _ = _capture({"oot_scores": scores})
        second, _ = _capture({"oot_scores": scores})
        self.assertEqual(first["fig-scores-vae"].to_json(), second["fig-scores-vae"].to_json())
        np.testing.assert_array_equal(scores, [0., 1., 2., 150000.])
