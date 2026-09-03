import unittest

import numpy as np
import pandas as pd

from ifvae_diag.metrics import evaluate_family_scores, evaluate_scores, positive_coverage


class MetricTests(unittest.TestCase):
    def test_metrics_at_operational_budget(self):
        frame = pd.DataFrame(
            {"label": [1, 0, 1, 0, 0], "score": [0.9, 0.8, 0.7, 0.2, 0.1]}
        )
        result = evaluate_scores(frame, "label", ["score"], [2])
        row = result.query("score_name == 'score' and k == 2").iloc[0]
        self.assertEqual(row["true_positives"], 1)
        self.assertAlmostEqual(row["precision_at_k"], 0.5)
        self.assertAlmostEqual(row["recall_at_k"], 0.5)
        self.assertAlmostEqual(row["lift_at_k"], 1.25)

    def test_positive_coverage_separates_shared_and_unique_hits(self):
        y = np.array([1, 1, 1, 0, 0])
        a = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
        b = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
        got = positive_coverage(y, a, b, k=2)
        self.assertEqual(got["shared_positive_hits"], 1)
        self.assertEqual(got["a_only_positive_hits"], 1)
        self.assertEqual(got["b_only_positive_hits"], 1)

    def test_ties_use_stable_id_not_row_order(self):
        a = pd.DataFrame({"id": ["b", "a"], "label": [0, 1], "score": [1.0, 1.0]})
        b = a.iloc[::-1].reset_index(drop=True)
        ra = evaluate_scores(a, "label", ["score"], [1], id_col="id")
        rb = evaluate_scores(b, "label", ["score"], [1], id_col="id")
        self.assertEqual(ra.iloc[0]["true_positives"], rb.iloc[0]["true_positives"])
        self.assertEqual(ra.iloc[0]["true_positives"], 1)

    def test_family_metrics_compare_family_positives_to_all_negatives(self):
        frame = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d", "e"],
                "label": [1, 1, 1, 0, 0],
                "family": ["A", "A", "B", "normal", "normal"],
                "score": [0.9, 0.8, 0.7, 0.6, 0.1],
            }
        )
        got = evaluate_family_scores(
            frame, "label", "family", ["score"], [2], id_col="id"
        )
        family_a = got.query("group_value == 'A'").iloc[0]
        self.assertEqual(family_a["cohort_rows"], 4)
        self.assertEqual(family_a["cohort_positives"], 2)
        self.assertEqual(family_a["true_positives"], 2)


if __name__ == "__main__":
    unittest.main()
