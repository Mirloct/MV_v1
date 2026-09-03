import unittest

import numpy as np
import pandas as pd

from ifvae_diag.diagnostics import (
    assign_disagreement_quadrant,
    build_autopsy,
    detect_score_orientation_risk,
)


class DiagnosticTests(unittest.TestCase):
    def test_assigns_if_only_quadrant(self):
        got = assign_disagreement_quadrant(
            np.array([0.99]), np.array([0.20]), threshold=0.95
        )
        self.assertEqual(got[0], "IF_ONLY")

    def test_threshold_is_inclusive(self):
        got = assign_disagreement_quadrant(
            np.array([0.95]), np.array([0.95]), threshold=0.95
        )
        self.assertEqual(got[0], "BOTH")

    def test_autopsy_ranks_standardized_feature_contributions(self):
        features = pd.DataFrame({"a": [10.0], "b": [2.0]})
        reconstruction = pd.DataFrame({"a": [0.0], "b": [0.0]})
        contributions = pd.DataFrame({"a": [100.0], "b": [1.0]})
        got = build_autopsy(
            pd.Series([True]), features, reconstruction, contributions, ["row-1"]
        )
        self.assertEqual(got.iloc[0]["feature"], "a")
        self.assertEqual(got.iloc[0]["rank"], 1)

    def test_orientation_risk_fires_when_positives_are_systematically_low(self):
        y = np.array([1, 1, 0, 0])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        warning = detect_score_orientation_risk(y, score)
        self.assertTrue(warning["risk"])
        self.assertLess(warning["positive_median"], warning["negative_median"])


if __name__ == "__main__":
    unittest.main()
