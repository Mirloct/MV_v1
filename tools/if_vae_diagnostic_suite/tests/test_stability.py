import unittest

import numpy as np

from ifvae_diag.stability import top_k_stability


class StabilityTests(unittest.TestCase):
    def test_reports_selection_probability_and_pairwise_overlap(self):
        scores = np.array(
            [[0.9, 0.8, 0.1, 0.0], [0.8, 0.9, 0.2, 0.0], [0.9, 0.7, 0.8, 0.0]]
        )
        result = top_k_stability(scores, k=2)
        np.testing.assert_allclose(result["selection_probability"][:2], [1.0, 2 / 3])
        self.assertGreater(result["mean_jaccard"], 0.3)
        self.assertLessEqual(result["mean_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()

