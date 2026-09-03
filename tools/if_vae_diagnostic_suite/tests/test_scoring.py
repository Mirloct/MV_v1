import unittest

import numpy as np

from ifvae_diag.scoring import (
    anomaly_percentile,
    latent_diagnostics,
    reconstruction_scores,
)


class ScoringTests(unittest.TestCase):
    def test_percentile_is_fit_only_on_reference_scores(self):
        reference = np.array([0.0, 1.0, 2.0, 3.0])
        scored = np.array([-1.0, 1.5, 100.0])
        got = anomaly_percentile(reference, scored)
        np.testing.assert_allclose(got, [0.0, 0.5, 1.0])

    def test_percentile_tie_counts_reference_values_at_or_below_score(self):
        got = anomaly_percentile(np.array([1.0, 2.0, 2.0, 3.0]), np.array([2.0]))
        np.testing.assert_allclose(got, [0.75])

    def test_top_k_prevents_many_normal_features_from_diluting_one_spike(self):
        reference_residuals = np.ones((20, 10)) * 0.1
        scored_residuals = np.ones((2, 10)) * 0.1
        scored_residuals[1, 0] = 10.0
        scores = reconstruction_scores(reference_residuals, scored_residuals, top_k=1)
        self.assertGreater(scores.loc[1, "recon_topk"], scores.loc[0, "recon_topk"] * 50)

    def test_latent_collapse_detects_inactive_units(self):
        mu = np.column_stack([np.linspace(-1, 1, 30), np.zeros(30), np.zeros(30)])
        logvar = np.zeros_like(mu)
        result = latent_diagnostics(mu, logvar, active_variance_threshold=1e-3)
        self.assertEqual(result["active_units"], 1)
        self.assertAlmostEqual(result["collapsed_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
