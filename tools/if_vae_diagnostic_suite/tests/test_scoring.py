import unittest

import numpy as np

from ifvae_diag.scoring import (
    MIN_SCALE_WHEN_FLOORED,
    anomaly_percentile,
    latent_diagnostics,
    reconstruction_scores,
    residual_contributions,
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

    def test_scale_floor_stops_a_near_perfect_variable_from_dominating_and_is_off_by_default(self):
        rng = np.random.default_rng(0)
        reference = np.column_stack([np.abs(rng.normal(1.0, 0.5, 500)),          # ordinary variable
                                     np.abs(rng.normal(0.0, 1e-9, 500))])        # reconstructed almost perfectly
        scored = np.array([[1.0, 0.0], [1.0, 0.05]])                             # row 1: small blip on variable 2
        raw = residual_contributions(reference, scored)                          # historical behaviour
        floored = residual_contributions(reference, scored, scale_floor_fraction=0.1)
        self.assertGreater(raw[1, 1], 1e6)                                       # exploded by a ~1e-9 scale
        self.assertLess(floored[1, 1], raw[1, 1] * 1e-3)
        self.assertTrue(np.isfinite(floored).all())
        np.testing.assert_allclose(floored[:, 0], raw[:, 0], rtol=0.05)          # ordinary variable ~unchanged
        self.assertEqual(MIN_SCALE_WHEN_FLOORED, 1e-6)                           # documented absolute floor
        scores = reconstruction_scores(reference, scored, top_k=1, scale_floor_fraction=0.1)
        self.assertEqual(len(scores), 2)

    def test_centering_makes_a_constant_contribution_zero_instead_of_permanently_high(self):
        rng = np.random.default_rng(1)
        constant = np.full(300, np.log(40.0))                       # NLL of a uniform 40-level variable
        ordinary = np.abs(rng.normal(1.0, 0.3, 300))
        reference = np.column_stack([constant, ordinary])
        scored = np.array([[np.log(40.0), 1.0], [np.log(40.0), 2.5]])
        plain = residual_contributions(reference, scored, scale_floor_fraction=0.1)
        centred = residual_contributions(reference, scored, scale_floor_fraction=0.1, center=True)
        np.testing.assert_allclose(plain[:, 0], 1.0)                # uncentred: a constant variable scores 1 on EVERY row
        np.testing.assert_allclose(centred[:, 0], 0.0, atol=1e-9)   # ... centred it is 0 (nothing unusual)
        self.assertGreater(centred[1, 1], centred[0, 1])            # a genuine excess still stands out
        self.assertEqual(len(reconstruction_scores(reference, scored, 1, 0.1, True)), 2)

    def test_latent_collapse_detects_inactive_units(self):
        mu = np.column_stack([np.linspace(-1, 1, 30), np.zeros(30), np.zeros(30)])
        logvar = np.zeros_like(mu)
        result = latent_diagnostics(mu, logvar, active_variance_threshold=1e-3)
        self.assertEqual(result["active_units"], 1)
        self.assertAlmostEqual(result["collapsed_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
