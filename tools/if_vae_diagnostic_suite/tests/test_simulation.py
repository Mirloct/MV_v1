import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ifvae_diag.pipeline import run_diagnostic
from ifvae_diag.simulation import generate_synthetic_case


class SimulationIntegrationTests(unittest.TestCase):
    def test_end_to_end_surfaces_detector_disagreement_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, scored, config = generate_synthetic_case(seed=7)
            result = run_diagnostic(reference, scored, config, root)
            quadrants = set(result.scored["quadrant"])
            self.assertIn("IF_ONLY", quadrants)
            self.assertIn("VAE_ONLY", quadrants)
            self.assertTrue((root / "report.md").is_file())
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "warnings.json").is_file())
            self.assertGreater(len(result.autopsies), 0)
            reloaded = pd.read_csv(root / "scored_diagnostics.csv")
            self.assertEqual(len(reloaded), len(scored))

    def test_synthetic_families_land_in_their_intended_disagreement_quadrants(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, scored, config = generate_synthetic_case(seed=11)
            result = run_diagnostic(reference, scored, config, Path(tmp))
            positive = result.scored.query("label == 1")
            expected = {
                "point_extrapolation": "IF_ONLY",
                "relationship_break": "VAE_ONLY",
                "shared_extreme": "BOTH",
            }
            for family, quadrant in expected.items():
                family_rows = positive.query("anomaly_family == @family")
                share = (family_rows["quadrant"] == quadrant).mean()
                self.assertGreaterEqual(share, 0.8, msg=f"{family} -> {quadrant}")

    def test_synthetic_normal_population_does_not_hide_a_generator_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, scored, config = generate_synthetic_case(seed=13)
            result = run_diagnostic(reference, scored, config, Path(tmp))
            self.assertLess(result.drift["ks_statistic"].max(), 0.25)

    def test_precomputed_if_score_path_marks_refit_stability_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, scored, config = generate_synthetic_case(seed=17)
            reference["production_if_normality"] = -reference["feature_0"].abs()
            scored["production_if_normality"] = -scored["feature_0"].abs()
            config = replace(
                config,
                if_score_col="production_if_normality",
                if_higher_is_anomalous=False,
            )
            result = run_diagnostic(reference, scored, config, Path(tmp))
            codes = {warning["code"] for warning in result.warnings}
            self.assertIn("if_stability_unavailable", codes)
            point_rows = result.scored.query("anomaly_family == 'point_extrapolation'")
            self.assertGreaterEqual(point_rows["if_percentile"].min(), 0.95)


if __name__ == "__main__":
    unittest.main()
