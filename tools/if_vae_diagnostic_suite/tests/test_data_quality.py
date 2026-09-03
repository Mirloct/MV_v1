import unittest

import pandas as pd

from ifvae_diag.data_quality import compare_populations, leakage_name_warnings


class DataQualityTests(unittest.TestCase):
    def test_detects_large_distribution_shift(self):
        reference = pd.DataFrame({"x": list(range(100))})
        scored = pd.DataFrame({"x": list(range(1000, 1100))})
        got = compare_populations(reference, scored, ["x"])
        self.assertGreater(got.loc[0, "ks_statistic"], 0.9)
        self.assertGreater(got.loc[0, "out_of_reference_range_rate"], 0.9)

    def test_flags_post_outcome_feature_names(self):
        warnings = leakage_name_warnings(
            ["amount_30d", "confirmed_fraud_result", "chargeback_after_90d"]
        )
        flagged = {x["feature"] for x in warnings}
        self.assertIn("confirmed_fraud_result", flagged)
        self.assertIn("chargeback_after_90d", flagged)
        self.assertNotIn("amount_30d", flagged)


if __name__ == "__main__":
    unittest.main()

