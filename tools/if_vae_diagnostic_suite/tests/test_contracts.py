import unittest

import numpy as np
import pandas as pd

from ifvae_diag.contracts import DataContractError, validate_frames


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.reference = pd.DataFrame({"id": ["r1", "r2"], "x": [0.0, 1.0]})
        self.scored = pd.DataFrame(
            {"id": ["s1", "s2"], "x": [2.0, 3.0], "label": [0, 1]}
        )

    def test_rejects_duplicate_ids(self):
        bad = self.scored.assign(id=["same", "same"])
        with self.assertRaisesRegex(DataContractError, "unique"):
            validate_frames(self.reference, bad, ["x"], "id", "label", None)

    def test_rejects_non_binary_labels(self):
        bad = self.scored.assign(label=[0, 2])
        with self.assertRaisesRegex(DataContractError, "binary"):
            validate_frames(self.reference, bad, ["x"], "id", "label", None)

    def test_rejects_infinite_feature_values(self):
        bad = self.scored.assign(x=[2.0, np.inf])
        with self.assertRaisesRegex(DataContractError, "finite"):
            validate_frames(self.reference, bad, ["x"], "id", "label", None)

    def test_temporal_overlap_is_reported_not_silently_accepted(self):
        reference = self.reference.assign(event_time=["2026-02-01", "2026-02-03"])
        scored = self.scored.assign(event_time=["2026-02-02", "2026-02-04"])
        result = validate_frames(
            reference, scored, ["x"], "id", "label", "event_time"
        )
        self.assertIn("temporal_overlap", {x["code"] for x in result})

    def test_rejects_all_missing_reference_feature(self):
        reference = self.reference.assign(x=[np.nan, np.nan])
        with self.assertRaisesRegex(DataContractError, "entirely missing"):
            validate_frames(reference, self.scored, ["x"], "id", "label", None)


if __name__ == "__main__":
    unittest.main()
