"""Behavior: `run_diagnostic` must support `label_col=None` (no ground truth).

Gap found integrating the suite against Modelo v0.1: the project's official,
real-data runs are unsupervised and carry no target column at all, but the
data contract hard-requires a binary `label_col` in `scored.csv`
(`contracts.py::_require_columns`), and every supervised computation
(`metrics.py`, `autopsies`, `coverage`, orientation-risk warnings) reads
`config.label_col` unconditionally. A user with no labels cannot produce a
valid `scored.csv` for this tool at all today.

This is not tautological: it asserts on concrete output content (which
columns exist, which are empty, which summary fields are None) that only
passes once the label-free path is implemented, and a second test guards
that supplying a real label column still enforces the existing binary-label
validation (the change must not weaken supervised-mode behavior).
"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ifvae_diag.contracts import DataContractError
from ifvae_diag.pipeline import run_diagnostic
from ifvae_diag.simulation import generate_synthetic_case


class UnsupervisedPipelineTests(unittest.TestCase):
    def test_run_diagnostic_without_labels_still_scores_and_skips_supervised_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, scored, config = generate_synthetic_case(seed=7)
            scored_unlabeled = scored.drop(columns=["label"])
            config = replace(config, label_col=None, family_col=None, segment_col=None)
            result = run_diagnostic(reference, scored_unlabeled, config, Path(tmp))

            # Label-free diagnostics still run: percentiles, quadrants,
            # latent/drift, the report, and the disagreement plot.
            self.assertIn("quadrant", result.scored.columns)
            self.assertIn("if_percentile", result.scored.columns)
            self.assertIn("vae_percentile", result.scored.columns)
            self.assertFalse(result.drift.empty)
            self.assertTrue((Path(tmp) / "report.md").is_file())
            self.assertTrue((Path(tmp) / "disagreement.png").is_file())

            # Supervised-only outputs degrade to empty/None, not a crash.
            self.assertTrue(result.metrics.empty)
            self.assertTrue(result.autopsies.empty)
            self.assertIsNone(result.summary["known_positives"])
            self.assertIsNone(result.summary["positive_quadrants"])
            codes = {warning["code"] for warning in result.warnings}
            self.assertNotIn("if_percentile_orientation", codes)
            self.assertNotIn("vae_percentile_orientation", codes)

    def test_run_diagnostic_still_enforces_binary_labels_when_a_label_col_is_given(self):
        # Regression guard: adding the label-free path must not weaken
        # validation for the existing, still-default supervised path.
        with tempfile.TemporaryDirectory() as tmp:
            reference, scored, config = generate_synthetic_case(seed=7)
            scored.loc[scored.index[0], "label"] = 2
            with self.assertRaises(DataContractError):
                run_diagnostic(reference, scored, config, Path(tmp))


if __name__ == "__main__":
    unittest.main()
