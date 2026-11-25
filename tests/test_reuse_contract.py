# Created: 2026-11-25
# Last modified: 2026-11-25
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.reuse_contract import (
    iforest_parameters, input_fingerprint, save_contract, save_if_training,
    validate_contract, validate_if_reusing,
)


class ReuseContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "reuse.json"
        self.models = {"iforest": self.root / "iforest.pkl", "vae": self.root / "vae.pt"}
        for name, path in self.models.items():
            path.write_bytes(f"opaque {name} model bytes".encode())
        self.X = np.array([[1., 2.], [3., 8.], [9., 4.]])
        self.names = ["balance", "income"]
        self.fingerprint = input_fingerprint(self.X, self.names)

    def save(self):
        save_contract(self.manifest, self.fingerprint, self.models)

    def test_feature_reorder_same_width_is_rejected(self):
        self.save()
        changed = input_fingerprint(self.X, self.names[::-1])
        with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
            validate_contract(self.manifest, changed, self.models)

    def test_loaded_forest_parameters_preserve_stacking_despite_stale_yaml(self):
        import yaml
        from src.models.iforest import IsolationForestDetector
        from src.models.stacking import build_stacked_matrix

        expected = {
            "n_estimators": 17, "max_samples": 32, "max_features": 0.75,
            "contamination": 0.1, "bootstrap": True,
        }
        X = np.random.default_rng(42).normal(size=(120, 4))
        train = np.arange(len(X)) < 80
        detector = IsolationForestDetector(random_state=42, n_jobs=1, **expected)
        detector.fit(X)
        detector.save(str(self.models["iforest"]))
        stale_path = self.root / "best_params.yaml"
        stale_path.write_text(
            "best_params:\n  n_estimators: 3\n  max_samples: 16\n"
            "  max_features: 1.0\n  contamination: 0.005\n  bootstrap: false\n",
            encoding="utf-8",
        )
        loaded = IsolationForestDetector.load(str(self.models["iforest"]))
        self.assertEqual(iforest_parameters(loaded), expected)
        np.testing.assert_array_equal(detector.score_samples(X), loaded.score_samples(X))

        baseline = IsolationForestDetector(random_state=42, n_jobs=1, **expected).fit(X[train])
        reused = IsolationForestDetector(
            random_state=42, n_jobs=1, **iforest_parameters(loaded),
        ).fit(X[train])
        baseline_scores = baseline.score_samples(X)
        reused_scores = reused.score_samples(X)
        np.testing.assert_array_equal(baseline_scores, reused_scores)
        original_matrix = build_stacked_matrix(X, baseline_scores, fit_mask=train)
        reused_matrix = build_stacked_matrix(X, reused_scores, fit_mask=train)
        np.testing.assert_array_equal(original_matrix.X, reused_matrix.X)

        stale_params = yaml.safe_load(stale_path.read_text(encoding="utf-8"))["best_params"]
        wrong = IsolationForestDetector(random_state=42, n_jobs=1, **stale_params).fit(X[train])
        self.assertFalse(np.array_equal(baseline_scores, wrong.score_samples(X)))
        self.assertEqual(iforest_parameters(loaded), expected)

    def test_scaler_changed_matrix_is_rejected(self):
        scaled = StandardScaler().fit_transform(self.X)
        original = input_fingerprint(scaled, self.names)
        save_contract(self.manifest, original, self.models)
        different = StandardScaler(with_mean=False).fit_transform(self.X)
        with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
            validate_contract(self.manifest, input_fingerprint(different, self.names), self.models)

    def test_row_order_and_single_value_changes_are_rejected(self):
        self.save()
        altered = self.X.copy()
        altered[0, 0] += 0.001
        for matrix in (self.X[::-1], altered):
            with self.subTest(matrix=matrix):
                with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
                    validate_contract(self.manifest, input_fingerprint(matrix, self.names), self.models)

    def test_feature_width_mismatch_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "width.*Retrain"):
            input_fingerprint(self.X, ["balance"])

    def test_fitted_preprocessor_drift_with_identical_training_matrix(self):
        original = MinMaxScaler(clip=False).fit(self.X)
        changed = MinMaxScaler(clip=True).fit(self.X)
        np.testing.assert_array_equal(original.transform(self.X), changed.transform(self.X))
        future = np.array([[100., 100.]])
        self.assertFalse(np.array_equal(original.transform(future), changed.transform(future)))
        fingerprint = input_fingerprint(original.transform(self.X), self.names, original)
        save_contract(self.manifest, fingerprint, self.models)
        with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
            validate_contract(
                self.manifest, input_fingerprint(changed.transform(self.X), self.names, changed), self.models,
            )
        equivalent = MinMaxScaler(clip=False).fit(self.X.copy())
        validate_contract(
            self.manifest, input_fingerprint(equivalent.transform(self.X), self.names, equivalent), self.models,
        )

    def test_roundtrip_json_contains_independent_model_sha256(self):
        with patch("joblib.dump", side_effect=AssertionError("must not save pickles")):
            self.save()
            validate_contract(self.manifest, input_fingerprint(self.X.copy(), tuple(self.names)), self.models)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["fingerprint"], self.fingerprint)
        self.assertEqual(manifest["models"], {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in self.models.items()
        })
        self.assertEqual({path.name for path in self.root.iterdir()}, {"reuse.json", "iforest.pkl", "vae.pt"})

    def test_model_tamper_is_rejected_for_each_detector(self):
        for name, path in self.models.items():
            with self.subTest(model=name):
                self.save()
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA-256.*Retrain"):
                    validate_contract(self.manifest, self.fingerprint, self.models)
                path.write_bytes(original)

    def test_model_inventory_change_and_missing_model_are_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "model names changed.*Retrain"):
            validate_contract(self.manifest, self.fingerprint, {"iforest": self.models["iforest"]})
        self.models["vae"].unlink()
        with self.assertRaisesRegex(ValueError, "vae.*cannot be read.*Retrain"):
            validate_contract(self.manifest, self.fingerprint, self.models)

    def test_missing_and_corrupt_manifests_fail_clear(self):
        with self.assertRaisesRegex(ValueError, "missing or unreadable.*Retrain"):
            validate_contract(self.manifest, self.fingerprint, self.models)
        for content in ('{broken', '[]', '{"version":99}', '{}', '{"version":1}'):
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Reuse.*Retrain"):
                    validate_contract(self.manifest, self.fingerprint, self.models)

    def test_atomic_replace_failure_keeps_previous_manifest_and_cleans_temp(self):
        self.save()
        previous = self.manifest.read_bytes()
        with patch("src.utils.reuse_contract.atomic_replace", side_effect=PermissionError("locked")):
            with self.assertRaisesRegex(ValueError, "could not be saved.*Retrain"):
                save_contract(self.manifest, "new-fingerprint", self.models)
        self.assertEqual(self.manifest.read_bytes(), previous)
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        validate_contract(self.manifest, self.fingerprint, self.models)

    def test_missing_model_save_leaves_previous_contract_unchanged(self):
        self.save()
        previous = self.manifest.read_bytes()
        self.models["vae"].unlink()
        with self.assertRaisesRegex(ValueError, "vae.*Retrain"):
            self.save()
        self.assertEqual(self.manifest.read_bytes(), previous)

    def test_nested_destination_and_contract_replacement(self):
        destination = self.root / "nested" / "reuse.json"
        save_contract(destination, self.fingerprint, self.models)
        self.models["vae"].write_bytes(b"newly trained VAE")
        save_contract(destination, "new-training-input", self.models)
        validate_contract(destination, "new-training-input", self.models)
        with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
            validate_contract(destination, self.fingerprint, self.models)

    def test_wrapper_training_and_reuse_flow(self):
        validate_if_reusing(False, self.manifest, self.fingerprint, self.models)
        self.assertFalse(self.manifest.exists())
        save_if_training(False, self.manifest, self.fingerprint, self.models)
        previous = self.manifest.read_bytes()
        validate_if_reusing(True, self.manifest, self.fingerprint, self.models)
        save_if_training(True, self.manifest, "changed", self.models)
        self.assertEqual(self.manifest.read_bytes(), previous)
        with self.assertRaisesRegex(ValueError, "fingerprint changed.*Retrain"):
            validate_if_reusing(True, self.manifest, "changed", self.models)

    def test_wrapper_skip_branches_do_not_access_models_or_manifest(self):
        save_if_training(True, self.manifest, None, {})
        validate_if_reusing(False, self.manifest, None, {})
        self.assertFalse(self.manifest.exists())
        with self.assertRaisesRegex(ValueError, "missing or unreadable.*Retrain"):
            validate_if_reusing(True, self.manifest, self.fingerprint, self.models)

    def test_empty_fingerprint_and_model_inventory_are_rejected(self):
        for fingerprint in ("", None, "   "):
            with self.subTest(fingerprint=fingerprint):
                with self.assertRaisesRegex(ValueError, "fingerprint.*Retrain"):
                    save_contract(self.manifest, fingerprint, self.models)
        with self.assertRaisesRegex(ValueError, "non-empty mapping.*Retrain"):
            save_contract(self.manifest, self.fingerprint, {})
        self.assertFalse(self.manifest.exists())


if __name__ == "__main__":
    unittest.main()
