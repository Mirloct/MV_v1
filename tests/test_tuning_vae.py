"""VAE tuning reliability: persisted schedule, safe resume, identity, and objective semantics.

Data sources / inputs: seeded in-memory matrices and temporary PyTorch/Optuna
checkpoints, SQLite studies, and YAML artifacts.
Created: 2026-09-25
Last modified: 2026-09-26
Changelog:
- 2026-09-25: Added regressions for full-data fingerprints, partial labels,
  objective direction, no-complete-trial failure, and early-stop resume state.
- 2026-09-26: Locked the non-tunable training schedule and added assertions
  for plateau outcomes and Optuna parameter exclusions.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
import yaml

from src.models.vae import VAEDetector, _data_fingerprint, _supervised_score, tune_vae

TINY = dict(latent_dim=3, hidden_dim=8, n_layers=1, batch_size=64, early_stopping_patience=None)


def _data(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    valid = np.zeros(n, dtype=bool)
    valid[-60:] = True
    return X, valid


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name


class TestPersistedSchedule(Base):
    def test_schedule_is_fixed_and_load_preserves_training_outcome(self):
        X, valid = _data()
        det = VAEDetector(epochs=99, kl_anneal_epochs=99, random_state=1,
                          **{**TINY, "early_stopping_patience": 99})
        det.fit(X, checkpoint_dir=os.path.join(self.d, "a"), valid_mask=valid)
        path = det.save(os.path.join(self.d, "vae.pt"))
        loaded = VAEDetector.load(path)
        self.assertEqual((loaded.epochs, loaded.kl_anneal_epochs,
                          loaded.early_stopping_patience), (15, 3, 3))
        self.assertAlmostEqual(loaded.early_stopping_min_delta_rel, 0.005)
        self.assertEqual(loaded.best_epoch_, det.best_epoch_)
        self.assertEqual(loaded.epochs_trained_, det.epochs_trained_)
        self.assertEqual(loaded.stopped_early_, det.stopped_early_)
        np.testing.assert_allclose(loaded.score_samples(X), det.score_samples(X), rtol=1e-5)

    def test_old_payload_without_schedule_falls_back_to_history_length(self):
        X, valid = _data()
        det = VAEDetector(epochs=2, random_state=1, **TINY)
        det.fit(X, checkpoint_dir=os.path.join(self.d, "a"), valid_mask=valid)
        path = det.save(os.path.join(self.d, "old.pt"))
        payload = torch.load(path, weights_only=False)
        for key in ("epochs", "kl_anneal_epochs", "early_stopping_patience", "data_fingerprint"):
            payload["config"].pop(key, None)
        torch.save(payload, path)
        self.assertEqual(VAEDetector.load(path).epochs, 2)

    def test_fit_refuses_zero_epochs(self):
        X, _ = _data()
        with self.assertRaises(ValueError):
            VAEDetector(epochs=0, **{k: v for k, v in TINY.items()}).fit(
                X, checkpoint_dir=os.path.join(self.d, "z"))


class TestCheckpointResume(Base):
    def _fit(self, ck, X, valid, resume=True, **kw):
        det = VAEDetector(epochs=2, random_state=1, **{**TINY, **kw})
        return det.fit(X, checkpoint_dir=ck, resume=resume, valid_mask=valid)

    def test_same_run_resumes_but_other_config_or_data_starts_fresh(self):
        X, valid = _data()
        ck = os.path.join(self.d, "ck")
        first = self._fit(ck, X, valid)
        ckpt = torch.load(os.path.join(ck, "checkpoint.pth"), weights_only=False)

        def probe(fingerprint=first.data_fingerprint_, **kw):
            det = VAEDetector(**{"epochs": 2, "random_state": 1, **TINY, **kw})
            det.input_dim_, det.data_fingerprint_ = first.input_dim_, fingerprint
            return det

        same = probe()
        self.assertTrue(same._checkpoint_compatible(ckpt))

        self.assertFalse(probe(beta=0.3)._checkpoint_compatible(ckpt))
        self.assertTrue(probe(kl_anneal_epochs=1)._checkpoint_compatible(ckpt))
        self.assertFalse(probe(fingerprint="somethingelse")._checkpoint_compatible(ckpt))

    def test_a_refit_on_other_data_in_a_used_directory_actually_trains(self):
        X, valid = _data(seed=0)
        Y, valid_y = _data(seed=5, n=320)
        ck = os.path.join(self.d, "ck")
        a = self._fit(ck, X, valid)
        b = self._fit(ck, Y, valid_y)  # same architecture, different matrix, resume=True
        self.assertNotEqual(a.data_fingerprint_, b.data_fingerprint_)
        self.assertEqual(len(b.history_), 2)   # trained, did not return the old model
        self.assertNotEqual(a.history_[-1]["train_loss"], b.history_[-1]["train_loss"])

    def test_legacy_checkpoint_without_the_new_keys_starts_fresh(self):
        X, valid = _data()
        ck = os.path.join(self.d, "ck")
        det = self._fit(ck, X, valid)
        path = os.path.join(ck, "checkpoint.pth")
        ckpt = torch.load(path, weights_only=False)
        for key in ("beta", "kl_anneal_epochs", "data_fingerprint", "random_state"):
            ckpt["config"].pop(key, None)
        self.assertFalse(det._checkpoint_compatible(ckpt))

    def test_fingerprint_changes_when_an_unsampled_row_changes(self):
        X, valid = _data()
        Y = X.copy()
        Y[1, 0] += 1.0
        self.assertNotEqual(_data_fingerprint(X, valid, 0.1),
                            _data_fingerprint(Y, valid, 0.1))

    def test_checkpoint_persists_early_stopping_progress(self):
        X, valid = _data()
        det = VAEDetector(epochs=2, random_state=1, **TINY)
        det.fit(X, checkpoint_dir=os.path.join(self.d, "patience"), valid_mask=valid)
        ckpt = torch.load(os.path.join(self.d, "patience", "checkpoint.pth"), weights_only=False)
        self.assertIn("epochs_without_improvement", ckpt)


class TestTuneVaeReliability(Base):
    def _tune(self, X, valid, name="run", **kw):
        kw.setdefault("n_trials", 2)
        kw.setdefault("max_epochs", 4)
        kw.setdefault("early_stopping_patience", None)
        study = tune_vae(
            X, valid_mask=valid, random_state=0,
            storage="sqlite:///" + os.path.join(self.d, f"{name}.db").replace("\\", "/"),
            best_params_path=os.path.join(self.d, f"{name}.yaml"),
            model_out=os.path.join(self.d, f"{name}.pt"),
            checkpoint_dir=os.path.join(self.d, "ckpts"), **kw,
        )
        return study, yaml.safe_load(open(os.path.join(self.d, f"{name}.yaml"), encoding="utf-8"))

    def test_refit_uses_fixed_schedule_and_optuna_excludes_non_tunables(self):
        X, valid = _data()
        study, payload = self._tune(X, valid, name="fin")
        model = VAEDetector.load(os.path.join(self.d, "fin.pt"))
        self.assertEqual(payload["status"], "final")
        for trial in study.trials:
            self.assertTrue({"epochs", "random_state", "threshold", "kl_anneal_epochs"}
                            .isdisjoint(trial.params))
        self.assertEqual(payload["fixed_training"]["max_epochs"], 4)
        self.assertEqual(payload["fixed_training"]["kl_warmup_epochs"], 3)
        self.assertEqual(payload["fixed_training"]["early_stopping_patience"], 3)
        self.assertAlmostEqual(payload["fixed_training"]["min_delta_relative"], 0.005)
        self.assertEqual(model.epochs, 4)
        self.assertEqual(model.kl_anneal_epochs, 3)
        for key in ("best_epoch", "epochs_trained", "stopped_early"):
            self.assertIn(key, payload["training_outcome"])

    def test_plateau_stops_after_three_epochs_and_reports_best_epoch(self):
        from unittest.mock import patch

        X, valid = _data(n=120)
        det = VAEDetector(epochs=15, random_state=1, **TINY)
        with patch.object(VAEDetector, "_evaluate", return_value=(1.0, 1.0, 0.0)):
            det.fit(X, checkpoint_dir=os.path.join(self.d, "plateau"),
                    resume=False, valid_mask=valid)
        self.assertEqual(det.best_epoch_, 1)
        self.assertEqual(det.epochs_trained_, 4)
        self.assertTrue(det.stopped_early_)

    def test_n_trials_is_a_total_budget_and_resume_is_idempotent(self):
        X, valid = _data()
        counts = []  # `study.trials` reads the DB live, so count right after each call
        for n in (2, 2, 3):
            study, _ = self._tune(X, valid, name="res", n_trials=n)
            counts.append(len(study.trials))
        self.assertEqual(counts, [2, 2, 3])

    def test_different_data_split_or_features_get_their_own_study_and_checkpoints(self):
        X, valid = _data()
        s1, _ = self._tune(X, valid, name="ids", n_trials=1)
        Y, valid_y = _data(seed=3, n=320)
        s2, _ = self._tune(Y, valid_y, name="ids", n_trials=1)
        s3, _ = self._tune(X, valid, name="ids", n_trials=1, feature_names=list("abcdef"))
        self.assertEqual(len({s1.study_name, s2.study_name, s3.study_name}), 3)
        dirs = set(os.listdir(os.path.join(self.d, "ckpts")))
        self.assertTrue({s1.study_name, s2.study_name, s3.study_name} <= dirs)

    def test_partial_labels_ignore_unknown_rows(self):
        score = _supervised_score(
            np.array([1.0, 0.0, np.nan, 1.0]),
            np.array([0.9, 0.1, 0.8, 0.7]),
            "average_precision",
        )
        self.assertTrue(np.isfinite(score))

    def test_label_free_metric_keeps_minimize_direction_even_if_labels_exist(self):
        X, valid = _data(n=180)
        y = np.zeros(len(X)); y[-20:] = 1
        study, payload = self._tune(
            X, valid, name="recon-dir", y=y, objective_metric="recon_p50",
            n_trials=1, max_epochs=1,
        )
        self.assertEqual(study.direction.name, "MINIMIZE")
        self.assertEqual(payload["direction"], "minimize")

    def test_no_completed_trial_raises_instead_of_reusing_old_outputs(self):
        X, valid = _data(n=180)
        with self.assertRaises(RuntimeError):
            self._tune(X, valid, name="empty", n_trials=0)


class TestAntiCollapseGuard(Base):
    """A trial whose latent space collapsed must never win the study, even though ELBO,
    reconstruction loss and PR-AUC/ROC-AUC all stay finite and plausible on a decoder that
    has learned to ignore the latent code -- see `tune_vae`'s "Anti-collapse guard" docstring.
    """

    def _tune(self, X, valid, name="run", **kw):
        kw.setdefault("n_trials", 2)
        kw.setdefault("max_epochs", 2)
        kw.setdefault("early_stopping_patience", None)
        study = tune_vae(
            X, valid_mask=valid, random_state=0,
            storage="sqlite:///" + os.path.join(self.d, f"{name}.db").replace("\\", "/"),
            best_params_path=os.path.join(self.d, f"{name}.yaml"),
            model_out=os.path.join(self.d, f"{name}.pt"),
            checkpoint_dir=os.path.join(self.d, "ckpts"), **kw,
        )
        return study, yaml.safe_load(open(os.path.join(self.d, f"{name}.yaml"), encoding="utf-8"))

    _COLLAPSED = {"latent_dim": 8, "active_units": 0, "inactive_units": 8,
                 "active_fraction": 0.0, "delta": 0.01, "mean_kl": 0.0002}
    _HEALTHY = {"latent_dim": 8, "active_units": 6, "inactive_units": 2,
               "active_fraction": 0.75, "delta": 0.01, "mean_kl": 0.6}

    def test_a_collapsed_trial_loses_to_a_healthy_one_regardless_of_its_own_metric(self):
        from unittest.mock import patch

        X, valid = _data(n=180)
        # Trial 0 "collapses" (per the mocked diagnostics); trial 1 does not. The guard must
        # force trial 0 to the worst possible value BEFORE its real (finite, otherwise
        # ordinary) ELBO is even looked at -- so trial 1 wins on every unsupervised run
        # regardless of which trial's real metric would have been numerically better.
        with patch.object(VAEDetector, "latent_diagnostics",
                          side_effect=[self._COLLAPSED, self._HEALTHY]):
            study, payload = self._tune(X, valid, name="guard")
        self.assertEqual([t.user_attrs["posterior_collapse"] for t in study.trials], [True, False])
        self.assertEqual(study.direction.name, "MINIMIZE")           # default unsupervised objective
        self.assertTrue(np.isinf(study.trials[0].value) and study.trials[0].value > 0)
        self.assertTrue(np.isfinite(study.trials[1].value))
        self.assertEqual(study.best_trial.number, 1)
        self.assertEqual(payload["best_trial_number"], 1)

    def test_every_trial_collapsing_is_reported_not_hidden(self):
        from unittest.mock import patch

        X, valid = _data(n=180)
        with patch.object(VAEDetector, "latent_diagnostics", return_value=self._COLLAPSED):
            study, payload = self._tune(X, valid, name="all-collapsed")
        self.assertTrue(all(t.user_attrs["posterior_collapse"] for t in study.trials))
        self.assertTrue(np.isinf(study.best_value) and study.best_value > 0)
        self.assertTrue(payload["best_value"] > 0 and payload["best_value"] == float("inf"))


if __name__ == "__main__":
    unittest.main()
