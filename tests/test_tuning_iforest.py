"""Behaviour of the redesigned IF tuner: fixed trees, absolute psi, anchors, selection, resume.

Data sources / inputs: seeded in-memory matrices and temporary Optuna SQLite,
YAML, and joblib artifacts.
Created: 2026-09-25
Last modified: 2026-09-26
Changelog:
- 2026-09-25: Added study-data identity and zero-complete-trial regressions.
- 2026-09-26: Locked the production contamination/tree contract and global
  VAE epoch caps across configuration entry points.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import joblib
import numpy as np
import yaml

from src.models import iforest as ifm
from src.models.iforest import (
    _detector_kwargs_from_params,
    _select_trial,
    _study_fingerprint,
    tune_iforest,
)

N_EST = 300


def _panel(seed: int = 0, n: int = 1600, n_pos: int = 60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.zeros(n)
    idx = rng.choice(np.arange(n // 2, n), n_pos, replace=False)  # anomalies in the validation half
    X[idx] += 3.5
    y[idx] = 1.0
    valid = np.zeros(n, dtype=bool)
    valid[n // 2:] = True
    return X, y, valid


class TunerBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name

    def run_tuner(self, X, valid, *, name="run", **kw):
        kw.setdefault("n_trials", 6)
        kw.setdefault("n_estimators", N_EST)
        kw.setdefault("max_samples_range", (128, 4096))
        kw.setdefault("noise_seeds", 2)
        kw.setdefault("early_stopping_patience", None)
        study = tune_iforest(
            X, valid_mask=valid,
            storage="sqlite:///" + os.path.join(self.d, f"{name}.db").replace("\\", "/"),
            best_params_path=os.path.join(self.d, f"{name}.yaml"),
            model_out=os.path.join(self.d, f"{name}.joblib"),
            **kw,
        )
        return study, yaml.safe_load(open(os.path.join(self.d, f"{name}.yaml"), encoding="utf-8"))


class TestSearchSpaceAndDeploy(TunerBase):
    def test_only_psi_and_max_features_are_searched(self):
        X, _, valid = _panel()
        study, _ = self.run_tuner(X, valid)
        for t in study.trials:
            self.assertEqual(set(t.params), {"max_samples", "max_features"})
            self.assertIsInstance(t.params["max_samples"], int)

    def test_psi_is_capped_to_fit_rows_and_never_a_fraction(self):
        X, _, valid = _panel(n=1600)  # 800 fit rows
        study, _ = self.run_tuner(X, valid, max_samples_range=(128, 32768))
        self.assertTrue(all(2 <= t.params["max_samples"] <= 800 for t in study.trials))

    def test_anchors_enqueued_first_and_clipped(self):
        X, _, valid = _panel(n=6000)  # 3000 fit rows
        study, _ = self.run_tuner(X, valid, max_samples_range=(128, 32768), n_trials=6)
        first = [t.params["max_samples"] for t in study.trials[:2]]
        self.assertEqual(first, [1024, 3000])  # 4096/16384/32768 clipped to the fit rows
        self.assertTrue(all(t.params["max_features"] == 1.0 for t in study.trials[:2]))

    def test_deployed_psi_is_identical_in_yaml_and_refit_model(self):
        X, _, valid = _panel()
        _, payload = self.run_tuner(X, valid, name="dep")
        model = joblib.load(os.path.join(self.d, "dep.joblib"))
        kwargs = payload["best_params"]
        self.assertEqual(kwargs["n_estimators"], N_EST)
        self.assertIsInstance(kwargs["max_samples"], int)
        self.assertEqual(payload["max_samples_absolute"], kwargs["max_samples"])
        self.assertEqual(model.max_samples, kwargs["max_samples"])
        self.assertEqual(model.n_estimators, N_EST)
        self.assertEqual(model.contamination, 0.005)
        self.assertEqual(payload["contamination"], 0.005)
        self.assertIs(payload["contamination_tuned"], False)
        self.assertIn("selection", payload)

    def test_kwargs_translation_new_and_legacy_layouts(self):
        new = _detector_kwargs_from_params({"max_samples": 2048, "max_features": 0.7})
        self.assertEqual((new["max_samples"], new["n_estimators"], new["bootstrap"]), (2048, 300, False))
        legacy = _detector_kwargs_from_params(
            {"n_estimators": 150, "max_samples_mode": "float", "max_samples": 0.53,
             "max_features": 0.53, "bootstrap": True})
        self.assertEqual((legacy["max_samples"], legacy["n_estimators"]), (0.53, 150))
        auto = _detector_kwargs_from_params(
            {"n_estimators": 100, "max_samples_mode": "auto", "max_features": 1.0, "bootstrap": False})
        self.assertEqual(auto["max_samples"], "auto")


class TestResume(TunerBase):
    def test_n_trials_is_a_total_budget(self):
        X, _, valid = _panel()
        s1, _ = self.run_tuner(X, valid, name="res", n_trials=4)
        self.assertEqual(len(s1.trials), 4)
        s2, _ = self.run_tuner(X, valid, name="res", n_trials=4)
        self.assertEqual(len(s2.trials), 4)           # nothing new to run
        s3, _ = self.run_tuner(X, valid, name="res", n_trials=6)
        self.assertEqual(len(s3.trials), 6)           # only the missing two

    def test_tree_count_override_is_ignored_and_does_not_change_study(self):
        X, _, valid = _panel()
        s1, _ = self.run_tuner(X, valid, name="fp", n_trials=3)
        s2, _ = self.run_tuner(X, valid, name="fp", n_trials=3, n_estimators=N_EST + 10)
        self.assertEqual(s1.study_name, s2.study_name)

    def test_fingerprint_depends_on_extra(self):
        X = np.zeros((10, 3))
        self.assertNotEqual(_study_fingerprint(X, None, "m", "maximize", "a"),
                            _study_fingerprint(X, None, "m", "maximize", "b"))

    def test_fingerprint_depends_on_matrix_values_not_only_shape(self):
        a = np.zeros((300, 6), dtype=np.float32)
        b = a.copy()
        b[1, 0] = 1.0
        self.assertNotEqual(_study_fingerprint(a, None, "m", "maximize"),
                            _study_fingerprint(b, None, "m", "maximize"))

    def test_no_completed_trial_is_a_failure_not_a_stale_artifact_success(self):
        X, _, valid = _panel(n=400)
        with self.assertRaises(RuntimeError):
            tune_iforest(
                X, n_trials=0, valid_mask=valid, n_estimators=N_EST,
                max_samples_range=(64, 128), early_stopping_patience=None,
                storage="sqlite:///" + os.path.join(self.d, "empty.db").replace("\\", "/"),
                best_params_path=os.path.join(self.d, "stale.yaml"),
                model_out=os.path.join(self.d, "stale.joblib"),
            )


class TestObjectiveModes(TunerBase):
    def test_default_is_label_free_tail_separation(self):
        X, _, valid = _panel()
        _, payload = self.run_tuner(X, valid)
        self.assertEqual(payload["objective_mode"], "unsupervised(tail_separation)")
        self.assertIsNone(payload["label_source"])

    def test_labels_with_unknown_rows_use_ap_on_known_rows(self):
        X, y, valid = _panel()
        y = y.copy()
        y[np.random.default_rng(1).random(y.size) < 0.3] = np.nan  # unknown rows are ignored
        _, payload = self.run_tuner(X, valid, y=y, label_source="reviewed_labels")
        self.assertEqual(payload["objective_mode"], "supervised(reviewed_labels)")
        self.assertEqual(payload["label_source"], "reviewed_labels")

    def test_too_few_positives_falls_back_to_label_free(self):
        X, y, valid = _panel(n_pos=3)
        _, payload = self.run_tuner(X, valid, y=y, min_eval_positives=10)
        self.assertEqual(payload["objective_mode"], "unsupervised(tail_separation)")

    def test_forced_rank_agreement_ignores_labels(self):
        X, y, valid = _panel()
        _, payload = self.run_tuner(X, valid, y=y, objective_metric="rank_agreement", n_trials=3)
        self.assertEqual(payload["objective_mode"], "unsupervised(rank_agreement)")

    def test_rank_agreement_accepts_several_top_k_fractions(self):
        X, _, _ = _panel()
        idx = np.arange(X.shape[0])
        kw = {"n_estimators": N_EST, "max_samples": 128, "max_features": 1.0,
              "contamination": 0.1, "bootstrap": False}
        one = ifm._rank_agreement(kw, X, idx[:800], idx[800:], 0, 0.02)
        many = ifm._rank_agreement(kw, X, idx[:800], idx[800:], 0, (0.005, 0.01, 0.02))
        self.assertTrue(0.0 <= one <= 1.0 and 0.0 <= many <= 1.0)


class TestSelectionRule(unittest.TestCase):
    @staticmethod
    def _trials(rows):
        return [SimpleNamespace(number=i, value=v, params={"max_samples": ms, "max_features": mf})
                for i, (v, ms, mf) in enumerate(rows)]

    cost = staticmethod(lambda p: p["max_samples"] * p["max_features"])

    def test_one_se_picks_cheapest_within_noise_of_best(self):
        trials = self._trials([(1.00, 16384, 1.0), (0.99, 1024, 0.6), (0.50, 256, 1.0)])
        sel = _select_trial(trials, {"mean": 0.5, "sd": 0.05}, sign=1.0, one_se_rule=True,
                            margin_sd=1.0, cost_of=self.cost)
        self.assertEqual(sel["best_trial"], 0)
        self.assertEqual(sel["picked_trial"], 1)
        self.assertTrue(sel["beats_reference"])

    def test_without_one_se_rule_argmax_wins(self):
        trials = self._trials([(1.00, 16384, 1.0), (0.99, 1024, 0.6)])
        sel = _select_trial(trials, {"mean": 0.5, "sd": 0.05}, sign=1.0, one_se_rule=False,
                            margin_sd=1.0, cost_of=self.cost)
        self.assertEqual(sel["picked_trial"], 0)

    def test_gain_inside_the_noise_keeps_the_reference(self):
        trials = self._trials([(0.52, 4096, 1.0)])
        sel = _select_trial(trials, {"mean": 0.50, "sd": 0.05}, sign=1.0, one_se_rule=True,
                            margin_sd=1.0, cost_of=self.cost)
        self.assertFalse(sel["beats_reference"])

    def test_minimize_direction(self):
        trials = self._trials([(0.10, 1024, 1.0), (0.90, 4096, 1.0)])
        sel = _select_trial(trials, {"mean": 0.9, "sd": 0.05}, sign=-1.0, one_se_rule=True,
                            margin_sd=1.0, cost_of=self.cost)
        self.assertEqual(sel["picked_trial"], 0)
        self.assertTrue(sel["beats_reference"])

    def test_zero_noise_reference_needs_strictly_better(self):
        trials = self._trials([(0.5, 4096, 1.0)])
        sel = _select_trial(trials, {"mean": 0.5, "sd": 0.0}, sign=1.0, one_se_rule=True,
                            margin_sd=1.0, cost_of=self.cost)
        self.assertFalse(sel["beats_reference"])

    def test_one_se_uses_replicated_best_uncertainty_and_paired_default_margin(self):
        trials = self._trials([(9.9, 16384, 1.0), (9.8, 1024, 0.6)])
        evaluations = {
            0: {"mean": 1.00, "se": 0.03, "values": [0.96, 1.00, 1.04]},
            1: {"mean": 0.98, "se": 0.01, "values": [0.97, 0.98, 0.99]},
        }
        reference = {"mean": 0.90, "sd": 0.10, "se": 0.058,
                     "values": [0.89, 0.90, 0.91]}
        sel = _select_trial(
            trials, reference, sign=1.0, one_se_rule=True, margin_sd=1.0,
            cost_of=self.cost, evaluations=evaluations,
        )
        self.assertEqual(sel["picked_trial"], 1)
        self.assertAlmostEqual(sel["best_se"], 0.03)
        self.assertTrue(sel["beats_reference"])


if __name__ == "__main__":
    unittest.main()


class TestPipelineConfigWiring(unittest.TestCase):
    """The tuning contract reaches `main.py`: absolute psi, fixed trees, label flags."""

    @staticmethod
    def _config(*argv):
        import main

        return main.config_from_args(main.build_arg_parser().parse_args(list(argv)))

    def test_defaults_are_fixed_trees_absolute_psi_and_labels_auto(self):
        cfg = self._config()
        self.assertEqual(cfg.iforest_params["n_estimators"], 300)
        self.assertIsInstance(cfg.iforest_params["max_samples"], int)
        self.assertEqual(cfg.iforest_params["max_samples"], 4096)
        self.assertEqual(tuple(cfg.iforest_max_samples_range), (1024, 32768))
        self.assertEqual((cfg.tune_with_labels, cfg.tune_min_positive_rows), ("auto", 10))
        self.assertEqual(tuple(cfg.diagnostic_experiment_contamination_grid), (0.02, 0.01, 0.005))
        self.assertEqual(cfg.iforest_params["contamination"], 0.005)
        self.assertEqual(cfg.vae_epochs, 15)
        self.assertEqual(cfg.diagnostic_experiment_epoch_cap, 15)

    def test_full_and_explicit_epoch_values_are_safely_capped(self):
        self.assertEqual(self._config("--full").vae_epochs, 15)
        self.assertEqual(self._config("--vae-epochs", "99").vae_epochs, 15)

        import main

        cfg = main.PipelineConfig(vae_epochs=99, diagnostic_experiment_epoch_cap=99)
        self.assertEqual((cfg.vae_epochs, cfg.diagnostic_experiment_epoch_cap), (15, 15))

    def test_flags_override_and_bad_values_are_rejected(self):
        cfg = self._config("--tune-with-labels", "off", "--tune-min-positive-rows", "25",
                           "--contamination", "0.1",
                           "--iforest-max-samples-range", "512", "2048")
        self.assertEqual((cfg.tune_with_labels, cfg.tune_min_positive_rows), ("off", 25))
        self.assertEqual(cfg.iforest_params["contamination"], 0.005)
        self.assertEqual(tuple(cfg.iforest_max_samples_range), (512, 2048))
        with self.assertRaises(SystemExit):
            self._config("--iforest-max-samples-range", "4096", "1024")
        with self.assertRaises(SystemExit):
            self._config("--tune-min-positive-rows", "0")
