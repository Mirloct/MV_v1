"""Section-9 experiment families: defaults, budgets, temporal isolation, and comparability.

Data sources / inputs: seeded synthetic panels produced by ``src.data`` and
temporary model checkpoints.
Created: 2026-09-25
Last modified: 2026-09-25
Changelog:
- 2026-09-25: Added regressions for duplicate capacity points, backtest
  calibration leakage, incomparable Jaccards, and exact three-period windows.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "if_vae_diagnostic_suite", "src"))

from src.evaluation.ifvae_contract import (  # noqa: E402
    STATUS_EXECUTED, STATUS_FAILED, STATUS_NOT_APPLICABLE, STATUS_NOT_REQUESTED, STATUS_UNAVAILABLE,
)
from src.evaluation.ifvae_experiments import (  # noqa: E402
    ALL_FAMILIES, FAMILY_LABELS, build_experiment_context, feature_families, run_experiment_families,
)

SETTINGS = dict(
    families=ALL_FAMILIES, vae_fit_budget=14, epoch_cap=3, max_fit_rows=100_000,
    capacity_grid=None, beta_grid=None, kl_anneal_grid=None,
    backtest_origins=4, backtest_vae_origins=1, backtest_min_fit_periods=4, seed=7,
)


class TestFeatureFamilies(unittest.TestCase):
    def test_rules_from_real_column_names(self):
        names = ["cat__segment_retail", "bool__flag", "cyc__x_month_sin",
                 "num__income_lag1", "num__balance_diff3", "num__age_own_z", "num__txn_amount_to_income",
                 "num__age", "iforest_score"]
        fam = feature_families(names, derived=["iforest_score"], ratio_names=["txn_amount_to_income"])
        self.assertEqual([fam[n] for n in names], [
            "cat", "bool", "cyc", "panel_hist", "panel_hist", "panel_hist",
            "ratios_negocio", "num_base", "derivada"])

    def test_capacity_points_never_repeat_the_production_hidden_shape(self):
        import src.evaluation.ifvae_experiments as mod

        base = type("Base", (), {"latent_dim": 4, "hidden_dims": [4], "hidden_dim": 4})()
        points = mod._capacity_points(base, n_features=10, override=None)
        self.assertNotIn(("hidden_dims=[4]", {"hidden_dims": [4]}), points)


class ExperimentsBase(unittest.TestCase):
    """One small labelled panel, fitted once, shared by the tests."""

    @classmethod
    def setUpClass(cls):
        from src.data import load_or_generate_panel
        from src.evaluation import chronological_split
        from src.models import IsolationForestDetector, VAEDetector
        from src.preprocessing import fit_transform_panel, split_matrix_for_model
        import logging

        cls._cwd, cls._tmp = os.getcwd(), tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        os.chdir(cls._tmp.name)
        logging.getLogger("modelo").setLevel(logging.ERROR)
        df, schema = load_or_generate_panel(
            data_path=os.path.join(cls._tmp.name, "d.csv"), n_individuals=120, n_periods=10, seed=3)
        split = chronological_split(df, time_col=schema.time_col, n_val_periods=2,
                                    n_test_periods=2, n_oot_periods=2)
        cls.masks = dict(train=split.train_mask, val=split.val_mask, test=split.test_mask, oot=split.oot_mask)
        cls.prep = dict(numeric_transform="yeo-johnson", categorical_encoding="onehot",
                        rare_min_frequency=0.001, impute_numeric="zero", add_panel_features=False,
                        random_state=3)
        X, keys, names, pipe = fit_transform_panel(df, schema, fit_mask=split.train_mask,
                                                   return_pipeline=True, **cls.prep)
        x_if, names_if = split_matrix_for_model(X, names, "iforest")
        x_if = x_if.toarray() if hasattr(x_if, "toarray") else np.asarray(x_if)
        x_all = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        in_mask = split.train_mask | split.val_mask
        valid_local = split.val_mask[in_mask]
        if_det = IsolationForestDetector(n_estimators=30, max_samples=128, random_state=3).fit(x_if[in_mask])
        vae_det = VAEDetector(epochs=3, latent_dim=4, hidden_dim=16, n_layers=1, batch_size=128,
                              random_state=3, early_stopping_patience=None)
        vae_det.fit(x_all[in_mask], valid_mask=valid_local,
                    checkpoint_dir=os.path.join(cls._tmp.name, "ck"), resume=False)
        cls.ctx = build_experiment_context(
            df=df, schema=schema, keys=keys, x_if_all=x_if, if_feature_names=names_if,
            x_vae_all=x_all, vae_feature_names=names, if_detector=if_det, vae_detector=vae_det,
            train_mask=split.train_mask, in_mask=in_mask, oot_mask=split.oot_mask, valid_local=valid_local,
            n_val_periods=2, derived_features=[], fitted_preprocessor=pipe, prep_kwargs=cls.prep)
        cls.ctx.update(threshold=0.9, top_k=5, alert_k=10)
        # Diagnostic-style frames (production alert sets come from these, like the bridge).
        from ifvae_diag.scoring import anomaly_percentile
        from src.evaluation.ifvae_diagnostic import _build_frame, _vae_forward

        mu, lv, rec = _vae_forward(vae_det, x_all)
        ent, per = keys[schema.entity_col].to_numpy(), keys[schema.time_col].astype(str).to_numpy()
        if_scores = if_det.score_samples(x_if)

        def frame(mask):
            return _build_frame(ent[mask], per[mask], x_all[mask], names, rec[mask], mu[mask], lv[mask],
                                if_scores[mask])

        cls.frames = {"reference": frame(split.train_mask), "scored": frame(split.oot_mask), "features": list(names)}
        ref_if, oot_if = if_scores[split.train_mask], if_scores[split.oot_mask]
        cls.prod_if_high = anomaly_percentile(ref_if, oot_if) >= 0.9
        cls.prod_if_scores = oot_if
        cls.prod_vae_high = np.random.default_rng(0).random(int(split.oot_mask.sum())) > 0.9

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls._tmp.cleanup()

    def run_families(self, **over):
        settings = {**SETTINGS, **over}
        return run_experiment_families(
            ctx=dict(self.ctx), settings=settings, frames=self.frames, prod_if_high=self.prod_if_high,
            prod_vae_high=self.prod_vae_high, prod_if_scores=self.prod_if_scores,
            noise_floor={"iforest": 0.9, "vae": 0.95})

    @classmethod
    def default_rows(cls):
        """The default matrix, run once per class: the tests that only read it must not each repeat it."""
        if "_default_rows" not in cls.__dict__:
            cls._default_rows = cls("run_families").run_families()
        return cls._default_rows


class TestAllSixFamiliesRunByDefault(ExperimentsBase):
    def test_every_family_produces_executed_rows_and_nothing_fails(self):
        rows = self.default_rows()
        self.assertFalse([r for r in rows if r["status"] == STATUS_FAILED], [r for r in rows if r["status"] == STATUS_FAILED])
        for label in FAMILY_LABELS.values():
            done = [r for r in rows if r["experiment"].startswith(label) and r["status"] == STATUS_EXECUTED]
            self.assertTrue(done, f"{label} did not execute")
        self.assertFalse([r for r in rows if r["status"] == STATUS_NOT_REQUESTED and "Desactivada" in (r["reason"] or "")])

    def test_row_format_matches_the_report_contract(self):
        for r in self.default_rows():
            self.assertEqual(set(r), {"experiment", "status", "configuration", "artifact", "detail", "reason"})
            if r["status"] == STATUS_EXECUTED:
                self.assertTrue(r["detail"])
            else:
                self.assertTrue(r["reason"], r)

    def test_vae_rows_carry_the_seed_noise_floor_and_the_production_comparison(self):
        rows = [r for r in self.run_families(epoch_cap=2)
                if r["experiment"].startswith(FAMILY_LABELS["capacity"] + ":")]
        self.assertTrue(rows)
        for r in rows:
            self.assertIn("control: misma configuración, otra semilla", r["detail"])
            self.assertIn("Jaccard vs producción", r["detail"])
            self.assertIn("épocas limitadas a 2", r["detail"])   # the cap is disclosed, not silent

    def test_loss_by_type_exposes_the_categorical_block(self):
        rows = {r["experiment"]: r for r in self.default_rows() if r["experiment"].startswith(FAMILY_LABELS["loss_by_type"])}
        cat = rows[FAMILY_LABELS["loss_by_type"] + ": cat"]
        self.assertIn("lugares top-K", cat["detail"])
        summary = rows[FAMILY_LABELS["loss_by_type"] + ": resumen de sesgo categórico"]
        self.assertEqual(summary["status"], STATUS_EXECUTED)
        self.assertIn("one-hot agregado", summary["detail"])


class TestSwitchesAndBudget(ExperimentsBase):
    def test_zero_budget_turns_vae_points_into_not_requested_with_the_reason_but_keeps_the_rest(self):
        rows = self.run_families(vae_fit_budget=0)
        vae_rows = [r for r in rows if r["experiment"].startswith((FAMILY_LABELS["capacity"], FAMILY_LABELS["beta_kl"]))]
        self.assertTrue(vae_rows)
        self.assertTrue(all(r["status"] == STATUS_NOT_REQUESTED and "presupuesto" in r["reason"] for r in vae_rows))
        self.assertTrue([r for r in rows if r["experiment"].startswith(FAMILY_LABELS["loss_by_type"]) and r["status"] == STATUS_EXECUTED])
        self.assertTrue([r for r in rows if "(IF)" in r["experiment"] and r["status"] == STATUS_EXECUTED])

    def test_budget_is_shared_and_never_exceeded(self):
        import src.evaluation.ifvae_experiments as mod
        calls = []
        real = mod._fit_vae

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        with mock.patch.object(mod, "_fit_vae", spy):
            rows = self.run_families(vae_fit_budget=3)
        self.assertEqual(len(calls), 3)                      # control + 2 points, then it stops
        self.assertTrue([r for r in rows if r["status"] == STATUS_NOT_REQUESTED and "presupuesto" in (r["reason"] or "")])

    def test_the_control_is_reported_and_used_as_the_noise_reference(self):
        rows = self.default_rows()
        control = [r for r in rows if r["experiment"].startswith("Control de ruido del VAE")]
        self.assertEqual(len(control), 1)
        self.assertEqual(control[0]["status"], STATUS_EXECUTED)
        self.assertIn("Jaccard vs producción", control[0]["detail"])
        ab = [r for r in rows if r["experiment"].startswith(FAMILY_LABELS["ablation"] + " (VAE)") and r["status"] == STATUS_EXECUTED]
        self.assertTrue(ab)
        self.assertTrue(all("-ELBO" not in r["detail"] for r in ab))      # other input dimension: not comparable
        cap = [r for r in rows if r["experiment"].startswith(FAMILY_LABELS["capacity"] + ":")]
        self.assertTrue(all("-ELBO" in r["detail"] for r in cap if r["status"] == STATUS_EXECUTED))

    def test_zero_origins_and_empty_family_list_are_visible_not_silent(self):
        bt = [r for r in self.run_families(families=("backtest",), backtest_origins=0)
              if r["experiment"] == FAMILY_LABELS["backtest"]]
        self.assertEqual(bt[0]["status"], STATUS_NOT_REQUESTED)
        self.assertIn("backtest_origins = 0", bt[0]["reason"])
        rows = self.run_families(families=())
        self.assertFalse([r for r in rows if r["status"] == STATUS_EXECUTED])       # [] = run none
        self.assertEqual(len([r for r in rows if "Desactivada" in (r["reason"] or "")]), len(ALL_FAMILIES))

    def test_families_can_be_switched_off_individually(self):
        rows = self.run_families(families=("loss_by_type",))
        off = [r for r in rows if r["status"] == STATUS_NOT_REQUESTED and "Desactivada" in (r["reason"] or "")]
        self.assertEqual(len(off), len(ALL_FAMILIES) - 1)
        self.assertFalse([r for r in rows if r["experiment"].startswith(FAMILY_LABELS["backtest"]) and r["status"] == STATUS_EXECUTED])

    def test_window_stability_needs_the_backtest(self):
        rows = self.run_families(families=("window_stability",))
        ws = [r for r in rows if r["experiment"].startswith(FAMILY_LABELS["window_stability"])]
        self.assertEqual(ws[0]["status"], STATUS_UNAVAILABLE)

    def test_empty_grid_switches_that_sweep_off_and_explicit_points_are_used(self):
        rows = self.run_families(capacity_grid=(), beta_grid=(3.0,), kl_anneal_grid=())
        cap = [r for r in rows if r["experiment"] == FAMILY_LABELS["capacity"]]
        self.assertEqual(cap[0]["status"], STATUS_NOT_REQUESTED)
        beta = [r for r in rows if r["experiment"].startswith(FAMILY_LABELS["beta_kl"] + ":")]
        self.assertEqual([r["configuration"] for r in beta], ["beta=3"])

    def test_absent_families_are_not_applicable_not_failures(self):
        rows = self.default_rows()
        ph = [r for r in rows if r["experiment"].endswith("sin panel_hist") and "(IF)" in r["experiment"]]
        self.assertEqual(ph[0]["status"], STATUS_NOT_APPLICABLE)      # panel features are off in this panel

    def test_too_few_periods_reports_unavailable(self):
        rows = self.run_families(backtest_min_fit_periods=20)
        bt = [r for r in rows if r["experiment"].startswith(FAMILY_LABELS["backtest"])]
        self.assertEqual(bt[0]["status"], STATUS_UNAVAILABLE)
        self.assertIn("periodos", bt[0]["reason"])


class TestNoLeakageInTheBacktest(ExperimentsBase):
    def test_each_origin_preprocesses_and_fits_only_on_strictly_earlier_periods(self):
        import src.preprocessing as prep
        calls = []
        real = prep.fit_transform_panel

        def spy(df, schema, logger=None, fit_mask=None, **kw):
            calls.append(np.asarray(fit_mask).copy())
            return real(df, schema, logger, fit_mask=fit_mask, **kw)

        with mock.patch.object(prep, "fit_transform_panel", spy):
            self.run_families(families=("backtest",), vae_fit_budget=0)
        period = self.ctx["period_idx"]
        self.assertGreaterEqual(len(calls), 2)
        origins = sorted(int(period[~m].min()) for m in calls)
        for mask, origin in zip(calls, origins):
            self.assertLess(int(period[mask].max()), origin)          # nothing at/after the origin
            self.assertTrue(np.array_equal(mask, period < origin))    # ... and everything before it

    def test_vae_percentile_reference_excludes_the_validation_periods(self):
        import src.evaluation.ifvae_experiments as mod

        calls = []

        def alerts(_det, _x, ref_idx, eval_idx, _top_k, _threshold, *_extra):
            calls.append((np.asarray(ref_idx), np.asarray(eval_idx)))
            return np.zeros(len(eval_idx), dtype=bool)

        with mock.patch.object(mod, "_recon_topk_alerts", side_effect=alerts):
            self.run_families(families=("backtest",), vae_fit_budget=1,
                              backtest_origins=2, backtest_vae_origins=1)
        self.assertEqual(len(calls), 1)
        ref_idx, eval_idx = calls[0]
        period = self.ctx["period_idx"]
        origin = int(period[eval_idx][0])
        self.assertTrue(np.all(period[ref_idx] < origin - self.ctx["n_val_periods"]))

    def test_backtest_does_not_publish_jaccard_when_its_gap_differs_from_production(self):
        rows = self.run_families(families=("backtest",), vae_fit_budget=0, backtest_origins=2)
        executed_if = [r for r in rows if "(IF)" in r["experiment"] and r["status"] == STATUS_EXECUTED]
        self.assertTrue(executed_if)
        self.assertTrue(all("Jaccard vs alertas de producción" not in r["detail"] for r in executed_if))


class TestWindowAggregation(unittest.TestCase):
    def test_aggregate_comparison_uses_the_last_two_exact_three_period_windows(self):
        import src.evaluation.ifvae_experiments as mod

        windows = {k: (np.arange(10), np.arange(10, dtype=float) + k) for k in range(8)}
        rows = mod._window_stability(windows, top_k_cfg=3)
        agg = next(r for r in rows if "ventanas agregadas" in r["experiment"])
        self.assertIn("periodos 3-5 vs 6-8", agg["configuration"])


class TestConfigurationDefaults(unittest.TestCase):
    def test_all_six_families_are_on_by_default_and_editable_in_the_single_file(self):
        import main
        from src.utils.config_file import ConfigFileError, apply_config_file

        cfg = main.PipelineConfig()
        self.assertEqual(tuple(cfg.diagnostic_experiment_families), ALL_FAMILIES)
        self.assertEqual(cfg.diagnostic_experiment_vae_fit_budget, 16)
        self.assertIsNone(cfg.diagnostic_experiment_capacity_grid)       # None = automatic points
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("experiments:\n  families: [capacity, backtest]\n  vae_fit_budget: 5\n"
                         "  capacity_grid: []\n  beta_grid:\n")
            apply_config_file(cfg, path)
            self.assertEqual(cfg.diagnostic_experiment_families, ("capacity", "backtest"))
            self.assertEqual(cfg.diagnostic_experiment_vae_fit_budget, 5)
            self.assertEqual(cfg.diagnostic_experiment_capacity_grid, ())    # [] = sweep off
            self.assertIsNone(cfg.diagnostic_experiment_beta_grid)           # empty key = automatic
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("experiments:\n  families: [capacidad]\n")
            with self.assertRaises(ConfigFileError) as ctx:
                apply_config_file(main.PipelineConfig(), path)
            self.assertIn("backtest", str(ctx.exception))                  # lists the valid names


if __name__ == "__main__":
    unittest.main()
