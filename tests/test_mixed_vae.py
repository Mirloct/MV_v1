"""Mixed-type VAE (embeddings + per-variable reconstruction): view, model, checkpoints, tuning, §9."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "if_vae_diagnostic_suite", "src"))

from src.evaluation.ifvae_contract import STATUS_EXECUTED, STATUS_FAILED, STATUS_NOT_APPLICABLE  # noqa: E402
from src.evaluation.ifvae_experiments import (  # noqa: E402
    ALL_FAMILIES, FAMILY_LABELS, build_experiment_context, run_experiment_families,
)
from src.models import IncompatibleCheckpointError, MixedVAEConfig, VAEDetector, tune_vae  # noqa: E402
from src.models.mixed_vae import NLL_CAP  # noqa: E402
from src.preprocessing.mixed_view import (  # noqa: E402
    MISSING_INDEX, UNKNOWN_INDEX, MixedLayout, MixedViewBuilder, MixedViewConfig, assert_no_onehot,
    categorical_sources,
)

TINY = dict(latent_dim=4, hidden_dim=16, n_layers=1, batch_size=128, early_stopping_patience=None)


def _panel(n=1500, seed=0, n_levels=40):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "flag_cat": rng.choice(["yes", "no"], n),                                   # binary categorical
        "puesto": rng.choice([f"p{i:02d}" for i in range(n_levels)], n),             # high cardinality
        "x": rng.normal(size=n), "y": rng.normal(size=n),
        "b": rng.random(n) > 0.8,
    })
    df.loc[rng.random(n) < 0.03, "puesto"] = None
    X = np.column_stack([df.x, df.y, df.b.astype(float)])
    names = ["num__x", "num__y", "bool__b"]
    return df, X, names


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name

    def build(self, n=1500, min_frequency=0.0, seed=0, n_levels=40):
        df, X, names = _panel(n, seed, n_levels)
        fit = np.arange(n) < int(n * 0.67)
        b = MixedViewBuilder(MixedViewConfig(min_frequency=min_frequency))
        M = b.fit_transform(df, X, names, fit, ["flag_cat", "puesto"])
        return df, X, names, fit, b, M

    def fit_detector(self, b, M, fit, epochs=3, seed=1, ckpt="ck", resume=False, **kw):
        det = VAEDetector(epochs=epochs, random_state=seed, layout=b.layout,
                          mixed_config=kw.pop("mixed_config", MixedVAEConfig()), **{**TINY, **kw})
        det.fit(M[fit], valid_mask=np.arange(fit.sum()) >= int(fit.sum() * 0.8),
                checkpoint_dir=os.path.join(self.d, ckpt), resume=resume)
        return det


class TestMixedView(Base):
    def test_binary_and_high_cardinality_variables_are_one_index_column_each(self):
        df, X, names, fit, b, M = self.build()
        assert_no_onehot(b.layout)
        self.assertEqual(M.shape[1], 3 + 2)                                # 3 numeric/binary + 2 index columns
        self.assertEqual([s.cardinality for s in b.layout.cat_specs()], [2 + 2, 2 + 40])
        self.assertTrue(all(not c.startswith("cat__") or r == "cat" for c, r in zip(b.layout.columns, b.layout.roles)))

    def test_missing_and_unknown_are_explicit_tokens(self):
        df, X, names, fit, b, M = self.build(min_frequency=0.0)
        col = b.layout.columns.index("cat__puesto")
        self.assertTrue(np.all(M[df.puesto.isna().to_numpy(), col] == MISSING_INDEX))
        df2 = df.copy()
        df2.loc[:9, "puesto"] = "never_seen"
        M2 = b.transform(df2, X)
        self.assertTrue(np.all(M2[:10, col] == UNKNOWN_INDEX))               # not a vocabulary member, not "normal"
        self.assertNotIn("never_seen", b.layout.categoricals["cat__puesto"].vocabulary)

    def test_rare_levels_of_the_fit_rows_become_unknown(self):
        df = pd.DataFrame({"c": ["a"] * 990 + ["rare"] * 10 + ["a"] * 1000, "x": np.arange(2000.0)})
        fit = np.arange(2000) < 1000
        b = MixedViewBuilder(MixedViewConfig(min_frequency=0.02))
        M = b.fit_transform(df, df[["x"]].to_numpy(), ["num__x"], fit, ["c"])
        self.assertEqual(b.layout.categoricals["cat__c"].vocabulary, ("a",))
        self.assertTrue(np.all(M[990:1000, -1] == UNKNOWN_INDEX))

    def test_a_category_that_only_exists_after_the_fit_rows_is_unknown_there_and_never_errors(self):
        df, X, names, fit, b, M = self.build()
        df.loc[~fit, "flag_cat"] = "only_in_validation"
        M2 = b.transform(df, X)                                              # no exception
        col = b.layout.columns.index("cat__flag_cat")
        self.assertTrue(np.all(M2[~fit, col] == UNKNOWN_INDEX))
        self.assertNotIn("only_in_validation", b.layout.categoricals["cat__flag_cat"].vocabulary)
        det = self.fit_detector(b, M, fit)
        self.assertTrue(np.isfinite(det.score_samples(M2)).all())            # the detector scores unseen levels

    def test_a_null_binary_value_is_scored_as_false_instead_of_aborting_the_scoring(self):
        # found by the end-to-end run: the "null" sensitivity scenario on a boolean input left NaN in the
        # matrix, the strict detector refused it and the whole post-training sensitivity phase died
        df, X, names, fit, b, M = self.build()
        X2 = X.copy()
        X2[:5, names.index("bool__b")] = np.nan
        M2 = b.transform(df, X2)
        col = b.layout.columns.index("bool__b")
        self.assertTrue(np.all(M2[:5, col] == 0.0))
        np.testing.assert_array_equal(M2[5:], M[5:])                          # nothing else moves
        det = self.fit_detector(b, M, fit)
        self.assertTrue(np.isfinite(det.score_samples(M2)).all())
        X3 = X.copy()
        X3[:5, names.index("num__x")] = np.nan                                # numeric NaN is still refused
        with self.assertRaises(ValueError):
            det.score_samples(b.transform(df, X3))

    def test_vocabulary_indices_do_not_depend_on_the_order_of_the_data(self):
        df, X, names, fit, b, M = self.build()
        perm = np.random.default_rng(3).permutation(len(df))
        b2 = MixedViewBuilder(MixedViewConfig(min_frequency=0.0))
        inv = np.argsort(perm)
        # same rows, shuffled: the fit rows are the same SET, so the vocabulary must be identical
        fit_perm = fit[perm]
        M2 = b2.fit_transform(df.iloc[perm].reset_index(drop=True), X[perm], names, fit_perm, ["flag_cat", "puesto"])
        self.assertEqual(b.layout.fingerprint(), b2.layout.fingerprint())
        np.testing.assert_array_equal(M2[inv], M)                             # same index for the same row
        levels = b.layout.categoricals["cat__puesto"].vocabulary
        self.assertEqual(list(levels), sorted(levels))                         # alphabetical, order-free

    def test_layout_serialisation_and_fingerprints_separate_vocabularies_and_order(self):
        _, _, _, _, b, _ = self.build()
        self.assertEqual(MixedLayout.from_dict(b.layout.to_dict()).fingerprint(), b.layout.fingerprint())
        _, _, _, _, b_other, _ = self.build(seed=5, n_levels=25)             # other vocabulary
        self.assertNotEqual(b.layout.fingerprint(), b_other.layout.fingerprint())
        cfg = MixedVAEConfig()
        self.assertNotEqual(cfg.fingerprint(b.layout), cfg.fingerprint(b_other.layout))
        self.assertNotEqual(cfg.fingerprint(b.layout), MixedVAEConfig(weight_categorical=2.0).fingerprint(b.layout))
        self.assertNotEqual(cfg.fingerprint(b.layout), MixedVAEConfig(numeric_loss="mse").fingerprint(b.layout))
        self.assertNotEqual(cfg.fingerprint(b.layout),
                            MixedVAEConfig(embedding_dimension_strategy="fixed", embedding_dimension=5).fingerprint(b.layout))
        reordered = MixedLayout(list(reversed(b.layout.columns)), list(reversed(b.layout.roles)), b.layout.categoricals)
        self.assertNotEqual(reordered.fingerprint(), b.layout.fingerprint())

    def test_categorical_sources_follow_the_pipeline_rule(self):
        df, _, _ = _panel(50)
        df["entity_id"], df["period"] = "e", pd.Timestamp("2024-01-01")
        schema = mock.Mock(entity_col="entity_id", time_col="period", target_col=None, identification_columns=())
        self.assertEqual(categorical_sources(df, schema), ["flag_cat", "puesto"])


class TestMixedModel(Base):
    def test_config_validation_and_embedding_rule(self):
        cfg = MixedVAEConfig()
        self.assertEqual(cfg.embedding_dim_for(4), 3)                           # binary + 2 tokens -> small
        self.assertLessEqual(cfg.embedding_dim_for(10_000), 32)                 # capped
        self.assertGreaterEqual(cfg.embedding_dim_for(2), 2)                    # floored
        self.assertEqual(MixedVAEConfig(embedding_dimension_strategy="fixed", embedding_dimension=6).embedding_dim_for(400), 6)
        for bad in (dict(aggregate_by_original_feature=False), dict(numeric_loss="l1"), dict(weight_categorical=0.0),
                    dict(unknown_category_policy="mode"), dict(embedding_min_dimension=9, embedding_max_dimension=3)):
            with self.assertRaises(ValueError):
                MixedVAEConfig(**bad)

    def test_exactly_one_contribution_per_original_variable_whatever_the_cardinality(self):
        _, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        C = det.contributions(M)
        self.assertEqual(C.shape, (len(M), 5))
        self.assertEqual(det.variable_names, ["num__x", "num__y", "bool__b", "cat__flag_cat", "cat__puesto"])
        self.assertTrue((C >= 0).all() and C[:, 3:].max() <= NLL_CAP + 1e-6)
        # the row score is the (unit-weight) mean of the per-variable contributions
        np.testing.assert_allclose(det.score_samples(M), C.mean(1), atol=1e-4)

    def test_loss_is_reported_by_family_and_by_original_categorical_variable(self):
        _, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        rec = det.history_[-1]
        for key in ("train_parts", "val_parts"):
            self.assertEqual(set(rec[key]), {"numeric", "boolean", "categorical", "by_categorical_variable"})
            self.assertEqual(len(rec[key]["by_categorical_variable"]), 2)
        self.assertGreater(rec["train_kl"], 0)
        self.assertAlmostEqual(rec["train_recon"],
                               sum(rec["train_parts"][k] for k in ("numeric", "boolean", "categorical")), places=3)

    def test_family_weights_do_not_change_with_the_number_of_columns_and_can_be_configured(self):
        _, _, _, fit, b, M = self.build()
        d1 = self.fit_detector(b, M, fit, seed=2)
        d2 = self.fit_detector(b, M, fit, seed=2, ckpt="ck2", mixed_config=MixedVAEConfig(weight_categorical=3.0))
        self.assertNotEqual(d1.architecture_fingerprint(), d2.architecture_fingerprint())
        self.assertGreater(d2.history_[-1]["train_recon"], d1.history_[-1]["train_recon"])

    def test_categorical_reconstruction_names_the_observed_category_and_its_probability(self):
        df, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        info = det.categorical_reconstruction(M[:5])
        self.assertEqual(set(info), {"flag_cat", "puesto"})
        self.assertEqual(info["flag_cat"]["observed"], list(df.flag_cat[:5]))
        self.assertTrue(((info["flag_cat"]["probability"] > 0) & (info["flag_cat"]["probability"] <= 1)).all())

    def test_invalid_category_indices_are_refused_not_silently_clamped(self):
        _, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        bad = M.copy()
        bad[0, -1] = 999
        with self.assertRaises(ValueError):
            det.score_samples(bad)
        with self.assertRaises(ValueError):
            det.score_samples(M[:, :-1])                                       # wrong number of columns


class TestCheckpoints(Base):
    def test_save_load_round_trip_scores_identically(self):
        _, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        path = det.save(os.path.join(self.d, "m.pt"))
        loaded = VAEDetector.load(path, expect_architecture="mixed_v1", expect_fingerprint=det.architecture_fingerprint())
        np.testing.assert_allclose(loaded.score_samples(M), det.score_samples(M), atol=1e-6)
        self.assertEqual(loaded.layout.fingerprint(), b.layout.fingerprint())
        self.assertEqual(loaded.mixed_config, det.mixed_config)

    def test_resumed_training_reproduces_the_uninterrupted_run_exactly(self):
        _, _, _, fit, b, M = self.build()
        straight = self.fit_detector(b, M, fit, epochs=4, ckpt="a")
        self.fit_detector(b, M, fit, epochs=2, ckpt="b")                        # "crash" after 2 epochs
        resumed = self.fit_detector(b, M, fit, epochs=4, ckpt="b", resume=True)  # continue in the same dir
        self.assertEqual(len(resumed.history_), 4)
        np.testing.assert_allclose(resumed.score_samples(M), straight.score_samples(M), atol=1e-5)

    def test_one_hot_checkpoints_are_rejected_never_loaded_partially(self):
        _, _, _, fit, b, M = self.build()
        onehot = VAEDetector(epochs=2, random_state=1, **TINY)
        onehot.fit(np.random.default_rng(0).normal(size=(400, 7)).astype(np.float32),
                   checkpoint_dir=os.path.join(self.d, "oh"), resume=False)
        oh_path = onehot.save(os.path.join(self.d, "oh.pt"))
        with self.assertRaises(IncompatibleCheckpointError) as ctx:
            VAEDetector.load(oh_path, expect_architecture="mixed_v1")
        self.assertIn("never loaded partially", str(ctx.exception))
        mixed_path = self.fit_detector(b, M, fit).save(os.path.join(self.d, "mx.pt"))
        with self.assertRaises(IncompatibleCheckpointError):
            VAEDetector.load(mixed_path, expect_architecture="onehot_v1")
        # a payload written before architectures existed is one-hot by definition
        legacy = torch.load(oh_path, weights_only=False)
        for k in ("architecture", "architecture_version", "layout", "mixed_config"):
            legacy.pop(k, None)
        legacy["config"] = {k: v for k, v in legacy["config"].items() if not k.startswith("architecture")}
        torch.save(legacy, oh_path)
        with self.assertRaises(IncompatibleCheckpointError):
            VAEDetector.load(oh_path, expect_architecture="mixed_v1")
        self.assertEqual(VAEDetector.load(oh_path).architecture, "onehot_v1")   # still loadable as what it is

    def test_a_layout_or_loss_change_is_a_different_architecture_and_is_refused(self):
        _, _, _, fit, b, M = self.build()
        path = self.fit_detector(b, M, fit).save(os.path.join(self.d, "mx.pt"))
        other = MixedVAEConfig(weight_numeric=2.0).fingerprint(b.layout)
        with self.assertRaises(IncompatibleCheckpointError):
            VAEDetector.load(path, expect_fingerprint=other)
        payload = torch.load(path, weights_only=False)
        payload["architecture_version"] = "0"
        torch.save(payload, path)
        with self.assertRaises(IncompatibleCheckpointError):
            VAEDetector.load(path, expect_architecture="mixed_v1")

    def test_resume_ignores_a_one_hot_or_other_layout_checkpoint_in_the_directory(self):
        _, _, _, fit, b, M = self.build()
        onehot = VAEDetector(epochs=2, random_state=1, **TINY)
        onehot.fit(np.random.default_rng(0).normal(size=(400, 5)).astype(np.float32),
                   checkpoint_dir=os.path.join(self.d, "shared"), resume=False)
        det = self.fit_detector(b, M, fit, epochs=2, ckpt="shared", resume=True)   # same dir, other architecture
        self.assertEqual(len(det.history_), 2)                                     # trained from scratch
        self.assertTrue(np.isfinite(det.score_samples(M)).all())


class TestTuning(Base):
    def _tune(self, M, fit, b, y=None, name="t", **kw):
        n_fit = int(fit.sum())
        return tune_vae(
            M[fit], n_trials=1, max_epochs=3, y=y, random_state=0, early_stopping_patience=None,
            valid_mask=np.arange(n_fit) >= int(n_fit * 0.8),
            storage="sqlite:///" + os.path.join(self.d, f"{name}.db").replace("\\", "/"),
            best_params_path=os.path.join(self.d, f"{name}.yaml"), model_out=os.path.join(self.d, f"{name}.pt"),
            checkpoint_dir=os.path.join(self.d, "ck"), layout=b.layout, mixed_config=MixedVAEConfig(), **kw)

    def test_study_fingerprints_separate_vocabularies_and_architectures(self):
        _, _, _, fit, b, M = self.build()
        s1 = self._tune(M, fit, b, name="a")
        _, _, _, fit2, b2, M2 = self.build(seed=5, n_levels=25)
        s2 = self._tune(M2, fit2, b2, name="a")
        s3 = tune_vae(M[fit][:, :3], n_trials=1, max_epochs=3, random_state=0, early_stopping_patience=None,
                      valid_mask=np.arange(int(fit.sum())) >= int(fit.sum() * 0.8),
                      storage="sqlite:///" + os.path.join(self.d, "a.db").replace("\\", "/"),
                      best_params_path=os.path.join(self.d, "o.yaml"), model_out=os.path.join(self.d, "o.pt"),
                      checkpoint_dir=os.path.join(self.d, "ck"))                       # one-hot MLP study
        self.assertEqual(len({s1.study_name, s2.study_name, s3.study_name}), 3)
        model = VAEDetector.load(os.path.join(self.d, "a.pt"), expect_architecture="mixed_v1")
        self.assertIsNotNone(model.layout)

    def test_partial_labels_with_nan_use_only_known_validation_rows(self):
        _, _, _, fit, b, M = self.build()
        n_fit = int(fit.sum())
        rng = np.random.default_rng(0)
        y = np.full(n_fit, np.nan)
        val = np.arange(n_fit) >= int(n_fit * 0.8)
        known = val & (rng.random(n_fit) < 0.7)                                   # only some validation rows are known
        y[known] = (rng.random(known.sum()) < 0.25).astype(float)
        s = self._tune(M, fit, b, y=y, name="lab", label_source="reviewed_labels")
        import yaml
        payload = yaml.safe_load(open(os.path.join(self.d, "lab.yaml"), encoding="utf-8"))
        self.assertEqual(payload["objective_mode"], "supervised")
        self.assertEqual(payload["label_source"], "reviewed_labels")
        self.assertEqual(payload["architecture"], "mixed_v1")

    def test_too_few_positives_fall_back_to_the_label_free_objective(self):
        _, _, _, fit, b, M = self.build()
        n_fit = int(fit.sum())
        y = np.full(n_fit, np.nan)
        y[-50:-2] = 0.0
        y[-2:] = 1.0                                                                # 2 positives < 10
        self._tune(M, fit, b, y=y, name="few")
        import yaml
        payload = yaml.safe_load(open(os.path.join(self.d, "few.yaml"), encoding="utf-8"))
        self.assertEqual(payload["objective_mode"], "unsupervised")


class MixedExperimentsBase(Base):
    """One small panel with real categoricals, fitted with the mixed VAE, feeding the §9 families."""

    @classmethod
    def setUpClass(cls):
        import logging
        from ifvae_diag.scoring import anomaly_percentile
        from src.data import load_or_generate_panel
        from src.evaluation import chronological_split
        from src.evaluation.ifvae_diagnostic import MIXED_CONTRIBUTION_SCALE_FLOOR, _build_frame, _vae_frame_arrays
        from src.models import IsolationForestDetector
        from src.preprocessing import fit_transform_panel, split_matrix_for_model

        cls._cwd, cls._tmpd = os.getcwd(), tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        os.chdir(cls._tmpd.name)
        logging.getLogger("modelo").setLevel(logging.ERROR)
        df, schema = load_or_generate_panel(data_path=os.path.join(cls._tmpd.name, "d.csv"),
                                            n_individuals=120, n_periods=10, seed=3)
        split = chronological_split(df, time_col=schema.time_col, n_val_periods=2, n_test_periods=2, n_oot_periods=2)
        cls.prep = dict(numeric_transform="yeo-johnson", categorical_encoding="onehot", rare_min_frequency=0.001,
                        impute_numeric="zero", add_panel_features=False, random_state=3)
        X, keys, names, pipe = fit_transform_panel(df, schema, fit_mask=split.train_mask, return_pipeline=True, **cls.prep)
        x_if, names_if = split_matrix_for_model(X, names, "iforest")
        x_if = x_if.toarray() if hasattr(x_if, "toarray") else np.asarray(x_if)
        x_onehot = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        in_mask = split.train_mask | split.val_mask
        valid_local = split.val_mask[in_mask]
        builder = MixedViewBuilder(MixedViewConfig(min_frequency=0.001))
        x_mixed = builder.fit_transform(df, X, names, split.train_mask, categorical_sources(df, schema))
        cls.builder, cls.x_mixed, cls.split, cls.df, cls.keys, cls.names = builder, x_mixed, split, df, keys, names
        if_det = IsolationForestDetector(n_estimators=30, max_samples=128, random_state=3).fit(x_if[in_mask])
        vae = VAEDetector(epochs=3, latent_dim=4, hidden_dim=16, n_layers=1, batch_size=128, random_state=3,
                          early_stopping_patience=None, layout=builder.layout, mixed_config=MixedVAEConfig())
        vae.fit(x_mixed[in_mask], valid_mask=valid_local, checkpoint_dir=os.path.join(cls._tmpd.name, "ck"), resume=False)
        cls.vae = vae
        cls.ctx = build_experiment_context(
            df=df, schema=schema, keys=keys, x_if_all=x_if, if_feature_names=names_if, x_vae_all=x_mixed,
            vae_feature_names=builder.layout.columns, if_detector=if_det, vae_detector=vae,
            train_mask=split.train_mask, in_mask=in_mask, oot_mask=split.oot_mask, valid_local=valid_local,
            n_val_periods=2, derived_features=[], fitted_preprocessor=pipe, prep_kwargs=cls.prep,
            x_onehot_all=x_onehot, onehot_feature_names=names, vae_builder=builder, mixed_config=MixedVAEConfig())
        cls.ctx.update(threshold=0.9, top_k=5, alert_k=10, scale_floor=MIXED_CONTRIBUTION_SCALE_FLOOR)
        mu, lv, vals, rec, var_names = _vae_frame_arrays(vae, x_mixed, builder.layout.columns)
        cls.var_names = var_names
        ent, per = keys[schema.entity_col].to_numpy(), keys[schema.time_col].astype(str).to_numpy()
        ifs = if_det.score_samples(x_if)

        def frame(mask):
            return _build_frame(ent[mask], per[mask], vals[mask], var_names, rec[mask], mu[mask], lv[mask], ifs[mask])

        cls.frames = {"reference": frame(split.train_mask), "scored": frame(split.oot_mask), "features": var_names}
        cls.prod_if_high = anomaly_percentile(ifs[split.train_mask], ifs[split.oot_mask]) >= 0.9
        cls.prod_if_scores = ifs[split.oot_mask]
        cls.prod_vae_high = np.random.default_rng(0).random(int(split.oot_mask.sum())) > 0.9

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        cls._tmpd.cleanup()

    def run_families(self, **over):
        settings = dict(families=ALL_FAMILIES, vae_fit_budget=16, epoch_cap=2, max_fit_rows=100_000,
                        capacity_grid=None, beta_grid=None, kl_anneal_grid=None, backtest_origins=4,
                        backtest_vae_origins=1, backtest_min_fit_periods=4, seed=7)
        settings.update(over)
        return run_experiment_families(ctx=dict(self.ctx), settings=settings, frames=self.frames,
                                       prod_if_high=self.prod_if_high, prod_vae_high=self.prod_vae_high,
                                       prod_if_scores=self.prod_if_scores)

    @classmethod
    def default_rows(cls):
        """The default matrix, run once per class: the tests that only read it must not each repeat it."""
        if "_default_rows" not in cls.__dict__:
            cls._default_rows = cls("run_families").run_families()
        return cls._default_rows


class TestDiagnosticFramesAreOriginalVariables(MixedExperimentsBase):
    def test_no_dimension_of_an_embedding_logit_or_dummy_becomes_a_feature(self):
        feats = self.frames["features"]
        self.assertEqual(feats, self.vae.variable_names)
        self.assertFalse([f for f in feats if "emb" in f or "logit" in f])
        self.assertFalse([f for f in feats if f.startswith("cat__") and f not in self.builder.layout.names("cat")])
        self.assertEqual(len([c for c in self.frames["scored"].columns if c.startswith("recon__")]), len(feats))
        assert_no_onehot(self.builder.layout)

    def test_recon_topk_selects_among_original_variables_only(self):
        from ifvae_diag.scoring import reconstruction_scores, residual_contributions
        from src.evaluation.ifvae_diagnostic import mixed_norm
        feats = self.frames["features"]
        ref, oot = self.frames["reference"], self.frames["scored"]
        rr = np.abs(ref[feats].to_numpy(float) - ref[[f"recon__{f}" for f in feats]].to_numpy(float))
        oo = np.abs(oot[feats].to_numpy(float) - oot[[f"recon__{f}" for f in feats]].to_numpy(float))
        norm = mixed_norm(self.vae)
        self.assertEqual(norm, {"scale_floor_fraction": 0.1, "center": True})        # documented normalisation
        C = residual_contributions(rr, oo, **norm)
        self.assertEqual(C.shape[1], len(feats))                                     # one column per original variable
        np.testing.assert_allclose(oo, self.vae.contributions(self.x_mixed[self.split.oot_mask]), atol=1e-4)
        top = np.argsort(-C, axis=1)[:, :5]
        self.assertTrue(((top >= 0) & (top < len(feats))).all())                     # slots index variables, not embedding/logit dims
        self.assertEqual(len(reconstruction_scores(rr, oo, 5, **norm)), oo.shape[0])
        self.assertEqual(mixed_norm(VAEDetector(epochs=1, **TINY)), {})               # one-hot keeps the historical rule

    def test_scale_floor_protects_against_a_zero_mad(self):
        from ifvae_diag.scoring import residual_contributions
        ref = np.zeros((200, 2))
        ref[:, 1] = np.abs(np.random.default_rng(0).normal(size=200))
        out = residual_contributions(ref, np.array([[0.02, 1.0]]), scale_floor_fraction=0.1)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(out[0, 0], 0.02 / 1e-6 + 1)                                    # bounded by the documented 1e-6 floor


class TestCardinalityIndependence(Base):
    """The top-k must not depend on how many levels a categorical variable has."""

    def test_centred_normalisation_gives_every_variable_a_comparable_share_of_the_top_k(self):
        from ifvae_diag.scoring import residual_contributions
        rng = np.random.default_rng(0)
        n = 5000
        z = rng.normal(size=(n, 2))
        df = pd.DataFrame({
            "c2": rng.choice(list("ab"), n, p=[.7, .3]),
            "c10": rng.choice([f"k{i}" for i in range(10)], n, p=np.r_[.4, .2, .1, .08, .06, .05, .04, .03, .02, .02]),
            "c40": rng.choice([f"q{i:02d}" for i in range(40)], n),             # uniform: constant NLL
        })
        X = np.column_stack([z[:, 0], z[:, 1], z[:, 0] * .5 + rng.normal(size=n) * .5, rng.normal(size=n)])
        fit = np.arange(n) < 3500
        b = MixedViewBuilder(MixedViewConfig(min_frequency=0.0))
        M = b.fit_transform(df, X, ["num__a", "num__b", "num__c", "num__d"], fit, ["c2", "c10", "c40"])
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=12, batch_size=128, kl_anneal_epochs=4,
                          early_stopping_patience=None, random_state=1, layout=b.layout, mixed_config=MixedVAEConfig())
        det.fit(M[fit], valid_mask=np.arange(int(fit.sum())) >= 2800, checkpoint_dir=os.path.join(self.d, "ck"), resume=False)
        ref, hold = det.contributions(M[fit]), det.contributions(M[~fit])
        names = det.variable_names

        def rates(**kw):
            C = residual_contributions(ref, hold, **kw)
            slots = np.argpartition(C, C.shape[1] - 3, axis=1)[:, -3:]
            return {nm: float((slots == i).sum()) / len(C) for i, nm in enumerate(names)}

        plain, centred = rates(scale_floor_fraction=0.1), rates(scale_floor_fraction=0.1, center=True)
        cats = [nm for nm in names if nm.startswith("cat__")]
        nums = [nm for nm in names if nm.startswith("num__")]
        # without centring the categorical terms (NLL is not centred at zero) monopolise the top-k ...
        self.assertGreater(min(plain[c] for c in cats), 4 * max(plain[v] for v in nums))
        # ... centred, every variable -- 2, 10 or 40 levels, or numeric -- gets a comparable share.
        self.assertLessEqual(max(centred.values()) / min(centred.values()), 2.0, centred)


class TestQualityValidatorRegressions(Base):
    """Defects found by the independent review of the mixed VAE (2026-09-25)."""

    def _uniform_big_panel(self, n=4000):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"big": rng.choice([f"q{i:02d}" for i in range(40)], n),        # uniform: NLL = log 40
                           "small": rng.choice(["a,b", "c"], n)})                          # a level containing a comma
        X = rng.normal(size=(n, 6))
        fit = np.arange(n) < int(n * 0.75)
        b = MixedViewBuilder(MixedViewConfig(min_frequency=0.0))
        M = b.fit_transform(df, X, [f"num__{i}" for i in range(6)], fit, ["big", "small"])
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=10, batch_size=128, kl_anneal_epochs=3,
                          early_stopping_patience=None, random_state=1, layout=b.layout, mixed_config=MixedVAEConfig())
        det.fit(M[fit], valid_mask=np.arange(int(fit.sum())) >= int(fit.sum() * 0.8),
                checkpoint_dir=os.path.join(self.d, "ck"), resume=False)
        return det, M, fit

    def test_analyst_explanations_rank_normalised_contributions_not_raw_nll(self):
        from src.interpretability import explain_rows_vae
        det, M, fit = self._uniform_big_panel()
        raw = det.contributions(M[~fit])
        raw_top = np.argsort(-raw, axis=1)[:, :5]
        big = det.variable_names.index("cat__big")
        self.assertGreater((raw_top == big).any(axis=1).mean(), 0.95)                 # the raw ranking IS monopolised ...
        out = explain_rows_vae(det, M[~fit][:400], top_k=5)
        self.assertLess(np.mean(["big=" in s for s in out]), 0.8)                     # ... the explanation is not
        self.assertTrue(all(len(s.split(", ")) == 5 for s in out))                     # a comma inside a category never adds a chip
        self.assertTrue(any("a;b" in s for s in out))

    def test_reference_normalisation_matches_the_suite_and_survives_save_load(self):
        from ifvae_diag.scoring import residual_contributions
        det, M, fit = self._uniform_big_panel()
        n_fit = int(fit.sum())
        train_rows = M[fit][: int(n_fit * 0.8)]                                       # the non-validation rows of fit()
        suite = residual_contributions(det.contributions(train_rows), det.contributions(M[~fit]), 0.1, True)
        np.testing.assert_allclose(det.normalized_contributions(M[~fit]), suite, rtol=1e-6, atol=1e-9)
        loaded = VAEDetector.load(det.save(os.path.join(self.d, "m.pt")))
        np.testing.assert_allclose(loaded.normalized_contributions(M[~fit]), det.normalized_contributions(M[~fit]))

    def test_categorical_sources_are_exactly_the_pipeline_categorical_branch(self):
        from sklearn.compose import make_column_selector
        df = pd.DataFrame({"o": ["a", "b", "a"], "c": pd.Categorical(["x", "y", "x"]), "s": pd.array(["p", "q", "p"], dtype="string"),
                           "td": pd.to_timedelta([1, 2, 3], unit="D"), "n": [1.0, 2.0, 3.0], "b": [True, False, True],
                           "e": ["k", "k", "k"], "t": pd.to_datetime(["2024-01-01"] * 3)})
        schema = mock.Mock(entity_col="e", time_col="t", target_col=None, identification_columns=())
        expected = list(make_column_selector(dtype_include=["object", "category"])(df.drop(columns=["e", "t"])))
        self.assertEqual(categorical_sources(df, schema), expected)
        self.assertNotIn("td", expected)

    def test_switching_architecture_never_overwrites_the_other_checkpoint(self):
        _, _, _, fit, b, M = self.build()
        shared = os.path.join(self.d, "shared")
        onehot = VAEDetector(epochs=2, random_state=1, **TINY)
        onehot.fit(np.random.default_rng(0).normal(size=(300, 4)).astype(np.float32), checkpoint_dir=shared, resume=False)
        with open(os.path.join(shared, "checkpoint.pth"), "rb") as fh:
            before = fh.read()
        self.fit_detector(b, M, fit, epochs=2, ckpt="shared", resume=True)
        moved = os.path.join(shared, "checkpoint.pth.incompatible-onehot_v1")
        self.assertTrue(os.path.isfile(moved))
        with open(moved, "rb") as fh:
            self.assertEqual(fh.read(), before)                                        # recoverable, byte for byte

    def test_strict_load_rules(self):
        _, _, _, fit, b, M = self.build()
        onehot = VAEDetector(epochs=2, random_state=1, **TINY)
        onehot.fit(np.random.default_rng(0).normal(size=(300, 4)).astype(np.float32),
                   checkpoint_dir=os.path.join(self.d, "oh"), resume=False)
        oh_path = onehot.save(os.path.join(self.d, "oh.pt"))
        with self.assertRaises(IncompatibleCheckpointError):                           # fingerprint expected, payload is one-hot
            VAEDetector.load(oh_path, expect_fingerprint="abc")
        det = self.fit_detector(b, M, fit)
        path = det.save(os.path.join(self.d, "mx.pt"))
        payload = torch.load(path, weights_only=False)
        payload["mixed_config"]["from_the_future"] = 1
        torch.save(payload, path)
        with self.assertRaises(IncompatibleCheckpointError) as ctx:                    # unknown config key: never a partial load
            VAEDetector.load(path)
        self.assertIn("partial", str(ctx.exception))
        # NLL cap and normalisation constants are part of the architecture identity
        import src.models.mixed_vae as mv
        fp = det.mixed_config.fingerprint(b.layout)
        with mock.patch.object(mv, "NLL_CAP", 20.0):
            self.assertNotEqual(det.mixed_config.fingerprint(b.layout), fp)

    def test_non_finite_or_non_binary_inputs_are_refused(self):
        _, _, _, fit, b, M = self.build()
        det = self.fit_detector(b, M, fit)
        bad = M.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            det.score_samples(bad)
        bad = M.copy()
        bad[0, b.layout.columns.index("bool__b")] = 2.5
        with self.assertRaises(ValueError):
            det.score_samples(bad)


class TestComparisonModule(unittest.TestCase):
    def test_arms_share_the_normalisation_and_criteria_are_measured(self):
        from src.evaluation.vae_representation_comparison import (
            ComparisonConfig, acceptance_criteria, promotion_decision, render_markdown, run_comparison,
        )
        s = run_comparison(ComparisonConfig(n_individuals=150, n_periods=10, epochs=3, seeds=(11, 23), n_unseen_rows=30))
        for arm in ("onehot", "embedding"):                                            # both normalisations for BOTH arms
            self.assertEqual(set(s["arms"][arm]["norm"]), {"historical", "centered"})
        crit = acceptance_criteria(s)
        names = " | ".join(c["criterion"] for c in crit)
        for needle in ("MISMA normalización", "pipeline actual", "control literal", "control equivalente", "score de producción del Excel"):
            self.assertIn(needle, names)
        self.assertEqual(s["arms"]["embedding"]["n_contributions"][0], s["layout"]["scored_variables"])
        self.assertEqual(s["arms"]["onehot"]["n_reference_rows"][0], s["calibration_leak_check"]["train_rows"])
        dec = promotion_decision(crit)
        self.assertEqual(dec["promote"], not dec["failed_criteria"])
        self.assertIn("Decisión de promoción", render_markdown(s, crit))


class TestSixFamiliesRunWithEmbeddings(MixedExperimentsBase):
    def test_every_family_executes_and_nothing_fails(self):
        rows = self.default_rows()
        self.assertFalse([r for r in rows if r["status"] == STATUS_FAILED], [r for r in rows if r["status"] == STATUS_FAILED])
        for label in FAMILY_LABELS.values():
            self.assertTrue([r for r in rows if r["experiment"].startswith(label) and r["status"] == STATUS_EXECUTED], label)

    def test_loss_by_type_is_expressed_in_original_variables_with_tokens_and_the_one_hot_control(self):
        rows = {r["experiment"]: r for r in self.default_rows() if r["experiment"].startswith(FAMILY_LABELS["loss_by_type"])}
        cat = rows[FAMILY_LABELS["loss_by_type"] + ": cat"]
        self.assertIn("variables originales", cat["configuration"])
        self.assertIn("pérdida media por variable", cat["detail"])
        self.assertIn("esperado por número de variables originales", cat["detail"])
        control = rows[FAMILY_LABELS["loss_by_type"] + ": control one-hot"]
        self.assertEqual(control["status"], STATUS_EXECUTED)
        self.assertIn("control one-hot", control["detail"])
        self.assertIn("Jaccard de alertas vs producción", control["detail"])
        for tok in ("MISSING", "UNKNOWN"):
            row = rows[f"{FAMILY_LABELS['loss_by_type']}: token {tok}"]
            self.assertIn(row["status"], (STATUS_EXECUTED, STATUS_NOT_APPLICABLE))       # identifiable, not silent
            self.assertTrue(row["detail"] or row["reason"])

    def test_vae_ablation_drops_variables_from_the_layout_not_only_the_matrix(self):
        seen = []
        import src.evaluation.ifvae_experiments as mod
        real = mod._fit_vae

        def spy(base, x_fit, valid, overrides, *a, **k):
            seen.append((overrides.get("layout"), x_fit.shape[1]))
            return real(base, x_fit, valid, overrides, *a, **k)

        with mock.patch.object(mod, "_fit_vae", spy):
            rows = self.run_families(families=("ablation",))
        ab = [r for r in rows if r["experiment"] == f"{FAMILY_LABELS['ablation']} (VAE): sin cat"]
        self.assertEqual(ab[0]["status"], STATUS_EXECUTED)
        no_cat = [(lay, w) for lay, w in seen if lay is not None and not lay.names("cat")]
        self.assertTrue(no_cat)
        self.assertEqual(no_cat[0][0].n_columns, no_cat[0][1])                            # layout and matrix agree

    def test_backtest_builds_each_origins_vocabulary_from_earlier_periods_only(self):
        fits = []
        real = MixedViewBuilder.fit

        def spy(self_, df, X, names, fit_mask, cat_sources):
            fits.append(np.asarray(fit_mask).copy())
            return real(self_, df, X, names, fit_mask, cat_sources)

        with mock.patch.object(MixedViewBuilder, "fit", spy):
            rows = self.run_families(families=("backtest",), backtest_origins=3, backtest_vae_origins=2)
        self.assertTrue(fits)
        period = self.ctx["period_idx"]
        for m in fits:
            origin = int(period[~m].min())
            self.assertTrue(np.array_equal(m, period < origin))
        self.assertTrue([r for r in rows if "(VAE)" in r["experiment"] and r["status"] == STATUS_EXECUTED])


class TestConsumersOfTheMixedDetector(MixedExperimentsBase):
    def test_explanations_use_original_names_with_observed_category_and_probability(self):
        from src.interpretability import explain_rows_vae
        out = explain_rows_vae(self.vae, self.x_mixed[self.split.oot_mask][:8], top_k=5)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(isinstance(s, str) and s for s in out))
        joined = " ".join(out)
        self.assertNotIn("emb", joined.replace("employment", ""))
        if "=" in joined:
            self.assertRegex(joined, r"\w+=\S+ \(p=0\.\d{3}\)")

    def test_reconstruction_error_by_feature_lists_original_variables(self):
        from src.interpretability import reconstruction_error_by_feature
        res = reconstruction_error_by_feature(self.vae, self.x_mixed, out_dir=os.path.join(self._tmpd.name, "fig"),
                                              categorical_columns=self.ctx["cat_sources"])
        self.assertEqual(set(res), set(self.vae.variable_names))

    def test_sensitivity_encodes_perturbed_frames_with_the_same_vocabularies(self):
        from src.evaluation.sensitivity import _stacked_matrix
        X = self.x_mixed[:5]
        det = mock.Mock()
        det.score_samples.return_value = np.arange(5, dtype=float)
        scaler = mock.Mock()
        scaler.transform.side_effect = lambda a: (a - 1.0)
        out = _stacked_matrix(X, X, {"detector": det, "scaler": scaler, "score_only_scaler": True})
        self.assertEqual(out.shape[1], X.shape[1] + 1)
        np.testing.assert_allclose(out[:, :-1], X)                                       # nothing else was rescaled
        np.testing.assert_allclose(out[:, -1], np.arange(5) - 1.0)


class TestConfigurationSection(unittest.TestCase):
    def test_vae_block_of_the_single_config_file(self):
        import main
        from src.utils.config_file import ConfigFileError, apply_config_file

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = os.path.join(d, "p.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("vae:\n  categorical_representation: embedding\n  categorical_embedding:\n"
                         "    dimension_strategy: fixed\n    dimension: 6\n    min_dimension: 2\n    max_dimension: 16\n"
                         "  reconstruction_loss:\n    numeric: mse\n    boolean: binary_cross_entropy\n"
                         "    categorical: cross_entropy\n    aggregate_by_original_feature: true\n"
                         "  loss_weights:\n    categorical: 2.0\n  unknown_category_policy: explicit_token\n"
                         "  missing_category_policy: explicit_token\n")
            cfg = main.PipelineConfig()
            apply_config_file(cfg, path)
            self.assertEqual(cfg.vae_categorical_representation, "embedding")
            self.assertEqual((cfg.vae_embedding_dimension_strategy, cfg.vae_embedding_dimension), ("fixed", 6))
            self.assertEqual((cfg.vae_embedding_min_dimension, cfg.vae_embedding_max_dimension), (2, 16))
            self.assertEqual((cfg.vae_numeric_loss, cfg.vae_weight_categorical), ("mse", 2.0))
            mixed = main._mixed_vae_config(cfg)
            self.assertEqual(mixed.embedding_dim_for(400), 6)
            for bad in ("vae:\n  categorical_representation: dummy\n", "vae:\n  loss_weights:\n    categorical: 0\n",
                        "vae:\n  unknown_category_policy: mode\n", "vae:\n  categorical_embedding:\n    typo: 1\n"):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(bad)
                with self.assertRaises(ConfigFileError, msg=bad):
                    apply_config_file(main.PipelineConfig(), path)

    def test_the_vae_representation_flag_is_registered_as_an_explicit_cli_value(self):
        import main
        parser = main.build_arg_parser()
        args = parser.parse_args(["--vae-categorical-representation", "embedding"])
        args._explicit_dests = main._explicit_dests(parser, ["--vae-categorical-representation", "embedding"])
        cfg = main.config_from_args(args)
        self.assertEqual(cfg.vae_categorical_representation, "embedding")
        self.assertIn("vae_categorical_representation", cfg.cli_explicit)


if __name__ == "__main__":
    unittest.main()
