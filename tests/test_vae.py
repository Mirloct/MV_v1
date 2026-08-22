"""Validation suite for `src.models.vae` (Variational Autoencoder detector).

Mirrors the iForest suite. Reuses the conftest session sandbox; all
checkpoints, sqlite dbs, best-params YAML and figures are directed at
tmp_path so nothing lands in the real repo. Tiny panel / epochs / trials keep
it fast (torch CPU).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import yaml

from src.data import load_or_generate_panel
from src.models import (
    VAEDetector,
    VAEModel,
    plot_latent_space,
    plot_reconstruction_error,
    tune_vae,
    vae_loss,
)
from src.preprocessing import fit_transform_panel

N_ENTITIES = 350
N_PERIODS = 6
SEED = 20260724


def _read_gt(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def _align_y(keys, gt, schema) -> np.ndarray:
    k = keys.copy()
    k[schema.entity_col] = k[schema.entity_col].astype(str)
    k[schema.time_col] = pd.to_datetime(k[schema.time_col])
    g = gt.copy()
    g["entity_id"] = g["entity_id"].astype(str)
    g["period"] = pd.to_datetime(g["period"])
    g["is_anomaly"] = g["is_anomaly"].astype(str).str.lower().isin(["true", "1"])
    merged = k.merge(g[["entity_id", "period", "is_anomaly"]],
                     left_on=[schema.entity_col, schema.time_col],
                     right_on=["entity_id", "period"], how="left")
    return merged["is_anomaly"].fillna(False).astype(int).to_numpy()


@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    dest = tmp_path_factory.mktemp("vae_panel")
    df, schema = load_or_generate_panel(
        data_path=str(dest / "data.csv"),
        ground_truth_path=str(dest / "ground_truth.parquet"),
        n_individuals=N_ENTITIES, n_periods=N_PERIODS, seed=SEED,
    )
    X, keys, names = fit_transform_panel(
        df, schema, numeric_transform="yeo-johnson", categorical_encoding="frequency")
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    gt = _read_gt(schema.ground_truth_path)
    y = _align_y(keys, gt, schema)
    return X.astype(np.float32), keys, names, y


# --------------------------------------------------------------------------- #
class TestVAEModelAndLoss:
    def test_forward_shapes_and_loss_decomposition(self):
        torch.manual_seed(0)
        m = VAEModel(input_dim=12, latent_dim=4, hidden_dim=16, n_layers=2)
        x = torch.randn(20, 12)
        x_recon, mu, logvar = m(x)
        assert x_recon.shape == x.shape
        assert mu.shape == (20, 4) and logvar.shape == (20, 4)
        total, recon, kl = vae_loss(x, x_recon, mu, logvar, beta=1.0)
        assert torch.isfinite(total) and torch.isfinite(recon) and torch.isfinite(kl)
        # total == recon + beta*kl (beta=1)
        assert torch.allclose(total, recon + kl, atol=1e-4)
        # beta scales the KL contribution
        total2, recon2, kl2 = vae_loss(x, x_recon, mu, logvar, beta=2.0)
        assert torch.allclose(total2, recon2 + 2.0 * kl2, atol=1e-4)


# --------------------------------------------------------------------------- #
class TestFitScore:
    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls, prep, tmp_path_factory):
        X, _, _, y = prep
        ck = tmp_path_factory.mktemp("vae_fit")
        det = VAEDetector(latent_dim=6, hidden_dim=32, n_layers=2, epochs=8,
                          batch_size=128, random_state=0)
        det.fit(X, checkpoint_dir=str(ck), resume=False)
        return det, X, y, ck

    def test_scores_finite_and_rank_anomalies_higher(self, fitted):
        det, X, y, _ = fitted
        s = det.score_samples(X)
        assert s.shape[0] == X.shape[0]
        assert np.isfinite(s).all()
        assert s[y == 1].mean() > s[y == 0].mean(), "recon-error did not rank anomalies higher"
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y, s) > 0.65, "VAE ROC-AUC unexpectedly low on injected anomalies"

    def test_encode_returns_latent_means(self, fitted):
        det, X, _, _ = fitted
        Z = np.asarray(det.encode(X))
        assert Z.shape == (X.shape[0], 6)
        assert np.isfinite(Z).all()

    def test_save_load_identical(self, fitted, tmp_path):
        det, X, _, _ = fitted
        p = tmp_path / "vae.pt"
        det.save(str(p))
        assert p.exists()
        det2 = VAEDetector.load(str(p))
        assert np.allclose(det.score_samples(X), det2.score_samples(X), atol=1e-5)

    def test_checkpoint_written(self, fitted):
        _, _, _, ck = fitted
        assert (Path(ck) / "checkpoint.pth").exists()


# --------------------------------------------------------------------------- #
class TestCheckpointResume:
    def test_resume_advances_epoch_state_not_restart(self, prep, tmp_path):
        X, _, _, _ = prep
        ck = tmp_path / "resume_ck"
        # first training run: 4 epochs
        VAEDetector(latent_dim=4, hidden_dim=16, n_layers=1, epochs=4,
                    batch_size=128, random_state=0).fit(X, checkpoint_dir=str(ck), resume=False)
        ckpt1 = torch.load(str(ck / "checkpoint.pth"), map_location="cpu", weights_only=False)
        epoch_after_first = int(ckpt1["epoch"])
        assert "config" in ckpt1 and "best_val_loss" in ckpt1
        # resume with a larger epoch budget: must continue, ending at a higher epoch
        VAEDetector(latent_dim=4, hidden_dim=16, n_layers=1, epochs=8,
                    batch_size=128, random_state=0).fit(X, checkpoint_dir=str(ck), resume=True)
        ckpt2 = torch.load(str(ck / "checkpoint.pth"), map_location="cpu", weights_only=False)
        assert int(ckpt2["epoch"]) > epoch_after_first, "resume did not advance epoch state"


# --------------------------------------------------------------------------- #
class TestSparse:
    def test_sparse_input_densified_and_scored(self, prep, tmp_path):
        X, _, _, _ = prep
        Xs = sp.csr_matrix(X)
        det = VAEDetector(latent_dim=4, hidden_dim=16, n_layers=1, epochs=3,
                          batch_size=128, random_state=0)
        det.fit(Xs, checkpoint_dir=str(tmp_path / "sp_ck"), resume=False)
        s = det.score_samples(Xs)
        assert s.shape[0] == X.shape[0] and np.isfinite(s).all()


# --------------------------------------------------------------------------- #
class TestTuneSupervised:
    @pytest.fixture(scope="class")
    @classmethod
    def tuned(cls, prep, tmp_path_factory):
        pytest.importorskip("optuna")
        X, _, _, y = prep
        wd = tmp_path_factory.mktemp("vae_tune")
        storage = "sqlite:///" + str(wd / "vae.db").replace("\\", "/")
        bp, mo = wd / "bp.yaml", wd / "m.pt"
        st1 = tune_vae(X, n_trials=2, y=y, storage=storage, study_name="vt",
                       best_params_path=str(bp), model_out=str(mo),
                       checkpoint_dir=str(wd / "ck"), random_state=1, max_epochs=3)
        n1 = len(st1.trials)
        st2 = tune_vae(X, n_trials=2, y=y, storage=storage, study_name="vt",
                       best_params_path=str(bp), model_out=str(mo),
                       checkpoint_dir=str(wd / "ck"), random_state=1, max_epochs=3)
        return n1, st2, bp, mo

    def test_resume_accumulates_trials(self, tuned):
        n1, st2, _, _ = tuned
        assert n1 == 2
        assert len(st2.trials) == 4, f"study did not resume: {len(st2.trials)} != 4"

    def test_best_params_yaml_and_model_written(self, tuned):
        _, st2, bp, mo = tuned
        assert bp.exists() and mo.exists()
        payload = yaml.safe_load(bp.read_text(encoding="utf-8"))
        best = payload["best_params"]
        for k in ("latent_dim", "hidden_dim", "n_layers", "dropout", "beta",
                  "lr", "optimizer", "batch_size"):
            assert k in best, f"missing tuned param {k}"
        assert VAEDetector.load(str(mo)) is not None

    def test_best_value_is_pr_auc_in_unit_interval(self, tuned):
        _, st2, _, _ = tuned
        assert 0.0 <= st2.best_value <= 1.0


# --------------------------------------------------------------------------- #
class TestTuneUnsupervised:
    def test_unsupervised_minimizes_finite_recon_loss(self, prep, tmp_path):
        pytest.importorskip("optuna")
        X, _, _, _ = prep
        storage = "sqlite:///" + str(tmp_path / "u.db").replace("\\", "/")
        st = tune_vae(X, n_trials=2, y=None, storage=storage, study_name="vu",
                      best_params_path=str(tmp_path / "bpu.yaml"),
                      model_out=str(tmp_path / "u.pt"),
                      checkpoint_dir=str(tmp_path / "uck"), random_state=2, max_epochs=3)
        assert len(st.trials) == 2
        assert st.direction == optuna_minimize(), "unsupervised study should minimize recon loss"
        assert np.isfinite(st.best_value)


def optuna_minimize():
    import optuna
    return optuna.study.StudyDirection.MINIMIZE


# --------------------------------------------------------------------------- #
class TestKLAnnealingAndEarlyStopping:
    """Fase 4 anti-collapse: ramp the KL weight, then stop when val stalls.

    The ELBO has a trivial optimum where the encoder returns the prior (KL = 0)
    and the decoder emits the dataset mean. The loss looks fine while the latent
    code carries nothing, so every row reconstructs equally badly and the anomaly
    score becomes noise. Ramping beta lets the decoder become useful before the
    full KL pressure arrives.
    """

    def test_beta_ramps_linearly_then_holds(self):
        det = VAEDetector(beta=1.0, kl_anneal_epochs=10)
        betas = [det._annealed_beta(e) for e in range(12)]
        assert betas[0] == pytest.approx(0.1)
        assert betas[9] == pytest.approx(1.0)
        assert betas[11] == pytest.approx(1.0)
        assert all(b2 >= b1 for b1, b2 in zip(betas, betas[1:])), "ramp must be monotone"

    def test_zero_anneal_epochs_disables_the_ramp(self):
        det = VAEDetector(beta=2.0, kl_anneal_epochs=0)
        assert det._annealed_beta(0) == pytest.approx(2.0)

    def test_history_records_the_beta_actually_used(self, prep, tmp_path):
        X, *_ = prep
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=4,
                          kl_anneal_epochs=4, random_state=0)
        det.fit(X, checkpoint_dir=str(tmp_path / "beta"), resume=False)
        betas = [h["beta"] for h in det.history_]
        assert betas == sorted(betas) and betas[-1] == pytest.approx(det.beta)

    def test_early_stopping_halts_before_the_epoch_budget(self, prep, tmp_path):
        X, *_ = prep
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=60,
                          kl_anneal_epochs=2, early_stopping_patience=2,
                          random_state=0)
        det.fit(X, checkpoint_dir=str(tmp_path / "es"), resume=False)
        assert len(det.history_) < 60, "early stopping never fired"
        assert np.isfinite(det.score_samples(X)).all()

    def test_patience_none_runs_the_full_budget(self, prep, tmp_path):
        X, *_ = prep
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=3,
                          kl_anneal_epochs=1, early_stopping_patience=None,
                          random_state=0)
        det.fit(X, checkpoint_dir=str(tmp_path / "nopat"), resume=False)
        assert len(det.history_) == 3


class TestTemporalValidation:
    """Fase 2/4: validation must be *later* periods, not shuffled rows."""

    @staticmethod
    def _mask(keys, schema):
        last = sorted(keys[schema.time_col].unique())[-2:]
        return keys[schema.time_col].isin(last).to_numpy()

    def test_valid_mask_is_used_and_scores_stay_finite(self, prep, tmp_path):
        X, keys, _, _ = prep
        schema = type("S", (), {"time_col": keys.columns[1]})
        vm = self._mask(keys, schema)
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=3,
                          random_state=0)
        det.fit(X, checkpoint_dir=str(tmp_path / "tv"), resume=False, valid_mask=vm)
        assert all(h["val_loss"] is not None for h in det.history_)
        assert np.isfinite(det.score_samples(X)).all()

    def test_degenerate_masks_are_rejected(self, prep, tmp_path):
        X, *_ = prep
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=1)
        for bad in (np.ones(len(X), bool), np.zeros(len(X), bool)):
            with pytest.raises(ValueError, match="some -- but not all"):
                det.fit(X, checkpoint_dir=str(tmp_path / "bad"), resume=False,
                        valid_mask=bad)

    def test_wrong_length_mask_is_rejected(self, prep, tmp_path):
        X, *_ = prep
        det = VAEDetector(latent_dim=4, hidden_dim=32, n_layers=1, epochs=1)
        with pytest.raises(ValueError, match="valid_mask has"):
            det.fit(X, checkpoint_dir=str(tmp_path / "len"), resume=False,
                    valid_mask=np.ones(7, bool))


class TestFigures:
    def test_figures_written_under_reports_figures(self, prep, tmp_path):
        pytest.importorskip("matplotlib")
        X, _, _, y = prep
        det = VAEDetector(latent_dim=4, hidden_dim=16, n_layers=1, epochs=3,
                          batch_size=128, random_state=0)
        det.fit(X, checkpoint_dir=str(tmp_path / "fck"), resume=False)
        out = tmp_path / "reports" / "figures" / "vae"
        paths = []
        try:
            paths.append(plot_reconstruction_error(det.score_samples(X), out_dir=str(out), y=y))
            paths.append(plot_latent_space(det, X, out_dir=str(out), y=y))
            for p in paths:
                assert os.path.exists(p) and p.lower().endswith(".png")
                assert "reports" in Path(p).parts and "figures" in Path(p).parts
        finally:
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
