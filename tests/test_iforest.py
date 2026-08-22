"""Validation suite for `src.models.iforest` (Isolation Forest detector + Optuna tuning).

Reuses the conftest session sandbox (cwd is a throwaway tmp dir) so any
relative-path defaults (configs/, models/, reports/figures/) never touch the
real repo. All tuning artifacts (sqlite db, best-params YAML, joblib model)
and figures are explicitly directed at tmp_path.

Panel is tiny (350 entities x 6 periods = 2100 rows) and n_trials is kept at
3-4 so the whole module runs in a few seconds.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from scipy.stats import spearmanr

from src.data import load_or_generate_panel
from src.models import (
    IsolationForestDetector,
    plot_score_distribution,
    tune_iforest,
)
from src.models.iforest import (
    _blocked_split,
    _detector_kwargs_from_params,
    _rank_agreement,
    _study_fingerprint,
)
from src.preprocessing import fit_transform_panel

N_ENTITIES = 350
N_PERIODS = 6
SEED = 20260724


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _read_ground_truth(path: str) -> pd.DataFrame:
    """Load the separate hidden ground-truth file (parquet or csv fallback)."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _align_y(keys: pd.DataFrame, gt: pd.DataFrame, schema) -> np.ndarray:
    """Join ground truth onto the preprocessing keys, row-for-row with X.

    A pandas left merge preserves the order of the left keys, so the returned
    label vector stays aligned with the feature-matrix rows.
    """
    k = keys.copy()
    k[schema.entity_col] = k[schema.entity_col].astype(str)
    k[schema.time_col] = pd.to_datetime(k[schema.time_col])

    g = gt.copy()
    g["entity_id"] = g["entity_id"].astype(str)
    g["period"] = pd.to_datetime(g["period"])
    # Robust to bool (parquet) or "True"/"False" strings (csv fallback).
    g["is_anomaly"] = g["is_anomaly"].astype(str).str.lower().isin(["true", "1"])

    merged = k.merge(
        g[["entity_id", "period", "is_anomaly"]],
        left_on=[schema.entity_col, schema.time_col],
        right_on=["entity_id", "period"],
        how="left",
    )
    return merged["is_anomaly"].fillna(False).astype(int).to_numpy()


def _to_1d(a) -> np.ndarray:
    return np.asarray(a).ravel()


# --------------------------------------------------------------------------- #
# Module fixtures (built once, shared)                                         #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel_and_schema(tmp_path_factory):
    dest = tmp_path_factory.mktemp("iforest_panel")
    df, schema = load_or_generate_panel(
        data_path=str(dest / "data.csv"),
        ground_truth_path=str(dest / "ground_truth.parquet"),
        n_individuals=N_ENTITIES, n_periods=N_PERIODS, seed=SEED,
    )
    return df, schema


@pytest.fixture(scope="module")
def ground_truth(panel_and_schema):
    _, schema = panel_and_schema
    assert schema.ground_truth_path is not None, "no ground-truth file was written"
    return _read_ground_truth(schema.ground_truth_path)


@pytest.fixture(scope="module")
def prep_dense(panel_and_schema):
    """Dense feature matrix (frequency encoding + standard scaling)."""
    df, schema = panel_and_schema
    X, keys, names = fit_transform_panel(
        df, schema,
        numeric_transform="standard",
        categorical_encoding="frequency",
        add_panel_features=True,
    )
    assert not sp.issparse(X), "expected a dense matrix from frequency encoding"
    return X, keys, names


@pytest.fixture(scope="module")
def y_labels(prep_dense, ground_truth, panel_and_schema):
    _, schema = panel_and_schema
    _, keys, _ = prep_dense
    y = _align_y(keys, ground_truth, schema)
    assert y.shape[0] == keys.shape[0]
    assert y.sum() > 0, "no injected anomalies landed in the panel; check the join"
    return y


@pytest.fixture(scope="module")
def fitted_default(prep_dense):
    X, _, names = prep_dense
    det = IsolationForestDetector(n_estimators=200, random_state=42)
    det.fit(X)
    return det, X, names


# --------------------------------------------------------------------------- #
# 1. Sign convention                                                           #
# --------------------------------------------------------------------------- #
class TestSignConvention:
    def test_anomalies_score_higher_and_scores_finite(self, fitted_default, y_labels):
        det, X, _ = fitted_default
        scores = det.score_samples(X)
        assert scores.shape[0] == X.shape[0]
        assert np.isfinite(scores).all(), "score_samples produced non-finite values"

        anom = scores[y_labels == 1]
        norm = scores[y_labels == 0]
        assert anom.size > 0 and norm.size > 0
        assert anom.mean() > norm.mean(), (
            f"mean anomaly score {anom.mean():.4f} !> mean normal score {norm.mean():.4f} "
            "(higher-is-more-anomalous convention violated)"
        )

    def test_score_samples_is_negated_sklearn_score(self, fitted_default):
        det, X, _ = fitted_default
        # Project convention: score_samples == -sklearn.score_samples.
        assert np.allclose(det.score_samples(X), -det.model_.score_samples(X))

    def test_decision_function_sign_opposes_score_samples(self, fitted_default, y_labels):
        det, X, _ = fitted_default
        dfun = det.decision_function(X)
        assert np.isfinite(dfun).all()
        # decision_function keeps sklearn semantics (negative = outlier), so
        # anomalies must have a LOWER mean decision_function than normals.
        assert dfun[y_labels == 1].mean() < dfun[y_labels == 0].mean()


# --------------------------------------------------------------------------- #
# 2. Shapes / predict semantics                                               #
# --------------------------------------------------------------------------- #
class TestShapesAndPredict:
    def test_score_length_matches_rows(self, fitted_default):
        det, X, _ = fitted_default
        assert det.score_samples(X).shape[0] == X.shape[0]
        assert det.decision_function(X).shape[0] == X.shape[0]

    def test_predict_is_binary_and_flags_contamination_fraction(self, prep_dense):
        X, _, _ = prep_dense
        det = IsolationForestDetector(
            n_estimators=150, contamination=0.05, random_state=42
        ).fit(X)
        pred = det.predict(X)
        assert pred.shape[0] == X.shape[0]
        assert set(np.unique(pred).tolist()) <= {0, 1}, "predict emitted values outside {0,1}"
        frac = float(pred.mean())
        assert 0.02 <= frac <= 0.09, (
            f"predict flagged {frac:.3%}; expected roughly the 5% contamination rate"
        )


# --------------------------------------------------------------------------- #
# 3. Sparse AND dense inputs accepted                                         #
# --------------------------------------------------------------------------- #
class TestSparseAndDense:
    def test_sparse_and_dense_both_fit_and_score(self, panel_and_schema):
        df, schema = panel_and_schema

        X_dense, _, _ = fit_transform_panel(
            df, schema, numeric_transform="standard", categorical_encoding="frequency"
        )
        assert not sp.issparse(X_dense)
        # Whether the one-hot ColumnTransformer returns sparse depends on the
        # density threshold (dense at this small scale), so exercise the sparse
        # code path explicitly by feeding a CSR view of the same matrix.
        X_sparse = sp.csr_matrix(X_dense)
        assert sp.issparse(X_sparse)

        for X in (X_sparse, X_dense):
            det = IsolationForestDetector(n_estimators=120, random_state=1).fit(X)
            s = det.score_samples(X)
            assert s.shape[0] == X.shape[0]
            assert np.isfinite(s).all()
            assert set(np.unique(det.predict(X)).tolist()) <= {0, 1}


# --------------------------------------------------------------------------- #
# 4. save / load round-trip                                                   #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_reloaded_detector_scores_identically(self, fitted_default, tmp_path):
        det, X, _ = fitted_default
        path = tmp_path / "iforest.joblib"
        returned = det.save(str(path))
        assert Path(returned).exists()

        reloaded = IsolationForestDetector.load(str(path))
        assert isinstance(reloaded, IsolationForestDetector)
        assert np.allclose(det.score_samples(X), reloaded.score_samples(X)), (
            "reloaded detector produced different scores"
        )


# --------------------------------------------------------------------------- #
# 5-7. Optuna tuning: resume, incremental YAML, supervised objective          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tuned_supervised(prep_dense, y_labels, tmp_path_factory):
    pytest.importorskip("optuna")
    X, _, _ = prep_dense
    workdir = tmp_path_factory.mktemp("tune_sup")
    db = workdir / "optuna_sup.db"
    storage = "sqlite:///" + str(db).replace("\\", "/")
    bp = workdir / "best_params.yaml"
    mo = workdir / "model.joblib"

    study1 = tune_iforest(
        X, n_trials=4, y=y_labels, storage=storage, study_name="resume_test",
        best_params_path=str(bp), model_out=str(mo), random_state=7,
    )
    n_after_first = len(study1.trials)

    # Second call, SAME storage + study_name -> must resume, not restart.
    study2 = tune_iforest(
        X, n_trials=3, y=y_labels, storage=storage, study_name="resume_test",
        best_params_path=str(bp), model_out=str(mo), random_state=7,
    )
    return {
        "study1_trials": n_after_first,
        "study2": study2,
        "best_params_path": bp,
        "model_out": mo,
    }


class TestOptunaResume:
    def test_second_run_resumes_and_accumulates_trials(self, tuned_supervised):
        assert tuned_supervised["study1_trials"] == 4, "first call did not run 4 trials"
        assert len(tuned_supervised["study2"].trials) == 7, (
            "study did not resume: expected 4 + 3 = 7 total trials, "
            f"got {len(tuned_supervised['study2'].trials)}"
        )


class TestIncrementalBestParams:
    _EXPECTED_KEYS = {
        "n_estimators", "max_samples", "max_features", "contamination", "bootstrap",
    }

    def test_best_params_yaml_written_with_tuned_keys(self, tuned_supervised):
        bp = tuned_supervised["best_params_path"]
        assert bp.exists(), "best_params YAML was never checkpointed"
        payload = yaml.safe_load(bp.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert "best_params" in payload, "no 'best_params' mapping in the YAML"
        best = payload["best_params"]
        assert isinstance(best, dict)
        assert set(best.keys()) == self._EXPECTED_KEYS, (
            f"best_params keys {set(best.keys())} != {self._EXPECTED_KEYS}"
        )

    def test_model_out_joblib_exists_and_loads(self, tuned_supervised):
        mo = tuned_supervised["model_out"]
        assert mo.exists(), "final refit model was not saved"
        det = IsolationForestDetector.load(str(mo))
        assert isinstance(det, IsolationForestDetector)
        assert det.model_ is not None, "loaded detector is not fitted"


class TestSupervisedObjective:
    def test_best_value_is_finite_pr_auc_in_unit_interval(self, tuned_supervised):
        best = tuned_supervised["study2"].best_value
        assert np.isfinite(best), "supervised best_value is not finite"
        assert 0.0 <= best <= 1.0, f"PR-AUC best_value {best} outside [0, 1]"


class TestUnsupervisedObjective:
    def test_unsupervised_runs_and_returns_finite_best_value(
        self, prep_dense, tmp_path
    ):
        pytest.importorskip("optuna")
        X, _, _ = prep_dense
        db = tmp_path / "optuna_unsup.db"
        storage = "sqlite:///" + str(db).replace("\\", "/")
        study = tune_iforest(
            X, n_trials=3, y=None, storage=storage, study_name="unsup_test",
            best_params_path=str(tmp_path / "unsup_best.yaml"),
            model_out=str(tmp_path / "unsup_model.joblib"),
            random_state=11,
        )
        assert len(study.trials) == 3
        assert np.isfinite(study.best_value), "unsupervised best_value is not finite"


# --------------------------------------------------------------------------- #
# 7b. Search space: contamination is not tunable                               #
# --------------------------------------------------------------------------- #
class TestContaminationIsNotSearched:
    """`contamination` cannot move a rank-based objective, so it is not searched.

    sklearn's `score_samples` ignores `offset_`; contamination only shifts the
    `decision_function`/`predict` threshold. Every objective here ranks by
    `score_samples`, and PR-AUC / ROC-AUC / Spearman are rank statistics --
    searching it burns TPE budget on a flat dimension and persists an arbitrary
    "best" value.
    """

    def test_contamination_absent_from_the_optuna_search_space(self, tuned_supervised):
        for trial in tuned_supervised["study2"].trials:
            assert "contamination" not in trial.params, (
                f"trial {trial.number} sampled contamination: {trial.params}"
            )

    def test_yaml_reports_contamination_as_a_fixed_operating_point(self, tuned_supervised):
        payload = yaml.safe_load(
            tuned_supervised["best_params_path"].read_text(encoding="utf-8")
        )
        assert "contamination" not in payload["raw_optuna_params"]
        assert payload["best_params"]["contamination"] == payload["contamination"]

    def test_scores_really_are_invariant_to_contamination(self, prep_dense):
        """The premise, asserted directly rather than assumed."""
        X, _, _ = prep_dense
        a = IsolationForestDetector(contamination=0.01, random_state=3).fit(X).score_samples(X)
        b = IsolationForestDetector(contamination=0.30, random_state=3).fit(X).score_samples(X)
        assert np.allclose(a, b), "score_samples changed with contamination"


# --------------------------------------------------------------------------- #
# 7c. Held-out objective, rank agreement, study fingerprint                    #
# --------------------------------------------------------------------------- #
class TestBlockedSplit:
    def test_split_is_disjoint_and_keeps_entities_whole(self, prep_dense, panel_and_schema):
        _, schema = panel_and_schema
        _, keys, _ = prep_dense
        groups = keys[schema.entity_col].to_numpy()
        fit_idx, eval_idx = _blocked_split(len(groups), groups, 0.3, 42)

        assert set(fit_idx).isdisjoint(eval_idx), "fit and eval rows overlap"
        assert len(fit_idx) + len(eval_idx) == len(groups)
        assert set(groups[fit_idx]).isdisjoint(groups[eval_idx]), (
            "an entity appears on both sides: rows of one customer share a "
            "latent level, so this leaks"
        )
        assert eval_idx.size > 0 and fit_idx.size > 0

    def test_falls_back_to_row_split_without_groups(self, prep_dense):
        X, _, _ = prep_dense
        fit_idx, eval_idx = _blocked_split(X.shape[0], None, 0.3, 42)
        assert set(fit_idx).isdisjoint(eval_idx)
        assert len(fit_idx) + len(eval_idx) == X.shape[0]


class TestRankAgreement:
    _KWARGS = dict(
        n_estimators=80, max_samples="auto", max_features=1.0,
        contamination=0.1, bootstrap=False,
    )

    def test_returns_zero_for_a_constant_score_vector(self):
        """The degeneracy guard: a collapsed model must not look 'perfectly
        stable'. This is exactly the pathology that makes a jitter-based
        stability proxy unusable as a selection objective."""
        X = np.ones((300, 5))
        fit_idx, eval_idx = _blocked_split(300, None, 0.3, 0)
        assert _rank_agreement(self._KWARGS, X, fit_idx, eval_idx, 0) == 0.0

    def test_is_positive_and_bounded_on_structured_data(self, prep_dense):
        X, _, _ = prep_dense
        fit_idx, eval_idx = _blocked_split(X.shape[0], None, 0.3, 0)
        value = _rank_agreement(self._KWARGS, X, fit_idx, eval_idx, 0)
        assert 0.0 < value <= 1.0, f"rank agreement {value} outside (0, 1]"

    def test_degenerate_index_sets_return_zero(self, prep_dense):
        X, _, _ = prep_dense
        assert _rank_agreement(self._KWARGS, X, np.arange(2), np.arange(2, 5), 0) == 0.0


class TestStudyFingerprint:
    def test_matrix_shape_changes_the_study_name(self, prep_dense):
        X, _, _ = prep_dense
        assert (
            _study_fingerprint(X, None, "supervised", "maximize")
            != _study_fingerprint(X[:50], None, "supervised", "maximize")
        )

    def test_feature_names_change_the_study_name(self, prep_dense):
        X, _, names = prep_dense
        other = [f"{n}_v2" for n in names]
        assert (
            _study_fingerprint(X, names, "supervised", "maximize")
            != _study_fingerprint(X, other, "supervised", "maximize")
        )

    def test_objective_mode_changes_the_study_name(self, prep_dense):
        X, _, _ = prep_dense
        assert (
            _study_fingerprint(X, None, "supervised", "maximize")
            != _study_fingerprint(X, None, "unsupervised", "maximize")
        )

    def test_tune_iforest_suffixes_the_study_name(self, prep_dense, y_labels, tmp_path):
        pytest.importorskip("optuna")
        X, _, _ = prep_dense
        storage = "sqlite:///" + str(tmp_path / "fp.db").replace("\\", "/")
        study = tune_iforest(
            X, n_trials=2, y=y_labels, storage=storage, study_name="iforest",
            best_params_path=str(tmp_path / "bp.yaml"),
            model_out=str(tmp_path / "m.joblib"), random_state=5,
        )
        assert study.study_name.startswith("iforest_")
        assert study.study_name != "iforest"

    def test_study_tag_overrides_the_fingerprint(self, prep_dense, y_labels, tmp_path):
        pytest.importorskip("optuna")
        X, _, _ = prep_dense
        storage = "sqlite:///" + str(tmp_path / "tag.db").replace("\\", "/")
        study = tune_iforest(
            X, n_trials=2, y=y_labels, storage=storage, study_name="iforest",
            best_params_path=str(tmp_path / "bp2.yaml"),
            model_out=str(tmp_path / "m2.joblib"), random_state=5,
            study_tag="explicit",
        )
        assert study.study_name == "iforest_explicit"


class TestObjectiveUsesHeldOutRows:
    def test_objective_scores_rows_the_trial_did_not_fit_on(
        self, prep_dense, y_labels, tmp_path
    ):
        """A trial fitted on every row would be scored in-sample; assert the
        objective value matches a held-out refit, not an in-sample one."""
        pytest.importorskip("optuna")
        from sklearn.metrics import average_precision_score

        X, _, _ = prep_dense
        storage = "sqlite:///" + str(tmp_path / "ho.db").replace("\\", "/")
        study = tune_iforest(
            X, n_trials=3, y=y_labels, storage=storage, study_name="holdout",
            best_params_path=str(tmp_path / "bp3.yaml"),
            model_out=str(tmp_path / "m3.joblib"), random_state=13,
        )
        best = study.best_trial
        kwargs = _detector_kwargs_from_params(best.params)
        fit_idx, eval_idx = _blocked_split(X.shape[0], None, 0.3, 13)

        det = IsolationForestDetector(random_state=13, n_jobs=-1, **kwargs)
        det.fit(X[fit_idx])
        held_out = float(
            average_precision_score(y_labels[eval_idx], det.score_samples(X[eval_idx]))
        )
        assert held_out == pytest.approx(best.value, abs=1e-9), (
            "objective value does not reproduce a held-out refit -- it is "
            "probably being computed in-sample"
        )


# --------------------------------------------------------------------------- #
# 7d. Theory guard: the IF is invariant to per-feature affine rescaling         #
# --------------------------------------------------------------------------- #
class TestAffineRescalingIsANoOp:
    """An IF split is `uniform(min_f, max_f)` on one feature. Under x -> a*x + b
    with a > 0 that uniform maps to the uniform of the rescaled range, so tree
    structure and the distribution of h(x) are unchanged. StandardScaler and
    RobustScaler are affine, hence provably neutral for this model.

    This is why the pipeline keeps scaling (it costs the IF nothing and the VAE
    needs it) and why "remove scaling to help the Isolation Forest" is a no-op.
    """

    def test_ranking_is_invariant_to_per_column_affine_rescaling(self, prep_dense):
        """The invariance is exact in real arithmetic, not in float64.

        A split is drawn as ``uniform(min_f, max_f)``; under ``a*x + b`` the
        equivalent draw is ``a*u + b``, and the two agree to within rounding.
        With a large shift relative to a column's own spread (here b up to +-50
        on columns whose range is ~0.3) that rounding can occasionally put a
        point on the other side of a cut, so the assertion is on the *ranking*
        -- which is what the score is used for -- rather than on bit equality.
        """
        X, _, _ = prep_dense
        Xd = np.asarray(X, dtype=float)
        rng = np.random.default_rng(0)
        a = rng.uniform(0.5, 20.0, size=Xd.shape[1])
        b = rng.normal(0.0, 50.0, size=Xd.shape[1])

        base = IsolationForestDetector(n_estimators=100, random_state=17).fit(Xd)
        scaled = IsolationForestDetector(n_estimators=100, random_state=17).fit(Xd * a + b)
        s_base = base.score_samples(Xd)
        s_scaled = scaled.score_samples(Xd * a + b)

        rho = spearmanr(s_base, s_scaled).correlation
        assert rho > 0.999, f"affine rescaling reordered the IF scores (spearman={rho})"

        # The decision surface -- the alert queue -- must be untouched.
        k = max(1, len(s_base) // 10)
        top_base = set(np.argsort(-s_base, kind="stable")[:k])
        top_scaled = set(np.argsort(-s_scaled, kind="stable")[:k])
        overlap = len(top_base & top_scaled) / k
        assert overlap == 1.0, f"top-decile changed under rescaling (overlap={overlap})"

    def test_pure_scaling_leaves_scores_numerically_identical(self, prep_dense):
        """Without the shift the arithmetic is benign, so equality does hold."""
        X, _, _ = prep_dense
        Xd = np.asarray(X, dtype=float)
        rng = np.random.default_rng(1)
        a = rng.uniform(0.9, 1.1, size=Xd.shape[1])

        base = IsolationForestDetector(n_estimators=100, random_state=17).fit(Xd)
        scaled = IsolationForestDetector(n_estimators=100, random_state=17).fit(Xd * a)
        assert np.allclose(
            base.score_samples(Xd), scaled.score_samples(Xd * a), atol=1e-9
        ), "pure per-column scaling changed the IF scores"


# --------------------------------------------------------------------------- #
# 8. Reproducibility                                                          #
# --------------------------------------------------------------------------- #
class TestReproducibility:
    def test_same_random_state_same_scores(self, prep_dense):
        X, _, _ = prep_dense
        a = IsolationForestDetector(n_estimators=120, random_state=99).fit(X).score_samples(X)
        b = IsolationForestDetector(n_estimators=120, random_state=99).fit(X).score_samples(X)
        assert np.allclose(a, b), "identical random_state produced different scores"


# --------------------------------------------------------------------------- #
# 9. plot_score_distribution                                                  #
# --------------------------------------------------------------------------- #
class TestPlotScoreDistribution:
    def test_writes_png_under_reports_figures(self, fitted_default, y_labels, tmp_path):
        pytest.importorskip("matplotlib")
        det, X, _ = fitted_default
        scores = det.score_samples(X)
        out_dir = tmp_path / "reports" / "figures" / "iforest"
        out_path = None
        try:
            out_path = plot_score_distribution(
                scores, out_dir=str(out_dir),
                filename="iforest_score_distribution.png", y=y_labels,
            )
            parts = Path(out_path).parts
            assert "reports" in parts and "figures" in parts
            assert os.path.exists(out_path)
            assert out_path.lower().endswith(".png")
        finally:
            if out_path and os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
