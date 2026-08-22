"""Validation suite for `src.models.stacking` (the IF -> VAE arrangement).

Reuses the conftest session sandbox. The panel is small and the VAE budgets are
tiny so the module runs in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.data import load_or_generate_panel
from src.evaluation import chronological_split
from src.models import IsolationForestDetector
from src.models.stacking import (
    DEFAULT_SCORE_FEATURE,
    build_stacked_matrix,
    score_shift_report,
)
from src.preprocessing import fit_transform_panel

N_ENTITIES = 250
N_PERIODS = 12
SEED = 20260816


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    dest = tmp_path_factory.mktemp("stack_panel")
    df, schema = load_or_generate_panel(
        data_path=str(dest / "data.csv"),
        ground_truth_path=str(dest / "ground_truth.parquet"),
        n_individuals=N_ENTITIES, n_periods=N_PERIODS, seed=SEED,
    )
    split = chronological_split(df, schema.time_col, n_val_periods=2, n_test_periods=3)
    X, keys, names = fit_transform_panel(df, schema, fit_mask=split.train_mask)
    X = np.asarray(X, dtype=float)
    forest = IsolationForestDetector(n_estimators=100, random_state=0).fit(X[split.train_mask])
    return {
        "X": X, "keys": keys, "names": names, "split": split,
        "scores": forest.score_samples(X),
    }


class TestBuildStackedMatrix:
    def test_appends_exactly_one_column_named_last(self, prepared):
        st = build_stacked_matrix(
            prepared["X"], prepared["scores"],
            fit_mask=prepared["split"].train_mask, feature_names=prepared["names"],
        )
        assert st.X.shape[1] == prepared["X"].shape[1] + 1
        assert st.X.shape[0] == prepared["X"].shape[0]
        assert st.feature_names[-1] == DEFAULT_SCORE_FEATURE
        assert len(st.feature_names) == st.X.shape[1]
        assert st.n_original == prepared["X"].shape[1]

    def test_column_order_is_identical_across_blocks(self, prepared):
        """The whole point of building one matrix and slicing it: train, val and
        test cannot drift apart because they are literally the same array."""
        st = build_stacked_matrix(
            prepared["X"], prepared["scores"],
            fit_mask=prepared["split"].train_mask, feature_names=prepared["names"],
        )
        sp_ = prepared["split"]
        blocks = [st.X[m] for m in (sp_.train_mask, sp_.val_mask, sp_.test_mask)]
        assert {b.shape[1] for b in blocks} == {st.X.shape[1]}
        assert sum(b.shape[0] for b in blocks) == st.X.shape[0]

    def test_scaler_is_fitted_on_the_train_block_only(self, prepared):
        st = build_stacked_matrix(
            prepared["X"], prepared["scores"],
            fit_mask=prepared["split"].train_mask, feature_names=prepared["names"],
        )
        train_block = st.X[prepared["split"].train_mask]
        # A StandardScaler fitted on train leaves *train* at mean 0 / std 1.
        assert np.allclose(train_block.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(train_block.std(axis=0), 1.0, atol=1e-6)

    def test_output_is_finite(self, prepared):
        st = build_stacked_matrix(
            prepared["X"], prepared["scores"], fit_mask=prepared["split"].train_mask)
        assert np.isfinite(st.X).all()

    def test_standardise_false_leaves_the_raw_score(self, prepared):
        st = build_stacked_matrix(
            prepared["X"], prepared["scores"],
            fit_mask=prepared["split"].train_mask, standardise=False)
        assert st.scaler is None
        assert np.allclose(st.X[:, -1], prepared["scores"])

    def test_sparse_input_stays_sparse(self, prepared):
        Xs = sp.csr_matrix(prepared["X"])
        st = build_stacked_matrix(
            Xs, prepared["scores"], fit_mask=prepared["split"].train_mask)
        assert sp.issparse(st.X)
        assert st.X.shape[1] == Xs.shape[1] + 1

    def test_non_finite_scores_are_repaired(self, prepared):
        bad = prepared["scores"].copy()
        bad[:5] = np.nan
        bad[5] = np.inf
        st = build_stacked_matrix(
            prepared["X"], bad, fit_mask=prepared["split"].train_mask)
        assert np.isfinite(st.X).all()

    def test_length_and_mask_mismatches_raise(self, prepared):
        with pytest.raises(ValueError, match="detector_scores has"):
            build_stacked_matrix(prepared["X"], np.zeros(7))
        with pytest.raises(ValueError, match="fit_mask has"):
            build_stacked_matrix(
                prepared["X"], prepared["scores"], fit_mask=np.ones(7, bool))
        with pytest.raises(ValueError, match="selects no rows"):
            build_stacked_matrix(
                prepared["X"], prepared["scores"],
                fit_mask=np.zeros(len(prepared["X"]), bool))


class TestScoreShiftReport:
    """The in-sample hazard has to be measured, not assumed."""

    def test_reports_a_shift_per_block(self, prepared):
        sp_ = prepared["split"]
        rep = score_shift_report(
            prepared["scores"], sp_.train_mask,
            {"validation": sp_.val_mask, "test": sp_.test_mask},
        )
        assert set(rep) == {"fit", "validation", "test"}
        for block in ("validation", "test"):
            for key in ("mean", "std", "shift", "n"):
                assert key in rep[block]
        assert rep["fit"]["shift"] == 0.0

    def test_detects_the_in_sample_shift_this_pipeline_actually_has(self, prepared):
        """A forest scores its own training rows lower than unseen ones. This is
        the documented cost of reusing one forest across the three sets; the
        assertion pins that it is *detected*, not that it is small."""
        sp_ = prepared["split"]
        rep = score_shift_report(
            prepared["scores"], sp_.train_mask, {"test": sp_.test_mask})
        assert rep["test"]["shift"] > 0.0, (
            "held-out rows should score as more anomalous than in-sample ones"
        )

    def test_identical_distributions_report_no_shift(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=2000)
        mask = np.zeros(2000, bool)
        mask[:1000] = True
        rep = score_shift_report(s, mask, {"other": ~mask})
        assert abs(rep["other"]["shift"]) < 0.2
