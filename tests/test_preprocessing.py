"""Validation suite for `src.preprocessing` (pipeline + transform diagnostics).

Turns the orchestrator's one-off smoke test into a permanent regression suite
and probes the edge cases the smoke test skipped (all-NaN / constant columns,
unknown categories at transform time, joblib round-trips, per-transform
finiteness). Reuses the conftest sandbox: relative-path defaults (logs/,
reports/figures/) land in a throwaway tmp dir and never touch the real repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data import PanelSchema, load_or_generate_panel
from src.preprocessing import (
    NUMERIC_TRANSFORMS,
    PanelFeatureEngineer,
    SignedLog1p,
    build_preprocessing_pipeline,
    categorical_feature_mask,
    compute_transform_diagnostics,
    fit_transform_panel,
    infer_numeric_features,
    make_numeric_transformer,
    plot_transform_diagnostics,
    recommend_transform,
    split_matrix_for_model,
)
from src.preprocessing.pipeline import (
    _DEFAULT_RATIO_FEATURES,
    AutoNumericTransformer,
    CATEGORICAL_ENCODINGS,
)

N_ENTITIES = 300
N_PERIODS = 6
SEED = 20260724
N_ROWS = N_ENTITIES * N_PERIODS
RATIO_NAMES = [name for _, _, name in _DEFAULT_RATIO_FEATURES]


def _fitted_auto_step(panel, schema) -> AutoNumericTransformer:
    """Fit the pipeline in "auto" mode and return its numeric transform step."""
    pipe = build_preprocessing_pipeline(schema, numeric_transform="auto")
    pipe.fit(panel)
    step = pipe.named_steps["column_transform"].named_transformers_["num"]
    return step.named_steps["transform"]

DIAG_COLUMNS = {
    "feature", "transform", "skewness", "excess_kurtosis", "jb_effect",
    "normaltest_stat", "normaltest_pvalue",
}


def _nonfinite_count(X) -> int:
    if sp.issparse(X):
        return int((~np.isfinite(X.tocsr().data)).sum())
    return int((~np.isfinite(np.asarray(X, dtype=np.float64))).sum())


def _n_rows(X) -> int:
    return X.shape[0]


def _width(X) -> int:
    return X.shape[1]


def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def _column(X, names, name) -> np.ndarray:
    idx = names.index(name)
    if sp.issparse(X):
        return np.asarray(X[:, idx].todense(), dtype=np.float64).ravel()
    return np.asarray(X, dtype=np.float64)[:, idx]


@pytest.fixture(scope="module")
def panel_and_schema(tmp_path_factory):
    dest = tmp_path_factory.mktemp("prep_panel")
    df, schema = load_or_generate_panel(
        data_path=str(dest / "data.csv"),
        ground_truth_path=str(dest / "ground_truth.parquet"),
        n_individuals=N_ENTITIES, n_periods=N_PERIODS, seed=SEED,
    )
    return df, schema


@pytest.fixture(scope="module")
def panel(panel_and_schema):
    return panel_and_schema[0]


@pytest.fixture(scope="module")
def schema(panel_and_schema):
    return panel_and_schema[1]


# ====================================================================== 1 ==
class TestNumericTransforms:
    @pytest.mark.parametrize("transform", NUMERIC_TRANSFORMS)
    def test_transform_produces_finite_aligned_matrix(self, panel, schema, transform):
        X, keys, names = fit_transform_panel(panel, schema, numeric_transform=transform)
        nf = _nonfinite_count(X)
        assert nf == 0, f"{transform}: {nf} non-finite values in the feature matrix"
        assert _width(X) == len(names), (
            f"{transform}: width {_width(X)} != len(names) {len(names)}"
        )
        assert _n_rows(X) == len(keys) == len(panel), (
            f"{transform}: rows X={_n_rows(X)} keys={len(keys)} df={len(panel)} disagree"
        )
        assert _width(X) > 0, f"{transform}: zero-width matrix"


# ====================================================================== 2 ==
class TestCategoricalEncodings:
    @pytest.mark.parametrize("encoding", CATEGORICAL_ENCODINGS)
    def test_encoding_runs_and_is_finite(self, panel, schema, encoding):
        X, keys, names = fit_transform_panel(panel, schema, categorical_encoding=encoding)
        assert _nonfinite_count(X) == 0, f"{encoding}: non-finite values"
        assert _width(X) == len(names)
        assert _n_rows(X) == len(panel)
        assert _width(X) > 0


class TestPerModelFeatureRouting:
    """Categoricals are withheld from the Isolation Forest, kept for the VAE.

    The split is driven purely by source dtype (object/category -> the
    ColumnTransformer's `cat__` branch), never by a hardcoded column list.
    """

    def test_iforest_view_drops_categoricals_vae_keeps_them(self, panel, schema):
        X, _keys, names = fit_transform_panel(panel, schema)
        cat_mask = categorical_feature_mask(names)
        assert cat_mask.any(), "fixture panel should contain categorical columns"

        X_if, names_if = split_matrix_for_model(X, names, "iforest")
        X_vae, names_vae = split_matrix_for_model(X, names, "vae")

        assert len(names_if) == len(names) - int(cat_mask.sum())
        assert not categorical_feature_mask(names_if).any(), "categorical leaked to iForest"
        assert _width(X_if) == len(names_if)
        # The VAE view is the untouched full matrix.
        assert names_vae == list(names)
        assert _width(X_vae) == len(names)

    def test_iforest_view_preserves_rows_and_numeric_values(self, panel, schema):
        X, _keys, names = fit_transform_panel(panel, schema)
        X_if, names_if = split_matrix_for_model(X, names, "iforest")
        assert _n_rows(X_if) == _n_rows(X) == len(panel)
        assert _nonfinite_count(X_if) == 0
        # Column values must be carried through unchanged, not recomputed:
        # compare the first retained feature against its source column.
        keep_idx = [i for i, n in enumerate(names) if not str(n).startswith("cat__")]
        dense_full = X.toarray() if sp.issparse(X) else np.asarray(X)
        dense_if = X_if.toarray() if sp.issparse(X_if) else np.asarray(X_if)
        np.testing.assert_allclose(dense_if, dense_full[:, keep_idx])

    def test_routing_follows_dtype_not_column_name(self, panel, schema):
        """A brand-new string column is routed to the categorical branch with
        no code change -- and a numeric column with a 'category-ish' name is
        not."""
        df = panel.copy()
        df["some_new_text_field"] = np.where(df.index % 2 == 0, "alpha", "beta")
        df["category_score_numeric"] = np.linspace(0.0, 1.0, len(df))

        X, _keys, names = fit_transform_panel(df, schema)
        cat_names = [n for n, c in zip(names, categorical_feature_mask(names)) if c]
        assert any("some_new_text_field" in n for n in cat_names), (
            "new object-dtype column was not routed to the categorical branch"
        )
        assert not any("category_score_numeric" in n for n in cat_names), (
            "numeric column was misrouted based on its name"
        )
        _X_if, names_if = split_matrix_for_model(X, names, "iforest")
        assert not any("some_new_text_field" in n for n in names_if)
        assert any("category_score_numeric" in n for n in names_if), (
            "numeric column must remain available to the Isolation Forest"
        )


# ====================================================================== 3 ==
class TestKeyLabelSeparation:
    def test_keys_excluded_from_features_and_row_aligned(self, panel, schema):
        X, keys, names = fit_transform_panel(panel, schema)
        assert schema.entity_col not in names, "entity_col leaked into feature_names"
        assert schema.time_col not in names, "time_col leaked into feature_names"
        assert list(keys.columns) == [schema.entity_col, schema.time_col]
        assert len(keys) == _n_rows(X) == len(panel)
        expected = panel[[schema.entity_col, schema.time_col]].reset_index(drop=True)
        assert keys.reset_index(drop=True).equals(expected), (
            "keys are not a row-for-row copy of the panel key columns"
        )


# ====================================================================== 4 ==
class TestMissingIndicators:
    def test_indicators_present_and_carry_signal(self, panel, schema):
        X, _, names = fit_transform_panel(
            panel, schema, categorical_encoding="frequency",
            numeric_transform="standard", add_missing_indicators=True,
        )
        assert _nonfinite_count(X) == 0, "imputation left NaNs even with indicators"
        indicator_names = [n for n in names if n.endswith("__missing")]
        assert indicator_names, "add_missing_indicators=True produced no indicators"
        target = next((n for n in indicator_names if "credit_score" in n), None)
        assert target is not None, f"no credit_score indicator among {indicator_names}"
        uniq = set(np.unique(_column(X, names, target)).tolist())
        assert uniq == {0.0, 1.0}, f"{target} carries no signal; values={sorted(uniq)}"

    def test_no_indicators_when_disabled(self, panel, schema):
        X, _, names = fit_transform_panel(
            panel, schema, categorical_encoding="frequency",
            numeric_transform="standard", add_missing_indicators=False,
        )
        leaked = [n for n in names if n.endswith("__missing")]
        assert leaked == [], f"add_missing_indicators=False still emitted {leaked}"
        assert _nonfinite_count(X) == 0


# ====================================================================== 5 ==
class TestPanelFeatures:
    def test_panel_features_present_and_widen_matrix(self, panel, schema):
        X_on, _, names_on = fit_transform_panel(panel, schema, add_panel_features=True)
        X_off, _, names_off = fit_transform_panel(panel, schema, add_panel_features=False)
        assert _nonfinite_count(X_on) == 0, "panel features leaked NaN/inf"
        for suffix in ("_lag1", "_diff1", "_own_z", "_month_sin", "_month_cos"):
            assert any(suffix in n for n in names_on), f"missing '*{suffix}'"
        assert _width(X_on) > _width(X_off), (
            f"panel features did not widen matrix ({_width(X_on)} !> {_width(X_off)})"
        )
        for suffix in ("_lag1", "_diff1", "_own_z", "_month_sin", "_month_cos"):
            assert not any(suffix in n for n in names_off), f"'*{suffix}' present when off"

    def test_lag_and_diff_reflect_within_entity_history(self):
        periods = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        df = pd.DataFrame({
            "entity_id": ["A", "A", "A", "B", "B", "B"],
            "period": list(periods) * 2,
            "account_balance": [10.0, 20.0, 35.0, 100.0, 250.0, 400.0],
        })
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        out = eng.fit_transform(df)
        assert out["account_balance_lag1"].tolist() == [0.0, 10.0, 20.0, 0.0, 100.0, 250.0]
        assert out["account_balance_diff1"].tolist() == [0.0, 10.0, 15.0, 0.0, 150.0, 150.0]
        assert np.isfinite(out["account_balance_own_z"].to_numpy()).all()
        assert out.index.tolist() == list(range(6))

    def test_lag_respects_time_order_when_rows_are_shuffled(self):
        periods = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        df = pd.DataFrame({
            "entity_id": ["A", "A", "A"],
            "period": [periods[2], periods[0], periods[1]],
            "account_balance": [30.0, 10.0, 20.0],
        })
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        out = eng.fit_transform(df)
        assert out["account_balance_lag1"].tolist() == [20.0, 0.0, 10.0]


# ===================================================================== 5b ==
class TestRatioFeatures:
    """Ratios give the axis-parallel splitter an axis to split on.

    An Isolation Forest cuts at `uniform(min_f, max_f)` of one feature, so it can
    only express unions of axis-parallel boxes. A ratio X/Y is constant along a
    ray through the origin -- an oblique boundary a box can only approximate with
    a staircase, at one split per stair. Materialising the ratio makes that
    structure axis-aligned.
    """

    def test_all_four_ratios_are_present_and_matrix_stays_aligned(self, panel, schema):
        X, _, names = fit_transform_panel(panel, schema)
        for expected in RATIO_NAMES:
            assert any(n.endswith(expected) for n in names), f"missing ratio {expected}"
        assert _width(X) == len(names)
        assert _nonfinite_count(X) == 0

    def test_ratios_widen_the_matrix_by_exactly_four_columns(self, panel, schema):
        X_on, _, _ = fit_transform_panel(panel, schema)
        X_off, _, names_off = fit_transform_panel(panel, schema, ratio_features=())
        assert _width(X_on) - _width(X_off) == len(RATIO_NAMES)
        for expected in RATIO_NAMES:
            assert not any(n.endswith(expected) for n in names_off)

    def test_zero_and_missing_denominator_yield_zero_not_inf_or_nan(self):
        df = pd.DataFrame({
            "entity_id": ["A", "A", "A", "A"],
            "period": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "monthly_transactions_amount": [50.0, 50.0, 50.0, 50.0],
            "income": [100.0, 0.0, -10.0, np.nan],
        })
        eng = PanelFeatureEngineer("entity_id", "period", False, [])
        out = eng.fit_transform(df)
        got = out["txn_amount_to_income"].tolist()
        assert got == [0.5, 0.0, 0.0, 0.0], f"guarded division misbehaved: {got}"
        assert np.isfinite(out["txn_amount_to_income"].to_numpy()).all()

    def test_ratio_value_matches_the_raw_quotient(self, panel):
        eng = PanelFeatureEngineer("entity_id", "period", False, [])
        out = eng.fit_transform(panel)
        den = panel["account_balance"].to_numpy(dtype=float)
        expected = np.where(den > 0, panel["withdrawal_amount"].to_numpy(dtype=float) / den, 0.0)
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
        assert np.allclose(out["withdrawal_to_balance"].to_numpy(), expected)


# ==================================================================== 5b2 ==
class TestMultiHorizonContrastFeatures:
    """Fase 1: contrast the current month against a longer history.

    A single-period lag under-reports a shock when the inputs are themselves
    smoothed; h=3 and h=6 recover the amplitude the smoothing hid.
    """

    @staticmethod
    def _toy(n_periods: int = 8, spike: float = 40.0):
        p = pd.date_range("2024-01-01", periods=n_periods, freq="MS")
        vals = [10.0] * (n_periods - 1) + [spike]
        return pd.DataFrame({
            "entity_id": ["A"] * n_periods, "period": p, "account_balance": vals,
        })

    def test_emits_lag_diff_and_ratio_for_every_horizon(self):
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        out = eng.fit_transform(self._toy())
        assert eng.lag_horizons_ == [1, 3, 6]
        for h in (1, 3, 6):
            for kind in ("lag", "diff", "ratio"):
                assert f"account_balance_{kind}{h}" in out.columns

    def test_contrast_values_are_arithmetically_right(self):
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        out = eng.fit_transform(self._toy())
        last = out.iloc[-1]
        for h in (1, 3, 6):
            assert last[f"account_balance_lag{h}"] == pytest.approx(10.0)
            assert last[f"account_balance_diff{h}"] == pytest.approx(30.0)
            assert last[f"account_balance_ratio{h}"] == pytest.approx(4.0)

    def test_rows_without_history_get_the_neutral_value(self):
        """0.0 for differences/levels, 1.0 for ratios -- a ratio of 0 would read
        as 'collapsed to nothing' and fake an anomaly."""
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        out = eng.fit_transform(self._toy())
        assert out["account_balance_diff6"].iloc[0] == 0.0
        assert out["account_balance_lag6"].iloc[0] == 0.0
        assert out["account_balance_ratio6"].iloc[0] == 1.0

    def test_horizons_deeper_than_the_panel_are_dropped(self):
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"])
        eng.fit(self._toy(n_periods=3))
        assert eng.lag_horizons_ == [1]
        assert not any("_lag6" in c for c in eng.get_feature_names_out())

    def test_horizons_are_resolved_against_the_fit_window(self, panel, schema):
        """The window the *scaler* sees decides which horizons are learnable: a
        horizon that is constant across the fit block would have ~zero fitted
        variance and then amplify unseen values without bound."""
        first_three = sorted(panel[schema.time_col].unique())[:3]
        mask = panel[schema.time_col].isin(first_three).to_numpy()
        eng = PanelFeatureEngineer(schema.entity_col, schema.time_col,
                                   fit_window_mask=mask)
        eng.fit(panel)
        assert eng.lag_horizons_ == [1], (
            f"a 3-period fit window cannot justify horizons {eng.lag_horizons_}"
        )

    def test_custom_horizons_are_honoured(self):
        eng = PanelFeatureEngineer("entity_id", "period", True, ["account_balance"],
                                   lag_horizons=(1, 2))
        out = eng.fit_transform(self._toy())
        assert eng.lag_horizons_ == [1, 2]
        assert "account_balance_lag2" in out.columns
        assert "account_balance_lag3" not in out.columns


# ==================================================================== 5b3 ==
class TestCyclicalFeaturesBypassScaling:
    """sin/cos seasonality must not be handed to a fitted scaler.

    A short chronological training window covers only a few months, so the
    encoding is near-constant there; a scaler fitted on it divides unseen months
    by a ~0 variance and explodes (4.9e18 was observed before this fix).
    """

    def test_cyclical_columns_stay_inside_their_natural_range(self, panel, schema):
        first_three = sorted(panel[schema.time_col].unique())[:3]
        mask = panel[schema.time_col].isin(first_three).to_numpy()
        X, _, names = fit_transform_panel(panel, schema, fit_mask=mask)
        cyc = [i for i, n in enumerate(names) if n.endswith(("_month_sin", "_month_cos"))]
        assert cyc, "no cyclical features found"
        vals = _to_dense(X)[:, cyc]
        assert np.abs(vals).max() <= 1.0 + 1e-9

    def test_no_feature_blows_up_under_a_short_fit_window(self, panel, schema):
        first_three = sorted(panel[schema.time_col].unique())[:3]
        mask = panel[schema.time_col].isin(first_three).to_numpy()
        X, _, names = fit_transform_panel(panel, schema, fit_mask=mask)
        peak = np.abs(_to_dense(X)).max()
        assert np.isfinite(peak) and peak < 1e6, (
            f"max|x|={peak:.3e}: a near-zero-variance column is being amplified"
        )


# ===================================================================== 5c ==
class TestAutoNumericTransform:
    """`numeric_transform="auto"` picks a transform per column, frozen at fit."""

    def test_auto_produces_a_finite_aligned_matrix(self, panel, schema):
        X, _, names = fit_transform_panel(panel, schema, numeric_transform="auto")
        assert _width(X) == len(names)
        assert _nonfinite_count(X) == 0
        assert _n_rows(X) == N_ROWS

    def test_auto_actually_varies_the_choice_across_columns(self, panel, schema):
        auto = _fitted_auto_step(panel, schema)
        picked = set(auto.chosen_transforms_)
        assert len(picked) > 1, (
            f"auto collapsed to a single transform {picked}; it is then just a "
            "renamed global option"
        )

    def test_quantile_is_never_selected(self, panel, schema):
        auto = _fitted_auto_step(panel, schema)
        assert "quantile" not in set(auto.chosen_transforms_), (
            "a rank-based normaliser flattens the tail that defines the anomaly"
        )

    def test_choice_is_frozen_at_fit_not_recomputed_on_transform(self, panel, schema):
        """Refitting on skewed data then transforming symmetric data (and vice
        versa) must apply the *fitted* mapping, never a fresh diagnosis."""
        auto = _fitted_auto_step(panel, schema)
        frozen = list(auto.chosen_transforms_)
        rng = np.random.default_rng(0)
        reshaped = rng.normal(size=(200, auto.n_features_in_))
        auto.transform(reshaped)
        assert auto.chosen_transforms_ == frozen

    def test_transform_is_deterministic_across_calls(self, panel, schema):
        auto = _fitted_auto_step(panel, schema)
        block = np.linspace(0.0, 10.0, 50 * auto.n_features_in_).reshape(50, -1)
        assert np.allclose(auto.transform(block), auto.transform(block))

    def test_make_numeric_transformer_rejects_auto_as_standalone(self):
        with pytest.raises(ValueError, match="pipeline-level mode"):
            make_numeric_transformer("auto")


# ===================================================================== 5d ==
class TestFitMask:
    """`fit_mask` fixes the OOT leak without destroying the panel features.

    The tempting shortcut -- fit on in-time rows, then `transform` the OOT rows
    alone -- silently zeroes every lag/diff/own_z on exactly the evaluated rows,
    because `shift(1)` finds no history inside a single-period subset.
    """

    @staticmethod
    def _masks(panel, schema):
        last = panel[schema.time_col].max()
        in_mask = (panel[schema.time_col] != last).to_numpy()
        return in_mask, ~in_mask

    def test_panel_features_on_oot_rows_survive_the_mask(self, panel, schema):
        in_mask, oot_mask = self._masks(panel, schema)
        eng_full = PanelFeatureEngineer(schema.entity_col, schema.time_col)
        features = eng_full.fit_transform(panel)
        panel_cols = [c for c in features.columns
                      if c.endswith(("_lag1", "_diff1", "_own_z"))]
        assert panel_cols, "no panel features to check"

        oot_block = features.loc[oot_mask, panel_cols].to_numpy()
        assert np.abs(oot_block).sum() > 0, "OOT panel features are identically zero"

    def test_naive_fit_in_transform_oot_destroys_them(self, panel, schema):
        """Guards the trap itself: if someone 'simplifies' fit_mask back to a
        plain fit/transform split, this documents what breaks."""
        in_mask, oot_mask = self._masks(panel, schema)
        eng = PanelFeatureEngineer(schema.entity_col, schema.time_col)
        eng.fit(panel[in_mask])
        naive = eng.transform(panel[oot_mask])
        panel_cols = [c for c in naive.columns
                      if c.endswith(("_lag1", "_diff1", "_own_z"))]
        assert np.abs(naive[panel_cols].to_numpy()).sum() == 0.0, (
            "the documented failure mode no longer reproduces -- update the "
            "fit_mask rationale in fit_transform_panel"
        )

    def test_column_transform_statistics_do_differ(self, panel, schema):
        in_mask, _ = self._masks(panel, schema)
        X_all, _, names_all = fit_transform_panel(panel, schema)
        X_masked, _, names_masked = fit_transform_panel(panel, schema, fit_mask=in_mask)
        assert names_all == names_masked
        assert X_all.shape == X_masked.shape
        assert not np.allclose(_to_dense(X_all), _to_dense(X_masked)), (
            "fit_mask changed nothing: the ColumnTransformer is still learning "
            "its statistics from the out-of-time rows"
        )

    def test_masked_output_is_finite_and_aligned(self, panel, schema):
        in_mask, _ = self._masks(panel, schema)
        X, keys, names = fit_transform_panel(panel, schema, fit_mask=in_mask)
        assert _width(X) == len(names)
        assert _n_rows(X) == N_ROWS == len(keys)
        assert _nonfinite_count(X) == 0

    def test_invalid_masks_are_rejected(self, panel, schema):
        with pytest.raises(ValueError, match="fit_mask has"):
            fit_transform_panel(panel, schema, fit_mask=np.ones(7, dtype=bool))
        with pytest.raises(ValueError, match="selects no rows"):
            fit_transform_panel(panel, schema, fit_mask=np.zeros(len(panel), dtype=bool))


# ====================================================================== 6 ==
class TestSignedLog1p:
    def test_matches_sign_times_log1p_abs(self):
        x = np.array([[-1e6], [-1000.0], [-1.0], [-1e-9], [0.0],
                      [1e-9], [1.0], [1000.0], [1e6]])
        out = SignedLog1p().fit(x).transform(x)
        expected = np.sign(x) * np.log1p(np.abs(x))
        assert np.allclose(out, expected), "SignedLog1p != sign(x)*log1p(|x|)"
        assert np.isfinite(out).all(), "SignedLog1p produced NaN/inf"
        assert out[np.where(x == 0.0)[0][0], 0] == 0.0
        assert (out[x < 0] <= 0).all()

    def test_multicolumn_and_no_overflow_on_large_values(self):
        x = np.array([[-1e12, 0.0, 1e12], [1e-3, -1e-3, 5.0]])
        out = SignedLog1p().fit_transform(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()
        assert np.allclose(out, np.sign(x) * np.log1p(np.abs(x)))


# ====================================================================== 7 ==
class TestJoblibRoundTrip:
    @pytest.mark.parametrize("encoding", ["onehot", "frequency"])
    def test_dump_load_transform_is_identical(self, panel, schema, tmp_path, encoding):
        pipe = build_preprocessing_pipeline(
            schema, categorical_encoding=encoding, numeric_transform="yeo-johnson")
        pipe.fit(panel)
        X1 = pipe.transform(panel)
        blob = tmp_path / f"pipeline_{encoding}.joblib"
        joblib.dump(pipe, blob)
        reloaded = joblib.load(blob)
        X2 = reloaded.transform(panel)
        assert sp.issparse(X1) == sp.issparse(X2), "sparsity changed across round-trip"
        assert np.allclose(_to_dense(X1), _to_dense(X2), equal_nan=True), (
            "joblib round-trip changed transform output -- inference reuse not exact"
        )
        assert list(pipe.get_feature_names_out()) == list(reloaded.get_feature_names_out())


# ====================================================================== 8 ==
class TestReproducibility:
    def test_same_input_same_config_identical_output(self, panel, schema):
        cfg = dict(numeric_transform="quantile", categorical_encoding="onehot")
        X1, k1, n1 = fit_transform_panel(panel, schema, **cfg)
        X2, k2, n2 = fit_transform_panel(panel, schema, **cfg)
        assert n1 == n2, "feature names differ between identical runs"
        assert k1.equals(k2), "keys differ between identical runs"
        assert np.allclose(_to_dense(X1), _to_dense(X2), equal_nan=True), (
            "two identical fit_transform_panel calls produced different matrices"
        )


# ====================================================================== 9 ==
class TestRobustnessEdgeCases:
    @pytest.fixture(scope="class")
    @classmethod
    def degenerate_panel(cls, panel):
        df = panel.copy()
        df["withdrawal_amount"] = np.nan   # entirely-NaN numeric column
        df["num_products"] = 3             # constant numeric column
        df["segment"] = "retail"           # constant categorical column
        return df

    @pytest.mark.parametrize("transform", NUMERIC_TRANSFORMS)
    def test_all_nan_and_constant_columns_do_not_crash(self, degenerate_panel, schema, transform):
        X, _, names = fit_transform_panel(degenerate_panel, schema, numeric_transform=transform)
        nf = _nonfinite_count(X)
        assert nf == 0, f"{transform}: {nf} non-finite from all-NaN/constant columns"
        assert _width(X) == len(names)

    @pytest.mark.parametrize("encoding", CATEGORICAL_ENCODINGS)
    def test_unknown_category_at_transform_time(self, panel, schema, encoding):
        held = panel["region"].value_counts().index[len(panel["region"].unique()) // 2]
        fit_df = panel[panel["region"] != held].copy()
        assert held not in set(fit_df["region"].unique())
        assert (panel["region"] == held).any()
        pipe = build_preprocessing_pipeline(
            schema, categorical_encoding=encoding, add_panel_features=False)
        pipe.fit(fit_df)
        X = pipe.transform(panel)
        assert _nonfinite_count(X) == 0, f"{encoding}: unknown category -> non-finite output"
        assert _n_rows(X) == len(panel)

    def test_degenerate_columns_with_indicators_and_panel_features(self, degenerate_panel, schema):
        X, _, names = fit_transform_panel(
            degenerate_panel, schema, numeric_transform="yeo-johnson",
            categorical_encoding="onehot", add_missing_indicators=True,
            add_panel_features=True,
        )
        assert _nonfinite_count(X) == 0
        assert any("withdrawal_amount" in n and n.endswith("__missing") for n in names), (
            "all-NaN column produced no missingness indicator"
        )


# ===================================================================== 10 ==
class TestStatisticsModule:
    @pytest.fixture(scope="class")
    @classmethod
    def diagnostics(cls, panel, schema):
        return compute_transform_diagnostics(panel, schema, random_state=SEED)

    def test_diagnostics_has_documented_columns_and_finite_shape_stats(self, diagnostics):
        assert DIAG_COLUMNS <= set(diagnostics.columns), (
            f"missing columns: {DIAG_COLUMNS - set(diagnostics.columns)}"
        )
        assert len(diagnostics) > 0
        raw = diagnostics[diagnostics["transform"] == "raw"]
        assert raw["skewness"].notna().any()
        assert np.isfinite(raw["skewness"].dropna().to_numpy()).all()
        assert np.isfinite(raw["excess_kurtosis"].dropna().to_numpy()).all()
        feat = "account_balance"
        raw_sk = diagnostics.query("feature == @feat and transform == 'raw'")["skewness"]
        yj_sk = diagnostics.query("feature == @feat and transform == 'yeo-johnson'")["skewness"]
        if len(raw_sk) and len(yj_sk):
            assert abs(float(yj_sk.iloc[0])) < abs(float(raw_sk.iloc[0])), (
                f"{feat}: yeo-johnson did not reduce |skew| "
                f"({float(yj_sk.iloc[0]):.3f} vs raw {float(raw_sk.iloc[0]):.3f})"
            )

    def test_recommend_returns_one_row_per_feature(self, diagnostics):
        recs = recommend_transform(diagnostics)
        features = set(diagnostics["feature"].unique())
        assert not recs.empty
        assert set(recs["feature"]) <= features
        assert not recs["feature"].duplicated().any(), "a feature was recommended twice"
        assert set(recs["recommended_transform"]) <= set(diagnostics["transform"].unique())
        assert {"feature", "recommended_transform"} <= set(recs.columns)

    def test_plot_writes_png_under_reports_figures(self, panel, schema):
        one_feature = infer_numeric_features(panel, schema)[:1]
        assert one_feature, "no numeric feature available to plot"
        paths = plot_transform_diagnostics(
            panel, schema, features=one_feature, kind="hist", random_state=SEED)
        try:
            assert paths, "plot_transform_diagnostics wrote no figures"
            for p in paths:
                parts = Path(p).parts
                assert "reports" in parts and "figures" in parts, (
                    f"figure written outside reports/figures/: {p}"
                )
                assert os.path.exists(p)
                assert p.lower().endswith(".png")
        finally:
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
