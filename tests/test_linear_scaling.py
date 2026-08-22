"""Validation suite for `src.preprocessing.linear_scaling`.

The module's contract is narrow and absolute, so the tests are organised
around the guarantees rather than around the functions: what gets touched,
what must never be touched, that degenerate columns cannot produce inf/NaN,
that the map really is affine, and that the two-step form is leak-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.linear_scaling import (
    apply_linear_scaling,
    is_key_column,
    minmax_scale,
    robust_scale,
    robust_scale_params,
    scaling_params,
    select_continuous_columns,
    standard_scale,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """A frame holding every column kind the selector must tell apart."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "codmes": [202401, 202401, 202402, 202402, 202403, 202403],
        "segmento": ["retail", "pyme", "retail", "corp", "pyme", "retail"],
        "activo": [True, True, False, True, False, True],
        "saldo": [1000.0, 1200.0, 950.0, 1100.0, 1050.0, 900_000.0],
        "edad": [31, 45, 28, 52, 39, 41],
        "constante": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        "con_nulos": [2.0, np.nan, 4.0, 6.0, np.nan, 8.0],
    })


class TestColumnSelection:
    def test_selects_numeric_and_ignores_everything_else(self, panel):
        assert select_continuous_columns(panel) == [
            "saldo", "edad", "constante", "con_nulos"
        ]

    @pytest.mark.parametrize("name", [
        "id", "ID", "codmes", "cod_mes", "COD_MES", "client_id", "id_cliente",
        "periodo", "period", "fecha", "date", "yyyymm", "anomes",
    ])
    def test_key_names_are_detected(self, name):
        assert is_key_column(name)

    @pytest.mark.parametrize("name", [
        "saldo", "edad", "income", "avg_txn_to_income", "balance_to_income",
        "identidad", "validez", "midpoint",
    ])
    def test_measurement_names_are_not_keys(self, name):
        """A name that merely *contains* the letters of a key token is a
        measurement -- matching is token-based, not substring-based."""
        assert not is_key_column(name)

    def test_booleans_are_excluded_despite_being_numeric_to_pandas(self, panel):
        assert pd.api.types.is_numeric_dtype(panel["activo"])
        assert "activo" not in select_continuous_columns(panel)

    def test_explicit_exclude_and_include(self, panel):
        assert "saldo" not in select_continuous_columns(panel, exclude=["saldo"])
        # `include` filters rather than overrides: a string stays out.
        assert select_continuous_columns(
            panel, include=["saldo", "segmento"]
        ) == ["saldo"]

    def test_min_unique_drops_low_cardinality_columns(self, panel):
        # `constante` has a single distinct value.
        assert "constante" not in select_continuous_columns(panel, min_unique=2)

    def test_detect_keys_can_be_disabled(self, panel):
        cols = select_continuous_columns(panel, detect_keys=False)
        assert "id" in cols and "codmes" in cols


class TestUntouchedColumns:
    """The absolute restriction: only continuous non-key columns change."""

    def test_strings_keys_and_booleans_pass_through_unchanged(self, panel):
        out = robust_scale(panel)
        for col in ("id", "codmes", "segmento", "activo"):
            pd.testing.assert_series_equal(out[col], panel[col])

    def test_dtypes_of_untouched_columns_are_preserved(self, panel):
        out = robust_scale(panel)
        for col in ("id", "codmes", "segmento", "activo"):
            assert out[col].dtype == panel[col].dtype

    def test_input_frame_is_not_mutated(self, panel):
        before = panel.copy(deep=True)
        robust_scale(panel)
        pd.testing.assert_frame_equal(panel, before)

    def test_inplace_mutates_and_returns_the_same_object(self, panel):
        out = robust_scale(panel, inplace=True)
        assert out is panel
        assert panel["saldo"].iloc[0] != 1000.0

    def test_column_order_is_preserved(self, panel):
        assert list(robust_scale(panel).columns) == list(panel.columns)


class TestRobustScaling:
    def test_matches_the_closed_form_definition(self, panel):
        out = robust_scale(panel)
        x = panel["saldo"].to_numpy(dtype=float)
        expected = (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25))
        np.testing.assert_allclose(out["saldo"].to_numpy(), expected)

    def test_median_maps_near_zero(self, panel):
        """Centring is what makes the median the origin -- so the scaled
        median is 0 by construction, for every non-degenerate column."""
        out = robust_scale(panel)
        assert out["saldo"].median() == pytest.approx(0.0, abs=1e-12)
        assert out["edad"].median() == pytest.approx(0.0, abs=1e-12)

    def test_the_anomaly_keeps_its_extremity(self, panel):
        """The point of a *linear* map: the outlier stays an outlier.

        A shape transform (rank/quantile) would pull the 900k row back into
        the bulk; here it must remain hundreds of IQRs out.
        """
        out = robust_scale(panel)
        assert out["saldo"].iloc[-1] > 100.0

    def test_rank_order_is_preserved(self, panel):
        """An affine map with a positive scale is strictly monotone."""
        out = robust_scale(panel)
        np.testing.assert_array_equal(
            out["saldo"].rank().to_numpy(), panel["saldo"].rank().to_numpy()
        )

    def test_transform_is_affine(self, panel):
        """Every scaled value satisfies y = a*x + b for one (a, b) per column."""
        out = robust_scale(panel)
        x = panel["edad"].to_numpy(dtype=float)
        y = out["edad"].to_numpy(dtype=float)
        a = (y[1] - y[0]) / (x[1] - x[0])
        b = y[0] - a * x[0]
        np.testing.assert_allclose(y, a * x + b)

    def test_custom_quantile_range(self, panel):
        out = robust_scale(panel, quantile_range=(10.0, 90.0))
        x = panel["edad"].to_numpy(dtype=float)
        expected = (x - np.median(x)) / (np.percentile(x, 90) - np.percentile(x, 10))
        np.testing.assert_allclose(out["edad"].to_numpy(), expected)

    @pytest.mark.parametrize("bad", [(75.0, 25.0), (-1.0, 50.0), (0.0, 101.0), (50.0, 50.0)])
    def test_invalid_quantile_range_raises(self, panel, bad):
        with pytest.raises(ValueError, match="quantile_range"):
            robust_scale(panel, quantile_range=bad)


class TestDegenerateColumns:
    """No input shape may produce inf or NaN where there was a finite value."""

    def test_zero_iqr_column_does_not_divide_by_zero(self, panel):
        out = robust_scale(panel)
        assert np.isfinite(out["constante"]).all()
        # x - median with scale forced to 1 -> exactly zero.
        np.testing.assert_allclose(out["constante"].to_numpy(), 0.0)

    def test_nans_stay_nan_and_do_not_contaminate_the_column(self, panel):
        out = robust_scale(panel)
        assert out["con_nulos"].isna().tolist() == panel["con_nulos"].isna().tolist()
        assert np.isfinite(out["con_nulos"].dropna()).all()

    def test_all_nan_column_survives(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
        out = robust_scale(df)
        assert out["x"].isna().all()
        assert np.isfinite(out["y"]).all()

    def test_single_row_frame(self):
        """IQR of one point is 0 -> scale 1 -> the row centres to zero."""
        out = robust_scale(pd.DataFrame({"x": [5.0], "s": ["a"]}))
        assert out["x"].tolist() == [0.0]
        assert out["s"].tolist() == ["a"]

    def test_frame_with_no_numeric_columns_is_returned_untouched(self):
        df = pd.DataFrame({"a": ["x", "y"], "id": [1, 2]})
        pd.testing.assert_frame_equal(robust_scale(df), df)

    def test_empty_frame(self):
        df = pd.DataFrame({"x": pd.Series(dtype=float)})
        assert len(robust_scale(df)) == 0


class TestLeakFreeTwoStepForm:
    def test_params_from_train_are_applied_to_every_row(self, panel):
        train = panel["codmes"] <= 202402
        params = robust_scale_params(panel[train])
        out = apply_linear_scaling(panel, params)

        # The constants come from the train block alone...
        assert params["center"]["saldo"] == pytest.approx(
            panel.loc[train, "saldo"].median()
        )
        # ...and the later rows are transformed with those same constants.
        x = panel["saldo"].to_numpy(dtype=float)
        expected = (x - params["center"]["saldo"]) / params["scale"]["saldo"]
        np.testing.assert_allclose(out["saldo"].to_numpy(), expected)

    def test_one_shot_and_two_step_differ_when_the_tail_is_out_of_sample(self, panel):
        """The regression guard for the leak this module warns about.

        Scaling the whole panel lets the 900k row (a test-period row) set the
        median/IQR applied to training rows. The two forms must therefore
        disagree -- if they ever stop disagreeing, the leak-free path is not
        actually doing anything.
        """
        train = panel["codmes"] <= 202402
        leaky = robust_scale(panel)
        clean = apply_linear_scaling(panel, robust_scale_params(panel[train]))
        assert not np.allclose(
            leaky.loc[train, "saldo"].to_numpy(), clean.loc[train, "saldo"].to_numpy()
        )

    def test_params_are_plain_json_serialisable_data(self, panel):
        """Not an object with .fit() -- it must survive a JSON round-trip."""
        import json

        params = robust_scale_params(panel)
        restored = json.loads(json.dumps(params))
        pd.testing.assert_frame_equal(
            apply_linear_scaling(panel, restored), apply_linear_scaling(panel, params)
        )

    def test_missing_column_is_skipped_not_raised(self, panel, caplog):
        import logging

        params = robust_scale_params(panel)
        reduced = panel.drop(columns=["edad"])
        logger = logging.getLogger("test-linear-scaling")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            out = apply_linear_scaling(reduced, params, logger=logger)
        assert "edad" in caplog.text
        assert np.isfinite(out["saldo"]).all()

    def test_applying_train_params_twice_is_idempotent_in_shape(self, panel):
        params = robust_scale_params(panel)
        once = apply_linear_scaling(panel, params)
        assert list(once.columns) == list(panel.columns)


class TestOtherMethods:
    def test_standard_scaling_definition(self, panel):
        out = standard_scale(panel)
        x = panel["edad"].to_numpy(dtype=float)
        np.testing.assert_allclose(out["edad"].to_numpy(), (x - x.mean()) / x.std())

    def test_minmax_maps_to_unit_interval(self, panel):
        out = minmax_scale(panel)
        assert out["edad"].min() == pytest.approx(0.0)
        assert out["edad"].max() == pytest.approx(1.0)

    def test_robust_beats_standard_at_preserving_the_anomaly(self, panel):
        """The reason robust is the default.

        The standard deviation is inflated by the very anomaly being detected,
        so dividing by it shrinks that point's distance from centre. The IQR
        ignores the outer quartiles, so it does not.
        """
        robust_out = robust_scale(panel)["saldo"].iloc[-1]
        standard_out = standard_scale(panel)["saldo"].iloc[-1]
        assert robust_out > standard_out * 100

    def test_scaling_params_dispatch(self, panel):
        for method in ("robust", "standard", "minmax"):
            assert scaling_params(panel, method=method)["method"] == method

    def test_unknown_method_raises(self, panel):
        with pytest.raises(ValueError, match="Unknown method"):
            scaling_params(panel, method="quantile")
