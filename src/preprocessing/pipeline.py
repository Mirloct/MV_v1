"""Modular, sklearn-compatible preprocessing pipeline for the banking panel.

The pipeline turns a raw panel DataFrame (as produced by ``src.data``) into a
model-ready feature matrix while keeping the ``(entity_id, period)`` keys aside
so the evaluation module can join detector scores back to the hidden ground
truth. Every knob that changes the *shape* of the transform -- the numeric
transformation, the categorical encoding, the imputation strategy, whether to
add missingness indicators or panel-derived features -- is a plain string/bool
argument to :func:`build_preprocessing_pipeline`, so an Optuna study can treat
each as a categorical hyperparameter.

Design
------
The returned object is a two-step sklearn :class:`~sklearn.pipeline.Pipeline`:

1. ``panel_features`` -- :class:`PanelFeatureEngineer`: a DataFrame->DataFrame
   step that (a) drops the key columns and stray datetimes, (b) normalises
   dtypes (nullable ``Int64`` -> ``float64``), and (c) optionally appends
   within-entity lag/diff/own-history-z features plus a cyclical month
   encoding. Keeping this a separate toggleable step means it can be ablated.
2. ``column_transform`` -- a :class:`~sklearn.compose.ColumnTransformer` that
   routes numeric / categorical / boolean columns (selected by dtype at fit
   time) into their respective sub-pipelines and, when enabled, emits a
   missingness-indicator branch. One-hot output is sparse so wide long-tail
   categoricals do not blow up memory at 1M rows.

The whole thing is ``fit``/``transform``-able and joblib-picklable, so it can be
persisted and reused unchanged at inference.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils import check_array
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted

from src.data.loader import PanelSchema
from src.utils.logging_config import log_phase, setup_logging

__all__ = [
    "NUMERIC_TRANSFORMS",
    "CATEGORICAL_ENCODINGS",
    "SignedLog1p",
    "RareCategoryGrouper",
    "FrequencyEncoder",
    "MissingnessIndicator",
    "PanelFeatureEngineer",
    "AutoNumericTransformer",
    "make_numeric_transformer",
    "build_preprocessing_pipeline",
    "fit_transform_panel",
    "categorical_feature_mask",
    "split_matrix_for_model",
]

#: Prefix the ColumnTransformer stamps on every output feature produced by the
#: **categorical** branch (`verbose_feature_names_out=True` prefixes each
#: output with its transformer's name; see `build_preprocessing_pipeline`).
#: Boolean columns get their own `bool__` prefix and are deliberately NOT
#: treated as categorical here: they are already 0/1 numerics that every
#: detector consumes without special handling.
CATEGORICAL_FEATURE_PREFIX = "cat__"

# Allowed values for the ``numeric_transform`` argument. These are the exact
# strings an Optuna study can enumerate as a categorical hyperparameter; each
# maps to a scikit-learn transformer in :func:`make_numeric_transformer`:
#   "standard"     -> StandardScaler (zero-mean/unit-variance; shape unchanged).
#   "robust"       -> RobustScaler (median/IQR centring; outlier-tolerant,
#                     shape unchanged).
#   "log1p"        -> SignedLog1p then StandardScaler (compresses heavy right
#                     tails; safe on zeros/negatives).
#   "yeo-johnson"  -> PowerTransformer (learned power transform toward Gaussian;
#                     the default, handles skew of either sign).
#   "quantile"     -> QuantileTransformer to a normal output (rank-based;
#                     strongest normaliser but stochastic -- see random_state).
#   "passthrough"  -> no transform (leaves the imputed values as-is).
#   "auto"         -> per-column selection driven by the skewness diagnostics
#                     (see AutoNumericTransformer); not a standalone transform.
NUMERIC_TRANSFORMS: tuple[str, ...] = (
    "standard",
    "robust",
    "log1p",
    "yeo-johnson",
    "quantile",
    "passthrough",
    "auto",
)

# Candidate transforms the "auto" mode chooses between, per numeric column.
# All three end in a scaler, so the choice is purely about *shape*:
# "standard" leaves the distribution untouched (skewness is affine-invariant),
# "log1p" compresses a heavy right tail by a fixed amount, and "yeo-johnson"
# fits an exponent per column. "quantile" is excluded by default -- see
# `_AUTO_EXCLUDED`.
_AUTO_CANDIDATES: tuple[str, ...] = ("standard", "log1p", "yeo-johnson")

# TEORÍA: a rank-based normaliser maps the top percentiles into a bounded
# normal range, flattening precisely the tail that constitutes the anomaly
# signal. More generally, any transform *fitted to gaussianise* is fitted on
# data that contains the anomalies, so it partially normalises them away. The
# goal here is to reduce skewness enough that the Isolation Forest's uniform
# axis-parallel cuts are informative, without destroying the tail that carries
# the signal -- so "quantile" is excluded and `abs_skewness` (not the
# distance-from-normality `jb_effect`) is the default criterion.
_AUTO_EXCLUDED: tuple[str, ...] = ("quantile",)
_AUTO_CRITERION = "abs_skewness"

# Columns with at most this many distinct values are treated as discrete
# (counts, scores, flags): a shape transform is not what they need, so "auto"
# assigns them the neutral scaler. Mirrors `statistics.infer_numeric_features`.
_AUTO_DISCRETE_MAX_NUNIQUE = 20
_AUTO_FALLBACK = "standard"

# Ratio features appended by `PanelFeatureEngineer` as (numerator, denominator,
# output name) triples. Pairs whose columns are absent or non-numeric are
# silently skipped, exactly like the panel-feature source columns above.
#
# TEORÍA: an Isolation Forest splits on `uniform(min_f, max_f)` of a single
# feature f, so every decision boundary it can express is a union of
# axis-parallel boxes. A ratio X/Y is constant along a *ray through the
# origin* -- an oblique boundary that boxes can only approximate with a
# staircase. Each stair costs a split, so the expected path length E[h(x)] of a
# point that is anomalous only in the ratio grows towards that of a normal
# point, and the score gap collapses. Materialising the ratio as its own
# coordinate turns that oblique structure axis-aligned, which is exactly what
# an Extended Isolation Forest buys via oblique cuts -- obtained here without a
# new dependency. The chosen pairs align with the injected geometries:
# balance/income and withdrawal/balance cover the "global" type and the
# decorrelated subgroup, txn_amount/income covers "collective".
# Horizons (in periods) for the within-entity contrast features. For each
# monetary column and each horizon h the engineer emits `{col}_lag{h}`,
# `{col}_diff{h}` and `{col}_ratio{h}`. Horizons that exceed the panel's own
# depth are dropped at fit time rather than producing constant columns.
#
# TEORÍA: a single-period lag only sees the most recent step. When the inputs
# are themselves smoothed (rolling averages/accumulations, as in most banking
# feature marts) a genuine shock is spread across several months, so its
# one-month difference is a fraction of its true size and the anomaly surfaces
# late or not at all. Contrasting the current value against a *longer* history
# (h=3, h=6) recovers the amplitude the smoothing hid. Both forms are kept
# because they answer different questions: the difference is the shock in the
# variable's own units (a 5,000 jump), while the ratio is scale-free (a
# doubling), and the ratio is what makes small and large customers comparable
# to a detector that splits on absolute thresholds.
_DEFAULT_LAG_HORIZONS: tuple[int, ...] = (1, 3, 6)

_DEFAULT_RATIO_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("monthly_transactions_amount", "income", "txn_amount_to_income"),
    ("withdrawal_amount", "account_balance", "withdrawal_to_balance"),
    ("account_balance", "income", "balance_to_income"),
    ("avg_transaction_amount", "income", "avg_txn_to_income"),
)

# Allowed values for the ``categorical_encoding`` argument (also an Optuna
# categorical). Each maps to a sub-pipeline in :func:`_make_categorical_pipeline`:
#   "onehot"    -> sparse OneHotEncoder with native infrequent-category folding
#                  (widest output; memory-safe via sparsity at 1M rows).
#   "ordinal"   -> RareCategoryGrouper then OrdinalEncoder (one integer column
#                  per feature; unknown -> -1, missing -> -2).
#   "frequency" -> FrequencyEncoder (one column per feature holding the training
#                  relative frequency; compact for very high cardinality).
CATEGORICAL_ENCODINGS: tuple[str, ...] = ("onehot", "ordinal", "frequency")

# Monetary/dynamic columns that panel feature engineering derives from when
# present. Anything not in the panel (or not numeric) is silently skipped.
_DEFAULT_PANEL_FEATURE_COLS: tuple[str, ...] = (
    "account_balance",
    "monthly_transactions_amount",
    "withdrawal_amount",
    "income",
    "avg_transaction_amount",
    "monthly_transactions_count",
)

_RARE_BUCKET = "__rare__"


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _to_float32(X):
    """Picklable helper: cast an array-like to float32 (used for booleans)."""
    return np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Custom transformers                                                         #
# --------------------------------------------------------------------------- #
class SignedLog1p(BaseEstimator, TransformerMixin):
    """Signed log1p transform: ``sign(x) * log1p(|x|)``.

    Unlike a plain ``log1p`` this is defined for zeros and negatives, so it is
    safe on any numeric column (differences, own-history deviations, ...), not
    just strictly-positive monetary features. It is a pure element-wise map:
    stateless apart from remembering the input width for feature naming.
    """

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, ensure_all_finite=False)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=False)
        return np.sign(X) * np.log1p(np.abs(X))

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        return np.asarray(input_features, dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Bucket categories occurring in < ``min_frequency`` of rows into ``__rare__``.

    Fitted per-column: at transform time any value not seen frequently enough
    in training (including genuinely unseen categories) is mapped to the shared
    ``__rare__`` token. Used as the rare-collapsing front-end for the ordinal
    encoder; the one-hot path uses scikit-learn's native ``min_frequency``
    instead.
    """

    def __init__(self, min_frequency: float = 0.001):
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        # Deliberately do NOT record feature_names_in_: this step is fitted on
        # the imputer's *ndarray* output (integer column labels), and storing
        # those would break get_feature_names_out name-threading through the
        # Pipeline. n_features_in_ is enough for _check_feature_names_in.
        Xdf = self._as_frame(X)
        self.n_features_in_ = Xdf.shape[1]
        n = len(Xdf)
        self.frequent_: dict = {}
        thresh = self.min_frequency * n if n else 0
        for col in Xdf.columns:
            counts = Xdf[col].value_counts(dropna=True)
            self.frequent_[col] = set(counts[counts >= thresh].index.tolist())
        return self

    def transform(self, X):
        check_is_fitted(self, "frequent_")
        Xdf = self._as_frame(X).copy()
        for col in Xdf.columns:
            keep = self.frequent_.get(col, set())
            s = Xdf[col].astype(object)
            Xdf[col] = s.where(s.isin(keep), other=_RARE_BUCKET)
        # Return an ndarray so a downstream OrdinalEncoder does not inherit
        # integer column names and reject the real names during naming.
        return Xdf.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X))


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode each category by its training relative frequency.

    A compact alternative to one-hot for very high-cardinality columns: output
    width equals the input width. Unseen/NaN values map to ``0.0`` (they were,
    by definition, never observed). Rare categories therefore encode to a small
    positive number, which itself is a usable "rareness" signal for the models.
    """

    def fit(self, X, y=None):
        # See RareCategoryGrouper.fit: fitted on ndarray, so keep only
        # n_features_in_ and let name-threading supply the real names.
        Xdf = RareCategoryGrouper._as_frame(X)
        self.n_features_in_ = Xdf.shape[1]
        n = len(Xdf)
        self.freq_maps_: dict = {}
        for col in Xdf.columns:
            self.freq_maps_[col] = (Xdf[col].value_counts(dropna=True) / max(n, 1)).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, "freq_maps_")
        Xdf = RareCategoryGrouper._as_frame(X)
        out = np.empty((len(Xdf), Xdf.shape[1]), dtype=np.float32)
        for j, col in enumerate(Xdf.columns):
            mapping = self.freq_maps_.get(col, {})
            out[:, j] = Xdf[col].map(mapping).fillna(0.0).to_numpy(dtype=np.float32)
        return out

    def get_feature_names_out(self, input_features=None):
        input_features = _check_feature_names_in(self, input_features)
        return np.asarray([f"{c}_freq" for c in input_features], dtype=object)


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Emit a 0/1 flag per column that carried NaNs in the training data.

    Upstream MNAR missingness is *informative* (large balances are redacted
    before release), so the fact that a value is missing is itself an anomaly
    cue. This transformer preserves that signal after imputation has erased it.
    With ``only_missing=True`` (default) it keeps flags only for columns that
    actually had missing values at fit time, so it adds nothing for complete
    columns.
    """

    def __init__(self, only_missing: bool = True):
        self.only_missing = only_missing

    def fit(self, X, y=None):
        Xdf = RareCategoryGrouper._as_frame(X)
        self.feature_names_in_ = np.asarray(Xdf.columns, dtype=object)
        self.n_features_in_ = Xdf.shape[1]
        na_any = Xdf.isna().any(axis=0)
        if self.only_missing:
            self.columns_ = [c for c in Xdf.columns if bool(na_any.get(c, False))]
        else:
            self.columns_ = list(Xdf.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        Xdf = RareCategoryGrouper._as_frame(X)
        if not self.columns_:
            return np.empty((len(Xdf), 0), dtype=np.float32)
        return Xdf[self.columns_].isna().to_numpy(dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray([f"{c}__missing" for c in self.columns_], dtype=object)


class PanelFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean keys/dtypes and (optionally) add within-entity panel features.

    Always: drops the ``entity_col`` / ``time_col`` keys and any stray datetime
    columns, and casts nullable-integer columns to ``float64`` so downstream
    scikit-learn steps accept them. Booleans are kept as ``bool`` and object
    columns as ``object`` so the ColumnTransformer's dtype selectors can route
    them.

    When ``add_panel_features`` is on, for each configured monetary column it
    appends, computed *within each entity in time order*:

    * ``{col}_lag1``  -- previous-period value,
    * ``{col}_diff1`` -- change vs the previous period,
    * ``{col}_own_z`` -- deviation from the entity's own expanding history
      (causal z-score using only strictly-prior periods),

    plus a cyclical ``{time}_month_sin`` / ``{time}_month_cos`` seasonality
    encoding. These directly serve the "local" (own-history) and "contextual"
    (seasonal) anomaly definitions. The first period per entity has no history,
    so the lag/diff/own-z features are NaN there and are explicitly filled with
    ``0.0`` (documented, not imputed downstream). Everything is vectorized via
    group-wise cumulative sums; row order is preserved.

    It also appends the ``ratio_features`` triples (see
    :data:`_DEFAULT_RATIO_FEATURES` for what they are and why they matter to an
    axis-parallel splitter). Ratios are row-local, so unlike the panel features
    they need no history and are unaffected by ``add_panel_features``.

    Every feature produced here is **causal**: lags and diffs look strictly
    backwards, ``own_z`` uses only strictly-prior periods, ratios are
    within-row, and the seasonality encoding is a function of the timestamp
    alone. Nothing is estimated from the data, so this step may -- and must --
    run over the whole panel even when downstream steps are fitted on a
    subset; see the ``fit_mask`` argument of :func:`fit_transform_panel`.
    """

    def __init__(
        self,
        entity_col: Optional[str],
        time_col: Optional[str],
        add_panel_features: bool = True,
        feature_cols: Optional[Sequence[str]] = None,
        ratio_features: Optional[Sequence[tuple[str, str, str]]] = None,
        lag_horizons: Optional[Sequence[int]] = None,
        fit_window_mask: Optional[np.ndarray] = None,
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.add_panel_features = add_panel_features
        self.feature_cols = feature_cols
        self.ratio_features = ratio_features
        self.lag_horizons = lag_horizons
        self.fit_window_mask = fit_window_mask

    # -- fit --------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PanelFeatureEngineer requires a pandas DataFrame")
        keys = {self.entity_col, self.time_col} - {None}
        kept, panel_cols = [], []
        for col in X.columns:
            if col in keys:
                continue
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                continue  # a stray datetime feature is dropped (kept only as a key)
            kept.append(col)

        default_cols = self.feature_cols if self.feature_cols is not None else _DEFAULT_PANEL_FEATURE_COLS
        if self.add_panel_features and self.entity_col is not None:
            for col in default_cols:
                if col in kept and pd.api.types.is_numeric_dtype(X[col]):
                    panel_cols.append(col)

        ratio_specs = []
        ratio_defs = (
            self.ratio_features if self.ratio_features is not None else _DEFAULT_RATIO_FEATURES
        )
        for num, den, name in ratio_defs:
            if (
                num in kept and den in kept
                and pd.api.types.is_numeric_dtype(X[num])
                and pd.api.types.is_numeric_dtype(X[den])
            ):
                ratio_specs.append((num, den, name))

        # Resolve the usable contrast horizons against the **fit window**, not
        # the whole panel.
        #
        # TEORÍA: a horizon h only carries information for rows that have h
        # prior periods. Measured on the full panel, h=6 looks usable on a
        # 15-month panel -- but if the estimators are fitted on a 3-month
        # training block (the chronological split), every training row sits at
        # the neutral fill value, the column's fitted variance is ~0, and the
        # scaler then amplifies the genuine test-set values without bound. The
        # horizon must therefore be justified by the window the scaler sees.
        requested = (
            self.lag_horizons if self.lag_horizons is not None else _DEFAULT_LAG_HORIZONS
        )
        max_history = 0
        if panel_cols and self.entity_col is not None and self.entity_col in X.columns:
            window = X
            if self.fit_window_mask is not None:
                mask = np.asarray(self.fit_window_mask, dtype=bool).ravel()
                if mask.shape[0] == len(X) and mask.any():
                    window = X.loc[mask]
            counts = window[self.entity_col].value_counts()
            max_history = int(counts.max()) if len(counts) else 0
        horizons = sorted({int(h) for h in requested if int(h) >= 1})
        if max_history:
            horizons = [h for h in horizons if h < max_history]
        if panel_cols and not horizons:
            horizons = [1]

        engineered = [name for _, _, name in ratio_specs]
        for col in panel_cols:
            for h in horizons:
                engineered += [f"{col}_lag{h}", f"{col}_diff{h}", f"{col}_ratio{h}"]
            engineered.append(f"{col}_own_z")
        if self.add_panel_features and self.time_col is not None:
            engineered += [f"{self.time_col}_month_sin", f"{self.time_col}_month_cos"]

        self.kept_columns_ = kept
        self.panel_cols_ = panel_cols
        self.ratio_specs_ = ratio_specs
        self.lag_horizons_ = horizons
        self.engineered_columns_ = engineered
        self.feature_names_out_ = np.asarray(kept + engineered, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    # -- transform --------------------------------------------------------- #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PanelFeatureEngineer requires a pandas DataFrame")
        df = X.reset_index(drop=True)
        n = len(df)

        data = {}
        for col in self.kept_columns_:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                data[col] = s.to_numpy(dtype=bool)
            elif pd.api.types.is_numeric_dtype(s):
                # Covers nullable Int64 (-> float64 with NaN) and plain floats.
                data[col] = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
            else:
                data[col] = s.astype(object).to_numpy()
        res = pd.DataFrame(data, index=df.index)

        for num, den, name in getattr(self, "ratio_specs_", []):
            numerator = pd.to_numeric(df[num], errors="coerce").to_numpy(dtype="float64")
            denominator = pd.to_numeric(df[den], errors="coerce").to_numpy(dtype="float64")
            # Guarded division: a non-positive or missing denominator yields 0.0
            # rather than inf/NaN. These are all non-negative monetary
            # quantities, so `den > 0` is the meaningful domain and NaN
            # comparisons are False, which folds missing values into the same
            # branch.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(denominator > 0, numerator / denominator, 0.0)
            res[name] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        if self.panel_cols_:
            self._add_panel_features(df, res, n)
        if self.add_panel_features and self.time_col is not None:
            months = pd.DatetimeIndex(pd.to_datetime(df[self.time_col], errors="coerce")).month.to_numpy(
                dtype="float64"
            )
            months = np.nan_to_num(months, nan=0.0)
            res[f"{self.time_col}_month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
            res[f"{self.time_col}_month_cos"] = np.cos(2.0 * np.pi * months / 12.0)

        # Guarantee the promised column order (and presence).
        return res.reindex(columns=list(self.feature_names_out_))

    def _add_panel_features(self, df: pd.DataFrame, res: pd.DataFrame, n: int) -> None:
        ent = df[self.entity_col].to_numpy()
        work = pd.DataFrame({"_g": ent, "_pos": np.arange(n)})
        sort_keys = ["_g"]
        if self.time_col is not None:
            work["_t"] = pd.to_datetime(df[self.time_col], errors="coerce").to_numpy()
            sort_keys = ["_g", "_t"]
        for col in self.panel_cols_:
            work[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").to_numpy()
        work = work.sort_values(sort_keys, kind="stable")
        grp = work.groupby("_g", sort=False)
        pos = work["_pos"].to_numpy()

        for col in self.panel_cols_:
            s = work[col]
            s_arr = s.to_numpy()

            # Multi-horizon contrast features. See `_DEFAULT_LAG_HORIZONS`.
            for h in self.lag_horizons_:
                lag_h = grp[col].shift(h)
                lag_arr = lag_h.to_numpy()
                self._scatter(res, pos, f"{col}_lag{h}", lag_arr)
                self._scatter(res, pos, f"{col}_diff{h}", s_arr - lag_arr)
                # Guarded ratio: 1.0 is the neutral value (no change), so an
                # unusable denominator must map there rather than to 0.0, which
                # would read as "collapsed to nothing" and fake an anomaly.
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(lag_arr > 0, s_arr / lag_arr, 1.0)
                self._scatter(res, pos, f"{col}_ratio{h}", ratio, fill=1.0)

            v0 = s.fillna(0.0)
            sq = v0 * v0
            nn = s.notna().astype("float64")
            work["_v0"], work["_sq"], work["_nn"] = v0.to_numpy(), sq.to_numpy(), nn.to_numpy()
            g2 = work.groupby("_g", sort=False)
            sum_prev = g2["_v0"].cumsum().to_numpy() - work["_v0"].to_numpy()
            sq_prev = g2["_sq"].cumsum().to_numpy() - work["_sq"].to_numpy()
            cnt_prev = g2["_nn"].cumsum().to_numpy() - work["_nn"].to_numpy()

            with np.errstate(divide="ignore", invalid="ignore"):
                mean_prev = np.where(cnt_prev > 0, sum_prev / cnt_prev, np.nan)
                var_prev = np.where(cnt_prev > 1, sq_prev / cnt_prev - mean_prev**2, np.nan)
                var_prev = np.clip(var_prev, 0.0, None)
                std_prev = np.sqrt(var_prev)
                own_z = np.where(std_prev > 0, (s.to_numpy() - mean_prev) / std_prev, 0.0)

            self._scatter(res, pos, f"{col}_own_z", own_z)

    @staticmethod
    def _scatter(
        res: pd.DataFrame,
        pos: np.ndarray,
        name: str,
        sorted_values: np.ndarray,
        fill: float = 0.0,
    ) -> None:
        """Place ``sorted_values`` (in sorted order) back into original order.

        ``fill`` is the neutral value for rows without enough history (the first
        ``h`` periods of an entity): ``0.0`` for differences and levels, ``1.0``
        for ratios.
        """
        arr = np.empty(len(pos), dtype="float64")
        arr[pos] = np.asarray(sorted_values, dtype="float64")
        res[name] = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


# --------------------------------------------------------------------------- #
# Factory helpers                                                             #
# --------------------------------------------------------------------------- #
def make_numeric_transformer(numeric_transform: str = "yeo-johnson", random_state: int = 0):
    """Return the scikit-learn transformer for a named numeric transform.

    ``log1p`` is the custom signed-log1p followed by standardisation (so its
    output is comparable to the other, standardised, transforms). ``passthrough``
    returns the literal string accepted by scikit-learn pipelines.
    """
    name = _normalize_name(numeric_transform)
    if name == "auto":
        raise ValueError(
            "'auto' is a pipeline-level mode, not a standalone transform; it "
            "resolves to one of "
            f"{_AUTO_CANDIDATES} per column via AutoNumericTransformer."
        )
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "log1p":
        return Pipeline([("signed_log1p", SignedLog1p()), ("scale", StandardScaler())])
    if name in ("yeo-johnson", "yeojohnson", "power"):
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=random_state)
    if name == "passthrough":
        return "passthrough"
    raise ValueError(
        f"Unknown numeric_transform {numeric_transform!r}; choose from {NUMERIC_TRANSFORMS}"
    )


class AutoNumericTransformer(BaseEstimator, TransformerMixin):
    """Pick one numeric transform **per column** from the skewness diagnostics.

    Every other ``numeric_transform`` option applies a single global choice to
    all numeric columns. That is a poor fit for a banking panel, where a
    lognormal balance and a bounded satisfaction score sit in the same matrix
    and want opposite treatment.

    On ``fit`` this reuses :func:`src.preprocessing.statistics.compute_transform_diagnostics`
    and :func:`~src.preprocessing.statistics.recommend_transform` -- the same
    machinery the pipeline already reports as evidence -- and makes their
    verdict load-bearing. The mapping is frozen at fit time and replayed
    verbatim on ``transform``, so a later batch with different skewness gets
    the transform learned on the training data, never a fresh one.

    Output is strictly one-to-one: column ``i`` in, column ``i`` out, which
    keeps ``get_feature_names_out`` aligned with the surrounding
    ColumnTransformer.

    Args:
        candidates: Transform names to choose between (default
            :data:`_AUTO_CANDIDATES`).
        criterion: Diagnostics column to minimise (default ``abs_skewness``;
            see :data:`_AUTO_EXCLUDED` for why not ``jb_effect``).
        exclude_transforms: Names dropped before ranking, as a hard guard.
        sample_size: Rows sampled per column for the diagnostics.
        random_state: Seed for the diagnostics subsample and any stochastic
            transform.
    """

    def __init__(
        self,
        candidates: Sequence[str] = _AUTO_CANDIDATES,
        criterion: str = _AUTO_CRITERION,
        exclude_transforms: Sequence[str] = _AUTO_EXCLUDED,
        sample_size: int = 20_000,
        random_state: int = 0,
    ):
        self.candidates = candidates
        self.criterion = criterion
        self.exclude_transforms = exclude_transforms
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X, y=None):
        # Imported lazily: `statistics` imports this module at import time, so a
        # module-level import here would be circular.
        from src.preprocessing.statistics import (
            compute_transform_diagnostics,
            recommend_transform,
        )

        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        n_cols = Xa.shape[1]
        col_ids = [f"c{i}" for i in range(n_cols)]
        frame = pd.DataFrame(Xa, columns=col_ids)

        candidates = [
            c for c in self.candidates
            if _normalize_name(c) not in {_normalize_name(e) for e in self.exclude_transforms}
        ] or [_AUTO_FALLBACK]

        # Discrete / constant columns get the neutral scaler: reshaping a count
        # or a 1-10 score is not what this diagnostic is for, and a fitted power
        # transform on a near-constant column is numerically fragile.
        diag_cols = [
            c for c in col_ids
            if frame[c].dropna().nunique() > _AUTO_DISCRETE_MAX_NUNIQUE
        ]
        chosen = {c: _AUTO_FALLBACK for c in col_ids}

        if diag_cols:
            diagnostics = compute_transform_diagnostics(
                frame,
                numeric_cols=diag_cols,
                transforms=tuple(candidates),
                sample_size=self.sample_size,
                random_state=self.random_state,
            )
            rec = recommend_transform(
                diagnostics,
                criterion=self.criterion,
                lower_is_better=True,
                exclude_transforms=tuple(self.exclude_transforms),
            )
            for feature, transform in zip(rec["feature"], rec["recommended_transform"]):
                chosen[str(feature)] = _normalize_name(str(transform))

        self.chosen_transforms_ = [chosen[c] for c in col_ids]

        # One fitted transformer per distinct choice. All candidates act
        # column-wise and independently, so fitting per group is equivalent to
        # fitting per column and far cheaper.
        chosen_arr = np.asarray(self.chosen_transforms_, dtype=object)
        self.groups_ = []
        for name in sorted(set(self.chosen_transforms_)):
            idx = np.flatnonzero(chosen_arr == name)
            transformer = make_numeric_transformer(name, random_state=self.random_state)
            if isinstance(transformer, str):  # "passthrough"
                self.groups_.append((idx, None))
                continue
            transformer.fit(Xa[:, idx])
            self.groups_.append((idx, transformer))

        self.n_features_in_ = n_cols
        return self

    def transform(self, X):
        check_is_fitted(self, "chosen_transforms_")
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(
                f"AutoNumericTransformer was fitted on {self.n_features_in_} columns, "
                f"got {Xa.shape[1]}"
            )
        out = np.empty_like(Xa)
        for idx, transformer in self.groups_:
            block = Xa[:, idx]
            out[:, idx] = block if transformer is None else np.asarray(
                transformer.transform(block), dtype=np.float64
            )
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "chosen_transforms_")
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)

    def chosen_transform_map(self, input_features=None) -> dict[str, str]:
        """``{feature name: chosen transform}``, for logging and diagnostics."""
        check_is_fitted(self, "chosen_transforms_")
        names = self.get_feature_names_out(input_features)
        return {str(n): t for n, t in zip(names, self.chosen_transforms_)}


def _make_numeric_imputer(impute_numeric: str) -> SimpleImputer:
    """Build the numeric imputer, translating the ``"zero"`` alias.

    ``"zero"`` is the project default and maps to sklearn's
    ``strategy="constant", fill_value=0.0``. It is a *statistic-free* fill:
    nothing is estimated from the data, so unlike ``"median"`` it cannot leak
    across the train/test boundary and cannot shift when the fit window
    changes.

    The trade-off is real and worth naming: a zero is a *value*, not a
    "missing" symbol. In a column where 0 already means something (an empty
    balance, no transactions) a filled zero is indistinguishable from a
    genuine one. That is why `add_missing_indicators` defaults to True --
    the 0/1 flag per NaN-bearing column keeps "this was absent" recoverable,
    so the pair (zero fill + indicator) loses no information even though the
    fill alone would. Turning indicators off while keeping zero fill is the
    combination to avoid.
    """
    if _normalize_name(impute_numeric) in ("zero", "zeros", "constant"):
        return SimpleImputer(
            strategy="constant", fill_value=0.0, keep_empty_features=True,
        )
    return SimpleImputer(strategy=impute_numeric, keep_empty_features=True)


def _make_numeric_pipeline(numeric_transform, impute_numeric, random_state) -> Pipeline:
    if _normalize_name(numeric_transform) == "auto":
        transform = AutoNumericTransformer(random_state=random_state)
    else:
        transform = make_numeric_transformer(numeric_transform, random_state)
    return Pipeline(
        [
            ("impute", _make_numeric_imputer(impute_numeric)),
            ("transform", transform),
        ]
    )


def _make_categorical_pipeline(
    categorical_encoding, impute_categorical, rare_min_frequency
) -> Pipeline:
    encoding = _normalize_name(categorical_encoding)
    if impute_categorical == "constant":
        imputer = SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)
    else:
        imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=rare_min_frequency,
            sparse_output=True,
            dtype=np.float32,
        )
        return Pipeline([("impute", imputer), ("encode", encoder)])
    if encoding == "ordinal":
        return Pipeline(
            [
                ("impute", imputer),
                ("rare", RareCategoryGrouper(min_frequency=rare_min_frequency)),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-2,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
    if encoding == "frequency":
        return Pipeline([("impute", imputer), ("encode", FrequencyEncoder())])
    raise ValueError(
        f"Unknown categorical_encoding {categorical_encoding!r}; choose from {CATEGORICAL_ENCODINGS}"
    )


# Suffixes of the cyclical seasonality encodings emitted by
# `PanelFeatureEngineer`.
_CYCLICAL_SUFFIXES: tuple[str, ...] = ("_month_sin", "_month_cos")


def _cyclical_selector(df: pd.DataFrame) -> list:
    """Columns holding a cyclical (sin/cos) seasonality encoding.

    TEORÍA: these are already normalised by construction -- ``sin``/``cos`` of
    the month angle live in ``[-1, 1]`` with a *meaningful* scale, so there is
    nothing for a scaler to fix. Worse, fitting one on them is actively unsafe
    under a chronological split: a short training window covers only a few
    months, so the encoding is near-constant there and its fitted variance is
    ~0. Every unseen month at transform time is then divided by that tiny
    number and explodes -- a 3-month training window (Nov/Dec/Jan) drove
    ``month_cos`` to 4.9e18 on the June test rows, which in turn blew up the
    VAE's MSE. Passing them through untouched keeps a bounded, comparable,
    genuinely cyclical feature.
    """
    return [
        c for c in df.columns
        if isinstance(c, str) and c.endswith(_CYCLICAL_SUFFIXES)
    ]


def _numeric_non_cyclical_selector(df: pd.DataFrame) -> list:
    """Numeric columns except the cyclical encodings (see :func:`_cyclical_selector`)."""
    cyclical = set(_cyclical_selector(df))
    return [
        c for c in df.columns
        if c not in cyclical
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def build_preprocessing_pipeline(
    schema: PanelSchema,
    numeric_transform: str = "yeo-johnson",
    categorical_encoding: str = "onehot",
    impute_numeric: str = "zero",
    impute_categorical: str = "most_frequent",
    add_missing_indicators: bool = True,
    add_panel_features: bool = True,
    rare_min_frequency: float = 0.001,
    panel_feature_cols: Optional[Sequence[str]] = None,
    ratio_features: Optional[Sequence[tuple[str, str, str]]] = None,
    lag_horizons: Optional[Sequence[int]] = None,
    fit_window_mask: Optional[np.ndarray] = None,
    sparse_threshold: float = 0.3,
    random_state: int = 0,
) -> Pipeline:
    """Assemble a fit/transform-able, picklable preprocessing pipeline.

    Args:
        schema: Panel schema; ``entity_col``/``time_col`` are treated as keys
            (dropped from the feature matrix, used for panel features).
        numeric_transform: One of :data:`NUMERIC_TRANSFORMS`. Selectable by
            name so an Optuna study can tune it as a categorical.
        categorical_encoding: One of :data:`CATEGORICAL_ENCODINGS`.
        impute_numeric: Numeric fill strategy. ``"zero"`` (the default) fills
            NaN with 0.0; ``"median"``/``"mean"``/``"most_frequent"`` use the
            corresponding sklearn statistic instead. See
            :func:`_make_numeric_imputer` for why zero is the default and what
            it costs. `main.py` exposes ``--no-zero-impute`` to switch to
            ``"median"`` without editing code.
        impute_categorical: "most_frequent" or "constant" (fills "__missing__").
        add_missing_indicators: Append a 0/1 flag per NaN-bearing column
            (defaults on -- upstream MNAR missingness is informative).
        add_panel_features: Append within-entity lag/diff/own-z + seasonality
            features (defaults on; toggle off to ablate).
        rare_min_frequency: Categories below this fraction are collapsed (into
            the one-hot infrequent bin, or the ``__rare__`` bucket for ordinal).
        panel_feature_cols: Override the monetary columns panel features derive
            from; ``None`` uses the sensible default set.
        ratio_features: Override the ``(numerator, denominator, name)`` ratio
            triples; ``None`` uses :data:`_DEFAULT_RATIO_FEATURES`, ``()``
            disables them.
        lag_horizons: Override the within-entity contrast horizons; ``None``
            uses :data:`_DEFAULT_LAG_HORIZONS` ``(1, 3, 6)``. Horizons at or
            beyond the panel's depth are dropped automatically.
        sparse_threshold: ColumnTransformer sparse/dense cut-over.
        random_state: Seed for the (stochastic) QuantileTransformer and for the
            ``"auto"`` transform diagnostics.

    Returns:
        A two-step ``Pipeline`` (``panel_features`` then ``column_transform``).
    """
    engineer = PanelFeatureEngineer(
        entity_col=schema.entity_col,
        time_col=schema.time_col,
        add_panel_features=add_panel_features,
        feature_cols=panel_feature_cols,
        ratio_features=ratio_features,
        lag_horizons=lag_horizons,
        fit_window_mask=fit_window_mask,
    )

    cat_selector = make_column_selector(dtype_include=["object", "category"])
    bool_selector = make_column_selector(dtype_include="bool")

    transformers = [
        ("num", _make_numeric_pipeline(numeric_transform, impute_numeric, random_state),
         _numeric_non_cyclical_selector),
        # Cyclical seasonality bypasses the scaler entirely -- see
        # `_cyclical_selector` for why fitting a scaler on it is unsafe.
        ("cyc", "passthrough", _cyclical_selector),
        ("cat", _make_categorical_pipeline(categorical_encoding, impute_categorical, rare_min_frequency), cat_selector),
        ("bool", FunctionTransformer(_to_float32, feature_names_out="one-to-one"), bool_selector),
    ]
    if add_missing_indicators:
        transformers.append(
            ("missing", MissingnessIndicator(only_missing=True), _numeric_non_cyclical_selector)
        )

    column_transform = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=True,
    )

    return Pipeline([("panel_features", engineer), ("column_transform", column_transform)])


def categorical_feature_mask(feature_names: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``feature_names``: ``True`` for categorical-derived columns.

    Identification is purely by **column dtype**, decided upstream and recorded
    in the name: `build_preprocessing_pipeline` routes columns to the
    categorical branch with
    ``make_column_selector(dtype_include=["object", "category"])``, and the
    ColumnTransformer stamps that branch's outputs with
    :data:`CATEGORICAL_FEATURE_PREFIX`. Nothing here inspects column *names* or
    keeps a hardcoded list, so a new string column in the source data is picked
    up automatically.

    Entity/period keys never appear: `PanelFeatureEngineer` drops them before
    the ColumnTransformer sees anything (they are keys, not features).
    """
    return np.array(
        [str(name).startswith(CATEGORICAL_FEATURE_PREFIX) for name in feature_names],
        dtype=bool,
    )


def group_name_by_source(name: str, categorical_columns: Sequence[str]) -> str:
    """Map one transformed feature name back to its *reporting group*.

    A categorical-derived name is grouped under its original source column
    (e.g. ``cat__region_North`` and ``cat__region_South`` both group under
    ``"region"``); everything else groups under itself unchanged. This exists
    because one-hot encoding turns a single string column into one column
    *per category*, and any per-feature ranking (VAE reconstruction error,
    SHAP importance) that does not undo that split lets a single
    high-cardinality categorical dominate a ranking by column *count*, not by
    how informative it actually is -- see `docs/models_vae.md` and
    `CONTEXT.md` on VAE feature attribution.

    ``categorical_columns`` must be the *original* (pre-transform) categorical
    column names (e.g. ``df.select_dtypes(include=["object", "category"]).
    columns``) -- matched longest-first so a name like ``"region_type"`` is
    preferred over ``"region"`` when both are real source columns and the
    derived name is ``cat__region_type_X``.
    """
    if not name.startswith(CATEGORICAL_FEATURE_PREFIX):
        return name
    stripped = name[len(CATEGORICAL_FEATURE_PREFIX):]
    for col in sorted(categorical_columns, key=len, reverse=True):
        if stripped == col or stripped.startswith(col + "_"):
            return col
    return name  # defensive: unrecognized shape (e.g. a future encoding), keep as its own group


def aggregate_attribution_by_source(
    per_feature: dict, categorical_columns: Sequence[str],
) -> dict:
    """Sum a ``{feature: value}`` attribution dict's one-hot-derived entries
    back under their original categorical source column.

    Values are **summed**, not averaged: a per-feature reconstruction error
    (or any additive attribution) is exactly what sums to the row's total
    score, so summing a source column's one-hot slices back together answers
    "how much of the total does reconstructing/attributing to *this original
    variable* cost" -- the fair, apples-to-apples comparison against a single
    numeric column that one-hot's column-count inflation otherwise breaks.
    Non-categorical entries pass through unchanged. Returned sorted
    descending, same convention as the ungrouped dict.
    """
    grouped: dict = {}
    for name, value in per_feature.items():
        key = group_name_by_source(name, categorical_columns)
        grouped[key] = grouped.get(key, 0.0) + float(value)
    return dict(sorted(grouped.items(), key=lambda kv: kv[1], reverse=True))


def split_matrix_for_model(X, feature_names: Sequence[str], model: str):
    """Return ``(X_model, names_model)`` -- the view of ``X`` a detector may use.

    THEORY -- why the two detectors get different columns:

    * **Isolation Forest** splits on ``uniform(min, max)`` of one feature at a
      time, so a split is an ordering statement: "value below / above this
      cut". A one-hot column is 0/1 with no meaningful interior, so every split
      on it degenerates to "has this level / does not", and with high-cardinality
      categoricals the encoding contributes many near-constant columns that
      dilute ``max_features`` sampling without adding isolable structure.
      Excluding them concentrates the forest on the continuous geometry it is
      actually good at.
    * **VAE** reconstructs its whole input vector, and a one-hot block is
      reconstructible in exactly the way the architecture expects (the decoder
      predicts the level's probability mass). Categorical context is genuinely
      informative for the "contextual" anomaly definition -- an amount that is
      ordinary for one channel and extreme for another -- so the VAE keeps it.

    Args:
        X: The full model-ready matrix from `fit_transform_panel` (dense or sparse).
        feature_names: Names aligned to ``X``'s columns.
        model: ``"iforest"`` (categoricals dropped) or anything else (full matrix).

    Returns:
        ``(X_model, names_model)``. For non-iforest models this is ``X`` and
        ``feature_names`` unchanged (no copy).
    """
    if model != "iforest":
        return X, list(feature_names)
    cat_mask = categorical_feature_mask(feature_names)
    if not cat_mask.any():
        return X, list(feature_names)
    keep = ~cat_mask
    X_kept = X[:, keep] if not sp.issparse(X) else X.tocsc()[:, keep].tocsr()
    kept_names = [n for n, k in zip(feature_names, keep) if k]
    return X_kept, kept_names


def _extract_keys(df: pd.DataFrame, schema: PanelSchema) -> pd.DataFrame:
    """Return the ``(entity_id, period)`` frame, row-aligned to ``df``."""
    key_cols = [c for c in (schema.entity_col, schema.time_col) if c is not None and c in df.columns]
    return df[key_cols].reset_index(drop=True)


def _matrix_profile(X) -> tuple[int, int, float, bool, int]:
    """Return (n_rows, n_cols, density, is_sparse, n_nonfinite)."""
    is_sparse = sp.issparse(X)
    if is_sparse:
        Xc = X.tocsr()
        nnz = Xc.nnz
        density = nnz / (Xc.shape[0] * Xc.shape[1]) if Xc.shape[1] else 0.0
        n_nonfinite = int((~np.isfinite(Xc.data)).sum())
        return Xc.shape[0], Xc.shape[1], density, True, n_nonfinite
    Xd = np.asarray(X)
    density = float(np.count_nonzero(Xd)) / Xd.size if Xd.size else 0.0
    return Xd.shape[0], Xd.shape[1], density, False, int((~np.isfinite(Xd)).sum())


# Any |value| above this in the transformed matrix is almost certainly a
# fitted-scaler blow-up rather than real signal: the other transforms all
# standardise, so legitimate columns land within a few dozen units.
_EXTREME_MAGNITUDE = 1e6


def _warn_on_extreme_magnitudes(X, feature_names: list, log: logging.Logger) -> None:
    """Log the columns whose transformed magnitude looks like a scaling blow-up."""
    if sp.issparse(X):
        Xc = X.tocsr()
        if Xc.nnz == 0:
            return
        col_max = np.zeros(Xc.shape[1], dtype=float)
        Xabs = abs(Xc).tocsc()
        for j in range(Xc.shape[1]):
            seg = Xabs.data[Xabs.indptr[j]:Xabs.indptr[j + 1]]
            col_max[j] = seg.max() if seg.size else 0.0
    else:
        Xd = np.asarray(X, dtype=float)
        if Xd.size == 0:
            return
        col_max = np.nanmax(np.abs(Xd), axis=0)

    bad = np.flatnonzero(col_max > _EXTREME_MAGNITUDE)
    if bad.size == 0:
        return
    worst = bad[np.argsort(-col_max[bad])][:5]
    named = ", ".join(
        f"{feature_names[j] if j < len(feature_names) else j}={col_max[j]:.3e}"
        for j in worst
    )
    log.warning(
        "%d feature(s) exceed |%.0e| after transformation: %s. This usually means "
        "a column had ~zero variance in the fit block and the scaler is amplifying "
        "unseen values -- check the fit_mask window.",
        int(bad.size), _EXTREME_MAGNITUDE, named,
    )


def fit_transform_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    logger: Optional[logging.Logger] = None,
    fit_mask: Optional[np.ndarray] = None,
    **config,
):
    """Fit the preprocessing pipeline on ``df`` and return ``(X, keys, names)``.

    Args:
        df: Raw panel DataFrame (must still contain the key columns so panel
            features can be computed).
        schema: The panel schema from the loader.
        logger: Optional logger; defaults to the project logger.
        fit_mask: Optional boolean row mask (typically the in-time rows) that
            restricts what the **estimated** part of the pipeline learns from.
            ``None`` fits everything on every row (the historical behaviour).
        **config: Forwarded verbatim to :func:`build_preprocessing_pipeline`.

    The ``fit_mask`` split, and why it is not a plain ``fit``/``transform``
    -------------------------------------------------------------------------
    The pipeline has two stages, and only one of them can leak:

    1. :class:`PanelFeatureEngineer` estimates **nothing**. Lags and diffs look
       strictly backwards, ``own_z`` uses only strictly-prior periods, ratios
       are within-row and the month encoding is a function of the timestamp.
       It runs over the **whole panel**.
    2. The ``ColumnTransformer`` estimates everything that can leak: imputation
       medians, scaler means/variances, the Yeo-Johnson exponents, one-hot
       categories, training frequencies, the ``"auto"`` per-column transform
       choice. Only this stage is fitted on ``df[fit_mask]``.

    TEORÍA: the tempting shortcut -- fit the pipeline on the in-time rows, then
    call ``transform`` on the out-of-time rows alone -- silently destroys the
    panel features. ``PanelFeatureEngineer`` computes lags *within entity across
    periods*; handed only the last period, ``shift(1)`` finds no history inside
    that subset, every lag/diff/own-z becomes NaN and is filled with ``0.0``.
    The 20 panel features would be identically zero on exactly the rows the
    model is evaluated on. Splitting by *stage* rather than by *call* fixes the
    leak without touching the time-series dependencies.

    Returns:
        ``(X, keys, feature_names)`` where ``X`` is the (possibly sparse)
        feature matrix, ``keys`` is the ``(entity_id, period)`` DataFrame
        aligned row-for-row with ``X`` for join-back to ground truth, and
        ``feature_names`` is a list matching ``X``'s column count.
    """
    log = logger or setup_logging()
    keys = _extract_keys(df, schema)

    with log_phase("preprocessing.fit_transform_panel", log):
        log.info(
            "Preprocessing input: %d rows x %d columns; keys=%s; config=%s",
            df.shape[0], df.shape[1],
            [c for c in (schema.entity_col, schema.time_col) if c],
            {k: config[k] for k in sorted(config)},
        )
        n_in_nan = int(df.isna().to_numpy().sum())

        # The fit window also decides which contrast horizons are learnable, so
        # it is handed to the engineer as well (it never restricts *computation*
        # -- stage 1 still runs over the full panel).
        if fit_mask is not None:
            config.setdefault("fit_window_mask", fit_mask)
        pipeline = build_preprocessing_pipeline(schema, **config)
        engineer = pipeline.named_steps["panel_features"]
        column_transform = pipeline.named_steps["column_transform"]

        if fit_mask is None:
            X = pipeline.fit_transform(df)
        else:
            mask = np.asarray(fit_mask, dtype=bool).ravel()
            if mask.shape[0] != len(df):
                raise ValueError(
                    f"fit_mask has {mask.shape[0]} entries but df has {len(df)} rows"
                )
            if not mask.any():
                raise ValueError("fit_mask selects no rows")
            # Stage 1 over the full panel (causal, estimates nothing) ...
            features = engineer.fit_transform(df)
            # ... stage 2 estimated on the masked subset only.
            column_transform.fit(features.loc[mask])
            X = column_transform.transform(features)
            log.info(
                "fit_mask: ColumnTransformer fitted on %d/%d rows (%.1f%%); panel "
                "features computed over the full panel so out-of-fit rows keep "
                "their causal history",
                int(mask.sum()), len(df), 100.0 * mask.mean(),
            )

        feature_names = [
            str(c) for c in column_transform.get_feature_names_out(
                engineer.get_feature_names_out()
            )
        ]

        n_rows, n_cols, density, is_sparse, n_nonfinite = _matrix_profile(X)
        log.info(
            "Panel feature engineering: added %d features from %d panel source "
            "column(s) and %d ratio(s)",
            len(engineer.engineered_columns_), len(engineer.panel_cols_),
            len(engineer.ratio_specs_),
        )
        numeric_step = column_transform.named_transformers_.get("num")
        auto_step = (
            numeric_step.named_steps.get("transform")
            if hasattr(numeric_step, "named_steps") else None
        )
        if isinstance(auto_step, AutoNumericTransformer):
            picked = auto_step.chosen_transforms_
            log.info(
                "Auto numeric transform: %s",
                {name: picked.count(name) for name in sorted(set(picked))},
            )
        log.info(
            "Preprocessing output: %d rows x %d columns (%s, density=%.4f); "
            "resolved %d input NaNs; %d non-finite values in the matrix",
            n_rows, n_cols, "sparse" if is_sparse else "dense", density, n_in_nan, n_nonfinite,
        )
        if len(feature_names) != n_cols:
            log.warning(
                "feature name count (%d) != matrix width (%d)", len(feature_names), n_cols
            )

        # Blow-up guard. A column whose training variance is ~0 makes any fitted
        # scaler amplify unseen values without limit; the symptom is a huge
        # max|x| that then destabilises gradient-based models downstream. This
        # is cheap and names the offending columns instead of leaving a silent
        # 1e18 in the matrix.
        _warn_on_extreme_magnitudes(X, feature_names, log)

    return X, keys, feature_names
