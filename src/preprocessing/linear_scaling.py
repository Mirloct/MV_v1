"""Stateless linear (affine) rescaling of the continuous block of a panel.

Every transform here is of the form ``y = a*x + b`` applied column-wise with
plain numpy/pandas -- no scikit-learn, no estimator classes, no ``.fit()``.
A function receives a DataFrame, computes its constants and returns the
rescaled DataFrame in one call.

Why a *linear* transform for these two detectors
------------------------------------------------
An affine map changes location and scale only; it leaves the distribution's
*shape* (skewness, kurtosis) and the rank order of every column untouched.
That is exactly what both detectors need, for different reasons:

* **Isolation Forest** splits on axis-parallel thresholds drawn uniformly
  between a column's observed min and max. It is invariant to any monotone
  per-column map, so rescaling cannot change its ranking -- but a *shape*
  transform can, and destructively: a rank/quantile normaliser pulls the far
  tail back into the bulk, which is precisely the signal being detected. A
  linear map keeps the outlier an outlier: after ``(x - median) / IQR`` a point
  6 IQRs out is still 6 IQRs out.

* **VAE** needs the opposite guarantee for a numerical reason. Its loss is a
  mean squared reconstruction error, so a feature on a 1e5 scale contributes
  ~1e10 to the gradient and dominates every other feature -- in this project
  that overflowed to ``NaN`` scores (see ``docs/leakage_free_pipeline.md``,
  "On RobustScaler"). Centring and dividing by a spread statistic puts all
  features in a comparable range, which keeps gradients finite and stops one
  column from monopolising the latent space.

Why the *median/IQR* pair specifically
--------------------------------------
Mean and standard deviation are computed from every row, anomalies included.
On a contaminated sample they are dragged toward the very points being
detected: the mean shifts and the standard deviation inflates, so dividing by
it *shrinks* the anomaly's distance from centre -- the estimator masks the
signal it is meant to expose. The median (breakdown point 50%) and the IQR
(which discards the outer quartiles by construction) are unaffected by a
minority of extreme values, so the scaled tail keeps its magnitude.

Leakage
-------
**The one-shot helpers estimate their constants from whatever rows they are
given.** Calling ``robust_scale(df)`` on a full panel therefore lets test-period
rows influence the median and IQR applied to training rows -- the exact leak
``docs/leakage_free_pipeline.md`` (Phase 3) forbids. For any chronologically
split run, use the two-step form instead, which is still fit-free in the sense
required (a plain dict of numbers, not a stateful object)::

    params = robust_scale_params(df[train_mask])   # constants from train only
    df = apply_linear_scaling(df, params)          # applied to every row

:func:`robust_scale` is the convenience path for exploratory work and for data
that is not time-split.
"""

from __future__ import annotations

import contextlib
import logging
import re
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "KEY_COLUMN_TOKENS",
    "SCALING_METHODS",
    "is_key_column",
    "select_continuous_columns",
    "robust_scale_params",
    "standard_scale_params",
    "minmax_scale_params",
    "scaling_params",
    "apply_linear_scaling",
    "robust_scale",
    "standard_scale",
    "minmax_scale",
]

#: Column names that identify an entity or a period rather than a measurement.
#: Matched against the whole (normalised) name and against its first/last
#: underscore-separated token, so ``id``, ``client_id``, ``id_cliente`` and
#: ``codmes`` are all caught while ``avg_txn_to_income`` (which merely
#: *contains* the letters) is not.
KEY_COLUMN_TOKENS: frozenset[str] = frozenset({
    "id", "ids", "key", "llave", "codigo", "cod",
    "codmes", "codmed", "anomes", "yyyymm", "periodo", "period", "mes",
    "fecha", "date", "datetime", "timestamp", "ts",
})

#: ``method`` values accepted by :func:`scaling_params`.
SCALING_METHODS: tuple[str, ...] = ("robust", "standard", "minmax")

#: Substituted for any non-positive or non-finite scale denominator, so a
#: constant column maps to ``x - centre`` (all zeros) instead of dividing by 0
#: and producing inf/NaN.
_SAFE_SCALE = 1.0

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalise(name: str) -> str:
    """Lowercase ``name`` and collapse punctuation to single underscores."""
    return _NON_ALNUM.sub("_", str(name).strip().lower()).strip("_")


def is_key_column(name: str) -> bool:
    """True when ``name`` reads as an identifier / period column.

    Checked against :data:`KEY_COLUMN_TOKENS` three ways: the full normalised
    name, its first token and its last token. ``"cod_mes"`` normalises to
    ``cod_mes`` whose tokens are ``cod``/``mes`` -- both keys -- so the
    compact and separated spellings behave identically.
    """
    norm = _normalise(name)
    if not norm:
        return False
    if norm in KEY_COLUMN_TOKENS:
        return True
    tokens = norm.split("_")
    return tokens[0] in KEY_COLUMN_TOKENS or tokens[-1] in KEY_COLUMN_TOKENS


def select_continuous_columns(
    df: pd.DataFrame,
    exclude: Sequence[str] = (),
    include: Optional[Sequence[str]] = None,
    detect_keys: bool = True,
    min_unique: int = 0,
) -> list[str]:
    """Columns eligible for linear rescaling.

    Eligible means: a real numeric dtype (``int``/``float``), not a key column,
    not excluded, and not degenerate.

    Args:
        df: Source frame.
        exclude: Column names to skip regardless of dtype -- pass the schema's
            entity/period/target columns here when their names do not follow
            the usual conventions.
        include: When given, restrict the search to these columns; any that are
            non-numeric are still dropped (with the same rules), so this is a
            filter and not an override.
        detect_keys: Apply :func:`is_key_column` name detection. Set False when
            the frame's measurement columns genuinely use key-like names and
            ``exclude`` already lists the real keys.
        min_unique: Drop numeric columns with fewer than this many distinct
            non-null values. Defaults to ``0`` (keep everything numeric); set
            e.g. ``20`` to skip small-count / flag-like integer columns.

    Returns:
        Column names, in the frame's own column order.

    Note:
        ``bool`` columns are excluded even though pandas reports them as
        numeric: they are already 0/1 on a fixed scale, and dividing a flag by
        its IQR makes it neither more comparable nor more interpretable.
    """
    excluded = {str(c) for c in exclude}
    candidates = list(df.columns) if include is None else [
        c for c in include if c in df.columns
    ]

    cols: list[str] = []
    for col in candidates:
        if str(col) in excluded:
            continue
        series = df[col]
        # bool first: is_numeric_dtype(bool) is True, which is not what we want.
        if pd.api.types.is_bool_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue  # strings, categories, datetimes -- left untouched
        if detect_keys and is_key_column(col):
            continue
        if min_unique > 0 and series.dropna().nunique() < min_unique:
            continue
        cols.append(col)
    return cols


def _column_block(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """The selected columns as one float64 matrix (a copy, NaNs preserved)."""
    if not columns:
        return np.empty((len(df), 0), dtype=np.float64)
    return df.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)


def _sanitise_scale(scale: np.ndarray) -> np.ndarray:
    """Replace non-finite or non-positive denominators with :data:`_SAFE_SCALE`.

    Covers the three ways a spread statistic degenerates: a constant column
    (spread 0), an all-NaN column (spread NaN), and -- for min-max -- a column
    whose range underflows. In every case the column becomes ``x - centre``,
    which is finite and constant rather than inf/NaN.
    """
    scale = np.asarray(scale, dtype=np.float64)
    bad = ~np.isfinite(scale) | (scale <= 0.0)
    if bad.any():
        scale = scale.copy()
        scale[bad] = _SAFE_SCALE
    return scale


@contextlib.contextmanager
def _quiet_nan_stats():
    """Silence the ``RuntimeWarning``s numpy raises on degenerate reductions.

    ``nanmedian``/``nanmean`` on an all-NaN or empty column warn and return
    NaN. That is the documented behaviour and this module handles the NaN
    explicitly (:func:`_sanitise_scale` and ``nan_to_num``), so the warning is
    noise. ``np.errstate`` does not cover these -- they are ``warnings.warn``
    calls, not floating-point error states.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
        with np.errstate(invalid="ignore", divide="ignore"):
            yield


def _degenerate_params(method: str, columns: Sequence[str]) -> dict:
    """Identity constants (centre 0, scale 1) for every column.

    Used when there is nothing to estimate from -- no columns, or no rows.
    Applying these leaves the data unchanged, which is the only defensible
    behaviour: an empty sample carries no information about location or
    spread, so inventing one would be worse than doing nothing.
    """
    n = len(columns)
    return _params(method, columns, np.zeros(n), np.ones(n))


def _params(
    method: str, columns: Sequence[str], center: np.ndarray, scale: np.ndarray
) -> dict:
    """Package the constants as a plain, JSON-serialisable dict.

    Deliberately not an object: it can be logged, stored beside the model and
    reloaded without unpickling any class, which is what keeps this module
    free of the ``.fit()`` / estimator machinery.
    """
    return {
        "method": str(method),
        "columns": [str(c) for c in columns],
        "center": {str(c): float(v) for c, v in zip(columns, center)},
        "scale": {str(c): float(v) for c, v in zip(columns, scale)},
    }


def robust_scale_params(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    **select_kwargs,
) -> dict:
    """Median / IQR constants for a robust rescaling: ``(x - median) / IQR``.

    Args:
        df: Rows the constants are estimated from. **For a chronologically
            split run pass only the training block** -- see the module
            docstring on leakage.
        columns: Columns to scale; inferred via
            :func:`select_continuous_columns` when omitted.
        quantile_range: Lower/upper percentile bounding the spread. The default
            ``(25, 75)`` is the interquartile range; widening it to e.g.
            ``(10, 90)`` uses more of the distribution and is therefore less
            robust to contamination.
        **select_kwargs: Forwarded to :func:`select_continuous_columns`.

    Returns:
        ``{"method", "columns", "center", "scale"}`` -- see :func:`_params`.
    """
    lo, hi = float(quantile_range[0]), float(quantile_range[1])
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError(
            f"quantile_range must satisfy 0 <= lo < hi <= 100; got {quantile_range!r}"
        )
    if columns is None:
        columns = select_continuous_columns(df, **select_kwargs)
    block = _column_block(df, columns)
    if block.shape[1] == 0 or block.shape[0] == 0:
        return _degenerate_params("robust", columns)

    with _quiet_nan_stats():
        center = np.nanmedian(block, axis=0)
        quantiles = np.nanpercentile(block, [lo, hi], axis=0)
    q_lo, q_hi = quantiles[0], quantiles[1]
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = _sanitise_scale(q_hi - q_lo)
    return _params("robust", columns, center, scale)


def standard_scale_params(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    **select_kwargs,
) -> dict:
    """Mean / standard-deviation constants: ``(x - mean) / std``.

    Provided for comparison. On data that contains the anomalies being
    detected, prefer :func:`robust_scale_params` -- both statistics here are
    computed from the contaminating points themselves.
    """
    if columns is None:
        columns = select_continuous_columns(df, **select_kwargs)
    block = _column_block(df, columns)
    if block.shape[1] == 0 or block.shape[0] == 0:
        return _degenerate_params("standard", columns)

    with _quiet_nan_stats():
        center = np.nanmean(block, axis=0)
        scale = np.nanstd(block, axis=0)
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    return _params("standard", columns, center, _sanitise_scale(scale))


def minmax_scale_params(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    **select_kwargs,
) -> dict:
    """Min / range constants mapping the observed range onto ``[0, 1]``.

    The least robust of the three: both constants are set by the single most
    extreme value in each direction, so one anomaly compresses every normal row
    into a narrow band. Included because it is occasionally required by a
    downstream bounded activation, not as a recommendation.
    """
    if columns is None:
        columns = select_continuous_columns(df, **select_kwargs)
    block = _column_block(df, columns)
    if block.shape[1] == 0 or block.shape[0] == 0:
        return _degenerate_params("minmax", columns)

    with _quiet_nan_stats():
        lo = np.nanmin(block, axis=0)
        hi = np.nanmax(block, axis=0)
    lo = np.nan_to_num(lo, nan=0.0, posinf=0.0, neginf=0.0)
    return _params("minmax", columns, lo, _sanitise_scale(hi - lo))


def scaling_params(
    df: pd.DataFrame,
    method: str = "robust",
    columns: Optional[Sequence[str]] = None,
    **kwargs,
) -> dict:
    """Dispatch to the ``*_scale_params`` function named by ``method``."""
    name = str(method).strip().lower()
    if name == "robust":
        return robust_scale_params(df, columns=columns, **kwargs)
    if name == "standard":
        return standard_scale_params(df, columns=columns, **kwargs)
    if name == "minmax":
        return minmax_scale_params(df, columns=columns, **kwargs)
    raise ValueError(f"Unknown method {method!r}; choose from {SCALING_METHODS}")


def apply_linear_scaling(
    df: pd.DataFrame,
    params: Mapping,
    inplace: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Apply precomputed constants: ``y = (x - center) / scale``, column-wise.

    Columns named in ``params`` but absent from ``df`` are skipped with a
    warning rather than raising, so a frame that legitimately lost a column
    (e.g. an all-null column dropped upstream) still scales.

    Args:
        df: Frame to transform.
        params: Output of any ``*_scale_params`` function.
        inplace: Mutate and return ``df`` itself instead of a copy. The default
            (``False``) copies, matching the rest of the codebase.
        logger: Optional logger for the skipped-column warning.

    Returns:
        The rescaled frame. Non-listed columns -- strings, keys, booleans --
        are carried through byte-for-byte.
    """
    columns = [str(c) for c in (params.get("columns") or [])]
    present = [c for c in columns if c in df.columns]
    if logger is not None and len(present) != len(columns):
        missing = sorted(set(columns) - set(present))
        logger.warning(
            "Linear scaling: %d column(s) in params not found in the frame; "
            "skipping them: %s", len(missing), ", ".join(missing),
        )

    out = df if inplace else df.copy()
    if not present:
        return out

    center = np.array([float(params["center"][c]) for c in present], dtype=np.float64)
    scale = _sanitise_scale(
        np.array([float(params["scale"][c]) for c in present], dtype=np.float64)
    )
    # One vectorised expression over the whole numeric block: (n, k) - (k,) then
    # / (k,) broadcasts across columns, so there is no per-column Python loop.
    block = _column_block(out, present)
    out[present] = (block - center) / scale
    return out


def robust_scale(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    inplace: bool = False,
    **select_kwargs,
) -> pd.DataFrame:
    """Robust-scale the continuous columns in one call and return the frame.

    Equivalent to ``apply_linear_scaling(df, robust_scale_params(df, ...))``.

    Estimates the median and IQR **from ``df`` itself**, so on a
    chronologically split panel use the two-step form instead -- see the module
    docstring.
    """
    params = robust_scale_params(
        df, columns=columns, quantile_range=quantile_range, **select_kwargs
    )
    return apply_linear_scaling(df, params, inplace=inplace)


def standard_scale(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    inplace: bool = False,
    **select_kwargs,
) -> pd.DataFrame:
    """Standard-scale (mean/std) the continuous columns; returns the frame."""
    params = standard_scale_params(df, columns=columns, **select_kwargs)
    return apply_linear_scaling(df, params, inplace=inplace)


def minmax_scale(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    inplace: bool = False,
    **select_kwargs,
) -> pd.DataFrame:
    """Min-max scale the continuous columns to ``[0, 1]``; returns the frame."""
    params = minmax_scale_params(df, columns=columns, **select_kwargs)
    return apply_linear_scaling(df, params, inplace=inplace)


if __name__ == "__main__":  # pragma: no cover - illustrative, not part of the API
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)

    # A panel with every column kind the selector has to tell apart: keys,
    # strings, a boolean flag, a constant column, one carrying NaN, and a
    # deliberate anomaly in the last row.
    demo = pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106],           # key -> untouched
        "codmes": [202401, 202401, 202402, 202402, 202403, 202403],  # key
        "segmento": ["retail", "pyme", "retail", "corp", "pyme", "retail"],  # str
        "activo": [True, True, False, True, False, True],   # bool -> untouched
        "saldo": [1_000.0, 1_200.0, 950.0, 1_100.0, 1_050.0, 900_000.0],  # anomaly
        "edad": [31, 45, 28, 52, 39, 41],
        "constante": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],       # IQR 0 -> scale 1
        "con_nulos": [2.0, np.nan, 4.0, 6.0, np.nan, 8.0],
    })

    print("=" * 78)
    print("ENTRADA")
    print("=" * 78)
    print(demo)
    print("\ndtypes:\n", demo.dtypes.to_string(), sep="")

    selected = select_continuous_columns(demo)
    print("\nColumnas continuas detectadas:", selected)
    print("Ignoradas:", [c for c in demo.columns if c not in selected])

    scaled = robust_scale(demo)

    print("\n" + "=" * 78)
    print("SALIDA (robust scaling: (x - mediana) / IQR)")
    print("=" * 78)
    print(scaled.round(4))

    # The three guarantees this module exists to provide.
    assert scaled["segmento"].equals(demo["segmento"]), "strings must not change"
    assert scaled["id"].equals(demo["id"]), "id must not change"
    assert scaled["codmes"].equals(demo["codmes"]), "codmes must not change"
    assert scaled["activo"].equals(demo["activo"]), "booleans must not change"
    assert np.isfinite(scaled["constante"]).all(), "IQR=0 must not divide by zero"
    print("\nComprobaciones: strings, id, codmes y booleanos intactos; "
          "columna constante finita.")

    # The anomaly keeps its magnitude -- the point of a linear transform.
    print(f"\nsaldo: la anomalía está a {scaled['saldo'].iloc[-1]:,.1f} IQRs de la "
          f"mediana (antes: {demo['saldo'].iloc[-1]:,.0f} unidades).")

    # Leak-free usage: constants from the first two periods only.
    print("\n" + "=" * 78)
    print("USO SIN FUGA (constantes solo del bloque de entrenamiento)")
    print("=" * 78)
    train_mask = demo["codmes"] <= 202402
    train_params = robust_scale_params(demo[train_mask])
    scaled_nl = apply_linear_scaling(demo, train_params)
    print("Constantes estimadas con", int(train_mask.sum()), "filas de entrenamiento:")
    for col in train_params["columns"]:
        print(f"  {col:<12} center={train_params['center'][col]:>12,.4f}  "
              f"scale={train_params['scale'][col]:>12,.4f}")
    print("\n", scaled_nl.round(4), sep="")
