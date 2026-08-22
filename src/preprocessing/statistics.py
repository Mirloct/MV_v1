"""Statistical justification of the chosen numeric transformation.

The project requires an explicit, statistical rationale for *why* a given
numeric transform is applied to a given feature. This module quantifies, for
each numeric feature x each candidate transform, how far the transformed
distribution is from Gaussian, and provides a ranking that recommends a
transform per feature under a configurable criterion.

Interpretability at scale
-------------------------
At 1,000,000 rows the p-value of any normality test
(:func:`scipy.stats.normaltest`, Anderson-Darling) is effectively always
~0: with that much data even a trivially small departure from normality is
"significant", so the p-value carries no decision value. We therefore report
the p-value for completeness but base recommendations on *effect-size*
measures that stay interpretable regardless of n:

* ``abs_skewness`` and ``excess_kurtosis`` -- scale-free shape descriptors
  (0 for a Gaussian),
* ``jb_effect`` = ``sqrt(skew**2 / 6 + excess_kurtosis**2 / 24)`` -- the
  per-observation Jarque-Bera statistic; a single, monotone "distance from
  normal" that does not grow with n,
* ``max_p99_ratio`` -- the surviving tail heaviness after transformation.

All plotting subsamples the data (histogramming 1M points at full resolution
is pointless and slow) and saves every figure under
``reports/figures/preprocessing/`` per the project-wide figures convention. The
non-interactive ``Agg`` backend is forced so plotting never blocks.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window, safe in any environment

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.data.loader import PanelSchema  # noqa: E402
from src.preprocessing.pipeline import NUMERIC_TRANSFORMS, make_numeric_transformer  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.logging_config import log_phase, setup_logging  # noqa: E402

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "infer_numeric_features",
    "apply_named_transform",
    "compute_transform_diagnostics",
    "recommend_transform",
    "plot_transform_diagnostics",
]

DEFAULT_FIGURE_DIR = paths.FIGURES_DIR

# Transforms diagnosed by default: every pipeline option except the trivial
# "standard"/"robust" rescalings (which do not change skew/kurtosis), plus the
# untouched "raw" baseline for comparison.
_DEFAULT_DIAG_TRANSFORMS: tuple[str, ...] = ("raw", "log1p", "yeo-johnson", "quantile")


def infer_numeric_features(
    df: pd.DataFrame, schema: Optional[PanelSchema] = None
) -> list[str]:
    """Return continuous numeric feature columns worth diagnosing.

    Excludes the key columns and booleans. Low-cardinality integer-like columns
    (<= 20 distinct values, e.g. small counts/scores) are dropped as well: a
    monotone transform of a near-discrete column is not what this diagnostic is
    for.
    """
    keys = set()
    if schema is not None:
        keys = {schema.entity_col, schema.time_col, schema.target_col} - {None}
    cols = []
    for col in df.columns:
        if col in keys:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            continue
        if s.dropna().nunique() <= 20:
            continue
        cols.append(col)
    return cols


def apply_named_transform(
    values: np.ndarray, transform: str, random_state: int = 0
) -> np.ndarray:
    """Apply a named numeric transform to a 1-D array (NaNs dropped upstream).

    ``"raw"``/``"passthrough"`` return the values unchanged. The remaining names
    are exactly the pipeline's :data:`NUMERIC_TRANSFORMS`, so the diagnostics
    reflect what the pipeline would actually do.
    """
    name = str(transform).strip().lower()
    values = np.asarray(values, dtype=np.float64)
    if name in ("raw", "passthrough") or values.size == 0:
        return values
    transformer = make_numeric_transformer(name, random_state=random_state)
    if transformer == "passthrough":
        return values
    arr = values.reshape(-1, 1)
    if name == "quantile":
        # n_quantiles must not exceed the sample size (else a noisy warning).
        transformer.set_params(n_quantiles=int(min(1000, max(2, arr.shape[0]))))
    out = transformer.fit_transform(arr)
    return np.asarray(out, dtype=np.float64).ravel()


def _distribution_stats(values: np.ndarray) -> dict:
    """Shape/normality descriptors for a 1-D sample; robust to degeneracy."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    out = {
        "n": n,
        "skewness": np.nan,
        "excess_kurtosis": np.nan,
        "jb_effect": np.nan,
        "normaltest_stat": np.nan,
        "normaltest_pvalue": np.nan,
        "anderson_stat": np.nan,
        "max_p99_ratio": np.nan,
        "abs_skewness": np.nan,
    }
    if n < 8 or np.allclose(v.std(), 0.0):
        return out  # constant / too small: shape stats are undefined

    skew = float(stats.skew(v, bias=False))
    kurt = float(stats.kurtosis(v, fisher=True, bias=False))
    out["skewness"] = skew
    out["abs_skewness"] = abs(skew)
    out["excess_kurtosis"] = kurt
    out["jb_effect"] = float(np.sqrt(skew**2 / 6.0 + kurt**2 / 24.0))
    try:
        nt = stats.normaltest(v)
        out["normaltest_stat"] = float(nt.statistic)
        out["normaltest_pvalue"] = float(nt.pvalue)
    except (ValueError, RuntimeWarning):
        pass
    try:
        out["anderson_stat"] = float(stats.anderson(v, dist="norm").statistic)
    except (ValueError, RuntimeWarning):
        pass
    p99 = float(np.percentile(v, 99))
    if p99 > 0:
        out["max_p99_ratio"] = float(np.max(v) / p99)
    return out


def _subsample(values: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if sample_size and values.size > sample_size:
        idx = rng.choice(values.size, size=sample_size, replace=False)
        return values[idx]
    return values


def compute_transform_diagnostics(
    df: pd.DataFrame,
    schema: Optional[PanelSchema] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    transforms: Sequence[str] = _DEFAULT_DIAG_TRANSFORMS,
    sample_size: int = 50_000,
    random_state: int = 0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Tidy (feature x transform) diagnostics dataframe.

    One row per ``(feature, transform)`` with skewness, excess kurtosis, a
    normality statistic + (deliberately reported-but-not-relied-on) p-value, an
    Anderson-Darling statistic, the ``max/p99`` tail ratio, and the scale-free
    ``jb_effect`` / ``abs_skewness`` effect sizes used for ranking. Data is
    subsampled (seeded) to ``sample_size`` rows per feature so the pass stays
    cheap at 1M rows.
    """
    log = logger or setup_logging()
    if numeric_cols is None:
        numeric_cols = infer_numeric_features(df, schema)
    rng = np.random.default_rng(random_state)

    rows = []
    with log_phase("preprocessing.compute_transform_diagnostics", log):
        log.info(
            "Diagnosing %d numeric feature(s) x %d transform(s) on up to %d sampled rows",
            len(numeric_cols), len(transforms), sample_size,
        )
        for col in numeric_cols:
            base = df[col].to_numpy(dtype=np.float64)
            base = base[np.isfinite(base)]
            base = _subsample(base, sample_size, rng)
            for transform in transforms:
                try:
                    transformed = apply_named_transform(base, transform, random_state=random_state)
                    stat = _distribution_stats(transformed)
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("Diagnostics failed for %s/%s: %s", col, transform, exc)
                    stat = _distribution_stats(np.array([]))
                rows.append({"feature": col, "transform": transform, **stat})

    diagnostics = pd.DataFrame(rows)
    log.info("Computed %d diagnostic rows", len(diagnostics))
    return diagnostics


def recommend_transform(
    diagnostics: pd.DataFrame,
    criterion: str = "jb_effect",
    lower_is_better: bool = True,
    exclude_transforms: Sequence[str] = (),
) -> pd.DataFrame:
    """Recommend one transform per feature under an explicit criterion.

    Args:
        diagnostics: Output of :func:`compute_transform_diagnostics`.
        criterion: Column to optimise (e.g. ``"jb_effect"``, ``"abs_skewness"``,
            ``"excess_kurtosis"``, ``"max_p99_ratio"``).
        lower_is_better: If True, the minimising transform wins (the default,
            appropriate for distance-from-normal criteria).
        exclude_transforms: Transforms to drop before ranking (e.g. ``("raw",)``
            to force a non-trivial transform).

    Returns:
        One row per feature: ``feature``, ``recommended_transform``,
        ``criterion``, ``value``.
    """
    if criterion not in diagnostics.columns:
        raise ValueError(f"criterion {criterion!r} not in diagnostics columns")
    data = diagnostics[~diagnostics["transform"].isin(set(exclude_transforms))].copy()
    data = data.dropna(subset=[criterion])

    recs = []
    for feature, grp in data.groupby("feature", sort=False):
        if grp.empty:
            continue
        idx = grp[criterion].idxmin() if lower_is_better else grp[criterion].idxmax()
        best = grp.loc[idx]
        recs.append(
            {
                "feature": feature,
                "recommended_transform": best["transform"],
                "criterion": criterion,
                "value": float(best[criterion]),
            }
        )
    return pd.DataFrame(recs)


def plot_transform_diagnostics(
    df: pd.DataFrame,
    schema: Optional[PanelSchema] = None,
    features: Optional[Sequence[str]] = None,
    transforms: Sequence[str] = _DEFAULT_DIAG_TRANSFORMS,
    kind: str = "hist",
    out_dir: str = DEFAULT_FIGURE_DIR,
    sample_size: int = 20_000,
    bins: int = 60,
    random_state: int = 0,
    logger: Optional[logging.Logger] = None,
) -> list[str]:
    """Save before/after diagnostic figures, one file per feature.

    Args:
        kind: ``"hist"`` (histograms) or ``"qq"`` (normal QQ plots).
        out_dir: Destination directory; defaults to
            ``reports/figures/preprocessing`` (created if missing). Every figure
            is written here per the project-wide figures convention.
        sample_size: Rows subsampled (seeded) before plotting.

    Returns:
        Absolute paths of the figures written.
    """
    log = logger or setup_logging()
    if features is None:
        features = infer_numeric_features(df, schema)
    if kind not in ("hist", "qq"):
        raise ValueError("kind must be 'hist' or 'qq'")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    saved: list[str] = []
    with log_phase("preprocessing.plot_transform_diagnostics", log):
        for col in features:
            base = df[col].to_numpy(dtype=np.float64)
            base = base[np.isfinite(base)]
            base = _subsample(base, sample_size, rng)
            if base.size < 8:
                log.warning("Skipping figure for '%s': too few finite values", col)
                continue

            n_panels = len(transforms)
            fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.6), squeeze=False)
            for ax, transform in zip(axes[0], transforms):
                values = apply_named_transform(base, transform, random_state=random_state)
                values = values[np.isfinite(values)]
                if values.size < 8 or np.allclose(values.std(), 0.0):
                    ax.text(0.5, 0.5, "degenerate", ha="center", va="center")
                    ax.set_title(f"{transform}")
                    continue
                if kind == "hist":
                    ax.hist(values, bins=bins, color="#3b6ea5", alpha=0.85)
                    skew = float(stats.skew(values, bias=False))
                    ax.set_title(f"{transform}\nskew={skew:.2f}")
                else:
                    stats.probplot(values, dist="norm", plot=ax)
                    ax.set_title(f"{transform}")
                ax.tick_params(labelsize=8)
            fig.suptitle(f"{col} - {kind} across transforms (n={base.size})", fontsize=11)
            fig.tight_layout()

            path = os.path.abspath(os.path.join(out_dir, f"{col}_{kind}.png"))
            fig.savefig(path, dpi=110)
            plt.close(fig)
            saved.append(path)
            log.info("Saved diagnostic figure: %s", path)

    log.info("Wrote %d diagnostic figure(s) to %s", len(saved), os.path.abspath(out_dir))
    return saved
