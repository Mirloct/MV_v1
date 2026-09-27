from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


EPSILON = 1e-12
#: Absolute lower bound of a contribution's scale when the relative floor is on (see
#: ``residual_contributions``): keeps a variable whose reference contributions are all ~0 from
#: turning a tiny numeric difference into a huge normalised value.
MIN_SCALE_WHEN_FLOORED = 1e-6


def anomaly_percentile(
    reference_scores: np.ndarray, scored_scores: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference_scores, dtype=float).reshape(-1)
    scored = np.asarray(scored_scores, dtype=float).reshape(-1)
    if reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("reference_scores must be finite and non-empty")
    if not np.isfinite(scored).all():
        raise ValueError("scored_scores must be finite")
    ordered = np.sort(reference)
    return np.searchsorted(ordered, scored, side="right") / ordered.size


def residual_contributions(
    reference_residuals: np.ndarray, residuals: np.ndarray, scale_floor_fraction: float = 0.0,
    center: bool = False,
) -> np.ndarray:
    """``|contribution| / scale`` per variable, ``scale`` = MAD of the reference (fallbacks below).

    ``scale_floor_fraction`` (default ``0.0`` = historical behaviour) puts a floor under the scale:
    ``scale >= scale_floor_fraction * mean(reference contribution)`` and ``>= MIN_SCALE_WHEN_FLOORED``
    (with ``center=True`` the mean is that of the *centred excess*, i.e. after the reference median is removed).
    Needed when the contributions are one-per-variable negative log-likelihoods or BCE terms: a
    variable the model reconstructs almost perfectly has a MAD close to zero, and dividing by it
    would let that one variable dominate every top-k. With ``0.1`` a contribution is never divided
    by less than a tenth of its variable's average reference contribution.

    ``center=True`` (default ``False`` = historical) first subtracts each variable's reference
    *median* and keeps the excess (``max(c - median_ref, 0)``) for both reference and scored values.
    A residual ``|x - x^|`` is naturally centred at zero, but a negative log-likelihood or a BCE term
    is not: a categorical whose contribution is (almost) constant -- e.g. a uniform many-level variable,
    NLL = log(cardinality) for every row -- has MAD ~ 0, and without centring it would look "high" on
    every row and monopolise the top-k. Centred, a constant contribution is 0 and only the *excess over
    the variable's usual level* competes. Known limit: a variable whose reference excess is essentially
    zero is normalised by a tiny scale (bounded below by ``MIN_SCALE_WHEN_FLOORED``), so a small real
    change in it is amplified.
    """
    reference = np.abs(np.asarray(reference_residuals, dtype=float))
    values = np.abs(np.asarray(residuals, dtype=float))
    if center:
        median = np.nanmedian(reference, axis=0)
        reference = np.maximum(reference - median, 0.0)
        values = np.maximum(values - median, 0.0)
    if reference.ndim != 2 or values.ndim != 2:
        raise ValueError("residual arrays must be 2-dimensional")
    if reference.shape[1] != values.shape[1]:
        raise ValueError("reference and scored residual feature counts must match")
    median = np.nanmedian(reference, axis=0)
    mad = np.nanmedian(np.abs(reference - median), axis=0) * 1.4826
    fallback = np.maximum(median, np.nanmean(reference, axis=0))
    scale = np.where(mad > EPSILON, mad, np.maximum(fallback, EPSILON))
    if scale_floor_fraction > 0.0:
        scale = np.maximum(scale, float(scale_floor_fraction) * np.nanmean(reference, axis=0))
        scale = np.maximum(scale, MIN_SCALE_WHEN_FLOORED)
    return values / scale


def _largest_k_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, values.shape[1])
    partitioned = np.partition(values, values.shape[1] - k, axis=1)
    return partitioned[:, -k:].mean(axis=1)


def reconstruction_scores(
    reference_residuals: np.ndarray,
    scored_residuals: np.ndarray,
    top_k: int,
    scale_floor_fraction: float = 0.0,
    center: bool = False,
) -> pd.DataFrame:
    contributions = residual_contributions(reference_residuals, scored_residuals, scale_floor_fraction, center)
    return pd.DataFrame({
        "recon_mean": contributions.mean(axis=1),
        "recon_topk": _largest_k_mean(contributions, top_k),
        "recon_max": contributions.max(axis=1),
    })


def row_kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    mu_array = np.asarray(mu, dtype=float)
    logvar_array = np.asarray(logvar, dtype=float)
    if mu_array.shape != logvar_array.shape or mu_array.ndim != 2:
        raise ValueError("mu and logvar must be equal-size 2-dimensional arrays")
    return -0.5 * np.sum(1.0 + logvar_array - mu_array**2 - np.exp(logvar_array), axis=1)


def latent_diagnostics(
    mu: np.ndarray,
    logvar: np.ndarray,
    active_variance_threshold: float = 1e-3,
) -> dict[str, object]:
    mu_array = np.asarray(mu, dtype=float)
    logvar_array = np.asarray(logvar, dtype=float)
    if mu_array.shape != logvar_array.shape or mu_array.ndim != 2:
        raise ValueError("mu and logvar must be equal-size 2-dimensional arrays")
    variances = np.var(mu_array, axis=0)
    kl_by_unit = -0.5 * np.mean(
        1.0 + logvar_array - mu_array**2 - np.exp(logvar_array), axis=0
    )
    active = variances > active_variance_threshold
    return {
        "latent_dimensions": int(mu_array.shape[1]),
        "active_units": int(active.sum()),
        "collapsed_fraction": float(1.0 - active.mean()),
        "mu_variance_by_unit": variances.tolist(),
        "mean_kl_by_unit": kl_by_unit.tolist(),
    }


def latent_mahalanobis(
    reference_mu: np.ndarray, scored_mu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference_mu, dtype=float)
    scored = np.asarray(scored_mu, dtype=float)
    if reference.ndim != 2 or scored.ndim != 2:
        raise ValueError("latent means must be 2-dimensional")
    if reference.shape[1] != scored.shape[1]:
        raise ValueError("latent dimensions must match")
    model = LedoitWolf().fit(reference)
    return model.mahalanobis(reference), model.mahalanobis(scored)

