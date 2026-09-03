from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


EPSILON = 1e-12


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
    reference_residuals: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    reference = np.abs(np.asarray(reference_residuals, dtype=float))
    values = np.abs(np.asarray(residuals, dtype=float))
    if reference.ndim != 2 or values.ndim != 2:
        raise ValueError("residual arrays must be 2-dimensional")
    if reference.shape[1] != values.shape[1]:
        raise ValueError("reference and scored residual feature counts must match")
    median = np.nanmedian(reference, axis=0)
    mad = np.nanmedian(np.abs(reference - median), axis=0) * 1.4826
    fallback = np.maximum(median, np.nanmean(reference, axis=0))
    scale = np.where(mad > EPSILON, mad, np.maximum(fallback, EPSILON))
    return values / scale


def _largest_k_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    k = min(top_k, values.shape[1])
    partitioned = np.partition(values, values.shape[1] - k, axis=1)
    return partitioned[:, -k:].mean(axis=1)


def reconstruction_scores(
    reference_residuals: np.ndarray,
    scored_residuals: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    contributions = residual_contributions(reference_residuals, scored_residuals)
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

