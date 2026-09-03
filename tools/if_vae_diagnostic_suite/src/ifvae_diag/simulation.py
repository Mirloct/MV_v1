from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DiagnosticConfig


def _normal_features(
    rng: np.random.Generator, rows: int, weights: np.ndarray
) -> np.ndarray:
    latent = rng.normal(size=(rows, 3))
    return latent @ weights + rng.normal(scale=0.25, size=(rows, weights.shape[1]))


def _attach_model_outputs(
    frame: pd.DataFrame,
    reconstruction: np.ndarray,
    mu: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    for index, feature in enumerate(features):
        result[f"recon__{feature}"] = reconstruction[:, index]
    for index in range(mu.shape[1]):
        result[f"mu__{index}"] = mu[:, index]
        result[f"logvar__{index}"] = 0.0
    return result


def generate_synthetic_case(
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, DiagnosticConfig]:
    rng = np.random.default_rng(seed)
    features = [f"feature_{index}" for index in range(8)]
    weights = rng.normal(size=(3, len(features)))
    reference_x = _normal_features(rng, 500, weights)
    scored_x = _normal_features(rng, 220, weights)
    labels = np.zeros(len(scored_x), dtype=int)
    families = np.full(len(scored_x), "normal", dtype=object)

    if_only = np.arange(180, 190)
    vae_only = np.arange(190, 200)
    shared = np.arange(200, 210)
    labels[np.r_[if_only, vae_only, shared]] = 1
    families[if_only] = "point_extrapolation"
    families[vae_only] = "relationship_break"
    families[shared] = "shared_extreme"
    extreme_features = [0, 3, 6]
    scored_x[np.ix_(if_only, extreme_features)] = 12.0
    scored_x[np.ix_(shared, extreme_features)] = 12.0

    reference_recon = reference_x + rng.normal(scale=0.05, size=reference_x.shape)
    scored_recon = scored_x + rng.normal(scale=0.05, size=scored_x.shape)
    scored_recon[if_only] = scored_x[if_only]
    scored_recon[vae_only, 1] = scored_x[vae_only, 1] - 8.0
    scored_recon[shared, 1] = scored_x[shared, 1] - 8.0

    reference_mu = reference_x[:, :3] / np.std(reference_x[:, :3], axis=0)
    scored_mu = scored_x[:, :3] / np.std(reference_x[:, :3], axis=0)
    reference = pd.DataFrame(reference_x, columns=features)
    scored = pd.DataFrame(scored_x, columns=features)
    reference.insert(0, "id", [f"ref-{i:04d}" for i in range(len(reference))])
    scored.insert(0, "id", [f"score-{i:04d}" for i in range(len(scored))])
    reference["event_time"] = pd.date_range("2026-01-01", periods=len(reference), freq="h")
    scored["event_time"] = pd.date_range("2026-02-01", periods=len(scored), freq="h")
    scored["label"] = labels
    scored["segment"] = np.where(np.arange(len(scored)) % 2, "retail", "business")
    scored["anomaly_family"] = families
    reference = _attach_model_outputs(reference, reference_recon, reference_mu, features)
    scored = _attach_model_outputs(scored, scored_recon, scored_mu, features)
    config = DiagnosticConfig(
        features=features,
        alert_budgets=[20, 30, 50],
        random_seeds=[7, 19, 43],
        if_n_estimators=160,
        top_k_residuals=1,
        percentile_threshold=0.95,
    )
    return reference, scored, config
