from __future__ import annotations

import numpy as np
import pandas as pd


def assign_disagreement_quadrant(
    if_percentile: np.ndarray,
    vae_percentile: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if_high = np.asarray(if_percentile) >= threshold
    vae_high = np.asarray(vae_percentile) >= threshold
    output = np.full(len(if_high), "NEITHER", dtype=object)
    output[if_high & ~vae_high] = "IF_ONLY"
    output[~if_high & vae_high] = "VAE_ONLY"
    output[if_high & vae_high] = "BOTH"
    return output


def build_autopsy(
    selected: pd.Series | np.ndarray,
    features: pd.DataFrame,
    reconstruction: pd.DataFrame,
    contributions: pd.DataFrame,
    ids: list[object] | np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mask = np.asarray(selected, dtype=bool)
    id_array = np.asarray(ids)
    for row_index in np.flatnonzero(mask):
        order = np.argsort(-contributions.iloc[row_index].to_numpy(), kind="stable")
        for rank, feature_index in enumerate(order, start=1):
            feature = features.columns[feature_index]
            rows.append({
                "id": id_array[row_index],
                "feature": feature,
                "rank": rank,
                "observed": float(features.iloc[row_index, feature_index]),
                "reconstructed": float(reconstruction.iloc[row_index, feature_index]),
                "absolute_residual": float(abs(
                    features.iloc[row_index, feature_index]
                    - reconstruction.iloc[row_index, feature_index]
                )),
                "standardized_contribution": float(
                    contributions.iloc[row_index, feature_index]
                ),
            })
    return pd.DataFrame(rows)


def detect_score_orientation_risk(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | bool | str]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    positive_scores = scores_array[labels_array == 1]
    negative_scores = scores_array[labels_array == 0]
    if not positive_scores.size or not negative_scores.size:
        return {"risk": False, "reason": "Both classes are required."}
    positive_median = float(np.median(positive_scores))
    negative_median = float(np.median(negative_scores))
    return {
        "risk": positive_median < negative_median,
        "positive_median": positive_median,
        "negative_median": negative_median,
        "reason": "Positive median is below negative median; verify score direction.",
    }

