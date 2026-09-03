from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


LEAKAGE_PATTERNS = (
    r"confirmed[_-]?fraud",
    r"investigation[_-]?(result|outcome)",
    r"chargeback[_-]?(after|future|90d)",
    r"case[_-]?disposition",
    r"label$",
    r"target$",
)


def _population_row(
    reference: pd.Series, scored: pd.Series, feature: str
) -> dict[str, float | str]:
    left = reference.dropna().to_numpy(dtype=float)
    right = scored.dropna().to_numpy(dtype=float)
    ks = ks_2samp(left, right) if left.size and right.size else None
    out_of_range = np.mean((right < np.min(left)) | (right > np.max(left))) if left.size else np.nan
    return {
        "feature": feature,
        "reference_missing_rate": float(reference.isna().mean()),
        "scored_missing_rate": float(scored.isna().mean()),
        "ks_statistic": float(ks.statistic) if ks else np.nan,
        "ks_pvalue": float(ks.pvalue) if ks else np.nan,
        "wasserstein_distance": float(wasserstein_distance(left, right)) if left.size and right.size else np.nan,
        "out_of_reference_range_rate": float(out_of_range),
    }


def compare_populations(
    reference: pd.DataFrame, scored: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = [_population_row(reference[f], scored[f], f) for f in features]
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def leakage_name_warnings(features: list[str]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    for feature in features:
        if any(re.search(pattern, feature, flags=re.IGNORECASE) for pattern in LEAKAGE_PATTERNS):
            warnings.append({
                "code": "possible_post_outcome_feature",
                "severity": "high",
                "feature": feature,
                "message": "Name suggests information that may not exist at scoring time.",
            })
    return warnings

