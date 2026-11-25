"""Numeric quantile policy for the operational OOT review union.

Data sources / inputs: full deduplicated OOT entity-score maps per detector.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — centralized inclusive quantile selection.
"""
from __future__ import annotations

import numpy as np


def review_ids(scores: dict, percentile: float = 99.0) -> set[str]:
    """Return all entities meeting the full-population empirical cutoff."""
    if not np.isfinite(percentile) or not 0 < percentile < 100:
        raise ValueError("review percentile must be finite and strictly between 0 and 100")
    if not scores:
        return set()
    values = np.asarray(list(scores.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OOT review scores must be finite; inspect detector scoring inputs")
    cutoff = float(np.percentile(values, percentile, method="linear"))
    return {str(key) for key, score in scores.items() if float(score) >= cutoff}


def review_union(if_scores: dict, vae_scores: dict, percentile: float = 99.0) -> set[str]:
    """Join each detector's own inclusive tail without averaging scores."""
    return review_ids(if_scores, percentile) | review_ids(vae_scores, percentile)


def split_cohort_mode(oot_mask, scores, threshold: float) -> str:
    """Describe the legacy minimum-ten threshold sample fallback accurately."""
    count = int((np.asarray(oot_mask) & (np.asarray(scores) >= threshold)).sum())
    if np.isfinite(threshold) and count >= 10:
        return "calibrated_threshold"
    return "top_score_fallback"
