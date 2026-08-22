"""Assemble a human-readable scored frame (keys + score + raw features).

The detectors score the *preprocessed* matrix ``X``, but the OOT Excel
deliverable must be interpretable by a human reviewer, so it needs the
**original** (pre-preprocessing) feature values, not the scaled/encoded ones.
:func:`build_scored_frame` joins the per-row anomaly ``scores`` back onto the
raw panel ``raw_df`` via the ``(entity_id, period)`` keys and returns a tidy
DataFrame of ``entity_id``, ``period``, the score, and the raw feature columns.

IMPORTANT: ``scores`` must be row-aligned with ``keys`` -- i.e. in the exact
order :func:`src.preprocessing.pipeline.fit_transform_panel` returned (which is
also the order the detector scored). ``keys`` carries no positional information
of its own, so a mis-ordered ``scores`` array would silently mislabel rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.utils.logging_config import setup_logging

__all__ = ["build_scored_frame"]


def build_scored_frame(
    raw_df: pd.DataFrame,
    keys: pd.DataFrame,
    scores,
    schema: PanelSchema,
    score_col: str = "anomaly_score",
) -> pd.DataFrame:
    """Join anomaly scores onto the raw panel features via the panel keys.

    Args:
        raw_df: The original (pre-preprocessing) panel DataFrame, still carrying
            the human-readable feature columns and the key columns.
        keys: The ``(entity_id, period)`` frame produced alongside ``X``;
            row-aligned with ``scores``.
        scores: Per-row anomaly scores (higher = more anomalous), row-aligned
            with ``keys``.
        schema: Panel schema (for ``entity_col`` / ``time_col``).
        score_col: Name for the score column in the output frame.

    Returns:
        A DataFrame with columns ``[entity_id, period, <score_col>,
        <raw feature columns...>]``, one row per ``keys`` row, in ``keys`` order.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"

    s = np.asarray(scores, dtype=float).ravel()
    if len(s) != len(keys):
        raise ValueError(
            f"scores length ({len(s)}) must match keys length ({len(keys)}); "
            "scores must be row-aligned with keys (fit_transform_panel order)."
        )

    base = keys.reset_index(drop=True).copy()
    base["__ord__"] = np.arange(len(base))
    base[score_col] = s

    # Coerce join keys so a datetime/string mismatch cannot break the merge.
    join_on = []
    if entity_col in base.columns and entity_col in raw_df.columns:
        base["__ent__"] = base[entity_col].astype(str).to_numpy()
        join_on.append("__ent__")
    if time_col in base.columns and time_col in raw_df.columns:
        base["__per__"] = pd.to_datetime(base[time_col], errors="coerce").to_numpy()
        join_on.append("__per__")

    # Raw features = everything in raw_df except the key columns (they are
    # already carried by `keys`); brought over on the coerced join keys.
    feature_cols = [c for c in raw_df.columns if c not in (entity_col, time_col)]
    right = raw_df[[c for c in (entity_col, time_col) if c in raw_df.columns] + feature_cols].copy()
    if entity_col in right.columns:
        right["__ent__"] = right[entity_col].astype(str).to_numpy()
    if time_col in right.columns:
        right["__per__"] = pd.to_datetime(right[time_col], errors="coerce").to_numpy()
    right = right.drop(columns=[c for c in (entity_col, time_col) if c in right.columns])
    right = right.drop_duplicates(subset=join_on, keep="first")

    merged = base.merge(right, on=join_on, how="left").sort_values("__ord__")

    ordered_cols = [c for c in (entity_col, time_col) if c in merged.columns]
    ordered_cols += [score_col] + feature_cols
    result = merged[ordered_cols].reset_index(drop=True)

    log.info(
        "Built scored frame: %d rows x %d columns (%d raw feature columns joined "
        "on %s)",
        len(result), result.shape[1], len(feature_cols), join_on,
    )
    return result
