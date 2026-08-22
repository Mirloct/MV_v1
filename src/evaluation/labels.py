"""Join the separate hidden ground-truth file back onto the panel rows.

Detection is unsupervised by default: the anomaly labels never live inside
``data.csv`` but in a sibling ground-truth file (parquet, or CSV when no
parquet engine is installed). For *offline evaluation* this module reads that
file at ``schema.ground_truth_path`` and left-joins it onto the ``keys`` frame
(``(entity_id, period)``, row-aligned with the feature matrix ``X``), returning
a 0/1 ``is_anomaly`` vector aligned to the ``X`` rows.

The join is robust to the two ways the ground-truth file can drift from the
loaded panel: the ``is_anomaly`` column may be a real bool or the strings
``"True"``/``"False"`` (CSV round-trip), and ``period`` may be a datetime or a
string. Both sides are coerced before matching. When no ground-truth file is
known (a genuinely unlabeled real dataset), an all-zero vector is returned with
a logged warning so downstream supervised metrics can simply be skipped.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.utils.logging_config import setup_logging

__all__ = ["load_ground_truth_labels", "load_ground_truth_types"]

_LABEL_COL = "is_anomaly"
_TYPE_COL = "anomaly_type"
_NO_TYPE = "none"
_TRUE_TOKENS = {"1", "true", "t", "yes", "y"}


def _read_ground_truth(path: str, log: logging.Logger) -> pd.DataFrame:
    """Read the ground-truth file, trying the CSV sibling if parquet fails."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - engine-dependent
            csv_alt = os.path.splitext(path)[0] + ".csv"
            if os.path.exists(csv_alt):
                log.warning(
                    "Failed to read parquet ground truth (%s); using CSV sibling %s",
                    exc, csv_alt,
                )
                return pd.read_csv(csv_alt)
            raise
    return pd.read_csv(path)


def _coerce_binary(series: pd.Series) -> np.ndarray:
    """Coerce a bool / numeric / string label column to a 0/1 int array."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy().astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.to_numpy().astype(float) > 0.5).astype(int)
    tokens = series.astype(str).str.strip().str.lower()
    return tokens.isin(_TRUE_TOKENS).to_numpy().astype(int)


def _join_ground_truth(
    schema: PanelSchema, keys: pd.DataFrame, log: logging.Logger
) -> Optional[pd.DataFrame]:
    """Left-join the hidden ground-truth file onto ``keys``, row-aligned.

    The single join shared by :func:`load_ground_truth_labels` and
    :func:`load_ground_truth_types`, so the two views of the ground truth can
    never fall out of alignment with each other or with ``X``.

    Returns:
        A frame of ``len(keys)`` rows with ``__lab__`` (0/1 int) and
        ``__type__`` (str, ``"none"`` where unlabeled or unmatched), or ``None``
        when no usable ground-truth file exists.
    """
    n = len(keys)
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"
    path = getattr(schema, "ground_truth_path", None)

    if not path or not os.path.exists(path):
        log.warning(
            "No ground-truth file available (path=%r); treat this run as "
            "unlabeled (supervised metrics skipped)",
            path,
        )
        return None

    gt = _read_ground_truth(path, log)
    if entity_col not in gt.columns or time_col not in gt.columns or _LABEL_COL not in gt.columns:
        log.warning(
            "Ground-truth file %s missing expected columns "
            "(need %r, %r, %r; found %s)",
            path, entity_col, time_col, _LABEL_COL, list(gt.columns),
        )
        return None

    # Build coercion-robust join frames. Entity ids compared as strings and
    # periods as datetimes so a parquet/CSV round-trip cannot break the match.
    left = pd.DataFrame(
        {
            "__ord__": np.arange(n),
            "__ent__": keys[entity_col].astype(str).to_numpy(),
            "__per__": pd.to_datetime(keys[time_col], errors="coerce").to_numpy(),
        }
    )
    right_data = {
        "__ent__": gt[entity_col].astype(str).to_numpy(),
        "__per__": pd.to_datetime(gt[time_col], errors="coerce").to_numpy(),
        "__lab__": _coerce_binary(gt[_LABEL_COL]),
    }
    if _TYPE_COL in gt.columns:
        right_data["__type__"] = gt[_TYPE_COL].astype(str).to_numpy()
    right = pd.DataFrame(right_data).drop_duplicates(
        subset=["__ent__", "__per__"], keep="first"
    )

    merged = left.merge(right, on=["__ent__", "__per__"], how="left").sort_values("__ord__")
    merged["__matched__"] = merged["__lab__"].notna()
    merged["__lab__"] = merged["__lab__"].fillna(0).astype(int)
    if "__type__" not in merged.columns:
        merged["__type__"] = _NO_TYPE
    merged["__type__"] = merged["__type__"].fillna(_NO_TYPE).astype(str)
    merged.attrs["path"] = path
    return merged.reset_index(drop=True)


def load_ground_truth_labels(schema: PanelSchema, keys: pd.DataFrame) -> np.ndarray:
    """Return a 0/1 ``is_anomaly`` vector row-aligned with ``keys`` (hence ``X``).

    Args:
        schema: Panel schema carrying ``entity_col``, ``time_col`` and the real
            ``ground_truth_path`` (or ``None`` when no labels exist).
        keys: The ``(entity_id, period)`` frame produced alongside ``X`` by
            :func:`src.preprocessing.pipeline.fit_transform_panel`.

    Returns:
        An ``int`` numpy array of length ``len(keys)``: ``1`` where the row is a
        ground-truth anomaly, ``0`` otherwise. All zeros (with a logged warning)
        when no ground-truth file is available.
    """
    log = setup_logging()
    n = len(keys)
    merged = _join_ground_truth(schema, keys, log)
    if merged is None:
        log.warning("Returning all-zero labels for %d rows", n)
        return np.zeros(n, dtype=int)

    labels = merged["__lab__"].to_numpy().astype(int)
    matched = int(merged["__matched__"].sum())
    log.info(
        "Loaded ground truth from %s: matched %d / %d panel rows; %d anomalies "
        "(%.3f%% positive rate)",
        merged.attrs.get("path"), matched, n, int(labels.sum()),
        100.0 * labels.sum() / n if n else 0.0,
    )
    if matched < n:
        log.warning(
            "%d panel rows had no ground-truth match and were labeled 0", n - matched
        )
    return labels


def load_ground_truth_types(schema: PanelSchema, keys: pd.DataFrame) -> np.ndarray:
    """Return the ``anomaly_type`` string per row, aligned with ``keys``.

    The generator emits four mutually exclusive anomaly geometries -- ``global``
    (extreme in the marginal), ``local`` (unremarkable globally but far from the
    entity's own history), ``contextual`` (normal for December, injected outside
    it) and ``collective`` (a synchronised group spike) -- plus ``"none"`` for
    clean rows. Until now only ``is_anomaly`` was ever read, so an aggregate
    PR-AUC could not say *which* geometry a detector was missing; this exposes
    the breakdown that :func:`src.evaluation.metrics.metrics_by_anomaly_type`
    consumes.

    Returns:
        A string array of length ``len(keys)``. Rows that are unmatched, or come
        from a ground-truth file without the column, are ``"none"``.
    """
    log = setup_logging()
    n = len(keys)
    merged = _join_ground_truth(schema, keys, log)
    if merged is None:
        return np.full(n, _NO_TYPE, dtype=object)

    types = merged["__type__"].to_numpy(dtype=object)
    counts = pd.Series(types).value_counts().to_dict()
    log.info("Ground-truth anomaly types: %s", {str(k): int(v) for k, v in counts.items()})
    return types
