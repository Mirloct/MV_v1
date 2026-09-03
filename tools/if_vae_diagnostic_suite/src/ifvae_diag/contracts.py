from __future__ import annotations

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when analysis inputs cannot support trustworthy diagnostics."""


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataContractError(f"{name} is missing columns: {missing}")


def _validate_ids(frame: pd.DataFrame, id_col: str, name: str) -> None:
    if frame[id_col].isna().any():
        raise DataContractError(f"{name}.{id_col} must not be null")
    if not frame[id_col].is_unique:
        raise DataContractError(f"{name}.{id_col} must be unique")


def _validate_numeric_finite(frame: pd.DataFrame, features: list[str], name: str) -> None:
    non_numeric = [f for f in features if not pd.api.types.is_numeric_dtype(frame[f])]
    if non_numeric:
        raise DataContractError(f"{name} features must be numeric: {non_numeric}")
    values = frame[features].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataContractError(f"{name} features must be finite or missing, not infinite")
    all_missing = [feature for feature in features if frame[feature].isna().all()]
    if all_missing:
        raise DataContractError(f"{name} features are entirely missing: {all_missing}")


def _validate_labels(frame: pd.DataFrame, label_col: str) -> None:
    labels = set(frame[label_col].dropna().unique().tolist())
    if not labels.issubset({0, 1, False, True}):
        raise DataContractError(f"{label_col} must be binary 0/1")


def _temporal_warning(
    reference: pd.DataFrame, scored: pd.DataFrame, time_col: str
) -> list[dict[str, str]]:
    reference_time = pd.to_datetime(reference[time_col], errors="coerce", utc=True)
    scored_time = pd.to_datetime(scored[time_col], errors="coerce", utc=True)
    if reference_time.isna().any() or scored_time.isna().any():
        raise DataContractError(f"{time_col} contains unparseable timestamps")
    if reference_time.max() >= scored_time.min():
        return [{
            "code": "temporal_overlap",
            "severity": "high",
            "message": "Reference time overlaps scoring time; investigate look-ahead leakage.",
        }]
    return []


def validate_frames(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    features: list[str],
    id_col: str,
    label_col: str | None,
    time_col: str | None,
) -> list[dict[str, str]]:
    if reference.empty or scored.empty:
        raise DataContractError("reference and scored frames must be non-empty")
    _require_columns(reference, [id_col, *features], "reference")
    scored_required = [id_col, *features] if label_col is None else [id_col, label_col, *features]
    _require_columns(scored, scored_required, "scored")
    _validate_ids(reference, id_col, "reference")
    _validate_ids(scored, id_col, "scored")
    _validate_numeric_finite(reference, features, "reference")
    _validate_numeric_finite(scored, features, "scored")
    if label_col is not None:
        _validate_labels(scored, label_col)
    warnings: list[dict[str, str]] = []
    if time_col and time_col in reference and time_col in scored:
        warnings.extend(_temporal_warning(reference, scored, time_col))
    if reference[features].isna().any().any() or scored[features].isna().any().any():
        warnings.append({
            "code": "missing_features",
            "severity": "medium",
            "message": "Missing values will be median-imputed from reference data only.",
        })
    return warnings
