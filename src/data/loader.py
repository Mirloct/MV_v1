"""Conditional loading of the banking panel dataset.

If the target CSV does not exist yet, it is generated on the fly via
`src.data.synthetic.generate_synthetic_panel`. Either way, the panel schema
(time column, entity column, optional target column) is inferred and logged
so downstream modules can consume the data without hardcoding column names.

Schema inference is name-hint-first with structural fallbacks: a column whose
name contains "period"/"date"/"time" wins the time slot, else the first
datetime-dtype column, else an object column that parses as a date for >95%
of its non-null values; the entity column is the first name containing
"entity"/"individual"/"id", else the highest-cardinality non-time column that
is still short of one value per row. Either can come back None, and callers
must handle that.

Ground truth is never expected inside the panel itself. For a generated
panel the loader takes the path the generator actually wrote; for a
pre-existing panel it probes `ground_truth.parquet` / `ground_truth.csv` next
to the data file. `PanelSchema.target_col` therefore stays None for synthetic
data and the pipeline runs unsupervised by default.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.data.synthetic import generate_synthetic_panel
from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

_TIME_NAME_HINTS = ("period", "date", "time")
_ENTITY_NAME_HINTS = ("entity", "individual", "id")
_TARGET_NAME_HINTS = ("target", "ground_truth", "groundtruth")

# Names probed for a sibling ground-truth file when the panel already exists.
_GROUND_TRUTH_BASENAMES = ("ground_truth.parquet", "ground_truth.csv")


@dataclass
class PanelSchema:
    """Inferred structure of a loaded panel DataFrame.

    Attributes:
        time_col: Name of the column identifying the time period, or None
            if it could not be inferred.
        entity_col: Name of the column identifying the individual/entity, or
            None if it could not be inferred.
        target_col: Name of a supervised target / ground-truth column found
            directly in the loaded DataFrame, or None if absent. Note: the
            synthetic generator writes ground truth to a *separate* file, so
            this will normally be None for generated data -- it exists to
            support future real datasets that may ship labels inline, and
            later modules branch supervised/unsupervised behavior on it.
        ground_truth_path: Path of the separate ground-truth file that goes
            with this panel, if one is known -- the path the generator
            actually wrote (it falls back from parquet to CSV when no
            parquet engine is installed), or a discovered sibling file.
            None when no ground truth could be located.
    """

    time_col: Optional[str]
    entity_col: Optional[str]
    target_col: Optional[str]
    ground_truth_path: Optional[str] = None


def _infer_time_col(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    for col in df.columns:
        if any(hint in col.lower() for hint in _TIME_NAME_HINTS):
            return col

    # Fall back: already-datetime dtype, or an object column that parses as
    # a date for the vast majority of its (non-null) values.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.95:
                return col
    logger.warning("Could not infer a time column")
    return None


def _infer_entity_col(df: pd.DataFrame, time_col: Optional[str], logger: logging.Logger) -> Optional[str]:
    candidates = [c for c in df.columns if c != time_col]

    for col in candidates:
        if any(hint in col.lower() for hint in _ENTITY_NAME_HINTS):
            return col

    # Fall back: pick the candidate with the highest cardinality that is
    # still strictly below the row count and above 1 (i.e. "far fewer
    # unique values than rows, but more than a plain constant/near-constant
    # low-cardinality category").
    n_rows = len(df)
    nunique = {col: df[col].nunique(dropna=True) for col in candidates}
    filtered = {col: n for col, n in nunique.items() if 1 < n < n_rows}
    if filtered:
        return max(filtered, key=filtered.get)

    logger.warning("Could not infer an entity/individual column")
    return None


def _infer_target_col(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    for col in df.columns:
        if any(hint in col.lower() for hint in _TARGET_NAME_HINTS):
            return col
    return None


#: Compact period formats that ``pd.to_datetime`` cannot infer on its own.
#: A bare 6-digit ``202401`` raises ("month must be in 1..12") because the
#: parser reads it as a year; an 8-digit ``20240115`` *is* inferred, but
#: naming it explicitly keeps the parse deterministic instead of relying on
#: dateutil's heuristics. Order matters: longest pattern first.
_COMPACT_PERIOD_FORMATS: tuple[tuple[str, str], ...] = (
    (r"^\d{8}$", "%Y%m%d"),   # 20240115
    (r"^\d{6}$", "%Y%m"),     # 202401  <- the yyyyMM case
)


def detect_period_format(series: pd.Series) -> Optional[str]:
    """Return the ``strftime`` format of a compact period column, else ``None``.

    ``None`` means "not a compact all-digit format" -- either it is already
    datetime-typed, or it carries separators (``2024-01-01``) that
    ``pd.to_datetime`` infers correctly on its own.
    """
    import re

    if pd.api.types.is_datetime64_any_dtype(series):
        return None
    sample = series.dropna()
    if sample.empty:
        return None
    # Ints and floats-that-are-really-ints both stringify with a stray
    # ``.0`` under ``astype(str)``, so normalise through Int64 first.
    if pd.api.types.is_numeric_dtype(sample):
        try:
            sample = sample.astype("Int64")
        except (TypeError, ValueError):
            return None
    as_text = sample.astype(str).str.strip()
    for pattern, fmt in _COMPACT_PERIOD_FORMATS:
        if as_text.str.match(pattern).all():
            return fmt
    return None


def parse_period_column(
    series: pd.Series, col_name: str, logger: logging.Logger
) -> pd.Series:
    """Parse a period column to datetime, handling compact ``yyyyMM`` inputs.

    ``pd.to_datetime(["202401"])`` raises rather than returning ``2024-01-01``:
    a 6-digit string is ambiguous, so pandas reads it as a year and rejects
    the impossible month. Real banking panels routinely ship the period this
    way, so the format is detected explicitly (see
    :func:`detect_period_format`) and passed to ``pd.to_datetime`` instead of
    being left to inference.

    A failure is logged and the column is returned **unchanged** rather than
    coerced -- ``errors="coerce"`` here would silently turn every period into
    ``NaT``, and downstream code (``chronological_split``, the panel feature
    engineer) would then split/lag on missing timestamps without ever
    raising. The assumption gate (``src.utils.assumptions.validate_panel``)
    is what turns an unparsed period column into a hard stop.
    """
    fmt = detect_period_format(series)
    try:
        if fmt is not None:
            parsed = pd.to_datetime(series, format=fmt)
            logger.info(
                "Parsed time column '%s' with explicit format %r (compact period "
                "format detected, e.g. %r -> %s).",
                col_name, fmt, str(series.dropna().iloc[0]),
                parsed.dropna().iloc[0].date() if len(parsed.dropna()) else "?",
            )
            return parsed
        return pd.to_datetime(series)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Column '%s' looked like a time column but failed to parse as datetime "
            "(%s); leaving it unchanged. The assumption gate will flag this before "
            "any model runs.", col_name, exc,
        )
        return series


def _discover_ground_truth(data_path: str, hint: Optional[str] = None) -> Optional[str]:
    """Return an existing ground-truth file for an already-present panel.

    Tries the requested path first (plus its `.csv` fallback sibling, which
    is what the generator writes when no parquet engine is installed), then
    the conventional names next to `data_path`.
    """
    parent = os.path.dirname(data_path) or "."
    candidates = []
    if hint:
        candidates += [hint, os.path.splitext(hint)[0] + ".csv"]
    candidates += [os.path.join(parent, name) for name in _GROUND_TRUTH_BASENAMES]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _infer_schema(
    df: pd.DataFrame, logger: logging.Logger, ground_truth_path: Optional[str] = None
) -> PanelSchema:
    time_col = _infer_time_col(df, logger)
    entity_col = _infer_entity_col(df, time_col, logger)
    target_col = _infer_target_col(df, logger)

    logger.info(
        "Inferred panel schema: time_col=%r, entity_col=%r, target_col=%r",
        time_col, entity_col, target_col,
    )
    if target_col is None:
        logger.info(
            "No target/ground_truth column found in the loaded data -- "
            "treating this as unsupervised (ground truth, if any, lives in a separate file)"
        )
    else:
        logger.info("Target/ground_truth column '%s' found -- supervised evaluation is possible", target_col)

    if ground_truth_path is not None:
        logger.info("Ground-truth labels for this panel live at '%s'", ground_truth_path)
    return PanelSchema(
        time_col=time_col,
        entity_col=entity_col,
        target_col=target_col,
        ground_truth_path=ground_truth_path,
    )


def load_or_generate_panel(
    data_path: str = paths.DATA_PATH, **generator_kwargs: Any
) -> tuple[pd.DataFrame, PanelSchema]:
    """Load the banking panel from `data_path`, generating it first if absent.

    Args:
        data_path: Path to the main panel CSV.
        **generator_kwargs: Forwarded to
            `src.data.synthetic.generate_synthetic_panel` if the file needs
            to be generated (e.g. `n_individuals`, `n_periods`,
            `ground_truth_path`, `seed`). `out_path` is always set to
            `data_path` and any `out_path` passed here is ignored. If no
            `ground_truth_path` is given, it defaults to a
            `ground_truth.parquet` *next to `data_path`* rather than to the
            generator's CWD-relative default.

    Returns:
        A `(df, schema)` tuple, where `schema` is a `PanelSchema` with the
        inferred `time_col`, `entity_col`, `target_col` (or None), and the
        real `ground_truth_path` if one is known.
    """
    logger = setup_logging()
    with log_phase("load_or_generate_panel", logger):
        generator_kwargs = dict(generator_kwargs)
        generator_kwargs.pop("out_path", None)
        # Keep the hidden labels with the data they describe: the generator's
        # own default is relative to the CWD, which would silently scatter
        # ground truth away from a `data_path` in another directory.
        generator_kwargs.setdefault(
            "ground_truth_path",
            os.path.join(os.path.dirname(data_path) or ".", "ground_truth.parquet"),
        )

        if not os.path.exists(data_path):
            logger.info("'%s' not found; generating synthetic panel", data_path)
            result = generate_synthetic_panel(out_path=data_path, **generator_kwargs)
            # Use the path the writer really produced (it may have fallen
            # back from parquet to CSV) instead of assuming the requested one.
            ground_truth_path = result.ground_truth_path
        else:
            logger.info("Loading existing panel from '%s'", data_path)
            ground_truth_path = _discover_ground_truth(
                data_path, hint=generator_kwargs.get("ground_truth_path")
            )

        df = pd.read_csv(data_path)
        logger.info("Loaded panel: %d rows x %d columns from '%s'", len(df), df.shape[1], data_path)

        schema = _infer_schema(df, logger, ground_truth_path=ground_truth_path)
        if schema.time_col is not None:
            df[schema.time_col] = parse_period_column(
                df[schema.time_col], schema.time_col, logger
            )

        return df, schema
