"""Out-of-time (OOT) split utilities for panel evaluation.

The banking panel is keyed by ``(entity_id, period)`` and balanced: every
entity is observed once per period. For a genuine *out-of-time* holdout the
**last period** in the panel is treated as the OOT month -- detectors are fit
on the in-time rows (all earlier periods) and score every row, while the
headline business deliverable focuses on the OOT month.

This module is the single source of truth for that split so the OOT month is
never hard-coded elsewhere. :func:`oot_split` returns row-aligned boolean masks
over ``keys`` (which is row-for-row aligned with the feature matrix ``X`` that
:func:`src.preprocessing.pipeline.fit_transform_panel` produced), and
:func:`oot_period` returns the OOT period value(s) themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logging

__all__ = ["oot_split", "oot_period", "chronological_split", "ChronologicalSplit"]


def _distinct_sorted_periods(keys: pd.DataFrame, time_col: str) -> np.ndarray:
    """Return the distinct period values of ``keys[time_col]`` in sorted order."""
    if time_col not in keys.columns:
        raise KeyError(
            f"time_col {time_col!r} not found in keys columns {list(keys.columns)}"
        )
    periods = keys[time_col]
    # pd.unique preserves dtype (datetime stays datetime); np.sort orders it.
    distinct = pd.unique(periods)
    # Drop NaT/NaN before sorting so a stray missing period cannot become "last".
    distinct = distinct[pd.notna(distinct)]
    return np.sort(distinct)


def oot_period(
    keys: pd.DataFrame, time_col: str = "period", n_oot_periods: int = 1
) -> np.ndarray:
    """Return the OOT period value(s): the last ``n_oot_periods`` distinct periods.

    Args:
        keys: The ``(entity_id, period)`` frame (row-aligned with ``X``).
        time_col: Name of the period column in ``keys``.
        n_oot_periods: Number of trailing distinct periods to treat as OOT.

    Returns:
        A numpy array of the OOT period value(s), in ascending order.
    """
    distinct = _distinct_sorted_periods(keys, time_col)
    n_oot_periods = max(1, int(n_oot_periods))
    return distinct[-n_oot_periods:]


@dataclass(frozen=True)
class ChronologicalSplit:
    """Row masks and period labels for a strictly chronological 3-way split."""

    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    train_periods: np.ndarray
    val_periods: np.ndarray
    test_periods: np.ndarray

    def describe(self) -> str:
        def _fmt(periods: np.ndarray) -> str:
            vals = [str(v)[:10] for v in np.atleast_1d(periods)]
            if not vals:
                return "(none)"
            return vals[0] if len(vals) == 1 else f"{vals[0]}..{vals[-1]}"

        return (
            f"train={_fmt(self.train_periods)} ({int(self.train_mask.sum())} rows) | "
            f"val={_fmt(self.val_periods)} ({int(self.val_mask.sum())} rows) | "
            f"test={_fmt(self.test_periods)} ({int(self.test_mask.sum())} rows)"
        )


def chronological_split(
    keys: pd.DataFrame,
    time_col: str = "period",
    n_val_periods: int = 2,
    n_test_periods: int = 3,
    logger: Optional[logging.Logger] = None,
) -> ChronologicalSplit:
    """Split the panel into train / validation / test by *time*, never at random.

    The last ``n_test_periods`` distinct periods become test, the
    ``n_val_periods`` immediately before them become validation, and everything
    earlier is train. Masks are boolean arrays row-aligned with ``keys`` (hence
    with ``X``).

    TEORÍA: in a panel every row carries a timestamp, and the model will be
    deployed on *future* months. A random split lets a model be selected using
    rows that come after the ones it was fitted on, which measures interpolation
    rather than forecasting and silently inflates every metric. Each set here
    also plays a distinct role, and mixing them is the classic leak:

    * **train** fits the preprocessing statistics and the models,
    * **validation** selects hyperparameters *and* calibrates the alert
      threshold -- it is spent on decisions, so it can no longer be an unbiased
      estimate,
    * **test** is touched exactly once, at the end, to report.

    Short panels are handled by shrinking the later blocks rather than failing:
    validation and test give up periods (in that order) until at least one train
    period survives. A 6-period panel with the defaults therefore yields
    3 train / 1 val / 2 test rather than an error.

    Args:
        keys: The ``(entity_id, period)`` frame produced alongside ``X`` (or any
            frame carrying ``time_col``, e.g. the raw panel).
        time_col: Name of the period column.
        n_val_periods: Requested number of validation periods.
        n_test_periods: Requested number of trailing test periods.
        logger: Optional logger; defaults to the project logger.

    Returns:
        A :class:`ChronologicalSplit`.

    Raises:
        ValueError: If the panel has fewer than 3 distinct periods, which cannot
            support a 3-way temporal split.
    """
    log = logger or setup_logging()
    distinct = _distinct_sorted_periods(keys, time_col)
    n_periods = len(distinct)
    if n_periods < 3:
        raise ValueError(
            f"chronological_split needs at least 3 distinct periods, got {n_periods}. "
            "Use oot_split for a 2-way in-time/out-of-time split."
        )

    n_test = max(1, int(n_test_periods))
    n_val = max(1, int(n_val_periods))
    # Shrink test first, then validation, until a train block remains.
    while n_test + n_val >= n_periods:
        if n_test > 1 and n_test >= n_val:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    test_periods = distinct[n_periods - n_test:]
    val_periods = distinct[n_periods - n_test - n_val: n_periods - n_test]
    train_periods = distinct[: n_periods - n_test - n_val]

    col = keys[time_col]
    test_mask = col.isin(test_periods).to_numpy()
    val_mask = col.isin(val_periods).to_numpy()
    train_mask = ~(test_mask | val_mask)

    split = ChronologicalSplit(
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        train_periods=train_periods, val_periods=val_periods, test_periods=test_periods,
    )
    if (n_test, n_val) != (int(n_test_periods), int(n_val_periods)):
        log.warning(
            "Panel has only %d periods; shrank the split to %d val / %d test "
            "period(s) so a train block survives.", n_periods, n_val, n_test,
        )
    log.info("Chronological split on %r: %s", time_col, split.describe())
    return split


def oot_split(
    keys: pd.DataFrame,
    time_col: str = "period",
    n_oot_periods: int = 1,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split panel rows into in-time / out-of-time boolean masks.

    The OOT mask selects every row whose period is among the last
    ``n_oot_periods`` distinct periods; the in-time mask selects the rest. Both
    masks are boolean numpy arrays row-aligned with ``keys`` (hence with ``X``).

    Args:
        keys: The ``(entity_id, period)`` frame produced alongside ``X``.
        time_col: Name of the period column in ``keys``.
        n_oot_periods: Number of trailing distinct periods held out as OOT.
        logger: Optional logger; defaults to the project logger.

    Returns:
        ``(in_time_mask, oot_mask)`` -- complementary boolean arrays of length
        ``len(keys)``.
    """
    log = logger or setup_logging()
    oot_vals = oot_period(keys, time_col=time_col, n_oot_periods=n_oot_periods)
    oot_mask = keys[time_col].isin(oot_vals).to_numpy()
    in_time_mask = ~oot_mask

    log.info(
        "OOT split on %r: OOT period(s)=%s -> %d in-time rows, %d OOT rows "
        "(%d total)",
        time_col,
        [str(v) for v in np.atleast_1d(oot_vals)],
        int(in_time_mask.sum()),
        int(oot_mask.sum()),
        len(keys),
    )
    return in_time_mask, oot_mask
