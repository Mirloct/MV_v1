"""Typed methodological-assumption checks, run before training either model.

Scope of this first pass (see `CONTEXT.md` for the running list): schema,
core data-quality, temporal-split integrity, and person-overlap diagnostics
on the panel, plus finite-matrix / config-sanity checks immediately before
each model's `.fit()`. It intentionally does not (yet) cover every bullet in
the original brief -- VAE-specific checks (posterior collapse, gradient
norms) belong next to the training loop that can actually observe them, not
here; this module owns what can be verified from data and config alone.

Every exception in the hierarchy below is raised *and* recorded as a failed
health check (`src.utils.observability`, category="assumption",
severity="critical") in the same call, so a blocking failure still leaves a
structured trace in `run_events.jsonl` -- "no silent assumption violations"
holds even when the pipeline stops.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils import observability


class AssumptionError(Exception):
    """Base class for every methodological assumption violation.

    `check`: short machine-readable name (e.g. "schema.required_columns").
    `observed`: the exact variables/columns/rows/values that caused the
    failure -- always populated, never a bare "something is wrong" message.
    """

    def __init__(self, message: str, *, check: str, observed: Any = None):
        super().__init__(message)
        self.check = check
        self.observed = observed
        observability.check(
            name=check,
            category="assumption",
            definition=message,
            expected="see message",
            severity="critical",
            passed=False,
            observed=observed,
            failure_action=f"{type(self).__name__} raised; downstream phases did not run.",
            evidence=str(observed),
        )


class SchemaAssumptionError(AssumptionError):
    """Required columns/keys are missing, ambiguous, or malformed."""


class DataQualityAssumptionError(AssumptionError):
    """Nulls, infinities, duplicates, or impossible values in the panel."""


class LeakageAssumptionError(AssumptionError):
    """Future information reached a fit that should only see past/train data."""


class TemporalSplitAssumptionError(AssumptionError):
    """Time ordering, period parsing, or split geometry is invalid."""


class IsolationForestAssumptionError(AssumptionError):
    """A precondition specific to fitting/scoring the Isolation Forest failed."""


class VAEAssumptionError(AssumptionError):
    """A precondition specific to fitting/scoring the VAE failed."""


class ArtifactGenerationError(AssumptionError):
    """A required output artifact was not produced or failed validation."""


def _ok(name: str, category: str, definition: str, expected: str, observed: Any, evidence: str = "") -> None:
    """Record a *passing* check (non-blocking checks call this directly)."""
    observability.check(
        name=name, category=category, definition=definition, expected=expected,
        severity="info", passed=True, observed=observed, evidence=evidence,
    )


def _warn(name: str, category: str, definition: str, expected: str, observed: Any, action: str, evidence: str = "") -> None:
    """Record a *non-blocking* failed check: logged and surfaced, execution continues."""
    observability.check(
        name=name, category=category, definition=definition, expected=expected,
        severity="warning", passed=False, observed=observed, failure_action=action, evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Panel-level gate: schema, data quality, temporal integrity, person overlap. #
# --------------------------------------------------------------------------- #
def validate_panel(
    df: pd.DataFrame,
    entity_col: Optional[str],
    time_col: Optional[str],
    *,
    null_rate_warn: float = 0.5,
) -> None:
    """Blocking + informational checks on the raw panel, before preprocessing.

    Raises:
        SchemaAssumptionError: `entity_col`/`time_col` unresolved, or the
            `(entity_col, time_col)` pair has duplicates (violates the
            balanced-panel contract in `CONTEXT.md`: one row per entity per
            period).
        TemporalSplitAssumptionError: `time_col` has unparseable (NaT) values.
    """
    if entity_col is None or time_col is None:
        raise SchemaAssumptionError(
            f"Panel schema could not resolve both keys: entity_col={entity_col!r}, "
            f"time_col={time_col!r}. Every downstream phase (split, panel features, "
            f"OOT export) assumes both are present.",
            check="schema.required_columns",
            observed={"entity_col": entity_col, "time_col": time_col, "columns": list(df.columns)},
        )

    if time_col in df.columns:
        n_nat = int(pd.isna(pd.to_datetime(df[time_col], errors="coerce")).sum())
        if n_nat > 0:
            raise TemporalSplitAssumptionError(
                f"Column {time_col!r} has {n_nat} value(s) that do not parse as a "
                f"date/period out of {len(df)} rows. A chronological split cannot "
                f"order rows it cannot parse.",
                check="temporal.parseable",
                observed={"column": time_col, "n_unparseable": n_nat, "n_rows": len(df)},
            )

    dup_mask = df.duplicated(subset=[entity_col, time_col], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        example_keys = df.loc[dup_mask, [entity_col, time_col]].head(5).to_dict("records")
        raise SchemaAssumptionError(
            f"{n_dup} row(s) share a duplicate ({entity_col}, {time_col}) key. The "
            f"data contract (CONTEXT.md) requires a balanced panel: exactly one row "
            f"per entity per period. Panel-feature lags/diffs and the OOT join are "
            f"both undefined under duplicate keys.",
            check="schema.no_duplicate_keys",
            observed={"n_duplicate_rows": n_dup, "example_keys": example_keys},
        )
    _ok(
        "schema.no_duplicate_keys", "data",
        f"No duplicate ({entity_col}, {time_col}) pairs.",
        "n_duplicate_rows == 0", {"n_duplicate_rows": 0},
    )

    # -- Informational (non-blocking) data-quality diagnostics -------------- #
    null_rates = df.isna().mean()
    bad_null_cols = null_rates[null_rates > null_rate_warn]
    if len(bad_null_cols):
        _warn(
            "data.high_null_rate", "data",
            f"Columns with a null rate above {null_rate_warn:.0%}.",
            f"null_rate <= {null_rate_warn:.0%} for every column",
            {c: round(float(r), 4) for c, r in bad_null_cols.items()},
            action="Review whether these columns should be dropped, imputed differently, "
                   "or are informatively missing (MNAR) as CONTEXT.md assumes by default.",
        )
    else:
        _ok("data.high_null_rate", "data", f"No column exceeds {null_rate_warn:.0%} nulls.",
            f"null_rate <= {null_rate_warn:.0%}", {"max_null_rate": round(float(null_rates.max()), 4) if len(null_rates) else 0.0})

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {c: int(np.isinf(df[c].to_numpy(dtype=float, na_value=0.0)).sum()) for c in numeric_cols}
    inf_cols = {c: n for c, n in inf_counts.items() if n > 0}
    if inf_cols:
        raise DataQualityAssumptionError(
            f"{sum(inf_cols.values())} infinite value(s) found across {len(inf_cols)} "
            f"numeric column(s). Both models require a finite feature matrix.",
            check="data.no_infinite_values", observed=inf_cols,
        )
    _ok("data.no_infinite_values", "data", "No +-inf in any numeric column.",
        "inf count == 0 for all numeric columns", {"numeric_columns_checked": len(numeric_cols)})

    const_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        _warn(
            "data.constant_features", "data",
            "Numeric columns with zero variance (a single distinct value).",
            "nunique > 1 for every numeric column", const_cols,
            action="A constant feature carries no signal for either model; confirm it "
                   "is expected (e.g. a synthetic-data artifact at this scale) before tuning.",
        )
    else:
        _ok("data.constant_features", "data", "No zero-variance numeric column.", "nunique > 1", {"n_checked": len(numeric_cols)})

    n_full_dupes = int(df.duplicated(keep=False).sum())
    if n_full_dupes:
        _warn(
            "data.duplicate_rows", "data",
            "Fully duplicate rows (every column identical).",
            "n_duplicate_rows == 0", {"n_duplicate_rows": n_full_dupes},
            action="Confirm these are legitimate (e.g. two entities with identical "
                   "profiles) rather than an upstream export bug.",
        )
    else:
        _ok("data.duplicate_rows", "data", "No fully duplicate rows.", "n_duplicate_rows == 0", {"n_duplicate_rows": 0})


def measure_person_overlap(
    df: pd.DataFrame, entity_col: str, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray,
) -> dict:
    """Diagnostic only -- never raises. See CONTEXT.md "Finding #4" discussion.

    THEORY CONTRAST: under a pure rescoring/forecasting objective on a closed,
    stationary population, person_overlap=100% is a *designed* property, not a
    defect -- this project's synthetic panel is exactly that (balanced: every
    entity observed in every period), so 100% is the expected reading, not
    evidence of a problem, by construction. It becomes a real risk claim only
    once churn/attrition/new-entity-inflow are part of the evaluated
    population; this function reports the number so that claim can be
    checked against real data instead of assumed either way.
    """
    train_e = set(df.loc[train_mask, entity_col].unique())
    val_e = set(df.loc[val_mask, entity_col].unique())
    test_e = set(df.loc[test_mask, entity_col].unique())
    overlap_train_test = len(train_e & test_e) / len(test_e) if test_e else float("nan")
    overlap_val_test = len(val_e & test_e) / len(test_e) if test_e else float("nan")
    new_in_test = len(test_e - train_e - val_e)
    result = {
        "n_train_entities": len(train_e),
        "n_val_entities": len(val_e),
        "n_test_entities": len(test_e),
        "train_test_person_overlap": round(overlap_train_test, 4) if test_e else None,
        "val_test_person_overlap": round(overlap_val_test, 4) if test_e else None,
        "entities_in_test_never_seen_in_train_or_val": new_in_test,
    }
    _ok(
        "temporal.person_overlap_measured", "assumption",
        "Fraction of OOT-period entities also present in train/val (diagnostic, not "
        "pass/fail -- see CONTEXT.md Finding #4 for the theoretical contrast).",
        "measured, not thresholded", result,
    )
    return result


# --------------------------------------------------------------------------- #
# Model-facing gates: run immediately before each model's `.fit()`.           #
# --------------------------------------------------------------------------- #
def validate_matrix_for_fit(X, model_name: str) -> None:
    """Blocking finiteness check on the feature matrix immediately before `.fit()`.

    Raises `IsolationForestAssumptionError`/`VAEAssumptionError` (selected by
    `model_name`) naming exactly how many NaN/Inf values were found and in
    how many of the matrix's columns, dense or sparse.
    """
    import scipy.sparse as sp

    arr = X.data if sp.issparse(X) else np.asarray(X)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    exc_cls = VAEAssumptionError if model_name == "vae" else IsolationForestAssumptionError
    if n_nan or n_inf:
        raise exc_cls(
            f"Feature matrix for {model_name!r} has {n_nan} NaN and {n_inf} Inf value(s) "
            f"out of {arr.size} entries immediately before fit -- both models require a "
            f"finite input matrix.",
            check=f"{model_name}.finite_input_matrix",
            observed={"n_nan": n_nan, "n_inf": n_inf, "total_entries": int(arr.size), "shape": tuple(X.shape)},
        )
    _ok(
        f"{model_name}.finite_input_matrix", "training",
        "Feature matrix has no NaN/Inf.", "n_nan == 0 and n_inf == 0",
        {"n_nan": 0, "n_inf": 0, "shape": tuple(X.shape)},
    )


def validate_iforest_config(contamination: float, n_estimators: int, max_samples: Any) -> None:
    """Blocking sanity check on Isolation Forest hyperparameters before fit/tune.

    `contamination` is a *fraction of rows treated as the positive class for
    thresholding*, not a probability -- sklearn requires `(0, 0.5]`; outside
    that range the forest either flags nothing or flags the majority.
    """
    if not (0.0 < float(contamination) <= 0.5):
        raise IsolationForestAssumptionError(
            f"contamination={contamination!r} is outside the valid range (0, 0.5]. "
            f"sklearn's IsolationForest interprets it as the fraction of samples "
            f"to flag as anomalous, so 0 flags nothing and >0.5 flags the majority "
            f"as the 'anomaly'.",
            check="iforest.contamination_in_range",
            observed={"contamination": contamination},
        )
    if int(n_estimators) < 10:
        _warn(
            "iforest.n_estimators_low", "training",
            "n_estimators below a size where score variance across seeds is typically small.",
            "n_estimators >= 10 (project default: 200)", {"n_estimators": n_estimators},
            action="Low tree counts increase score instability across random seeds; "
                   "acceptable for a smoke test, not for a reported metric.",
        )
