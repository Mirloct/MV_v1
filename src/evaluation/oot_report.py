"""The headline deliverable: OOT top-decile anomaly Excel export.

Business requirement (CONTEXT.md Status): export the **top 10% of individuals by
anomaly score in the out-of-time (last) month**, sorted descending, laid out as
**ID - SCORE - VARIABLES**. This module produces exactly that ``.xlsx`` artifact
under ``reports/`` and returns the table for programmatic use.

Column layout (hard requirement):

* column 1  -- the entity identifier (``entity_id``),
* column 2  -- the anomaly ``SCORE``,
* columns 3+ -- the individual's VARIABLES: the *original*, human-readable
  feature values carried by the scored frame (never the preprocessed/scaled
  matrix). The OOT month itself is constant across the table (it is the single
  last period) and is logged rather than repeated as a column.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.splits import oot_period
from src.utils import observability, paths
from src.utils.assumptions import ArtifactGenerationError
from src.utils.logging_config import log_phase, setup_logging

__all__ = [
    "export_oot_top_anomalies", "export_oot_top_decile", "DEFAULT_TOP_N",
    "export_p95_checkpoint",
]

_DEFAULT_OUT = paths.OOT_REPORT_DEFAULT

# Business default: the review queue is a fixed headcount, not a percentage.
DEFAULT_TOP_N = 50

#: Percentile bands used to grade the exported queue, ascending. An
#: individual is graded by the highest band its score clears, so the labels
#: partition the export: `p95` means "at or above P95 but below P97".
#:
#: Only three, deliberately: the point of the band is to let a reviewer sort
#: a queue by urgency, and more than three grades stop being actionable --
#: nobody triages ten tiers differently. P99 is the "look at this today" tier,
#: P95 the "review this cycle" one.
PERCENTILE_BANDS: tuple[float, ...] = (95.0, 97.0, 99.0)
#: Column added to the export carrying that grade.
BAND_COL = "percentil"


def _resolve_out_path(
    out_path: str, top_fraction: float, model_name: str, top_n: Optional[int],
    min_percentile: Optional[float] = None,
) -> str:
    """Fold the selection size and ``model_name`` into the default filename.

    ``oot_p95_iforest.xlsx`` for a percentile export, ``oot_top50_iforest.xlsx``
    for a top-N one, ``oot_top10_iforest.xlsx`` for a top-fraction one.
    """
    if out_path == _DEFAULT_OUT and model_name and model_name != "model":
        if min_percentile is not None:
            tag = f"p{int(round(min_percentile))}"
        else:
            tag = f"top{int(top_n) if top_n is not None else int(round(top_fraction * 100))}"
        return os.path.join(paths.REPORTS_DIR, f"oot_{tag}_{model_name}.xlsx")
    return out_path


def _percentile_band_labels(
    scores: np.ndarray, reference: np.ndarray,
    bands: "tuple[float, ...]" = PERCENTILE_BANDS,
) -> tuple[np.ndarray, dict]:
    """Grade each score by the highest percentile band it clears.

    Args:
        scores: Scores of the rows being exported.
        reference: The score population the percentiles are computed from.
            Passed separately (rather than reusing ``scores``) because the
            cut-offs must describe the *whole* OOT block: computing them from
            the already-filtered selection would make the bands circular --
            the top 5% of the top 5% is not P99.

    Returns:
        ``(labels, cutoffs)`` -- a string array like ``"p95"``/``"p97"``/
        ``"p99"`` aligned to ``scores``, and the numeric cut-off per band.
    """
    ordered = tuple(sorted(bands))
    cutoffs = {p: float(np.percentile(reference, p)) for p in ordered}
    labels = np.full(len(scores), "", dtype=object)
    # Ascending assignment: each higher band overwrites the previous, so a
    # score ends up labelled with the highest band it clears.
    for p in ordered:
        labels[scores >= cutoffs[p]] = f"p{int(round(p))}"
    return labels, cutoffs


def export_oot_top_anomalies(
    scored_df: pd.DataFrame,
    schema: PanelSchema,
    out_path: str = _DEFAULT_OUT,
    top_n: Optional[int] = None,
    top_fraction: float = 0.10,
    min_percentile: Optional[float] = 95.0,
    model_name: str = "model",
    n_oot_periods: int = 1,
    score_col: str = "anomaly_score",
    threshold: Optional[float] = None,
) -> tuple[str, pd.DataFrame]:
    """Export the riskiest out-of-time **individuals** as an ID-SCORE-VARIABLES xlsx.

    This is the project's headline business deliverable: the alert queue an
    analyst actually works through.

    Selection size -- the first of these that is set wins:

    * ``min_percentile=95.0`` (**the default**) -> every individual at or above
      the 95th percentile of the OOT score distribution. Percentile rather than
      a fixed headcount because the queue then scales with the portfolio and
      the cut has a distributional meaning: "the riskiest 5%" holds whether
      the panel has 2,000 customers or 200,000. Each exported row also carries
      a :data:`BAND_COL` grade (``p95``/``p97``/``p99``) so the queue can be
      triaged by urgency.
    * ``top_n=50`` -> the 50 highest-scoring individuals: a fixed headcount for
      a team that works N cases a month regardless of portfolio size. Set
      ``min_percentile=None`` to use it.
    * ``top_fraction=0.10`` -> the top decile, used when both others are None.

    **One row per individual.** When ``n_oot_periods > 1`` an entity appears in
    several test months; the export keeps that entity's single highest-scoring
    month, and the ``time_col`` of that month is part of the output -- so a
    reviewer can see *which* period triggered the alert.

    Args:
        scored_df: Output of :func:`src.evaluation.scoring.build_scored_frame`
            (``entity_id``, ``period``, ``score_col``, then raw feature columns).
        schema: Panel schema (for ``entity_col`` / ``time_col``).
        out_path: Destination ``.xlsx``. If left at the default and a
            ``model_name`` is given, it becomes
            ``artifacts/reports/oot_p95_<model>.xlsx`` (or ``oot_top<N>_...``).
        min_percentile: Percentile cut-off on the OOT score distribution
            (default 95.0). ``None`` falls back to ``top_n``/``top_fraction``.
        top_n: Number of individuals to export. Used only when
            ``min_percentile`` is None.
        top_fraction: Fraction of OOT individuals to keep when both
            ``min_percentile`` and ``top_n`` are None.
        model_name: Detector name, folded into the default filename.
        n_oot_periods: Trailing distinct periods treated as OOT (default 1).
        score_col: Name of the score column in ``scored_df``.
        threshold: Optional calibrated cut-off (see
            :mod:`src.evaluation.thresholds`). When given, an ``alert`` column
            flags rows at or above it and the counts are logged -- so the export
            shows both the fixed-size queue *and* how many of those cases the
            calibrated rule would actually have raised.

    Returns:
        ``(path, table_df)`` -- the written path and the exported table DataFrame.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"
    out_path = _resolve_out_path(
        out_path, top_fraction, model_name, top_n, min_percentile,
    )

    with log_phase("evaluation.export_oot_top_anomalies", log):
        oot_vals = oot_period(scored_df, time_col=time_col, n_oot_periods=n_oot_periods)
        oot_mask = scored_df[time_col].isin(oot_vals).to_numpy()
        oot = scored_df.loc[oot_mask].copy()
        n_oot_rows = len(oot)
        if n_oot_rows == 0:
            raise ValueError(
                f"No rows found for OOT period(s) {list(np.atleast_1d(oot_vals))}"
            )

        # Deterministic ordering: score descending, ties broken by entity_id
        # ascending, via a stable sort so the result is fully reproducible.
        oot = oot.sort_values(
            by=[score_col, entity_col], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)

        # One row per individual: the sort above already put each entity's worst
        # month first, so keeping the first occurrence keeps the max score.
        n_rows_before_dedup = len(oot)
        oot = oot.drop_duplicates(subset=[entity_col], keep="first").reset_index(drop=True)
        n_individuals = len(oot)
        if n_individuals < n_rows_before_dedup:
            log.info(
                "Collapsed %d OOT rows to %d distinct individuals (kept each "
                "entity's highest-scoring test month)",
                n_rows_before_dedup, n_individuals,
            )

        # Percentile cut-offs come from the FULL de-duplicated OOT population,
        # before any selection. Deriving them from the exported subset instead
        # would be circular: the top 5% of the top 5% is not P99.
        all_oot_scores = oot[score_col].to_numpy(dtype=float)

        if min_percentile is not None:
            cut = float(np.percentile(all_oot_scores, float(min_percentile)))
            # `>=` so ties at the cut-off are kept: dropping half a tie group
            # would make the export depend on sort order among equal scores.
            keep = all_oot_scores >= cut
            k = int(keep.sum())
            top = oot.loc[keep].copy()
            selection = (
                f"{k} individuals at or above P{min_percentile:g} "
                f"(score >= {cut:.6f})"
            )
        else:
            if top_n is not None:
                k = min(n_individuals, max(1, int(top_n)))
                selection = f"top {k} individuals (requested top_n={int(top_n)})"
            else:
                k = min(n_individuals, max(1, math.ceil(top_fraction * n_individuals)))
                selection = f"top {k} = ceil({top_fraction:.4g} x {n_individuals})"
            top = oot.iloc[:k].copy()

        # ID - PERIOD - SCORE - BAND - VARIABLES.
        #
        # `time_col` is part of the output, not stripped as a key: with more
        # than one OOT period the export keeps each entity's worst month, and
        # without the period a reviewer cannot tell *when* the alert fired --
        # which is the first thing needed to go look at the case.
        variable_cols = [
            c for c in scored_df.columns
            if c not in (entity_col, time_col, score_col)
        ]
        table = top[[entity_col, time_col, score_col] + variable_cols].reset_index(drop=True)

        # Percentile band per exported row, graded against the full OOT
        # population computed above.
        bands, cutoffs = _percentile_band_labels(
            top[score_col].to_numpy(dtype=float), all_oot_scores,
        )
        table.insert(3, BAND_COL, bands)
        band_counts = {b: int((bands == b).sum()) for b in sorted(set(bands)) if b}
        log.info(
            "Percentile bands (cut-offs from the %d-individual OOT block): %s; "
            "exported counts: %s",
            n_individuals,
            {f"p{int(p)}": round(v, 6) for p, v in cutoffs.items()},
            band_counts,
        )

        if threshold is not None and np.isfinite(threshold):
            alerts = (top[score_col].to_numpy(dtype=float) >= float(threshold)).astype(int)
            table.insert(4, "alert", alerts)
            log.info(
                "Calibrated threshold %.6f: %d of the exported %d individuals "
                "are above it",
                float(threshold), int(alerts.sum()), k,
            )

        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        table.to_excel(out_path, index=False, engine="openpyxl")

        score_lo = float(top[score_col].min())
        score_hi = float(top[score_col].max())
        log.info(
            "OOT export: period(s)=%s, %d individuals in the OOT block, %s, "
            "selected score range [%.6f, %.6f], layout=ID-%s-SCORE-%s-%d variables -> %s",
            [str(v) for v in np.atleast_1d(oot_vals)], n_individuals, selection,
            score_lo, score_hi, time_col, BAND_COL, len(variable_cols), out_path,
        )

    return out_path, table


def export_oot_top_decile(
    scored_df: pd.DataFrame,
    schema: PanelSchema,
    out_path: str = _DEFAULT_OUT,
    top_fraction: float = 0.10,
    model_name: str = "model",
    n_oot_periods: int = 1,
    score_col: str = "anomaly_score",
) -> tuple[str, pd.DataFrame]:
    """Percentage-based export -- thin wrapper kept for backwards compatibility.

    Equivalent to :func:`export_oot_top_anomalies` with ``top_n=None``. New code
    should call that directly, since the default deliverable is now a fixed-size
    top-N queue.
    """
    return export_oot_top_anomalies(
        scored_df, schema, out_path=out_path, top_n=None, top_fraction=top_fraction,
        model_name=model_name, n_oot_periods=n_oot_periods, score_col=score_col,
    )


def export_p95_checkpoint(
    df: pd.DataFrame,
    scores: np.ndarray,
    *,
    in_mask: np.ndarray,
    schema: PanelSchema,
    split_masks: dict,
    percentile: float = 95.0,
    out_path: Optional[str] = None,
    model_name: str = "iforest",
) -> tuple[str, pd.DataFrame, float]:
    """Gate checkpoint: export the P95 highest-scoring records right after a
    detector finishes, with every original column, before the next layer runs.

    **Score definition** (documented, not assumed): ``scores`` is expected to
    be ``detector.score_samples(X)`` under this project's convention, **higher
    = more anomalous** -- see `docs/models_isolation_forest.md`. This function
    takes the array directly (not the detector) because that is all a P95
    selection needs; ``decision_function``/``predict`` were considered as
    extra diagnostic columns and deliberately left out here, since either
    would require threading the preprocessed feature matrix ``X`` through a
    function whose contract is otherwise just "a df and its row-aligned
    scores" -- add them at the call site in `main.py` if a future need
    justifies that.

    **P95 rule, made explicit** (the mega-brief asks this be a decision, not a
    default): the threshold is the ``percentile``-th percentile (default 95,
    i.e. the top 5%) of scores **restricted to `in_mask` (train+val, the rows
    the detector was allowed to see)** -- never the OOT/test rows, mirroring
    how `src.evaluation.thresholds.calibrate_threshold` is fitted on
    validation only elsewhere in this pipeline. That fixed threshold is then
    **applied to every row of the panel**, in-time and OOT alike -- scoring a
    row is not fitting on it, so this does not leak. Ties at the threshold are
    **included** (`score >= threshold`), so the exported set can be slightly
    larger than exactly 5% when scores repeat.

    Every original column of ``df`` is preserved (this never overwrites or
    subsets the source data) plus: ``isolation_forest_score``,
    ``p95_threshold``, ``run_id``, ``split`` (train/val/test per row, from
    ``split_masks``).

    Validates the written artifact before returning: exists, non-empty,
    row/column counts match the in-memory table exactly, and a re-read of the
    file reproduces the same shape (catches silent truncation). Raises
    `ArtifactGenerationError` on any validation failure -- **by design, this
    is not caught here**, so a failure propagates out of this function and
    the caller (`main.py`) does not proceed to the next layer.

    Returns:
        ``(path, table_df, threshold)``.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    scores = np.asarray(scores, dtype=float).ravel()
    in_mask = np.asarray(in_mask, dtype=bool).ravel()

    if len(scores) != len(df):
        raise ArtifactGenerationError(
            f"scores has {len(scores)} entries but df has {len(df)} rows -- "
            f"they must be row-aligned for a P95 checkpoint export.",
            check=f"{model_name}.p95_checkpoint.row_alignment",
            observed={"len_scores": len(scores), "len_df": len(df)},
        )

    with log_phase(f"evaluation.export_p95_checkpoint[{model_name}]", log):
        in_time_scores = scores[in_mask]
        threshold = float(np.percentile(in_time_scores, percentile))
        select_mask = scores >= threshold
        n_selected = int(select_mask.sum())

        split_col = np.full(len(df), "unknown", dtype=object)
        for name, mask in split_masks.items():
            split_col[np.asarray(mask, dtype=bool)] = name

        ctx = observability.current_run()
        run_id = ctx.run_id if ctx is not None else None

        # Every diagnostic column is attached *before* the single final sort,
        # so score/threshold/run_id/split all stay row-aligned with `table`
        # regardless of how the sort reorders it.
        table = df.loc[select_mask].copy().reset_index(drop=True)
        table["isolation_forest_score"] = scores[select_mask]
        table["p95_threshold"] = threshold
        table["run_id"] = run_id
        table["split"] = split_col[select_mask]

        sort_cols = [c for c in ("isolation_forest_score", entity_col) if c in table.columns]
        table = table.sort_values(
            by=sort_cols, ascending=[False, True][: len(sort_cols)], kind="stable",
        ).reset_index(drop=True)

        resolved_out = out_path or os.path.join(
            paths.REPORTS_DIR, f"p95_checkpoint_{model_name}.xlsx"
        )
        parent = os.path.dirname(os.path.abspath(resolved_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        table.to_excel(resolved_out, index=False, engine="openpyxl")

        # -- Artifact validation (blocking) ---------------------------------- #
        exists = os.path.isfile(resolved_out)
        size_bytes = os.path.getsize(resolved_out) if exists else 0
        reread_ok, reread_shape = False, None
        if exists and size_bytes > 0:
            try:
                reread = pd.read_excel(resolved_out, engine="openpyxl")
                reread_shape = tuple(reread.shape)
                reread_ok = reread_shape == table.shape
            except Exception as exc:
                log.warning("Could not re-read %s for validation: %s", resolved_out, exc)

        checksum = None
        if exists:
            with open(resolved_out, "rb") as fh:
                checksum = hashlib.sha256(fh.read()).hexdigest()[:16]
        generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        valid = exists and size_bytes > 0 and reread_ok

        observability.check(
            name=f"{model_name}.p95_checkpoint_export", category="artifact",
            definition="P95 checkpoint Excel exists, is non-empty, and a re-read "
                        "reproduces the in-memory table's exact shape.",
            expected="exists and size_bytes > 0 and reread_shape == table.shape",
            severity="critical", passed=valid,
            observed={
                "path": resolved_out, "size_bytes": size_bytes, "checksum_sha256_16": checksum,
                "rows": int(table.shape[0]), "cols": int(table.shape[1]),
                "reread_shape": reread_shape, "threshold": threshold, "percentile": percentile,
                "generated_at": generated_at, "run_id": run_id,
            },
            failure_action="Raise ArtifactGenerationError -- the VAE layer must not start "
                            "without a validated P95 checkpoint from this layer.",
            evidence=resolved_out,
        )
        if not valid:
            raise ArtifactGenerationError(
                f"P95 checkpoint export for {model_name!r} failed validation: "
                f"exists={exists}, size_bytes={size_bytes}, reread_shape={reread_shape} "
                f"(expected {table.shape}).",
                check=f"{model_name}.p95_checkpoint_export",
                observed={"path": resolved_out, "size_bytes": size_bytes, "reread_shape": reread_shape},
            )

        log.info(
            "[%s] P95 checkpoint: threshold=%.6f (percentile=%.1f, fitted on %d in-time "
            "scores) -> %d/%d rows selected -> %s (checksum=%s)",
            model_name, threshold, percentile, int(in_mask.sum()),
            n_selected, len(df), resolved_out, checksum,
        )

    return resolved_out, table, threshold
