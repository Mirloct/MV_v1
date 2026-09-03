from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DiagnosticConfig
from .contracts import DataContractError, validate_frames
from .data_quality import compare_populations, leakage_name_warnings
from .diagnostics import (
    assign_disagreement_quadrant,
    build_autopsy,
    detect_score_orientation_risk,
)
from .metrics import evaluate_family_scores, evaluate_scores, positive_coverage
from .modeling import isolation_forest_scores, prepare_features
from .reporting import (
    disagreement_plot,
    metrics_bar_plot,
    write_json,
    write_markdown_report,
)
from .scoring import (
    anomaly_percentile,
    latent_diagnostics,
    latent_mahalanobis,
    reconstruction_scores,
    residual_contributions,
    row_kl_divergence,
)
from .stability import top_k_stability


@dataclass
class DiagnosticResult:
    scored: pd.DataFrame
    metrics: pd.DataFrame
    drift: pd.DataFrame
    autopsies: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[dict[str, Any]]


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _prefixed_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([column for column in frame.columns if column.startswith(prefix)])


def _reconstruction_arrays(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    config: DiagnosticConfig,
    reference_x: np.ndarray,
    scored_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = [f"{config.reconstruction_prefix}{f}" for f in config.features]
    missing = [c for c in columns if c not in reference or c not in scored]
    if missing:
        raise DataContractError(f"Missing VAE reconstruction columns: {missing}")
    reference_recon = reference[columns].to_numpy(dtype=float)
    scored_recon = scored[columns].to_numpy(dtype=float)
    if not np.isfinite(reference_recon).all() or not np.isfinite(scored_recon).all():
        raise DataContractError("VAE reconstruction outputs must be finite")
    return (
        np.abs(reference_x - reference_recon),
        np.abs(scored_x - scored_recon),
        reference_recon,
        scored_recon,
    )


def _if_outputs(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    reference_x: np.ndarray,
    scored_x: np.ndarray,
    config: DiagnosticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if config.if_score_col:
        column = config.if_score_col
        if column not in reference or column not in scored:
            raise DataContractError(f"Precomputed IF score column {column!r} is missing")
        direction = 1.0 if config.if_higher_is_anomalous else -1.0
        return (
            direction * reference[column].to_numpy(dtype=float),
            direction * scored[column].to_numpy(dtype=float),
            None,
        )
    return isolation_forest_scores(reference_x, scored_x, config)


def _score_reconstruction_candidates(
    reference_residual: np.ndarray,
    scored_residual: np.ndarray,
    config: DiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    reference_scores = reconstruction_scores(
        reference_residual, reference_residual, config.top_k_residuals
    )
    scored_scores = reconstruction_scores(
        reference_residual, scored_residual, config.top_k_residuals
    )
    contributions = residual_contributions(reference_residual, scored_residual)
    return reference_scores, scored_scores, contributions


def _add_latent_candidates(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    config: DiagnosticConfig,
    reference_scores: pd.DataFrame,
    scored_scores: pd.DataFrame,
) -> dict[str, Any] | None:
    mu_ref_columns = _prefixed_columns(reference, config.latent_mu_prefix)
    mu_score_columns = _prefixed_columns(scored, config.latent_mu_prefix)
    if not mu_ref_columns and not mu_score_columns:
        return None
    if mu_ref_columns != mu_score_columns:
        raise DataContractError("Reference/scored latent mean columns do not match")
    reference_mu = reference[mu_ref_columns].to_numpy(dtype=float)
    scored_mu = scored[mu_score_columns].to_numpy(dtype=float)
    ref_distance, score_distance = latent_mahalanobis(reference_mu, scored_mu)
    reference_scores["latent_mahalanobis"] = ref_distance
    scored_scores["latent_mahalanobis"] = score_distance
    logvar_columns = _prefixed_columns(reference, config.latent_logvar_prefix)
    if not logvar_columns:
        return {"latent_dimensions": reference_mu.shape[1], "kl_available": False}
    if logvar_columns != _prefixed_columns(scored, config.latent_logvar_prefix):
        raise DataContractError("Reference/scored latent log-variance columns do not match")
    reference_logvar = reference[logvar_columns].to_numpy(dtype=float)
    scored_logvar = scored[logvar_columns].to_numpy(dtype=float)
    reference_scores["vae_kl"] = row_kl_divergence(reference_mu, reference_logvar)
    scored_scores["vae_kl"] = row_kl_divergence(scored_mu, scored_logvar)
    return latent_diagnostics(
        reference_mu, reference_logvar, config.active_variance_threshold
    )


def _add_percentiles(
    output: pd.DataFrame,
    reference_scores: pd.DataFrame,
    scored_scores: pd.DataFrame,
) -> list[str]:
    percentile_columns: list[str] = []
    for column in scored_scores.columns:
        output[column] = scored_scores[column].to_numpy()
        percentile_column = f"{column}_percentile"
        output[percentile_column] = anomaly_percentile(
            reference_scores[column].to_numpy(), scored_scores[column].to_numpy()
        )
        percentile_columns.append(percentile_column)
    return percentile_columns


def _group_metrics(
    scored: pd.DataFrame,
    config: DiagnosticConfig,
    score_columns: list[str],
) -> pd.DataFrame:
    if config.label_col is None:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    if config.segment_col and config.segment_col in scored:
        for group_value, group in scored.groupby(config.segment_col, dropna=False):
            table = evaluate_scores(
                group, config.label_col, score_columns, config.alert_budgets, config.id_col
            )
            table.insert(0, "cohort_positives", int(group[config.label_col].sum()))
            table.insert(0, "cohort_rows", len(group))
            table.insert(0, "group_value", str(group_value))
            table.insert(0, "group_column", config.segment_col)
            rows.append(table)
    if config.family_col and config.family_col in scored:
        family_table = evaluate_family_scores(
            scored,
            config.label_col,
            config.family_col,
            score_columns,
            config.alert_budgets,
            config.id_col,
        )
        if not family_table.empty:
            rows.append(family_table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _supervised_outputs(
    output: pd.DataFrame,
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    scored_x: np.ndarray,
    score_recon: np.ndarray,
    contributions: np.ndarray,
    score_columns: list[str],
    config: DiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, int]], pd.DataFrame, int | None, dict[str, int] | None]:
    """Everything that requires a known-positive label: metrics, group
    metrics, coverage, and autopsies. Returns empty/None placeholders when
    ``config.label_col`` is ``None`` (label-free mode) instead of raising --
    the label-free path still gets percentiles, quadrants, latent, and drift
    diagnostics from the caller, just not these."""
    if config.label_col is None:
        return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame(), None, None
    metrics = evaluate_scores(
        output, config.label_col, score_columns, config.alert_budgets, config.id_col
    )
    group_metrics = _group_metrics(output, config, score_columns)
    coverage = [
        positive_coverage(
            output[config.label_col], output["if_percentile"], output["vae_percentile"], k
        )
        for k in config.alert_budgets
    ]
    reconstruction_frame = pd.DataFrame(score_recon, columns=config.features)
    contribution_frame = pd.DataFrame(contributions, columns=config.features)
    autopsies = build_autopsy(
        output[config.label_col] == 1,
        pd.DataFrame(scored_x, columns=config.features),
        reconstruction_frame,
        contribution_frame,
        output[config.id_col].tolist(),
    )
    known_positives = int(output[config.label_col].sum())
    positive_quadrants = output.loc[
        output[config.label_col] == 1, "quadrant"
    ].value_counts().to_dict()
    return metrics, group_metrics, coverage, autopsies, known_positives, positive_quadrants


def _build_warnings(
    base: list[dict[str, Any]],
    scored: pd.DataFrame,
    config: DiagnosticConfig,
    latent: dict[str, Any] | None,
    has_stability: bool,
) -> list[dict[str, Any]]:
    warnings = [*base, *leakage_name_warnings(config.features)]
    if config.label_col is not None:
        for score in ["if_percentile", "vae_percentile"]:
            risk = detect_score_orientation_risk(scored[config.label_col], scored[score])
            if risk.get("risk"):
                warnings.append({"code": f"{score}_orientation", "severity": "high", **risk})
    if latent and latent.get("collapsed_fraction", 0.0) > 0.5:
        warnings.append({
            "code": "possible_posterior_collapse",
            "severity": "high",
            "message": "More than half of latent units are inactive on the reference population.",
        })
    if not has_stability:
        warnings.append({
            "code": "if_stability_unavailable",
            "severity": "medium",
            "message": "Precomputed IF scores cannot be refit across seeds by this run.",
        })
    return warnings


def _write_outputs(
    output_dir: Path,
    result: DiagnosticResult,
    group_metrics: pd.DataFrame,
    coverage: list[dict[str, int]],
    stability: dict[str, Any] | None,
    config: DiagnosticConfig,
) -> None:
    result.scored.to_csv(output_dir / "scored_diagnostics.csv", index=False)
    result.metrics.to_csv(output_dir / "metrics.csv", index=False)
    result.drift.to_csv(output_dir / "drift.csv", index=False)
    result.autopsies.to_csv(output_dir / "autopsies.csv", index=False)
    group_metrics.to_csv(output_dir / "metrics_by_group.csv", index=False)
    write_json(result.summary, output_dir / "summary.json")
    write_json({"warnings": result.warnings}, output_dir / "warnings.json")
    write_json({"coverage": coverage}, output_dir / "coverage.json")
    write_json({"stability": stability}, output_dir / "if_stability.json")
    write_json(config.to_dict(), output_dir / "resolved_config.json")
    disagreement_plot(result.scored, output_dir / "disagreement.png", config.label_col)
    has_metrics_plot = metrics_bar_plot(result.metrics, output_dir / "metrics.png")
    write_markdown_report(
        output_dir / "report.md", result.summary, result.metrics, result.drift,
        result.warnings, has_metrics_plot,
    )


def run_diagnostic(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    config: DiagnosticConfig,
    output_dir: str | Path,
) -> DiagnosticResult:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base_warnings = validate_frames(
        reference, scored, config.features, config.id_col, config.label_col, config.time_col
    )
    reference_x, scored_x = prepare_features(reference, scored, config.features)
    if_reference, if_scored, if_runs = _if_outputs(
        reference, scored, reference_x, scored_x, config
    )
    ref_residual, score_residual, _, score_recon = _reconstruction_arrays(
        reference, scored, config, reference_x, scored_x
    )
    reference_scores, scored_scores, contributions = _score_reconstruction_candidates(
        ref_residual, score_residual, config
    )
    latent = _add_latent_candidates(reference, scored, config, reference_scores, scored_scores)
    if config.vae_primary_score not in scored_scores:
        raise DataContractError(
            f"vae_primary_score {config.vae_primary_score!r} is unavailable; "
            f"choose one of {list(scored_scores.columns)}"
        )
    output = scored.copy()
    _add_percentiles(output, reference_scores, scored_scores)
    output["if_score"] = if_scored
    output["if_percentile"] = anomaly_percentile(if_reference, if_scored)
    output["vae_percentile"] = output[f"{config.vae_primary_score}_percentile"]
    output["ensemble_max"] = output[["if_percentile", "vae_percentile"]].max(axis=1)
    output["ensemble_mean"] = output[["if_percentile", "vae_percentile"]].mean(axis=1)
    output["quadrant"] = assign_disagreement_quadrant(
        output["if_percentile"], output["vae_percentile"], config.percentile_threshold
    )
    score_columns = ["if_percentile", "vae_percentile", "ensemble_max", "ensemble_mean"]
    metrics, group_metrics, coverage, autopsies, known_positives, positive_quadrants = (
        _supervised_outputs(
            output, reference, scored, scored_x, score_recon, contributions,
            score_columns, config,
        )
    )
    stability = top_k_stability(if_runs, max(config.alert_budgets)) if if_runs is not None else None
    warnings = _build_warnings(base_warnings, output, config, latent, stability is not None)
    drift = compare_populations(reference, scored, config.features)
    summary = {
        "rows_scored": len(output),
        "known_positives": known_positives,
        "vae_primary_score": config.vae_primary_score,
        "percentile_threshold": config.percentile_threshold,
        "quadrants": output["quadrant"].value_counts().to_dict(),
        "positive_quadrants": positive_quadrants,
        "latent_diagnostics": latent,
        "warning_count": len(warnings),
        "reference_fingerprint": _frame_fingerprint(reference),
        "scored_fingerprint": _frame_fingerprint(scored),
    }
    result = DiagnosticResult(output, metrics, drift, autopsies, summary, warnings)
    _write_outputs(destination, result, group_metrics, coverage, stability, config)
    return result
