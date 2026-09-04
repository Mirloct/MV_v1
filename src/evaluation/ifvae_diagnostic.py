"""In-process bridge to the vendored IF-VAE Diagnostic Suite
(``tools/if_vae_diagnostic_suite``), plus the data-gathering layer that feeds
the report chapter's two contracts: the factual ficha
(``ifvae_contract.build_diagnostic_contract``) and the interpretation layer
(``ifvae_interpretation.build_interpretation_contract``).

Builds the suite's ``reference``/``scored`` frame contract directly from this
pipeline's own live, already-fitted IF + VAE objects -- no refit for the
scoring path, no CSV round-trip -- and runs it label-free. This project's
official runs carry no target, so label-free is the only mode wired into
``main.py``; a labeled run remains a manual, suite-side validation exercise
(``tools/export_diagnostic_suite_inputs.py``).

Everything downstream of ``run_diagnostic`` here is *reading and arranging*
what the suite already computed, PLUS one genuinely new measurement: seeded
stability refits (§ below). No score used for agreement/candidates/latent is
redefined, and nothing in this module ranks the detectors or interprets a
result for the factual contract -- see ``ifvae_interpretation`` for where
interpretation is deliberately allowed to happen, in a separate, labelled
layer.

**Stability refits are a real, additional cost.** Unlike every other section,
which only reads what ``run_diagnostic`` already computed, IF and VAE
stability each refit their detector ``stability_refits`` times (default 3)
with different seeds on the SAME architecture/hyperparameters this run's own
production detector used (read off the fitted instance's own public
attributes, e.g. ``vae_detector.latent_dim``) to measure top-K alert-set
Jaccard across seeds -- mirroring the suite's own
``ifvae_diag.stability.top_k_stability``, applied to a detector class the
suite itself does not refit (VAE) or does not refit when the score is
precomputed (IF, always true here since the "production" IF score is reused
rather than let the suite fit a throwaway one). VAE refits in particular are
full training runs, not just scoring, so this is the single most expensive
part of Phase 9c -- see ``main.py``'s ``--diagnostic-stability-refits`` flag.

The suite package (``ifvae_diag``) is an optional, vendored dependency
(``pip install -e tools/if_vae_diagnostic_suite``), not part of this
project's core requirements, so the import is lazy. Any failure here is the
caller's to catch (see ``main.py`` "Phase 9c").
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.loader import PanelSchema
from src.evaluation.ifvae_contract import (
    STATUS_EXECUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_REQUESTED,
    STATUS_UNAVAILABLE,
    REASON_NO_ENTITY_VIEW,
    REASON_NO_LABELS,
    REASON_NO_SEGMENT,
    REASON_NO_SENSITIVITY_GRID,
    REASON_NO_STABILITY_REFITS,
    REASON_NO_EXPERIMENT_TRACKING,
    _artifact_entry,
    feature_family,
    build_diagnostic_contract,
)
from src.evaluation.ifvae_interpretation import build_interpretation_contract

__all__ = ["run_ifvae_diagnostic_suite", "diagnose_frames"]

#: Every file the suite writes, checked for existence and non-emptiness so the
#: chapter never links to a missing or zero-byte artifact.
SUITE_ARTIFACTS = (
    "report.md", "summary.json", "warnings.json", "resolved_config.json",
    "scored_diagnostics.csv", "drift.csv", "metrics.csv", "metrics_by_group.csv",
    "autopsies.csv", "coverage.json", "if_stability.json", "disagreement.png",
    "metrics.png",
)

#: Upper bound on points serialised into the scatter block -- a rendering
#: budget, not a statistical choice. The sample is uniform and its size is
#: reported in the chart's own caption.
SCATTER_POINT_BUDGET = 4000

#: Below this many observations a per-period/per-segment row carries a size
#: warning. Structural guard so small groups are visibly flagged rather than
#: silently read as comparable to large ones.
SMALL_GROUP_WARNING_THRESHOLD = 30

QUADRANT_KEYS = ("BOTH", "IF_ONLY", "VAE_ONLY", "NEITHER")

#: Read off the FITTED detector's own public attributes (not off a possibly
#: partial ``best_params`` dict) so a stability refit uses exactly this run's
#: production architecture, tuned or not. ``random_state`` is deliberately
#: excluded -- that is the one thing each refit varies.
_IF_REFIT_PARAMS = ("n_estimators", "max_samples", "max_features", "contamination", "bootstrap")
_VAE_REFIT_PARAMS = (
    "latent_dim", "hidden_dim", "n_layers", "hidden_dims", "dropout", "activation",
    "beta", "lr", "optimizer", "batch_size", "weight_decay", "score_kl_weight",
    "epochs", "kl_anneal_epochs", "early_stopping_patience",
)


def _densify(x, dtype=np.float64) -> np.ndarray:
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=dtype)


def _vae_forward(detector: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mu, logvar, reconstruction)`` for every row of ``x``, batched.

    Same convention as ``tools/export_diagnostic_suite_inputs.py::_vae_forward``
    and ``VAEDetector.score_samples``/``latent_diagnostics``: a direct call
    into the fitted torch model (``detector._check_fitted()``) rather than a
    CSV round-trip, since this runs inside the same process that already
    holds the fitted detector and its input matrix in memory.
    """
    import torch

    model = detector._check_fitted()
    xd = _densify(x, dtype=np.float32)
    model.eval()
    n, bs = xd.shape[0], detector.batch_size
    mus = np.empty((n, model.latent_dim), dtype=np.float64)
    logvars = np.empty((n, model.latent_dim), dtype=np.float64)
    recon = np.empty_like(xd)
    with torch.no_grad():
        for start in range(0, n, bs):
            chunk = xd[start:start + bs]
            xb = torch.from_numpy(chunk).to(detector.device)
            mu, logvar = model.encode(xb)
            xr = model.decode(mu)
            end = start + xb.size(0)
            mus[start:end] = mu.cpu().numpy()
            logvars[start:end] = logvar.cpu().numpy()
            recon[start:end] = xr.cpu().numpy()
    return mus, logvars, recon


def _build_frame(
    entity_ids: np.ndarray,
    periods: np.ndarray,
    features_matrix: np.ndarray,
    feature_names: list,
    recon: np.ndarray,
    mu: np.ndarray,
    logvar: np.ndarray,
    if_score: np.ndarray,
    segment: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    blocks = [
        pd.DataFrame({"alert_id": [f"{e}__{p}" for e, p in zip(entity_ids, periods)]}),
        pd.DataFrame({"entity_id": entity_ids}),
        pd.DataFrame({"scoring_timestamp": periods}),
        pd.DataFrame(features_matrix, columns=feature_names),
        pd.DataFrame(recon, columns=[f"recon__{f}" for f in feature_names]),
        pd.DataFrame(mu, columns=[f"mu__{i}" for i in range(mu.shape[1])]),
        pd.DataFrame(logvar, columns=[f"logvar__{i}" for i in range(logvar.shape[1])]),
        pd.DataFrame({"if_score": if_score}),
    ]
    if segment is not None:
        blocks.append(pd.DataFrame({"segment": segment}))
    return pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)


# --------------------------------------------------------------------------- #
# Descriptive statistics over what the suite already computed                 #
# --------------------------------------------------------------------------- #
def _population_meta(frame: pd.DataFrame, name: str, source: str) -> dict:
    periods = frame["scoring_timestamp"]
    return {
        "name": name,
        "source": source,
        "rows": int(len(frame)),
        "entities": int(frame["entity_id"].nunique()),
        "periods": int(periods.nunique()),
        "period_min": str(periods.min()) if len(periods) else None,
        "period_max": str(periods.max()) if len(periods) else None,
    }


def _rank_correlation(a: pd.Series, b: pd.Series) -> tuple[Optional[float], Optional[str]]:
    """Spearman rho over two percentile columns, or the reason it is undefined.

    A descriptive statistic about rank agreement. It is reported as a number
    and nothing is concluded from its magnitude here.
    """
    if len(a) < 2:
        return None, "Se requieren al menos dos observaciones."
    if a.nunique() < 2 or b.nunique() < 2:
        return None, ("Al menos una de las series de percentiles es constante; "
                      "la correlación de rangos no está definida.")
    from scipy.stats import spearmanr

    value = float(spearmanr(a.to_numpy(), b.to_numpy()).statistic)
    if value != value:  # NaN
        return None, "El cálculo devolvió un valor no definido."
    return value, None


def _quadrant_counts(if_high: np.ndarray, vae_high: np.ndarray) -> dict:
    return {
        "BOTH": int(np.sum(if_high & vae_high)),
        "IF_ONLY": int(np.sum(if_high & ~vae_high)),
        "VAE_ONLY": int(np.sum(~if_high & vae_high)),
        "NEITHER": int(np.sum(~if_high & ~vae_high)),
    }


def _set_relations(counts: dict) -> dict:
    """Intersection/union/Jaccard, arithmetic on the quadrant counts."""
    intersection = counts["BOTH"]
    union = counts["BOTH"] + counts["IF_ONLY"] + counts["VAE_ONLY"]
    return {
        "if_alerts": counts["BOTH"] + counts["IF_ONLY"],
        "vae_alerts": counts["BOTH"] + counts["VAE_ONLY"],
        "intersection": intersection,
        "union": union,
        "jaccard": (intersection / union) if union else None,
    }


def _build_agreement(scored_diag: pd.DataFrame, threshold: float) -> dict:
    counts = {k: int((scored_diag["quadrant"] == k).sum()) for k in QUADRANT_KEYS}
    relations = _set_relations(counts)
    rho, rho_reason = _rank_correlation(
        scored_diag["if_percentile"], scored_diag["vae_percentile"]
    )
    total = int(len(scored_diag))
    sample = scored_diag
    sampled = False
    if total > SCATTER_POINT_BUDGET:
        sample = scored_diag.sample(SCATTER_POINT_BUDGET, random_state=0)
        sampled = True
    caption = (
        f"{len(sample):,} de {total:,} observaciones"
        + (" (muestra uniforme para acotar el tamaño del gráfico)" if sampled else "")
        + "."
    )
    return {
        "quadrants": counts,
        "total": total,
        **relations,
        "rank_correlation": rho,
        "rank_correlation_reason": rho_reason or "",
        "scatter": {
            "points": {
                "x": [round(float(v), 4) for v in sample["if_percentile"]],
                "y": [round(float(v), 4) for v in sample["vae_percentile"]],
                "quadrant": [str(v) for v in sample["quadrant"]],
            },
            "x_label": "Percentil del puntaje IF (respecto de la referencia)",
            "y_label": "Percentil del puntaje VAE (respecto de la referencia)",
            "threshold": float(threshold),
            "alt_text": (
                "Dispersión de observaciones por percentil IF (eje horizontal) y "
                "percentil VAE (eje vertical), con líneas en el umbral configurado "
                "que dividen los cuadrantes BOTH, IF_ONLY, VAE_ONLY y NEITHER. "
                "Los conteos de cada cuadrante se listan en la tabla anterior."
            ),
            "caption": caption,
        },
    }


def _build_candidates(scored_diag: pd.DataFrame, threshold: float, config: dict,
                      artifacts: dict) -> list[dict]:
    """One row per VAE score candidate the suite wrote, same statistics for all.

    Candidates are discovered from the columns present in
    ``scored_diagnostics.csv`` -- not from a hardcoded list.
    """
    if_high = (scored_diag["if_percentile"] >= threshold).to_numpy()
    artifact = "scored_diagnostics.csv" if artifacts.get(
        "scored_diagnostics.csv", {}).get("status") == STATUS_EXECUTED else None
    top_k = config.get("top_k_residuals")
    parameters = {"recon_topk": f"top_k_residuals={top_k}"} if top_k is not None else {}

    candidates = []
    percentile_columns = [c for c in scored_diag.columns if c.endswith("_percentile")]
    for column in percentile_columns:
        name = column[: -len("_percentile")]
        if name in ("if", "vae"):
            continue
        series = scored_diag[column]
        if series.isna().all():
            candidates.append({
                "name": name, "status": STATUS_UNAVAILABLE,
                "reason": "La columna existe pero no tiene valores.",
                "artifact": artifact, "total": len(scored_diag),
                "alerts": 0, "jaccard": None, "rank_correlation": None,
            })
            continue
        vae_high = (series >= threshold).to_numpy()
        counts = _quadrant_counts(if_high, vae_high)
        relations = _set_relations(counts)
        rho, _ = _rank_correlation(scored_diag["if_percentile"], series)
        candidates.append({
            "name": name,
            "status": STATUS_EXECUTED,
            "parameters": parameters.get(name, ""),
            "total": int(len(scored_diag)),
            "alerts": int(vae_high.sum()),
            "quadrant_text": " / ".join(str(counts[k]) for k in QUADRANT_KEYS),
            "intersection": relations["intersection"],
            "union": relations["union"],
            "jaccard": relations["jaccard"],
            "rank_correlation": rho,
            "artifact": artifact,
        })
    return sorted(candidates, key=lambda c: c["name"])


def _build_sensitivity(scored_diag: pd.DataFrame, grid: Optional[Sequence[float]],
                       config: dict, budgets: Optional[Sequence[int]]) -> dict:
    """Quadrant counts recomputed across a configured threshold grid."""
    source = "main.py::PipelineConfig.diagnostic_sensitivity_grid"
    if not grid:
        return {"status": STATUS_NOT_REQUESTED, "reason": REASON_NO_SENSITIVITY_GRID,
                "source": source, "rows": [], "grid": None}
    primary = config.get("vae_primary_score")
    rows = []
    total = len(scored_diag)
    budget_by_index = list(budgets or [])
    for index, threshold in enumerate(grid):
        if_high = (scored_diag["if_percentile"] >= threshold).to_numpy()
        vae_high = (scored_diag["vae_percentile"] >= threshold).to_numpy()
        counts = _quadrant_counts(if_high, vae_high)
        relations = _set_relations(counts)
        rho, _ = _rank_correlation(
            scored_diag["if_percentile"], scored_diag["vae_percentile"]
        )
        rows.append({
            "threshold": threshold,
            "score_candidate": primary,
            **counts,
            "if_rate": f"{100.0 * relations['if_alerts'] / total:.2f}%" if total else None,
            "vae_rate": f"{100.0 * relations['vae_alerts'] / total:.2f}%" if total else None,
            "union": relations["union"],
            "jaccard": relations["jaccard"],
            "rank_correlation": rho,
            "budget": budget_by_index[index] if index < len(budget_by_index) else None,
        })
    return {"status": STATUS_EXECUTED, "reason": None, "source": source,
            "rows": rows, "grid": list(grid)}


def _build_latent(summary: dict, config: dict, reference: pd.DataFrame) -> dict:
    source = "ifvae_diagnostics/summary.json::latent_diagnostics"
    latent = summary.get("latent_diagnostics")
    mu_available = any(c.startswith("mu__") for c in reference.columns)
    logvar_available = any(c.startswith("logvar__") for c in reference.columns)
    if not isinstance(latent, dict) or "active_units" not in latent:
        reason = (
            "Las poblaciones no incluyen columnas de log-varianza latente, por "
            "lo que la suite no calculó unidades activas ni KL por unidad."
            if mu_available and not logvar_available else
            "La corrida no expuso diagnóstico latente."
        )
        return {"status": STATUS_UNAVAILABLE, "reason": reason, "source": source}
    by_unit = list(latent.get("mean_kl_by_unit") or [])
    return {
        "status": STATUS_EXECUTED, "reason": None, "source": source,
        "latent_dimensions": int(latent.get("latent_dimensions", 0)),
        "active_units": int(latent.get("active_units", 0)),
        "collapsed_fraction": latent.get("collapsed_fraction"),
        "mu_variance_by_unit": list(latent.get("mu_variance_by_unit") or []),
        "mean_kl_by_unit": by_unit,
        "mean_kl": (sum(by_unit) / len(by_unit)) if by_unit else None,
        "active_threshold": config.get("active_variance_threshold"),
        "mu_available": "Sí" if mu_available else "No",
        "logvar_available": "Sí" if logvar_available else "No",
        "artifact": "summary.json",
    }


def _seeded_refit_stability(
    detector_cls: Any, base_detector: Any, param_names: Sequence[str],
    fit_x: np.ndarray, score_x: np.ndarray, seeds: Sequence[int], k: int,
    fit_kwargs: Optional[dict] = None,
) -> dict:
    """Refit ``detector_cls`` with each of ``seeds`` (same hyperparameters as
    ``base_detector``, read off its own public attributes) and measure top-K
    alert-set stability across refits via the suite's own
    ``top_k_stability`` -- the identical metric the suite uses for IF when it
    fits its own throwaway forest, just applied here to a class/detector the
    suite does not refit on its own.
    """
    from ifvae_diag.stability import top_k_stability

    params = {name: getattr(base_detector, name) for name in param_names}
    runs = []
    for seed in seeds:
        detector = detector_cls(random_state=seed, **params)
        detector.fit(fit_x, **(fit_kwargs or {}))
        runs.append(np.asarray(detector.score_samples(score_x), dtype=float))
    return top_k_stability(np.vstack(runs), k)


def _build_stability(
    *,
    if_detector: Any, x_if_fit: np.ndarray, x_if_score: np.ndarray,
    vae_detector: Any, x_vae_fit: np.ndarray, x_vae_score: np.ndarray,
    valid_mask: Optional[np.ndarray], config_dict: dict,
    stability_refits: int, base_seed: int,
) -> dict:
    """IF and VAE stability, both measured by genuinely refitting the
    detector -- see the module docstring for the added cost this implies.

    Seeds are derived from ``base_seed`` (this run's own seed) rather than
    hardcoded, so two different official runs use two different stability
    seed sets and neither collides with the production fit's own seed.
    """
    k = max(config_dict.get("alert_budgets") or [config_dict.get("top_k_residuals", 25)])
    if stability_refits <= 0:
        empty = {"status": STATUS_UNAVAILABLE, "reason": REASON_NO_STABILITY_REFITS,
                 "source": "main.py::PipelineConfig.diagnostic_stability_refits"}
        return {"iforest": empty, "vae": dict(empty)}

    seeds = tuple(base_seed + 1000 * (i + 1) for i in range(stability_refits))

    from src.models import IsolationForestDetector, VAEDetector

    try:
        if_result = _seeded_refit_stability(
            IsolationForestDetector, if_detector, _IF_REFIT_PARAMS,
            x_if_fit, x_if_score, seeds, k,
        )
        iforest = {
            "status": STATUS_EXECUTED, "reason": None,
            "source": "src/evaluation/ifvae_diagnostic.py (reajuste multisemilla)",
            "seeds": list(seeds), "refits": len(seeds), "top_k": if_result["k"],
            "mean_jaccard": if_result["mean_jaccard"],
            "min_jaccard": if_result["min_jaccard"],
            "resampling_unit": "Observación entidad–periodo (población evaluada)",
            "artifact": None,
        }
    except Exception as exc:  # noqa: BLE001 - stability is best-effort by design
        iforest = {"status": STATUS_UNAVAILABLE,
                   "reason": f"El reajuste multisemilla falló: {exc}",
                   "source": "src/evaluation/ifvae_diagnostic.py"}

    try:
        vae_result = _seeded_refit_stability(
            VAEDetector, vae_detector, _VAE_REFIT_PARAMS,
            x_vae_fit, x_vae_score, seeds, k,
            fit_kwargs={"valid_mask": valid_mask} if valid_mask is not None else None,
        )
        vae = {
            "status": STATUS_EXECUTED, "reason": None,
            "source": "src/evaluation/ifvae_diagnostic.py (reajuste multisemilla)",
            "seeds": list(seeds), "refits": len(seeds), "top_k": vae_result["k"],
            "mean_jaccard": vae_result["mean_jaccard"],
            "min_jaccard": vae_result["min_jaccard"],
            "resampling_unit": "Observación entidad–periodo (población evaluada)",
            "artifact": None,
        }
    except Exception as exc:  # noqa: BLE001
        vae = {"status": STATUS_UNAVAILABLE,
               "reason": f"El reajuste multisemilla falló: {exc}",
               "source": "src/evaluation/ifvae_diagnostic.py"}

    return {"iforest": iforest, "vae": vae}


def _group_quadrants(scored_diag: pd.DataFrame, group_column: str) -> list:
    rows = []
    for group, block in scored_diag.groupby(group_column, sort=True):
        counts = {k: int((block["quadrant"] == k).sum()) for k in QUADRANT_KEYS}
        relations = _set_relations(counts)
        n = len(block)
        rows.append([
            str(group), n, counts["BOTH"], counts["IF_ONLY"], counts["VAE_ONLY"],
            counts["NEITHER"],
            f"{100.0 * relations['if_alerts'] / n:.2f}%" if n else "",
            f"{100.0 * relations['vae_alerts'] / n:.2f}%" if n else "",
            (f"Menos de {SMALL_GROUP_WARNING_THRESHOLD} observaciones"
             if n < SMALL_GROUP_WARNING_THRESHOLD else ""),
        ])
    return rows


def _experiment_matrix() -> list[dict]:
    """§9 tracking rows -- see the section's own caption for why this stays
    NOT_REQUESTED rather than implemented: a full experiment-tracking harness
    (contamination/capacity/beta/preprocessing/ablation/ensemble/temporal
    sweeps, per the suite's own EXPERIMENT_MATRIX.md) is out of scope for
    this report-only change; this list keeps the backlog explicit instead of
    silently dropping it.
    """
    families = (
        "Variantes de contaminación", "Variantes de reconstrucción",
        "Pérdidas por tipo de feature", "Capacidad y dimensión latente",
        "Beta y programación KL", "Preprocesamiento",
        "Ablación de familias de features", "Ensembles",
        "Backtests temporales", "Estabilidad entre ventanas",
    )
    return [
        {"experiment": name, "status": STATUS_NOT_REQUESTED,
         "reason": REASON_NO_EXPERIMENT_TRACKING}
        for name in families
    ]


def _drift_signal(drift_frame: Optional[pd.DataFrame], derived_features: Sequence[str],
                  alpha: float = 0.05) -> dict:
    """Distilled drift signal for the interpretation layer only (the raw,
    row-per-feature table was removed from the factual ficha by request).

    Uses Benjamini-Hochberg FDR correction across every feature's KS p-value
    rather than flagging each nominally-significant test on its own -- with
    dozens of features tested simultaneously, an uncorrected 0.05 cutoff
    produces several false "drifted" features by construction. This is the
    standard remedy for multiple hypothesis testing (Benjamini & Hochberg,
    1995) and directly replaces the earlier, explicitly-flagged gap ("no se
    declaró una regla de severidad") with a principled one.
    """
    if drift_frame is None or drift_frame.empty or "ks_pvalue" not in drift_frame:
        return {"available": False, "n_features": 0, "n_flagged": 0, "top": []}
    frame = drift_frame.dropna(subset=["ks_pvalue"]).sort_values("ks_pvalue")
    m = len(frame)
    if m == 0:
        return {"available": False, "n_features": 0, "n_flagged": 0, "top": []}
    ranks = np.arange(1, m + 1)
    bh_threshold = ranks / m * alpha
    passed = frame["ks_pvalue"].to_numpy() <= bh_threshold
    # BH: the largest rank whose p-value clears its own threshold, and every
    # smaller rank, are flagged.
    flagged_upto = np.max(np.flatnonzero(passed)) if passed.any() else -1
    frame = frame.reset_index(drop=True)
    frame["flagged"] = frame.index <= flagged_upto
    flagged = frame[frame["flagged"]]
    return {
        "available": True,
        "n_features": int(m),
        "n_flagged": int(len(flagged)),
        "alpha": alpha,
        "method": "Benjamini-Hochberg FDR",
        "top": [
            {"feature": row["feature"], "family": feature_family(row["feature"], derived_features),
             "ks_statistic": round(float(row["ks_statistic"]), 4),
             "ks_pvalue": round(float(row["ks_pvalue"]), 6)}
            for _, row in flagged.head(5).iterrows()
        ],
    }


def run_ifvae_diagnostic_suite(
    keys: pd.DataFrame,
    schema: PanelSchema,
    if_detector: Any,
    if_scores: np.ndarray,
    x_if: Any,
    in_mask: np.ndarray,
    valid_mask: np.ndarray,
    vae_detector: Any,
    x_vae: Any,
    vae_feature_names: list,
    train_mask: np.ndarray,
    oot_mask: np.ndarray,
    out_dir: str,
    run_meta: Optional[dict] = None,
    top_k_residuals: int = 5,
    percentile_threshold: float = 0.95,
    sensitivity_grid: Optional[Sequence[float]] = None,
    entity_view: bool = False,
    stability_refits: int = 3,
    base_seed: int = 42,
    segment: Optional[np.ndarray] = None,
) -> dict:
    """Run the suite label-free against this run's own OOT window and return
    ``{"contract": ..., "interpretation": ..., ...run-level fields}``.

    ``reference`` = the train block (``train_mask``); ``scored`` = the true
    OOT block (``oot_mask``), one row per (entity, period).

    Raises on any failure other than a single detector's stability refit
    (best-effort, caught internally); the caller (``main.py`` "Phase 9c") is
    responsible for catching everything else and logging.
    """
    from ifvae_diag import run_diagnostic
    from ifvae_diag.config import DiagnosticConfig

    run_meta = dict(run_meta or {})
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"

    mu, logvar, recon = _vae_forward(vae_detector, x_vae)
    x_vae_dense = _densify(x_vae)
    entity_ids = keys[entity_col].to_numpy()
    periods = keys[time_col].astype(str).to_numpy()

    def _slice(mask: np.ndarray) -> pd.DataFrame:
        return _build_frame(
            entity_ids[mask], periods[mask], x_vae_dense[mask], vae_feature_names,
            recon[mask], mu[mask], logvar[mask], if_scores[mask],
            segment=(segment[mask] if segment is not None else None),
        )

    reference = _slice(train_mask)
    scored = _slice(oot_mask)

    return diagnose_frames(
        reference, scored, list(vae_feature_names), out_dir,
        run_meta=run_meta,
        top_k_residuals=top_k_residuals,
        percentile_threshold=percentile_threshold,
        sensitivity_grid=sensitivity_grid,
        entity_view=entity_view,
        segment_col=("segment" if segment is not None else None),
        stability=(
            {
                "if_detector": if_detector, "x_if_fit": x_if[in_mask],
                "x_if_score": x_if[oot_mask], "vae_detector": vae_detector,
                "x_vae_fit": x_vae[in_mask], "x_vae_score": x_vae[oot_mask],
                "valid_mask": valid_mask, "stability_refits": stability_refits,
                "base_seed": base_seed,
            }
        ),
    )


def diagnose_frames(
    reference: pd.DataFrame,
    scored: pd.DataFrame,
    features: list,
    out_dir: str,
    *,
    run_meta: Optional[dict] = None,
    top_k_residuals: int = 5,
    percentile_threshold: float = 0.95,
    sensitivity_grid: Optional[Sequence[float]] = None,
    entity_view: bool = False,
    label_col: Optional[str] = None,
    segment_col: Optional[str] = None,
    vae_primary_score: str = "recon_topk",
    stability: Optional[dict] = None,
) -> dict:
    """Run the suite over ready-made frames and assemble both contracts.

    Split out from ``run_ifvae_diagnostic_suite`` so the whole contract path
    is exercisable without a fitted VAE in the process -- the pipeline builds
    its frames from live detectors and calls this; the tests build small
    synthetic frames and call the very same code. ``stability`` (a dict of
    fitted detectors + matrices) is optional so tests can exercise everything
    else without paying for real refits; omitting it reports stability as
    NOT_REQUESTED with a stated reason, not as a silently-passing default.
    """
    from ifvae_diag import run_diagnostic
    from ifvae_diag.config import DiagnosticConfig

    run_meta = dict(run_meta or {})
    config = DiagnosticConfig(
        features=list(features),
        id_col="alert_id",
        label_col=label_col,
        time_col="scoring_timestamp",
        segment_col=segment_col,
        family_col=None,
        if_score_col="if_score",
        if_higher_is_anomalous=True,
        vae_primary_score=vae_primary_score,
        top_k_residuals=top_k_residuals,
        percentile_threshold=percentile_threshold,
    )
    result = run_diagnostic(reference, scored, config, out_dir)

    out_dir_abs = os.path.abspath(str(out_dir))
    artifacts = {name: _artifact_entry(out_dir_abs, name) for name in SUITE_ARTIFACTS}
    for entry in artifacts.values():
        if entry["path"]:
            entry["relative_path"] = os.path.join(
                os.path.basename(out_dir_abs), entry["name"]
            ).replace(os.sep, "/")

    config_dict = config.to_dict()
    threshold = float(config.percentile_threshold)
    scored_diag = result.scored
    derived_features = list(run_meta.get("derived_features") or [])

    agreement = _build_agreement(scored_diag, threshold)
    candidates = _build_candidates(scored_diag, threshold, config_dict, artifacts)
    sensitivity = _build_sensitivity(
        scored_diag, sensitivity_grid, config_dict, config_dict.get("alert_budgets")
    )
    latent = _build_latent(result.summary, config_dict, reference)

    if stability:
        stability_result = _build_stability(config_dict=config_dict, **stability)
    else:
        empty = {"status": STATUS_NOT_REQUESTED,
                 "reason": "No se proporcionaron detectores ajustados para el "
                           "reajuste de estabilidad en esta llamada.",
                 "source": "src/evaluation/ifvae_diagnostic.py"}
        stability_result = {"iforest": empty, "vae": dict(empty)}

    period_rows = _group_quadrants(scored_diag, "scoring_timestamp")
    temporal = {
        "status": STATUS_EXECUTED if period_rows else STATUS_UNAVAILABLE,
        "reason": None if period_rows else "La población evaluada no expone periodos.",
        "source": "ifvae_diagnostics/scored_diagnostics.csv",
        "rows": period_rows,
        "caption": (
            f"Los grupos con menos de {SMALL_GROUP_WARNING_THRESHOLD} "
            "observaciones se marcan en la última columna."
        ),
    }
    if segment_col and segment_col in scored.columns:
        segment_rows = _group_quadrants(
            scored_diag.assign(**{segment_col: scored[segment_col].to_numpy()}),
            segment_col,
        )
        segmentation = {"status": STATUS_EXECUTED, "reason": None,
                        "source": "ifvae_diagnostics/scored_diagnostics.csv",
                        "rows": segment_rows}
    else:
        segmentation = {"status": STATUS_NOT_APPLICABLE, "reason": REASON_NO_SEGMENT,
                        "source": "ifvae_diag.config::segment_col", "rows": []}

    populations = {
        "reference": _population_meta(
            reference, "Bloque de entrenamiento",
            "main.py::chronological_split (train_mask)"),
        "scored": _population_meta(
            scored, "Ventana fuera de tiempo (OOT)",
            "main.py::chronological_split (oot_mask)"),
        "observation_unit": "Observación entidad–periodo",
    }

    if entity_view:
        rule = run_meta.get("entity_aggregation_rule")
        if rule:
            aggregated = scored_diag.assign(
                entity_id=scored["entity_id"].to_numpy()
            ).groupby("entity_id")[["if_percentile", "vae_percentile"]].max()
            run_meta = {**run_meta, "_entity_count": int(len(aggregated))}

    drift_signal = _drift_signal(result.drift, derived_features)

    contract = build_diagnostic_contract(
        populations=populations,
        config=config_dict,
        run_meta={**run_meta,
                 "suite_version": _suite_version(),
                 "stability_refits": (stability or {}).get("stability_refits")},
        agreement=agreement,
        candidates=candidates,
        sensitivity=sensitivity,
        latent=latent,
        stability=stability_result,
        temporal=temporal,
        segmentation=segmentation,
        experiments=_experiment_matrix(),
    )

    interpretation = build_interpretation_contract(
        agreement=agreement,
        candidates=candidates,
        sensitivity=sensitivity,
        latent=latent,
        stability=stability_result,
        temporal=temporal,
        segmentation=segmentation,
        drift_signal=drift_signal,
        warnings=list(result.warnings),
        config=config_dict,
        run_meta=run_meta,
        populations=populations,
    )

    return {
        "contract": contract,
        "interpretation": interpretation,
        "rows_reference": int(len(reference)),
        "rows_scored": int(len(scored)),
        "quadrants": agreement["quadrants"],
        "report_dir": out_dir_abs,
        "report_md_path": os.path.join(out_dir_abs, "report.md"),
    }


def _suite_version() -> Optional[str]:
    try:
        import ifvae_diag

        return getattr(ifvae_diag, "__version__", None)
    except Exception:  # noqa: BLE001
        return None
