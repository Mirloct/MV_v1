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

The suite package (``ifvae_diag``) is an optional, vendored dependency, not
part of the core requirements. ``ensure_suite_installed`` validates the lazy
import and installs the repository's own copy automatically when needed; it
never resolves a package name from an index. Any remaining failure is the
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
    STATUS_FAILED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_REQUESTED,
    STATUS_UNAVAILABLE,
    REASON_NO_ENTITY_VIEW,
    REASON_NO_LABELS,
    REASON_NO_SEGMENT,
    REASON_NO_SENSITIVITY_GRID,
    REASON_NO_STABILITY_REFITS,
    _artifact_entry,
    feature_family,
    build_diagnostic_contract,
)
from src.evaluation.ifvae_interpretation import build_interpretation_contract
from src.utils.logging_config import setup_logging

__all__ = ["run_ifvae_diagnostic_suite", "diagnose_frames", "ensure_suite_installed"]

#: Path to the vendored suite, relative to the repository root. The package is
#: shipped inside this repo, so a missing install is a setup gap, not a
#: missing third-party dependency to go fetch from an index.
SUITE_SOURCE_DIR = os.path.join("tools", "if_vae_diagnostic_suite")


def ensure_suite_installed(auto_install: bool = True) -> dict:
    """Make ``ifvae_diag`` importable, installing the vendored copy if needed.

    The suite lives inside this repository (``tools/if_vae_diagnostic_suite``),
    so "not installed" means "this checkout was never `pip install -e`'d",
    not "a third-party package is missing". Rather than making every operator
    remember that one command, this checks and -- when ``auto_install`` --
    runs the editable install itself, from the vendored path only.

    Deliberately narrow, because auto-installing is a side effect: it only
    ever installs THIS repo's own vendored directory, never a name resolved
    from an index, and it is a no-op when the import already works.

    Returns:
        ``{"available": bool, "action": str, "detail": str}`` -- ``action`` is
        one of ``already_installed`` / ``installed_now`` / ``install_failed``
        / ``missing_source`` / ``not_attempted``, so the caller can report
        exactly what happened instead of only whether it worked.
    """
    import importlib
    import subprocess
    import sys

    log = setup_logging()

    def _importable() -> bool:
        try:
            importlib.import_module("ifvae_diag")
            return True
        except ImportError:
            return False

    if _importable():
        return {"available": True, "action": "already_installed",
                "detail": "El paquete ifvae_diag ya es importable."}

    source = os.path.abspath(SUITE_SOURCE_DIR)
    if not os.path.isdir(source):
        return {"available": False, "action": "missing_source",
                "detail": f"No existe el directorio vendorizado {SUITE_SOURCE_DIR}."}
    if not auto_install:
        return {"available": False, "action": "not_attempted",
                "detail": "La instalación automática está desactivada "
                          "(--no-auto-install-suite)."}

    log.info(
        "ifvae_diag no está instalado; instalando la copia vendorizada desde %s "
        "(pip install -e). Esto ocurre una sola vez por entorno.", source,
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", source],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:  # noqa: BLE001 - a failed install must not raise here
        return {"available": False, "action": "install_failed",
                "detail": f"No se pudo ejecutar pip: {exc}"}
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return {"available": False, "action": "install_failed",
                "detail": "pip devolvió código "
                          f"{completed.returncode}: {' | '.join(tail[-3:])}"}
    # `pip install -e` (PEP 660) writes an editable finder/`.pth` entry into
    # site-packages, but that only gets *read* when the `site` module runs at
    # interpreter start-up -- a plain `importlib.invalidate_caches()` refreshes
    # finder caches, it does not re-scan site-packages for new `.pth`/finder
    # files, so the freshly-installed package is still invisible in THIS
    # process. Rather than exec-ing a new interpreter, add the vendored
    # source directory straight onto `sys.path`: we already know exactly
    # where it is (we just installed from it), so this is not a guess.
    importlib.invalidate_caches()
    vendored_src = os.path.join(source, "src")
    if not _importable() and os.path.isdir(vendored_src) and vendored_src not in sys.path:
        sys.path.insert(0, vendored_src)
        importlib.invalidate_caches()
    if not _importable():
        return {"available": False, "action": "install_failed",
                "detail": "pip terminó sin error pero ifvae_diag sigue sin "
                          "poder importarse en este proceso (ni siquiera tras "
                          f"agregar {vendored_src} a sys.path). Puede requerir "
                          "reiniciar el intérprete."}
    log.info("ifvae_diag instalado correctamente desde la copia vendorizada.")
    return {"available": True, "action": "installed_now",
            "detail": f"Instalado con pip install -e {SUITE_SOURCE_DIR}."}


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


#: Families genuinely out of bounds for a report-only change (they need
#: core model/preprocessing code, or a multi-window pipeline structure this
#: bridge does not have) -- each reason names exactly why, not a blanket
#: "not implemented".
_OUT_OF_SCOPE_EXPERIMENTS = (
    ("Pérdidas por tipo de feature",
     "Requiere reescribir la función de pérdida del VAE por tipo de "
     "variable (Gaussiana/Bernoulli/categórica) -- código de modelo, fuera "
     "de alcance de un cambio solo de reporte."),
    ("Preprocesamiento",
     "Requiere re-ejecutar `fit_transform_panel` con transformaciones "
     "distintas -- código de preprocesamiento, fuera de alcance de un "
     "cambio solo de reporte."),
    ("Ablación de familias de features",
     "Requiere reconstruir la matriz de features excluyendo familias "
     "completas -- código de preprocesamiento, fuera de alcance de un "
     "cambio solo de reporte."),
    ("Backtests temporales",
     "Requiere reajustar en múltiples cortes temporales (origen rodante) -- "
     "una estructura de corridas distinta a la de un solo pipeline. La "
     "sección 8 ya desglosa por mes la ventana OOT de ESTA corrida, pero "
     "eso no reajusta nada en cortes anteriores."),
    ("Estabilidad entre ventanas",
     "Requiere comparar rankings entre múltiples ventanas OOT históricas, "
     "que esta corrida no conserva -- ver limitación de \"Backtests "
     "temporales\" arriba."),
)


def _swept_refit_experiment(
    detector_cls: Any, base_detector: Any, param_names: Sequence[str],
    sweep_param: str, sweep_grid: Sequence, fit_x: np.ndarray, score_x: np.ndarray,
    reference_high: np.ndarray, percentile_threshold: float, base_seed: int,
    fit_kwargs: Optional[dict] = None,
) -> list[dict]:
    """For each value in ``sweep_grid``, refit ``detector_cls`` with the
    production detector's own hyperparameters except ``sweep_param``
    (varied), on the SAME fit/score data and a FIXED seed (``base_seed`` --
    the point here is the hyperparameter's effect, not seed variance, which
    ``_seeded_refit_stability`` already measures separately). Each variant's
    own alert set (recomputed via the suite's own ``anomaly_percentile``, its
    own reference distribution) is compared against ``reference_high`` (the
    PRODUCTION detector's alert set at the same threshold) via Jaccard -- a
    real, executed comparison, never a placeholder.
    """
    from ifvae_diag.scoring import anomaly_percentile

    base_params = {name: getattr(base_detector, name) for name in param_names}
    rows = []
    for value in sweep_grid:
        try:
            detector = detector_cls(random_state=base_seed,
                                    **{**base_params, sweep_param: value})
            detector.fit(fit_x, **(fit_kwargs or {}))
            fit_scores = np.asarray(detector.score_samples(fit_x), dtype=float)
            variant_scores = np.asarray(detector.score_samples(score_x), dtype=float)
            variant_pct = anomaly_percentile(fit_scores, variant_scores)
            variant_high = variant_pct >= percentile_threshold
            counts = _quadrant_counts(reference_high, variant_high)
            relations = _set_relations(counts)
            rows.append({
                "value": value, "status": STATUS_EXECUTED,
                "alerts": int(variant_high.sum()), "total": int(len(variant_high)),
                "jaccard_vs_production": relations["jaccard"],
            })
        except Exception as exc:  # noqa: BLE001 - one bad grid point must not kill the sweep
            rows.append({"value": value, "status": STATUS_FAILED, "reason": str(exc)})
    return rows


def _build_experiments(
    *, scored_diag: pd.DataFrame, threshold: float,
    if_detector: Optional[Any] = None, x_if_fit: Optional[np.ndarray] = None,
    x_if_score: Optional[np.ndarray] = None,
    vae_detector: Optional[Any] = None, x_vae_fit: Optional[np.ndarray] = None,
    x_vae_score: Optional[np.ndarray] = None, valid_mask: Optional[np.ndarray] = None,
    contamination_grid: Sequence[float] = (),
    capacity_grid: Sequence[int] = (), beta_grid: Sequence[float] = (),
    base_seed: int = 42,
) -> list[dict]:
    """§9 tracking rows -- genuinely executed where the underlying comparison
    is cheap or already computed elsewhere; explicitly out of bounds (with a
    specific, per-family reason, see ``_OUT_OF_SCOPE_EXPERIMENTS``) where it
    would require touching model or preprocessing code, which is outside the
    scope of a report-only change.
    """
    experiments: list[dict] = []
    if_high = (scored_diag["if_percentile"] >= threshold).to_numpy()
    vae_high = (scored_diag["vae_percentile"] >= threshold).to_numpy()

    # -- Ensembles: max()/mean() of the two detectors' percentiles, already
    #    computed by the suite itself (`ifvae_diag.pipeline`) -- zero new
    #    cost, just surfaced here instead of silently dropped. -------------
    for name, label in (("ensemble_max", "máximo"), ("ensemble_mean", "promedio")):
        if name not in scored_diag.columns:
            experiments.append({
                "experiment": f"Ensembles ({label} IF/VAE)",
                "status": STATUS_UNAVAILABLE,
                "reason": f"La suite no escribió la columna {name}.",
            })
            continue
        combo_high = (scored_diag[name] >= threshold).to_numpy()
        counts = _quadrant_counts(if_high, combo_high)
        relations = _set_relations(counts)
        experiments.append({
            "experiment": f"Ensembles ({label} IF/VAE)",
            "status": STATUS_EXECUTED,
            "configuration": f"umbral={threshold:.3f}",
            "artifact": "scored_diagnostics.csv",
            "detail": f"{int(combo_high.sum())} alertas; Jaccard vs. IF solo = "
                     f"{relations['jaccard']:.3f}" if relations["jaccard"] is not None
                     else f"{int(combo_high.sum())} alertas",
        })

    # -- Variantes de reconstrucción: recon_mean/topk/max, distancia latente
    #    y KL ya se comparan en la sección 4 -- ejecutado ahí, no se repite.
    experiments.append({
        "experiment": "Variantes de reconstrucción (VAE)",
        "status": STATUS_EXECUTED,
        "configuration": "recon_mean, recon_topk, recon_max, latent_mahalanobis, KL",
        "artifact": "scored_diagnostics.csv",
        "detail": "Ver sección 4 (Comparación de puntajes VAE) -- misma "
                 "corrida, no se repite el cómputo aquí.",
    })

    # -- Variantes de contaminación (IF): reajuste real y barato (el forest
    #    no necesita reentrenamiento profundo), comparado contra la alerta
    #    de producción. --------------------------------------------------
    if contamination_grid and if_detector is not None and x_if_fit is not None:
        from src.models import IsolationForestDetector

        sweep = _swept_refit_experiment(
            IsolationForestDetector, if_detector, _IF_REFIT_PARAMS,
            "contamination", contamination_grid, x_if_fit, x_if_score,
            if_high, threshold, base_seed,
        )
        for value, row in zip(contamination_grid, sweep):
            if row["status"] == STATUS_EXECUTED:
                detail = (f"{row['alerts']} alertas de {row['total']}; Jaccard vs. "
                         f"producción = {row['jaccard_vs_production']:.3f}"
                         if row["jaccard_vs_production"] is not None
                         else f"{row['alerts']} alertas de {row['total']}")
            else:
                detail = None
            experiments.append({
                "experiment": f"Variantes de contaminación (IF, contamination={value})",
                "status": row["status"],
                "configuration": f"contamination={value}",
                "artifact": None,
                "detail": detail,
                "reason": row.get("reason"),
            })
    else:
        experiments.append({
            "experiment": "Variantes de contaminación (IF)",
            "status": STATUS_NOT_REQUESTED,
            "reason": "No se configuró una malla de contaminación "
                     "(--diagnostic-experiment-contamination-grid).",
        })

    # -- Capacidad y dimensión latente / Beta y programación KL (VAE):
    #    ejecutables con la misma clase VAEDetector, pero cada punto de la
    #    malla es un entrenamiento completo -- apagado por defecto, opt-in.
    for label, param, grid, refit_param_names in (
        ("Capacidad y dimensión latente (VAE)", "latent_dim", capacity_grid, _VAE_REFIT_PARAMS),
        ("Beta y programación KL (VAE)", "beta", beta_grid, _VAE_REFIT_PARAMS),
    ):
        if grid and vae_detector is not None and x_vae_fit is not None:
            from src.models import VAEDetector

            sweep = _swept_refit_experiment(
                VAEDetector, vae_detector, refit_param_names, param, grid,
                x_vae_fit, x_vae_score, vae_high, threshold, base_seed,
                fit_kwargs={"valid_mask": valid_mask} if valid_mask is not None else None,
            )
            for value, row in zip(grid, sweep):
                if row["status"] == STATUS_EXECUTED:
                    detail = (f"{row['alerts']} alertas de {row['total']}; Jaccard vs. "
                             f"producción = {row['jaccard_vs_production']:.3f}"
                             if row["jaccard_vs_production"] is not None
                             else f"{row['alerts']} alertas de {row['total']}")
                else:
                    detail = None
                experiments.append({
                    "experiment": f"{label}: {param}={value}",
                    "status": row["status"],
                    "configuration": f"{param}={value}",
                    "artifact": None,
                    "detail": detail,
                    "reason": row.get("reason"),
                })
        else:
            experiments.append({
                "experiment": label,
                "status": STATUS_NOT_REQUESTED,
                "reason": "No se configuró una malla (opt-in: cada punto "
                         "reentrena el VAE por completo). Ver "
                         "--diagnostic-experiment-capacity-grid / "
                         "--diagnostic-experiment-beta-grid.",
            })

    for name, reason in _OUT_OF_SCOPE_EXPERIMENTS:
        experiments.append({"experiment": name, "status": STATUS_NOT_REQUESTED,
                            "reason": reason})

    return experiments


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
    auto_install_suite: bool = True,
    experiment_contamination_grid: Sequence[float] = (0.01, 0.02, 0.05),
    experiment_capacity_grid: Sequence[int] = (),
    experiment_beta_grid: Sequence[float] = (),
) -> dict:
    """Run the suite label-free against this run's own OOT window and return
    ``{"contract": ..., "interpretation": ..., ...run-level fields}``.

    ``reference`` = the train block (``train_mask``); ``scored`` = the true
    OOT block (``oot_mask``), one row per (entity, period).

    Raises on any failure other than a single detector's stability refit
    (best-effort, caught internally); the caller (``main.py`` "Phase 9c") is
    responsible for catching everything else and logging.
    """
    # Checked here too (not just inside `diagnose_frames` below), so a
    # missing/unfixable install fails fast -- before the VAE forward pass
    # and stability refits below do real, potentially slow work for nothing.
    install = ensure_suite_installed(auto_install=auto_install_suite)
    if not install["available"]:
        raise RuntimeError(
            "El paquete vendorizado ifvae_diag no está disponible: "
            f"{install['detail']}"
        )

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
        auto_install_suite=auto_install_suite,
        experiment_contamination_grid=experiment_contamination_grid,
        experiment_capacity_grid=experiment_capacity_grid,
        experiment_beta_grid=experiment_beta_grid,
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
    auto_install_suite: bool = True,
    experiment_contamination_grid: Sequence[float] = (0.01, 0.02, 0.05),
    experiment_capacity_grid: Sequence[int] = (),
    experiment_beta_grid: Sequence[float] = (),
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
    install = ensure_suite_installed(auto_install=auto_install_suite)
    if not install["available"]:
        raise RuntimeError(
            "El paquete vendorizado ifvae_diag no está disponible: "
            f"{install['detail']}"
        )

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

    experiments = _build_experiments(
        scored_diag=scored_diag, threshold=threshold,
        if_detector=(stability or {}).get("if_detector"),
        x_if_fit=(stability or {}).get("x_if_fit"),
        x_if_score=(stability or {}).get("x_if_score"),
        vae_detector=(stability or {}).get("vae_detector"),
        x_vae_fit=(stability or {}).get("x_vae_fit"),
        x_vae_score=(stability or {}).get("x_vae_score"),
        valid_mask=(stability or {}).get("valid_mask"),
        contamination_grid=experiment_contamination_grid,
        capacity_grid=experiment_capacity_grid,
        beta_grid=experiment_beta_grid,
        base_seed=(stability or {}).get("base_seed", 42),
    )

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
        experiments=experiments,
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
