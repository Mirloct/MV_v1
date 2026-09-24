"""Supervised challengers trained on reviewed event labels -- never a replacement
for IF/VAE, and never run without evidence.

Three deliberately small models (P1/P4 of ``reports/Supervisión temporal por
eventos.md``), all L2-penalised logistic regressions with natural prevalence and
no resampling:

``logit_scores_head``
    a head over the **frozen** IF/VAE scores only (2 parameters): asks whether
    reviewed feedback adds signal over the unsupervised detectors.
``ridge_logistic``
    ridge logistic regression on the model matrix, capped to ``max_features``
    columns chosen on training rows only.
``discrete_hazard``
    the same ridge model fitted on person-period data for "a NEW episode starts
    within the next H months" (rows already inside an episode are not at risk;
    rows whose horizon is not fully observed are censored out, not negatives).

Leakage rules enforced here, not left to the caller: fit rows must end at least
``purge`` months before the first evaluation month, and any episode that touches
that embargo is dropped from fitting entirely (an episode never crosses folds).
The regularisation strength is chosen on the validation months only.

Fallbacks, each reported with its reason instead of failing the run: gate not
authorising (unless ``force``), too few training episodes, a single class in the
fit rows, a failed fit, or an evaluation window without positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from src.evaluation.event_evaluation import calibration_summary, evaluate_ranking
from src.evaluation.event_labels import EventLabels
from src.utils.logging_config import log_phase, setup_logging
from src.utils.progress import track

__all__ = ["ChallengerConfig", "run_event_challengers"]

FAMILIES = ("logit_scores_head", "ridge_logistic", "discrete_hazard")


@dataclass(frozen=True)
class ChallengerConfig:
    hazard_horizon_months: int = 3
    c_grid: tuple[float, ...] = (0.01, 0.05, 0.2, 1.0)
    default_c: float = 0.05
    max_features: int = 40
    min_train_positive_episodes: int = 5
    review_capacity_k: Optional[int] = None
    n_boot: int = 100
    min_positive_episodes_conclusive: int = 20
    seed: int = 42


def _dense(X, rows: np.ndarray, cols: Sequence[int]) -> np.ndarray:
    sub = X[rows][:, list(cols)] if sp.issparse(X) else np.asarray(X)[np.ix_(rows, list(cols))]
    return np.asarray(sub.toarray() if sp.issparse(sub) else sub, dtype=float)


def _select_columns(X, rows: np.ndarray, y_rows: np.ndarray, max_features: int) -> list[int]:
    """Train-only univariate screen: the ``max_features`` columns most correlated
    (absolute) with the target. Constant columns score 0."""
    n_cols = X.shape[1]
    if n_cols <= max_features:
        return list(range(n_cols))
    block = _dense(X, rows, range(n_cols))
    yc = y_rows - y_rows.mean()
    xc = block - block.mean(axis=0)
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.abs(np.divide(xc.T @ yc, denom, out=np.zeros(n_cols), where=denom > 0))
    return sorted(np.argsort(-corr)[:max_features].tolist())


def _fit_logit(Xtr: np.ndarray, ytr: np.ndarray, c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=2000).fit(scaler.transform(Xtr), ytr)
    return scaler, model


def _predict(scaler, model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def _ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def _fit_rows(labels: EventLabels, y: np.ndarray, eligible: np.ndarray, in_fit: np.ndarray,
              eval_mask: np.ndarray, purge: int) -> np.ndarray:
    """Rows that may be fitted on: eligible, inside the fit blocks, ending at
    least ``purge`` months before the first evaluation month, and not part of an
    episode that touches that embargo."""
    first_eval = int(labels.month[eval_mask].min()) if eval_mask.any() else int(labels.month.max()) + 1
    ok = eligible & in_fit & (labels.month <= first_eval - 1 - purge)
    if len(labels.episodes):
        touching = labels.episodes.loc[labels.episodes["end_m"] >= first_eval - purge, "episode_id"]
        ok &= ~np.isin(labels.episode_id, touching.to_numpy())
    return np.flatnonzero(ok)


def _evaluate_windows(p_by_row, y, eligible, labels, episode, windows, cfg, n_boot=None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in windows.items():
        res = evaluate_ranking(
            p_by_row, y, eligible & mask, labels.entity, episode, k=cfg.review_capacity_k,
            n_boot=cfg.n_boot if n_boot is None else n_boot, seed=cfg.seed,
            min_positive_episodes=cfg.min_positive_episodes_conclusive,
        )
        sel = eligible & mask & np.isfinite(p_by_row)
        if res.get("status") == "ok":
            res["calibration"] = calibration_summary(p_by_row[sel], y[sel])
        out[name] = res
    return out


def _run_one(name, X, feature_names, labels, frozen, y, eligible, episode, masks, cfg) -> dict[str, Any]:
    """Fit one family and evaluate it (and the IF/VAE baselines) on each window."""
    windows = {"test": masks["test"], "oot": masks["oot"]}
    eval_any = masks["test"] | masks["oot"]
    purge = max(labels.spec.maturity_lag_months, cfg.hazard_horizon_months if name == "discrete_hazard" else 0)
    fit_idx = _fit_rows(labels, y, eligible, masks["train"] | masks["val"], eval_any, purge)
    train_idx = np.intersect1d(fit_idx, np.flatnonzero(masks["train"]))
    val_idx = np.intersect1d(fit_idx, np.flatnonzero(masks["val"]))
    y_fit = y[fit_idx]
    train_episodes = len(np.unique(episode[fit_idx][y_fit == 1]))
    result: dict[str, Any] = {"family": name, "fit_rows": int(len(fit_idx)),
                              "fit_positive_rows": int(y_fit.sum()), "fit_positive_episodes": int(train_episodes),
                              "purge_months": int(purge)}
    if len(np.unique(y_fit)) < 2:
        return {**result, "status": "skipped", "reason": "Las filas de ajuste tienen una sola clase tras la purga temporal."}
    if train_episodes < cfg.min_train_positive_episodes:
        return {**result, "status": "skipped",
                "reason": f"Solo {train_episodes} episodio(s) positivo(s) de ajuste (< {cfg.min_train_positive_episodes})."}

    if name == "logit_scores_head":
        cols = list(frozen)
        feature = lambda rows: np.column_stack([frozen[c][rows] for c in cols])  # noqa: E731
        names = cols
    else:
        screen_rows = train_idx if len(train_idx) else fit_idx
        cols_idx = _select_columns(X, screen_rows, y[screen_rows], cfg.max_features)
        feature = lambda rows: _dense(X, rows, cols_idx)  # noqa: E731
        names = [str(feature_names[i]) for i in cols_idx]

    chosen_c = cfg.default_c
    c_source = "por defecto (validación sin positivos)"
    if name != "logit_scores_head" and len(val_idx) and y[val_idx].sum() > 0 and len(train_idx) and len(np.unique(y[train_idx])) == 2:
        scored = []
        for c in cfg.c_grid:
            scaler, model = _fit_logit(feature(train_idx), y[train_idx], c)
            scored.append((_ap(y[val_idx], _predict(scaler, model, feature(val_idx))), c))
        chosen_c = max(scored, key=lambda t: (np.nan_to_num(t[0], nan=-1.0), -t[1]))[1]
        c_source = "elegido en los meses de validación"
    scaler, model = _fit_logit(feature(fit_idx), y_fit, chosen_c)

    # Score every row of the evaluation windows (only those are ever read).
    p = np.full(len(y), np.nan)
    rows = np.flatnonzero(eval_any & eligible)
    if len(rows):
        p[rows] = _predict(scaler, model, feature(rows))
    coef = model.coef_[0]
    n_params = len(names) + 1
    result.update(
        status="executed", C=float(chosen_c), C_source=c_source, n_features=len(names),
        events_per_parameter=train_episodes / n_params,
        top_coefficients=[{"feature": names[i], "standardized_coef": float(coef[i])}
                          for i in np.argsort(-np.abs(coef))[:10]],
        warnings=[],
    )
    if result["events_per_parameter"] < 10:
        result["warnings"].append(
            "Menos de 10 episodios por parámetro: modelo exploratorio; la regularización, no el "
            "tamaño de muestra, es lo único que limita el sobreajuste.")
    if _ap(y_fit, model.predict_proba(scaler.transform(feature(fit_idx)))[:, 1]) >= 0.999 and y_fit.sum() < 50:
        result["warnings"].append("Posible separación casi perfecta en el ajuste; considerar Firth como sensibilidad.")
    result["windows"] = _evaluate_windows(p, y, eligible, labels, episode, windows, cfg)
    baselines = {}
    for base_name, base_scores in frozen.items():
        baselines[base_name] = _evaluate_windows(np.asarray(base_scores, dtype=float), y, eligible, labels,
                                                 episode, windows, cfg)
    result["baselines"] = baselines
    return result


def run_event_challengers(
    *,
    X: Any,
    feature_names: Sequence[str],
    labels: EventLabels,
    masks: dict[str, np.ndarray],
    frozen_scores: dict[str, np.ndarray],
    gate: dict[str, Any],
    cfg: ChallengerConfig = ChallengerConfig(),
    force: bool = False,
) -> dict[str, Any]:
    """Run every authorised (or forced) family; see the module docstring.

    Args:
        X: Model matrix aligned with the panel rows (sparse or dense).
        masks: Boolean row masks ``train``, ``val``, ``test``, ``oot``.
        frozen_scores: ``{"if_score": ..., "vae_score": ...}`` from the already
            fitted detectors (never refitted here).
        gate: The acta from :func:`src.evaluation.label_gate.decide_gate`.
        force: Run despite a red gate (exploration); the result says so.
    """
    log = setup_logging()
    if not gate.get("supervised_challengers_allowed") and not force:
        return {"status": "skipped", "forced": False,
                "reason": f"Compuerta {gate.get('level')}: se mantiene IF/VAE "
                          f"({gate.get('action', '')})", "families": {}}
    allowed = set(gate.get("implemented_families", [])) | ({*FAMILIES} if force else set())
    families: dict[str, Any] = {}
    frozen = {k: np.asarray(v, dtype=float) for k, v in frozen_scores.items()}
    y_cur, elig_cur = labels.target, labels.usable
    y_haz, elig_haz, epi_haz = labels.horizon_target(cfg.hazard_horizon_months)
    for name in track([f for f in FAMILIES if f in allowed], desc="event_challengers[families]",
                      unit="familia", label=str):
        y, elig, epi = ((y_haz, elig_haz, epi_haz) if name == "discrete_hazard"
                        else (labels.target, elig_cur, labels.episode_id))
        try:
            with log_phase(f"event_challenger.{name}"):
                families[name] = _run_one(name, X, feature_names, labels, frozen, np.nan_to_num(y), elig,
                                          epi, masks, cfg)
        except Exception as exc:  # noqa: BLE001 - a challenger must never break the run
            log.warning("Challenger %s failed (%s); IF/VAE results are unaffected.", name, exc)
            families[name] = {"family": name, "status": "failed", "reason": str(exc)}
    executed = [n for n, r in families.items() if r.get("status") == "executed"]
    return {
        "status": "executed" if executed else "skipped",
        "forced": bool(force and not gate.get("supervised_challengers_allowed")),
        "reason": None if executed else "Ninguna familia pudo ajustarse (ver cada motivo).",
        "hazard_horizon_months": cfg.hazard_horizon_months,
        "families": families,
        "promotion": "Ningún challenger se promueve desde este acta: la promoción exige mejora "
                     "repetida en varios orígenes y semillas frente a IF/VAE (Fase 5/6 del plan).",
    }
