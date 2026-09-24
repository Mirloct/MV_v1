"""Gate 4.5: is there enough temporal evidence to try a supervised challenger?

Implements the governance semaphore of ``reports/Supervisión temporal por
eventos.md``. The decision is **not** a row-rate threshold: it counts *mature,
independent positive episodes*, distinct positive entities, temporal origins and
out-of-sample positives, and any veto wins over the count.

=================  =============================  ==============================
Level              Mature positive episodes       Work authorised
=================  =============================  ==============================
``rojo``           ``< 30`` or ``< 10`` entities  IF/VAE only; fix labels/audit
``ambar_1``        ``30-99``                      + penalised logistic, discrete
                                                  hazard, head over frozen scores
``ambar_2``        ``100-199``                    + restricted boosting (pilot)
``verde_condic.``  ``>= 200`` **and** the formal  + balanced ensembles as
                   sample-size calculation OK     challengers
=================  =============================  ==============================

The cut-offs are the document's *internal governance heuristics*, explicitly not
published universal thresholds; they live in :class:`GateThresholds` so a local
precision/power calculation can replace them. Only the families this repository
implements are listed in ``implemented_families``; the rest are reported as
authorised-but-not-built, never silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.evaluation.event_labels import EventLabels

__all__ = ["GateThresholds", "LEVELS", "decide_gate", "sufficiency_metrics"]

LEVELS = ("rojo", "ambar_1", "ambar_2", "verde_condicionado")

_FAMILIES = {
    "rojo": ("iforest", "vae"),
    "ambar_1": ("logit_scores_head", "ridge_logistic", "discrete_hazard"),
    "ambar_2": ("gradient_boosting_restricted",),
    "verde_condicionado": ("easy_ensemble", "balanced_forest"),
}
IMPLEMENTED_FAMILIES = ("iforest", "vae", "logit_scores_head", "ridge_logistic", "discrete_hazard")

_ACTIONS = {
    "rojo": "Mantener IF/VAE como detectores; mejorar el contrato de labels, la madurez, "
            "los episodios y la auditoría de no-alertas antes de cualquier modelo supervisado.",
    "ambar_1": "Exploración restringida: regresión logística penalizada, hazard discreto o una "
               "cabeza simple sobre los scores congelados de IF/VAE, con muy pocos grados de "
               "libertad. Sin ranking concluyente; IF/VAE siguen siendo el baseline.",
    "ambar_2": "Piloto: baseline supervisado simple y boosting fuertemente restringido, con "
               "intervalos amplios; no llamarlo validación confirmatoria.",
    "verde_condicionado": "Candidato: comparar logística/hazard, boosting con costes y ensembles "
                          "balanceados frente a IF/VAE con los mismos cortes temporales y K.",
}


@dataclass(frozen=True)
class GateThresholds:
    """Internal governance heuristics (see module docstring)."""

    ambar_1_min_episodes: int = 30
    ambar_1_min_entities: int = 10
    ambar_1_min_origins: int = 2
    ambar_2_min_episodes: int = 100
    ambar_2_min_entities: int = 20
    ambar_2_min_origins: int = 3
    ambar_2_min_oos_episodes: int = 50
    verde_min_episodes: int = 200
    verde_min_entities: int = 30
    verde_min_origins: int = 3
    verde_min_oos_episodes: int = 50
    max_single_month_share: float = 0.5
    min_fold_positive_episodes: int = 20
    origin_step_months: int = 3


# --------------------------------------------------------------------------- #
# Sufficiency metrics                                                         #
# --------------------------------------------------------------------------- #
def _icc_design_factor(entity: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """One-way ANOVA intraclass correlation of ``y`` across entities and the
    approximate design factor ``1 + (m-1) rho`` (Debray et al. 2023 as a
    diagnostic only: it does not turn rows into episodes)."""
    frame = pd.DataFrame({"e": entity, "y": y})
    groups = frame.groupby("e")["y"]
    n_i, mean_i = groups.size().to_numpy(dtype=float), groups.mean().to_numpy(dtype=float)
    k, n_tot = len(n_i), float(n_i.sum())
    if k < 2 or n_tot <= k:
        return None, None
    grand = float(frame["y"].mean())
    ssb = float((n_i * (mean_i - grand) ** 2).sum())
    ssw = float(((frame["y"] - groups.transform("mean")) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n_tot - k)
    m0 = (n_tot - float((n_i ** 2).sum()) / n_tot) / (k - 1)
    denom = msb + (m0 - 1) * msw
    if denom <= 0 or m0 <= 1:
        return None, None
    icc = float(np.clip((msb - msw) / denom, 0.0, 1.0))
    return icc, float(1.0 + (n_tot / k - 1.0) * icc)


def sufficiency_metrics(
    labels: EventLabels,
    eval_mask: Optional[np.ndarray] = None,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """The derived counts the document asks each cut to materialise.

    ``eval_mask`` marks the rows of the untouched evaluation windows (test/OOT);
    positives inside it are the "out-of-sample" episodes. Fields the data cannot
    supply (audited non-alerts, regimes) are ``None`` -- unknown, never invented.
    """
    usable = labels.usable
    pos_rows = usable & (labels.target == 1.0)
    neg_rows = usable & (labels.target == 0.0)
    episodes = labels.episodes
    mature = episodes[episodes["mature"]] if len(episodes) else episodes
    n_usable = int(usable.sum())

    icc, design = _icc_design_factor(labels.entity[usable], labels.target[usable]) if n_usable else (None, None)
    starts = mature["start_m"].to_numpy() if len(mature) else np.array([], dtype=int)
    shares = pd.Series(starts).value_counts(normalize=True) if len(starts) else pd.Series(dtype=float)
    blocks = np.unique(starts // max(1, thresholds.origin_step_months)) if len(starts) else np.array([])
    oos = 0
    if eval_mask is not None and len(mature):
        oos = int(len(np.unique(labels.episode_id[np.asarray(eval_mask, dtype=bool) & labels.episode_mature])))
    return {
        "eligible_rows": n_usable,
        "event_rate_row": (float(pos_rows.sum()) / n_usable) if n_usable else None,
        "n_positive_rows": int(pos_rows.sum()),
        "n_positive_episodes": int(len(episodes)),
        "n_mature_positive_episodes": int(len(mature)),
        "n_immature_positive_episodes": int(len(episodes) - len(mature)),
        "n_positive_entities": int(mature["entity"].nunique()) if len(mature) else 0,
        "episodes_per_positive_entity": (float(len(mature)) / mature["entity"].nunique()) if len(mature) else None,
        "n_confirmed_negative_rows": int(neg_rows.sum()),
        "n_confirmed_negative_entities": int(pd.Series(labels.entity[neg_rows]).nunique()),
        "n_unreviewed_rows": int((~labels.known).sum()),
        "n_audited_nonalerts": None,   # needs an audit stream outside the alert queue
        "months_covered": int(len(np.unique(starts))),
        "regimes_covered": None,       # needs an agreed regime definition
        "n_temporal_origins": int(max(0, len(blocks) - 1)),
        "max_single_month_share": float(shares.max()) if len(shares) else None,
        "oos_positive_episodes": oos,
        "intraclass_correlation": icc,
        "design_factor": design,
        "effective_sample_size_estimate": (n_usable / design) if design else None,
    }


# --------------------------------------------------------------------------- #
# Decision                                                                    #
# --------------------------------------------------------------------------- #
def _veto_list(metrics: dict, audit: dict, unlisted_attested: bool) -> list[dict[str, str]]:
    vetoes: list[dict[str, str]] = []

    def add(code: str, message: str) -> None:
        vetoes.append({"code": code, "message": message})

    if metrics["eligible_rows"] == 0:
        add("no_usable_labels", "Ninguna fila tiene un label utilizable y maduro.")
    elif metrics["n_positive_rows"] and not metrics["n_confirmed_negative_rows"]:
        add("no_confirmed_negatives", "Hay positivos pero ningún negativo confirmado y maduro.")
    if audit.get("invalid_targets"):
        add("invalid_target_values", f"{audit['invalid_targets']} target(s) fuera de {{0,1}} (target ambiguo).")
    if audit.get("conflicting_duplicate_rows"):
        add("label_conflicts", f"{audit['conflicting_duplicate_rows']} fila(s) duplicadas con targets contradictorios.")
    if audit.get("unlisted_policy") == "negative" and not unlisted_attested:
        add("unreviewed_converted_to_negative",
            "Las filas ausentes del archivo se tratan como negativas sin auditoría atestiguada: "
            "'unreviewed' nunca es un negativo.")
    if audit.get("positive_rows_without_episode_id"):
        add("episodes_not_collapsed",
            f"{audit['positive_rows_without_episode_id']} positivo(s) sin episode_id: episodios no colapsados de forma fiable.")
    return vetoes


def _level_by_count(m: dict, th: GateThresholds, formal_ok: bool) -> tuple[str, list[str]]:
    """Highest level whose count AND evidence requirements are all met, with the
    unmet requirements of the next level up (so the acta says what is missing)."""
    checks = [
        ("ambar_1", [
            (m["n_mature_positive_episodes"] >= th.ambar_1_min_episodes, f"episodios maduros >= {th.ambar_1_min_episodes}"),
            (m["n_positive_entities"] >= th.ambar_1_min_entities, f"entidades positivas >= {th.ambar_1_min_entities}"),
            (m["n_temporal_origins"] >= th.ambar_1_min_origins, f"orígenes temporales >= {th.ambar_1_min_origins}"),
            ((m["max_single_month_share"] or 0.0) <= th.max_single_month_share, f"ningún mes concentra > {th.max_single_month_share:.0%} de los episodios"),
        ]),
        ("ambar_2", [
            (m["n_mature_positive_episodes"] >= th.ambar_2_min_episodes, f"episodios maduros >= {th.ambar_2_min_episodes}"),
            (m["n_positive_entities"] >= th.ambar_2_min_entities, f"entidades positivas >= {th.ambar_2_min_entities}"),
            (m["n_temporal_origins"] >= th.ambar_2_min_origins, f"orígenes temporales >= {th.ambar_2_min_origins}"),
            (m["oos_positive_episodes"] >= th.ambar_2_min_oos_episodes, f"positivos OOS >= {th.ambar_2_min_oos_episodes}"),
        ]),
        ("verde_condicionado", [
            (m["n_mature_positive_episodes"] >= th.verde_min_episodes, f"episodios maduros >= {th.verde_min_episodes}"),
            (m["n_positive_entities"] >= th.verde_min_entities, f"entidades positivas >= {th.verde_min_entities}"),
            (m["n_temporal_origins"] >= th.verde_min_origins, f"orígenes temporales >= {th.verde_min_origins}"),
            (m["oos_positive_episodes"] >= th.verde_min_oos_episodes, f"positivos OOS >= {th.verde_min_oos_episodes}"),
            (formal_ok, "cálculo formal de tamaño muestral satisfecho (--labels-formal-calc-ok)"),
        ]),
    ]
    level, unmet = "rojo", []
    for name, requirements in checks:
        missing = [text for ok, text in requirements if not ok]
        if missing:
            return level, missing
        level = name
    return level, unmet


def decide_gate(
    metrics: dict[str, Any],
    audit: dict[str, Any],
    thresholds: GateThresholds = GateThresholds(),
    *,
    formal_calculation_ok: bool = False,
    unlisted_attested: bool = False,
) -> dict[str, Any]:
    """The acta of gate 4.5: level, unmet requirements, vetoes, authorised
    families and the action -- a reproducible decision, not a model."""
    count_level, unmet = _level_by_count(metrics, thresholds, formal_calculation_ok)
    vetoes = _veto_list(metrics, audit, unlisted_attested)
    level = "rojo" if vetoes else count_level
    authorised: list[str] = []
    for name in LEVELS[: LEVELS.index(level) + 1]:
        authorised.extend(_FAMILIES[name])
    notes = [
        "Las cifras 30/100/200 y los mínimos de orígenes/entidades son heurísticas internas de "
        "gobierno, no cortes publicados.",
        "Un fold con menos de "
        f"{thresholds.min_fold_positive_episodes} episodios positivos no decide qué modelo gana.",
    ]
    if metrics["n_audited_nonalerts"] is None:
        notes.append("No hay corriente de auditoría de no-alertas: la tasa poblacional de falsos "
                     "negativos no es identificable si solo se revisó el top-K de IF/VAE.")
    if level != "verde_condicionado" and count_level == "ambar_2" and not formal_calculation_ok:
        notes.append("El nivel verde exige además el cálculo formal de tamaño (Riley/pmsampsize).")
    return {
        "level": level,
        "level_by_count": count_level,
        "vetoed": bool(vetoes),
        "vetoes": vetoes,
        "unmet_for_next_level": unmet,
        "authorized_families": authorised,
        "implemented_families": [f for f in authorised if f in IMPLEMENTED_FAMILIES],
        "not_implemented_families": [f for f in authorised if f not in IMPLEMENTED_FAMILIES],
        "supervised_challengers_allowed": level != "rojo",
        "action": _ACTIONS[level],
        "notes": notes,
        "formal_calculation_ok": bool(formal_calculation_ok),
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }
