"""Phase 8c: reviewed event labels -> evaluation of IF/VAE, gate 4.5, challengers.

The single entry point ``main.py`` calls after the detectors are fitted and
scored. It never trains IF/VAE and never lets labels influence a score:

1. **Load** the external labels file (``entity_id``, ``codmes``, ``target`` plus
   optional status/episode/maturity columns) and align it to the panel.
2. **Gate 4.5**: count mature independent positive episodes, entities, temporal
   origins and out-of-sample positives; decide the level and the authorised
   families (:mod:`src.evaluation.label_gate`).
3. **Evaluate** the frozen IF and VAE scores against those labels on the test and
   OOT windows, for two targets: the *current-month* label and *a new episode
   starts within H months*. Results are descriptive unless a window has enough
   positive episodes (:mod:`src.evaluation.event_evaluation`).
4. **Challengers** (:mod:`src.models.event_challenger`) only when the gate
   authorises them (or ``force``), each family with its own fallback.

Fallbacks, so a missing or unusable file never stops the run: no file found ->
``no_labels_file``; unreadable/contract-violating file -> ``contract_error``;
red gate -> IF/VAE evaluation only. In every case the pipeline continues
unsupervised exactly as before.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.event_evaluation import evaluate_ranking
from src.evaluation.event_labels import (
    EventLabels,
    EventLabelsContractError,
    EventLabelsEmptyError,
    EventLabelsError,
    LabelSpec,
    find_labels_file,
    load_event_labels,
)
from src.evaluation.label_gate import GateThresholds, decide_gate, sufficiency_metrics
from src.models.event_challenger import ChallengerConfig, run_event_challengers
from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = ["EventSupervisionConfig", "run_event_supervision"]

CHALLENGER_MODES = ("off", "auto", "force")


@dataclass(frozen=True)
class EventSupervisionConfig:
    labels_path: Optional[str] = None
    labels_dir: Optional[str] = paths.REVIEWED_LABELS_DIR
    spec: LabelSpec = field(default_factory=LabelSpec)
    thresholds: GateThresholds = field(default_factory=GateThresholds)
    challenger: ChallengerConfig = field(default_factory=ChallengerConfig)
    challenger_mode: str = "auto"
    formal_calculation_ok: bool = False
    unlisted_attested: bool = False
    out_dir: str = paths.REPORTS_DIR


def _windows(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"test": masks["test"], "oot": masks["oot"]}


def _evaluate_baselines(labels: EventLabels, scores: dict[str, np.ndarray],
                        masks: dict[str, np.ndarray], cfg: EventSupervisionConfig) -> dict[str, Any]:
    """IF/VAE scores against the current-month target and the onset-in-H target."""
    ch = cfg.challenger
    y_haz, elig_haz, epi_haz = labels.horizon_target(ch.hazard_horizon_months)
    targets = {
        "current_month": (np.nan_to_num(labels.target), labels.usable, labels.episode_id),
        f"onset_within_{ch.hazard_horizon_months}m": (y_haz, elig_haz, epi_haz),
    }
    out: dict[str, Any] = {}
    for target_name, (y, eligible, episode) in targets.items():
        out[target_name] = {}
        for model, s in scores.items():
            out[target_name][model] = {
                window: evaluate_ranking(
                    np.asarray(s, dtype=float), y, eligible & mask, labels.entity, episode,
                    k=ch.review_capacity_k, n_boot=ch.n_boot, seed=ch.seed,
                    min_positive_episodes=ch.min_positive_episodes_conclusive,
                ) for window, mask in _windows(masks).items()
            }
    return out


def _flatten(evaluation: dict, challengers: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(target, model, window, res, kind):
        ci = res.get("ci") or {}
        cal = res.get("calibration") or {}
        rows.append({
            "kind": kind, "target": target, "model": model, "window": window,
            **{k: res.get(k) for k in (
                "status", "eligible_rows", "n_positive_rows", "n_positive_episodes",
                "n_positive_entities", "event_rate_row", "k", "ap", "ap_baseline", "roc_auc",
                "precision_at_k", "recall_at_k", "episode_recall_at_k", "lift_at_k",
                "false_positives_at_k", "fp_per_1000_eligible", "conclusive")},
            "ap_ci_low": (ci.get("ap") or [None, None])[0], "ap_ci_high": (ci.get("ap") or [None, None])[1],
            "precision_at_k_ci_low": (ci.get("precision_at_k") or [None, None])[0],
            "precision_at_k_ci_high": (ci.get("precision_at_k") or [None, None])[1],
            **{f"cal_{k}": v for k, v in cal.items()}, "note": res.get("note"),
        })

    for target, by_model in evaluation.items():
        for model, by_window in by_model.items():
            for window, res in by_window.items():
                add(target, model, window, res, "detector")
    for family, result in (challengers.get("families") or {}).items():
        if result.get("status") != "executed":
            continue
        target = (f"onset_within_{challengers.get('hazard_horizon_months')}m"
                  if family == "discrete_hazard" else "current_month")
        for window, res in result["windows"].items():
            add(target, family, window, res, "challenger")
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_event_supervision(
    *,
    cfg: EventSupervisionConfig,
    schema: PanelSchema,
    keys: pd.DataFrame,
    masks: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    X: Any,
    feature_names: list[str],
) -> dict[str, Any]:
    """Run the whole phase (see module docstring) and write its artifacts.

    Args:
        masks: Boolean row masks ``train``, ``val``, ``test`` and ``oot``.
        scores: ``{"if_score": ..., "vae_score": ...}``, higher = more anomalous.
        X, feature_names: The matrix the ridge/hazard challengers may use.
    """
    log = setup_logging()
    if cfg.challenger_mode not in CHALLENGER_MODES:
        raise ValueError(f"challenger_mode must be one of {CHALLENGER_MODES}")

    with log_phase("event_supervision.load_labels"):
        path = find_labels_file(cfg.labels_path, cfg.labels_dir)
        if path is None:
            where = cfg.labels_path or cfg.labels_dir
            reason = (f"No se encontró el archivo de labels ({where!r}); la corrida sigue sin "
                      "supervisión, solo con IF/VAE.")
            log.info(reason)
            return {"status": "no_labels_file", "reason": reason, "searched": [cfg.labels_path, cfg.labels_dir]}
        try:
            labels = load_event_labels(path, schema, keys, cfg.spec)
        except EventLabelsEmptyError as exc:
            reason = (f"El archivo de labels {path!r} no tiene contenido utilizable ({exc}); se "
                      "ignora y la corrida sigue solo con IF/VAE.")
            log.warning(reason)
            return {"status": "empty_labels_file", "reason": reason, "source_path": path}
        except EventLabelsContractError as exc:
            reason = (f"El archivo de labels {path!r} no cumple el contrato mínimo: {exc}; se "
                      "ignora y la corrida sigue solo con IF/VAE.")
            log.warning(reason)
            return {"status": "contract_error", "reason": reason, "source_path": path}
        except EventLabelsError as exc:  # any future subclass: ignore, never crash
            reason = f"Labels no utilizables ({exc}); se ignoran y la corrida sigue solo con IF/VAE."
            log.warning(reason)
            return {"status": "contract_error", "reason": reason, "source_path": path}

    with log_phase("event_supervision.gate_4_5"):
        eval_rows = masks["test"] | masks["oot"]
        metrics = sufficiency_metrics(labels, eval_rows, cfg.thresholds)
        gate = decide_gate(metrics, labels.audit, cfg.thresholds,
                           formal_calculation_ok=cfg.formal_calculation_ok,
                           unlisted_attested=cfg.unlisted_attested)
        log.info("Compuerta 4.5: nivel %s (por conteo: %s) con %d episodio(s) positivo(s) maduro(s), "
                 "%d entidad(es); vetos: %s.", gate["level"], gate["level_by_count"],
                 metrics["n_mature_positive_episodes"], metrics["n_positive_entities"],
                 [v["code"] for v in gate["vetoes"]] or "ninguno")

    with log_phase("event_supervision.evaluate_detectors"):
        evaluation = _evaluate_baselines(labels, scores, masks, cfg)

    challengers: dict[str, Any] = {"status": "skipped", "reason": "Modo de challengers desactivado.", "families": {}}
    if cfg.challenger_mode != "off":
        with log_phase("event_supervision.challengers"):
            challengers = run_event_challengers(
                X=X, feature_names=feature_names, labels=labels, masks=masks,
                frozen_scores=scores, gate=gate, cfg=cfg.challenger,
                force=(cfg.challenger_mode == "force"),
            )

    with log_phase("event_supervision.write_artifacts"):
        os.makedirs(cfg.out_dir, exist_ok=True)
        gate_path = os.path.join(cfg.out_dir, os.path.basename(paths.LABEL_GATE_JSON))
        eval_path = os.path.join(cfg.out_dir, os.path.basename(paths.EVENT_EVALUATION_CSV))
        chal_path = os.path.join(cfg.out_dir, os.path.basename(paths.EVENT_CHALLENGERS_JSON))
        acta = {"source_path": labels.source_path, "audit": labels.audit, "gate": gate,
                "spec": {k: v for k, v in labels.spec.__dict__.items()}}
        with open(gate_path, "w", encoding="utf-8") as fh:
            json.dump(acta, fh, ensure_ascii=False, indent=2, default=_json_default)
        _flatten(evaluation, challengers).to_csv(eval_path, index=False)
        with open(chal_path, "w", encoding="utf-8") as fh:
            json.dump(challengers, fh, ensure_ascii=False, indent=2, default=_json_default)
    return {
        "status": "executed", "source_path": labels.source_path, "audit": labels.audit, "gate": gate,
        "evaluation": evaluation, "challengers": challengers,
        "artifacts": {"label_gate": os.path.abspath(gate_path),
                      "event_evaluation": os.path.abspath(eval_path),
                      "event_challengers": os.path.abspath(chal_path)},
    }
