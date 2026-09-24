"""Post-training sensitivity and data-quality stress testing.

The fitted preprocessor and fitted detectors are reused exactly as they are;
no scenario refits either model.  This distinction is load-bearing for a
post-training validation: changes in score must come from lost information,
not from a model adapting to the perturbed data.

Raw-variable "removal" is represented by a training-window neutral value
(median for numeric inputs, mode for categorical/boolean inputs).  Explicit
null and zero scenarios are evaluated separately.  The model input schema is
therefore never changed -- a trained estimator cannot accept fewer columns --
while the three business questions remain distinguishable.
"""

from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ks_2samp, spearmanr

from src.data.loader import PanelSchema
from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging
from src.utils.progress import Bar, track

__all__ = ["run_post_training_sensitivity"]


@dataclass(frozen=True)
class _ModelRuntime:
    detector: Any
    baseline_scores: np.ndarray
    threshold: float


def _iforest_matrix(X, feature_names: Sequence[str]):
    """Mirror the pipeline's dtype-based IF routing without importing EDA.

    Importing ``src.preprocessing`` also imports plotting diagnostics.  This
    post-training module only needs the established ``cat__`` naming contract,
    so keeping this tiny selection local avoids making report generation
    depend on matplotlib.
    """
    categorical = np.asarray(
        [str(name).startswith("cat__") for name in feature_names], dtype=bool
    )
    if not categorical.any():
        return X
    keep = ~categorical
    return X.tocsc()[:, keep].tocsr() if sp.issparse(X) else X[:, keep]


def _classification_metrics(labels: Optional[np.ndarray], scores: np.ndarray,
                            threshold: float) -> dict[str, Optional[float]]:
    if labels is None or len(labels) != len(scores):
        return {k: None for k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(scores >= threshold, dtype=int)
    out: dict[str, Optional[float]] = {}
    try:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        has_both = np.unique(y).size == 2
        out["roc_auc"] = float(roc_auc_score(y, scores)) if has_both else None
        out["pr_auc"] = float(average_precision_score(y, scores)) if y.sum() else None
        out["precision"] = float(precision_score(y, pred, zero_division=0))
        out["recall"] = float(recall_score(y, pred, zero_division=0))
        out["f1"] = float(f1_score(y, pred, zero_division=0))
    except Exception:
        out = {k: None for k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    return out


def _top_jaccard(a: np.ndarray, b: np.ndarray, fraction: float = 0.10) -> float:
    k = max(1, int(math.ceil(fraction * len(a))))
    ia = set(np.argpartition(a, -k)[-k:].tolist())
    ib = set(np.argpartition(b, -k)[-k:].tolist())
    union = ia | ib
    return len(ia & ib) / len(union) if union else 1.0


def _compare_scores(base: np.ndarray, scenario: np.ndarray, threshold: float,
                    labels: Optional[np.ndarray]) -> dict[str, Any]:
    base = np.asarray(base, dtype=float)
    scenario = np.asarray(scenario, dtype=float)
    eps = max(float(np.nanmedian(np.abs(base))) * 1e-6, 1e-9)
    pct = 100.0 * (scenario - base) / np.maximum(np.abs(base), eps)
    delta = scenario - base
    base_alert = base >= threshold
    scenario_alert = scenario >= threshold
    if len(base) <= 1:
        rho = 1.0
    elif np.ptp(base) == 0 or np.ptp(scenario) == 0:
        rho = 1.0 if np.allclose(base, scenario) else 0.0
    else:
        rho = spearmanr(base, scenario).statistic
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    base_perf = _classification_metrics(labels, base, threshold)
    perf = _classification_metrics(labels, scenario, threshold)
    out: dict[str, Any] = {
        "n_records": int(len(base)),
        "mean_score": float(np.mean(scenario)),
        "score_q05": float(np.quantile(scenario, 0.05)),
        "score_q50": float(np.quantile(scenario, 0.50)),
        "score_q95": float(np.quantile(scenario, 0.95)),
        "mean_signed_change_pct": float(np.mean(pct)),
        "median_abs_change_pct": float(np.median(np.abs(pct))),
        "p95_abs_change_pct": float(np.quantile(np.abs(pct), 0.95)),
        "direction_up_pct": float(100.0 * np.mean(delta > eps)),
        "direction_down_pct": float(100.0 * np.mean(delta < -eps)),
        "rank_correlation": rho,
        "alert_flip_rate": float(np.mean(base_alert != scenario_alert)),
        "alert_rate": float(np.mean(scenario_alert)),
        "alert_rate_delta_pp": float(100.0 * (np.mean(scenario_alert) - np.mean(base_alert))),
        "top10_jaccard": float(_top_jaccard(base, scenario)),
    }
    for key, value in perf.items():
        out[key] = value
        b = base_perf.get(key)
        out[f"{key}_delta"] = (float(value - b) if value is not None and b is not None else None)
    return out


def _zero_or_missing_mask(series: pd.Series) -> np.ndarray:
    missing = series.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        return missing | (numeric == 0)
    text = series.astype("string").str.strip().str.lower()
    return missing | text.isin(("", "0", "0.0", "false", "null", "none")).fillna(True).to_numpy()


def _neutral_values(df: pd.DataFrame, columns: Sequence[str], train_mask: np.ndarray) -> dict:
    train = df.loc[np.asarray(train_mask, dtype=bool)]
    values = {}
    for col in columns:
        s = train[col]
        non_null = s.dropna()
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values[col] = float(pd.to_numeric(non_null, errors="coerce").median()) if len(non_null) else 0.0
        elif len(non_null):
            mode = non_null.mode(dropna=True)
            values[col] = mode.iloc[0] if len(mode) else non_null.iloc[0]
        else:
            values[col] = False if pd.api.types.is_bool_dtype(s) else "__missing__"
    return values


def _apply_loss(frame: pd.DataFrame, columns: Sequence[str], kind: str,
                neutral: dict) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if kind == "ablation":
            out[col] = neutral[col]
        elif kind == "null_replacement":
            out[col] = np.nan
        elif kind == "zero":
            if pd.api.types.is_bool_dtype(frame[col]):
                out[col] = False
            elif pd.api.types.is_numeric_dtype(frame[col]):
                out[col] = 0
        elif kind == "information_loss":
            if pd.api.types.is_bool_dtype(frame[col]):
                out[col] = False
            elif pd.api.types.is_numeric_dtype(frame[col]):
                out[col] = 0
            else:
                out[col] = np.nan
        else:
            raise ValueError(f"unknown perturbation kind {kind!r}")
    return out


def _stacked_matrix(X, X_if, stack_context: Optional[dict]):
    if not stack_context:
        return X
    stack_scores = stack_context["detector"].score_samples(X_if)
    score_col = np.asarray(stack_scores, dtype=float).reshape(-1, 1)
    if sp.issparse(X):
        augmented = sp.hstack([X.tocsr(), sp.csr_matrix(score_col)], format="csr")
    else:
        augmented = np.hstack([np.asarray(X, dtype=float), score_col])
    scaler = stack_context.get("scaler")
    return scaler.transform(augmented) if scaler is not None else augmented


def _score_frame(frame: pd.DataFrame, preprocessor, feature_names: Sequence[str],
                 runtimes: dict[str, _ModelRuntime], stack_context: Optional[dict],
    eval_local_mask: np.ndarray) -> dict[str, np.ndarray]:
    X = preprocessor.transform(frame)
    X_if = _iforest_matrix(X, feature_names)
    matrices = {"iforest": X_if, "vae": _stacked_matrix(X, X_if, stack_context)}
    return {
        name: np.asarray(runtime.detector.score_samples(matrices[name][eval_local_mask]), dtype=float)
        for name, runtime in runtimes.items()
    }


def _representative_context(df: pd.DataFrame, schema: PanelSchema, test_mask: np.ndarray,
                            max_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    test_pos = np.flatnonzero(np.asarray(test_mask, dtype=bool))
    if test_pos.size > max_rows:
        rng = np.random.default_rng(seed)
        test_pos = np.sort(rng.choice(test_pos, size=max_rows, replace=False))
    entity_col = schema.entity_col
    if entity_col and entity_col in df.columns:
        chosen = set(df.iloc[test_pos][entity_col].astype(str))
        context_pos = np.flatnonzero(df[entity_col].astype(str).isin(chosen).to_numpy())
    else:
        context_pos = test_pos
    local_by_original = {int(pos): i for i, pos in enumerate(context_pos)}
    eval_local = np.asarray([local_by_original[int(pos)] for pos in test_pos], dtype=int)
    local_mask = np.zeros(len(context_pos), dtype=bool)
    local_mask[eval_local] = True
    return df.iloc[context_pos].reset_index(drop=True), local_mask, test_pos


def _scenario_record(scenario_id: str, scenario_type: str, variables: Sequence[str],
                     missing_fraction: float, model: str, metrics: dict) -> dict:
    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "variables": " | ".join(variables),
        "n_variables": len(variables),
        "missing_fraction": float(missing_fraction),
        "model": model,
        **metrics,
    }


def _variable_ranking(scenarios: pd.DataFrame) -> pd.DataFrame:
    singles = scenarios.loc[
        scenarios["scenario_type"].isin(("ablation", "null_replacement", "zero"))
        & (scenarios["n_variables"] == 1)
    ].copy()
    if singles.empty:
        return pd.DataFrame()
    rows = []
    for (model, variable), block in singles.groupby(["model", "variables"], sort=False):
        abs_change = float(block["median_abs_change_pct"].max())
        flip = float(block["alert_flip_rate"].max())
        rank_loss = float((1.0 - block["rank_correlation"].min()) * 100.0)
        perf_loss = 0.0
        for col in ("pr_auc_delta", "roc_auc_delta", "f1_delta"):
            vals = pd.to_numeric(block.get(col), errors="coerce")
            if vals.notna().any():
                perf_loss = max(perf_loss, float(max(0.0, -vals.min()) * 100.0))
        leverage = abs_change + 100.0 * flip + rank_loss + perf_loss
        if flip >= 0.10 or block["rank_correlation"].min() < 0.90 or perf_loss >= 5.0:
            recommendation = "INDISPENSABLE — conservar"
        elif abs_change < 1.0 and flip < 0.01 and block["rank_correlation"].min() >= 0.99 and perf_loss < 1.0:
            recommendation = "MARGINAL — candidata a eliminación"
        else:
            recommendation = "CONSERVAR — impacto intermedio"
        rows.append({
            "model": model,
            "variable": variable,
            "leverage_score": leverage,
            "max_median_abs_change_pct": abs_change,
            "max_alert_flip_rate": flip,
            "min_rank_correlation": float(block["rank_correlation"].min()),
            "max_performance_loss_pp": perf_loss,
            "recommendation": recommendation,
        })
    ranking = pd.DataFrame(rows)
    ranking["rank"] = ranking.groupby("model")["leverage_score"].rank(
        method="dense", ascending=False
    ).astype(int)
    return ranking.sort_values(["model", "rank", "variable"]).reset_index(drop=True)


def _high_zero_analysis(df: pd.DataFrame, schema: PanelSchema, input_columns: Sequence[str],
                        zero_ratio: np.ndarray, test_mask: np.ndarray, train_mask: np.ndarray,
                        labels: Optional[np.ndarray], runtimes: dict[str, _ModelRuntime],
                        ranking: pd.DataFrame, cutoff: float) -> tuple[pd.DataFrame, list[dict]]:
    high = zero_ratio >= cutoff
    keys = [c for c in (schema.entity_col, schema.time_col) if c and c in df.columns]
    record_rows: list[dict] = []
    summaries: list[dict] = []
    test = np.asarray(test_mask, dtype=bool)
    train = np.asarray(train_mask, dtype=bool)
    for model, runtime in runtimes.items():
        scores = runtime.baseline_scores
        test_scores, high_test = scores[test], high[test]
        y_test = np.asarray(labels)[test] if labels is not None else None
        rest_scores = test_scores[~high_test]
        high_scores = test_scores[high_test]
        all_perf = _classification_metrics(y_test, test_scores, runtime.threshold)
        kept_perf = _classification_metrics(
            y_test[~high_test] if y_test is not None else None,
            rest_scores,
            runtime.threshold,
        )
        if len(high_scores) >= 2 and len(rest_scores) >= 2:
            ks = ks_2samp(high_scores, rest_scores)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            ks_stat, ks_p = None, None
        improvement = 0.0
        for metric in ("pr_auc", "roc_auc", "f1"):
            before, after = all_perf.get(metric), kept_perf.get(metric)
            if before is not None and after is not None:
                improvement = max(improvement, float(after - before))
        high_alert_rate = float(np.mean(high_scores >= runtime.threshold)) if len(high_scores) else None
        rest_alert_rate = float(np.mean(rest_scores >= runtime.threshold)) if len(rest_scores) else None
        structurally_different = bool(ks_p is not None and ks_p < 0.05 and ks_stat >= 0.20)
        if len(high_scores) == 0:
            recommendation = "Sin registros >=90%; mantener y monitorear"
        elif improvement >= 0.02 and structurally_different:
            recommendation = "Excluir completamente en futuras evaluaciones"
        elif structurally_different or len(high_scores) < 30:
            recommendation = "Mantener solo en prueba/monitoreo hasta reunir evidencia"
        else:
            recommendation = "Mantener en entrenamiento y prueba"
        summaries.append({
            "model": model,
            "zero_missing_cutoff": cutoff,
            "n_high_zero_all": int(high.sum()),
            "n_high_zero_train": int((high & train).sum()),
            "n_high_zero_test": int((high & test).sum()),
            "test_share_high_zero": float(np.mean(high_test)) if len(high_test) else 0.0,
            "high_zero_alert_rate": high_alert_rate,
            "rest_alert_rate": rest_alert_rate,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "performance_improvement_after_exclusion": improvement,
            "recommendation": recommendation,
        })

        critical = ranking.loc[ranking["model"].eq(model)].sort_values("rank").head(5)
        critical_vars = critical["variable"].tolist()
        test_positions = np.flatnonzero(test & high)
        reference = np.sort(test_scores)
        test_pos_to_local = {int(pos): i for i, pos in enumerate(np.flatnonzero(test))}
        for pos in test_positions:
            local = test_pos_to_local[int(pos)]
            score = float(scores[pos])
            cdf = float(np.searchsorted(reference, score, side="right") / max(len(reference), 1))
            row = {k: df.iloc[pos][k] for k in keys}
            row.update({
                "model": model,
                "row_position": int(pos),
                "zero_missing_ratio": float(zero_ratio[pos]),
                "score": score,
                "score_cdf": cdf,
                "alert": int(score >= runtime.threshold),
                "post_label": (int(np.asarray(labels)[pos]) if labels is not None else None),
                "critical_variables": " | ".join(critical_vars),
                "critical_values": json.dumps(
                    {v: (None if pd.isna(df.iloc[pos][v]) else str(df.iloc[pos][v]))
                     for v in critical_vars if v in df.columns},
                    ensure_ascii=False,
                ),
            })
            record_rows.append(row)
    return pd.DataFrame(record_rows), summaries


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _write_html_report(out_path: str, scenarios: pd.DataFrame, ranking: pd.DataFrame,
                       high_zero: pd.DataFrame, summary: dict,
                       score_samples: dict[str, dict[str, list[float]]]) -> None:
    figures = []
    try:
        import plotly.graph_objects as go

        for model, payload in score_samples.items():
            fig = go.Figure()
            for name, values in payload.items():
                fig.add_trace(go.Histogram(
                    x=values, name=name, opacity=0.55, histnorm="probability density"
                ))
            fig.update_layout(
                barmode="overlay", title=f"Distribución de scores por escenario — {model}",
                xaxis_title="Score", yaxis_title="Densidad", template="plotly_white",
            )
            figures.append(fig)

        for model, block in scenarios.loc[scenarios["scenario_type"].eq("information_loss")].groupby("model"):
            fig = go.Figure()
            grouped = block.groupby("missing_fraction").agg(
                median=("rank_correlation", "median"),
                q10=("rank_correlation", lambda s: s.quantile(.10)),
                q90=("rank_correlation", lambda s: s.quantile(.90)),
                flip=("alert_flip_rate", "median"),
            ).reset_index()
            x = 100 * grouped["missing_fraction"]
            fig.add_trace(go.Scatter(x=x, y=grouped["q90"], line={"width": 0}, showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=grouped["q10"], fill="tonexty", name="P10–P90",
                                     line={"width": 0}, fillcolor="rgba(42,120,214,.20)"))
            fig.add_trace(go.Scatter(x=x, y=grouped["median"], name="Correlación mediana"))
            fig.add_trace(go.Scatter(x=x, y=grouped["flip"], name="Tasa de cambio de clase",
                                     yaxis="y2"))
            perf = block.groupby("missing_fraction")["pr_auc"].median()
            if perf.notna().any():
                fig.add_trace(go.Scatter(
                    x=100 * perf.index.to_numpy(), y=perf.to_numpy(),
                    name="PR-AUC post-label", line={"dash": "dot"},
                ))
            fig.update_layout(
                title=f"Fan chart: estabilidad ante pérdida de información — {model}",
                xaxis_title="Variables ausentes o en cero (%)", yaxis_title="Correlación de rangos",
                yaxis2={"title": "Flip rate", "overlaying": "y", "side": "right"},
                template="plotly_white",
            )
            figures.append(fig)

        for model, block in ranking.groupby("model"):
            top = block.sort_values("leverage_score").tail(20)
            fig = go.Figure(go.Bar(x=top["leverage_score"], y=top["variable"], orientation="h"))
            fig.update_layout(title=f"Leverage por variable — {model}", xaxis_title="Impacto compuesto",
                              template="plotly_white", height=max(420, 24 * len(top) + 140))
            figures.append(fig)
    except Exception:
        figures = []

    figure_html = []
    for i, fig in enumerate(figures):
        figure_html.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))

    required = summary.get("required_variables", {})
    conclusions = "".join(
        f"<li><b>{html.escape(model)}</b>: {int(block.get('required_count', 0))} de "
        f"{int(block.get('total_count', 0))} variables necesarias bajo el criterio "
        "ρ≥0.98, flip≤5% y pérdida de performance≤2 pp.</li>"
        for model, block in required.items()
    )
    hz = "".join(
        f"<li><b>{html.escape(str(row['model']))}</b>: {html.escape(str(row['recommendation']))} "
        f"({int(row['n_high_zero_test'])} registros de prueba con ≥90% cero/faltante).</li>"
        for row in summary.get("high_zero_summary", [])
    )
    css = """
    body{font-family:Arial,sans-serif;margin:0;background:#eef1f5;color:#172033}
    main{max-width:1500px;margin:auto;padding:28px}.card{background:white;border-radius:12px;padding:20px;margin:16px 0}
    table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px;border-bottom:1px solid #dde3eb;text-align:left}
    th{position:sticky;top:0;background:#f7f9fb}.scroll{max-height:520px;overflow:auto}h1,h2{margin-top:0}
    """
    body = f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Prueba de sensibilidad post-entrenamiento</title><style>{css}</style></head><body><main>
    <h1>Prueba de sensibilidad post-entrenamiento</h1>
    <div class='card'><h2>Conclusiones ejecutivas</h2><ul>{conclusions}{hz}</ul>
    <p>“Eliminar” neutraliza la variable con una referencia aprendida exclusivamente en entrenamiento; null y cero se prueban por separado. Ningún escenario reajusta los modelos.</p></div>
    {''.join(f"<div class='card'>{block}</div>" for block in figure_html)}
    <div class='card'><h2>Ranking de variables</h2><div class='scroll'>{ranking.to_html(index=False, escape=True)}</div></div>
    <div class='card'><h2>Escenarios y métricas versus base</h2><div class='scroll'>{scenarios.to_html(index=False, escape=True)}</div></div>
    <div class='card'><h2>Registros con ≥90% de información cero/faltante</h2><div class='scroll'>{high_zero.head(500).to_html(index=False, escape=True)}</div></div>
    </main></body></html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(body)


def run_post_training_sensitivity(
    df: pd.DataFrame,
    schema: PanelSchema,
    preprocessor,
    feature_names: Sequence[str],
    models: dict[str, Any],
    baseline_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    stack_context: Optional[dict] = None,
    max_test_rows: int = 5000,
    combination_top_k: int = 6,
    random_subsets_per_level: int = 5,
    missing_levels: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    high_zero_cutoff: float = 0.90,
    random_state: int = 42,
    out_dir: Optional[str] = None,
) -> dict:
    """Evaluate fitted detectors under variable-loss and zero/missing scenarios.

    Labels, when supplied, are used only here after training to measure whether
    score instability translates into degraded post-hoc classification.  They
    never flow into preprocessing, fitting, scenario generation or scoring.
    """
    log = setup_logging()
    resolved_dir = out_dir or paths.REPORTS_DIR
    os.makedirs(resolved_dir, exist_ok=True)
    key_cols = {c for c in (schema.entity_col, schema.time_col, schema.target_col) if c}
    input_columns = [c for c in df.columns if c not in key_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if not input_columns:
        raise ValueError("sensitivity analysis requires at least one model input column")
    runtimes = {
        name: _ModelRuntime(
            detector=detector,
            baseline_scores=np.asarray(baseline_scores[name], dtype=float),
            threshold=float(thresholds[name]),
        )
        for name, detector in models.items()
        if name in baseline_scores and name in thresholds and np.isfinite(thresholds[name])
    }
    if not runtimes:
        raise ValueError("no model has detector, baseline scores and finite threshold")

    context_df, eval_local_mask, eval_original_pos = _representative_context(
        df, schema, test_mask, max_test_rows, random_state
    )
    y_eval = np.asarray(labels)[eval_original_pos] if labels is not None else None
    neutral = _neutral_values(df, input_columns, train_mask)
    baseline_eval = {
        name: runtime.baseline_scores[eval_original_pos] for name, runtime in runtimes.items()
    }
    scenario_rows: list[dict] = []
    score_samples: dict[str, dict[str, list[float]]] = {
        name: {"base": base[:2000].tolist()} for name, base in baseline_eval.items()
    }

    def evaluate(scenario_id: str, scenario_type: str, variables: Sequence[str],
                 fraction: float = 0.0, capture: bool = False) -> None:
        perturbation = "ablation" if scenario_type == "pruning_path" else scenario_type
        perturbed = _apply_loss(context_df, variables, perturbation, neutral)
        scored = _score_frame(
            perturbed, preprocessor, feature_names, runtimes, stack_context, eval_local_mask
        )
        for name, values in scored.items():
            metrics = _compare_scores(
                baseline_eval[name], values, runtimes[name].threshold, y_eval
            )
            scenario_rows.append(_scenario_record(
                scenario_id, scenario_type, variables, fraction, name, metrics
            ))
            if capture:
                score_samples[name][scenario_id] = values[:2000].tolist()

    # Individual stress: neutral ablation, explicit null, and zero where the
    # raw dtype makes zero a meaningful representable value.
    # Each block below is its own bar: the whole study is hundreds to thousands
    # of scorings of the fitted models, and a phase-level timer cannot say
    # which block it is in or how far along it is.
    for col in track(input_columns, desc="sensitivity[single-variable]",
                     unit="variable", label=str):
        evaluate(f"ablate::{col}", "ablation", [col], 1 / len(input_columns))
        evaluate(f"null::{col}", "null_replacement", [col], 1 / len(input_columns))
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            evaluate(f"zero::{col}", "zero", [col], 1 / len(input_columns))

    initial = pd.DataFrame(scenario_rows)
    initial_ranking = _variable_ranking(initial)
    leverage = initial_ranking.groupby("variable")["leverage_score"].max().sort_values(ascending=False)
    top = leverage.head(max(2, min(combination_top_k, len(leverage)))).index.tolist()
    pairs = [(left, right) for i, left in enumerate(top) for right in top[i + 1:]]
    for left, right in track(pairs, desc="sensitivity[variable-pairs]", unit="pair",
                             label=lambda pair: " + ".join(pair)):
        evaluate(f"pair::{left}+{right}", "ablation", [left, right], 2 / len(input_columns))

    # Random information-loss levels provide a distribution, not one lucky or
    # unlucky subset, which is what the fan chart needs.
    rng = np.random.default_rng(random_state)
    reps_per_level = max(1, int(random_subsets_per_level))
    loss_bar = Bar(desc="sensitivity[information-loss]", unit="scenario",
                   total=len(missing_levels) * reps_per_level)
    try:
        for level in missing_levels:
            k = max(1, min(len(input_columns), int(round(float(level) * len(input_columns)))))
            for rep in range(reps_per_level):
                loss_bar.set_postfix_str(f"{int(100 * level)}% de variables, réplica {rep + 1}")
                cols = sorted(rng.choice(input_columns, size=k, replace=False).tolist())
                evaluate(f"loss::{int(100*level):02d}::{rep+1}", "information_loss", cols,
                         k / len(input_columns), capture=(rep == 0))
                loss_bar.update(1)
    finally:
        loss_bar.close()

    # Cumulative removal from least to most influential estimates the smallest
    # stable scope under explicit operational thresholds.
    least_first = leverage.sort_values(ascending=True).index.tolist()
    for k in track(range(1, len(least_first) + 1), desc="sensitivity[pruning-path]",
                   unit="step", label=lambda step: f"{step} variables retiradas"):
        cols = least_first[:k]
        evaluate(f"prune::{k:03d}", "pruning_path", cols, k / len(input_columns))

    scenarios = pd.DataFrame(scenario_rows)
    ranking = _variable_ranking(scenarios)
    sensitivity_matrix = (
        scenarios.loc[
            scenarios["scenario_type"].isin(("ablation", "null_replacement", "zero"))
            & scenarios["n_variables"].eq(1)
        ]
        .pivot_table(
            index=["model", "variables"], columns="scenario_type",
            values=["median_abs_change_pct", "alert_flip_rate", "rank_correlation"],
            aggfunc="max",
        )
    )
    sensitivity_matrix.columns = [f"{metric}__{scenario}" for metric, scenario in sensitivity_matrix.columns]
    sensitivity_matrix = sensitivity_matrix.reset_index().rename(columns={"variables": "variable"})
    required: dict[str, dict] = {}
    for model, block in scenarios.loc[scenarios["scenario_type"].eq("pruning_path")].groupby("model"):
        stable = block.loc[
            (block["rank_correlation"] >= 0.98)
            & (block["alert_flip_rate"] <= 0.05)
            & (pd.to_numeric(block["pr_auc_delta"], errors="coerce").fillna(0) >= -0.02)
            & (pd.to_numeric(block["f1_delta"], errors="coerce").fillna(0) >= -0.02)
        ]
        removable = int(stable["n_variables"].max()) if len(stable) else 0
        required[model] = {
            "total_count": len(input_columns),
            "max_stably_removable": removable,
            "required_count": len(input_columns) - removable,
        }

    zero_matrix = np.column_stack([_zero_or_missing_mask(df[c]) for c in input_columns])
    zero_ratio = zero_matrix.mean(axis=1)
    with log_phase("sensitivity.high_zero_analysis"):
        high_zero_records, high_zero_summary = _high_zero_analysis(
            df, schema, input_columns, zero_ratio, test_mask, train_mask, labels,
            runtimes, ranking, high_zero_cutoff,
        )
    summary = {
        "method": "post_training_no_refit",
        "evaluated_input_variables": input_columns,
        "n_test_records_sampled": int(len(eval_original_pos)),
        "post_training_labels_available": bool(labels is not None and np.unique(labels).size > 1),
        "stability_rule": {"rank_correlation_min": 0.98, "alert_flip_rate_max": 0.05,
                           "performance_loss_max": 0.02},
        "required_variables": required,
        "high_zero_summary": high_zero_summary,
    }

    def _in_output(default_path: str) -> str:
        return os.path.join(resolved_dir, os.path.basename(default_path))

    scenario_csv = _in_output(paths.SENSITIVITY_SCENARIOS_CSV)
    ranking_csv = _in_output(paths.SENSITIVITY_VARIABLES_CSV)
    matrix_csv = _in_output(paths.SENSITIVITY_MATRIX_CSV)
    high_zero_csv = _in_output(paths.SENSITIVITY_HIGH_ZERO_CSV)
    workbook = _in_output(paths.SENSITIVITY_WORKBOOK_DEFAULT)
    summary_json = _in_output(paths.SENSITIVITY_SUMMARY_JSON)
    html_path = _in_output(paths.SENSITIVITY_HTML_DEFAULT)
    with log_phase("sensitivity.write_tables"):
        scenarios.to_csv(scenario_csv, index=False)
        ranking.to_csv(ranking_csv, index=False)
        sensitivity_matrix.to_csv(matrix_csv, index=False)
        high_zero_records.to_csv(high_zero_csv, index=False)
    with log_phase("sensitivity.write_workbook"), pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="variables", index=False)
        sensitivity_matrix.to_excel(writer, sheet_name="matriz_sensibilidad", index=False)
        scenarios.to_excel(writer, sheet_name="escenarios", index=False)
        high_zero_records.to_excel(writer, sheet_name="registros_90pct_cero", index=False)
        pd.DataFrame(high_zero_summary).to_excel(writer, sheet_name="recomendacion_registros", index=False)
        pd.DataFrame([
            {"model": model, **values} for model, values in required.items()
        ]).to_excel(writer, sheet_name="alcance_variables", index=False)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, ensure_ascii=False, indent=2)
    with log_phase("sensitivity.write_html_report"):
        _write_html_report(html_path, scenarios, ranking, high_zero_records, summary, score_samples)

    artifacts = {
        "html": os.path.abspath(html_path),
        "workbook": os.path.abspath(workbook),
        "scenarios_csv": os.path.abspath(scenario_csv),
        "variables_csv": os.path.abspath(ranking_csv),
        "matrix_csv": os.path.abspath(matrix_csv),
        "high_zero_csv": os.path.abspath(high_zero_csv),
        "summary_json": os.path.abspath(summary_json),
    }
    log.info(
        "Post-training sensitivity: %d scenarios x %d model(s), %d input variables, "
        "%d records sampled -> %s",
        scenarios["scenario_id"].nunique(), len(runtimes), len(input_columns),
        len(eval_original_pos), html_path,
    )
    return {**summary, "artifacts": artifacts}
