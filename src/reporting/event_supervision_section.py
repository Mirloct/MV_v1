"""Report chapter for Phase 8c: reviewed event labels, gate 4.5, challengers.

Renders what ``src.evaluation.event_supervision.run_event_supervision`` returned,
in Markdown and HTML, without computing anything itself. When the phase did not
use labels (no file, empty file, contract error) the chapter still appears as a
short notice, so a reader can tell "no labels were used" from "the phase never
ran". When it is absent from the context (phase disabled) nothing is rendered.
"""

from __future__ import annotations

import html
import os
from typing import Any, Optional, Sequence

__all__ = ["event_supervision_html", "event_supervision_markdown"]

_LEVEL_LABELS = {
    "rojo": "Rojo — sin base para supervisión",
    "ambar_1": "Ámbar 1 — exploración restringida",
    "ambar_2": "Ámbar 2 — piloto",
    "verde_condicionado": "Verde condicionado — candidato",
}
_METRIC_LABELS = (
    ("eligible_rows", "Filas elegibles (conocidas y maduras)"),
    ("event_rate_row", "Tasa fila-mes (prevalencia cruda)"),
    ("n_positive_rows", "Filas positivas"),
    ("n_positive_episodes", "Episodios positivos"),
    ("n_mature_positive_episodes", "Episodios positivos maduros"),
    ("n_immature_positive_episodes", "Episodios positivos inmaduros (excluidos)"),
    ("n_positive_entities", "Entidades positivas distintas"),
    ("n_confirmed_negative_rows", "Filas negativas confirmadas y maduras"),
    ("n_unreviewed_rows", "Filas sin revisar (nunca se cuentan como negativas)"),
    ("n_audited_nonalerts", "No-alertas auditadas"),
    ("months_covered", "Meses con episodios"),
    ("n_temporal_origins", "Orígenes temporales"),
    ("max_single_month_share", "Mayor concentración en un solo mes"),
    ("oos_positive_episodes", "Episodios positivos en evaluación intacta (test/OOT)"),
    ("effective_sample_size_estimate", "Tamaño efectivo estimado (bajo clustering)"),
    ("regimes_covered", "Regímenes cubiertos"),
)
_EVAL_HEADERS = ("Objetivo", "Modelo", "Ventana", "Filas", "Episodios +", "AP (IC 95%)",
                 "Precisión@K", "Recall@K", "Recall de episodios@K", "Lift@K", "FP@K", "Concluyente")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(res: dict) -> str:
    ci = (res.get("ci") or {}).get("ap")
    ap = res.get("ap")
    if ap is None:
        return "n/d"
    return f"{ap:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else f"{ap:.3f}"


def _eval_row(target: str, model: str, window: str, res: dict) -> tuple[str, ...]:
    if res.get("status") != "ok":
        note = {"no_positives": "sin positivos maduros", "no_eligible_rows": "sin filas elegibles"}.get(
            res.get("status"), str(res.get("status")))
        return (target, model, window, _fmt(res.get("eligible_rows")), _fmt(res.get("n_positive_episodes")),
                note, "", "", "", "", "", "no")
    return (target, model, window, _fmt(res["eligible_rows"]), _fmt(res["n_positive_episodes"]), _ci(res),
            _fmt(res["precision_at_k"]), _fmt(res["recall_at_k"]), _fmt(res.get("episode_recall_at_k")),
            _fmt(res["lift_at_k"], 2), _fmt(res["false_positives_at_k"]), _fmt(res.get("conclusive")))


def _evaluation_rows(payload: dict) -> list[tuple[str, ...]]:
    rows = []
    for target, by_model in (payload.get("evaluation") or {}).items():
        for model, by_window in by_model.items():
            for window, res in by_window.items():
                rows.append(_eval_row(target, model, window, res))
    chal = payload.get("challengers") or {}
    for family, result in (chal.get("families") or {}).items():
        if result.get("status") != "executed":
            continue
        target = f"onset_within_{chal.get('hazard_horizon_months')}m" if family == "discrete_hazard" else "current_month"
        for window, res in result["windows"].items():
            rows.append(_eval_row(target, family, window, res))
    return rows


def _challenger_rows(payload: dict) -> list[tuple[str, ...]]:
    rows = []
    for family, result in ((payload.get("challengers") or {}).get("families") or {}).items():
        if result.get("status") == "executed":
            rows.append((family, "ejecutado", _fmt(result.get("C"), 3), _fmt(result.get("n_features")),
                         _fmt(result.get("fit_positive_episodes")), _fmt(result.get("events_per_parameter"), 1),
                         " / ".join(result.get("warnings") or []) or "—"))
        else:
            rows.append((family, str(result.get("status")), "", "", _fmt(result.get("fit_positive_episodes")),
                         "", str(result.get("reason") or "")))
    return rows


def _metric_rows(gate: dict) -> list[tuple[str, str]]:
    metrics = gate.get("metrics") or {}
    rows = []
    for key, label in _METRIC_LABELS:
        value = metrics.get(key)
        if key in ("event_rate_row", "max_single_month_share") and value is not None:
            rows.append((label, f"{100 * value:.2f}%"))
        else:
            rows.append((label, _fmt(value, 1)))
    return rows


def _notice(payload: dict) -> str:
    return {
        "no_labels_file": "No se encontró el archivo de labels.",
        "empty_labels_file": "El archivo de labels está vacío o sin valores utilizables.",
        "contract_error": "El archivo de labels no cumple el contrato mínimo o no se pudo leer.",
    }.get(payload.get("status"), "El archivo de labels no se utilizó.") + " " + str(payload.get("reason") or "")


def _md_table(headers: Sequence[str], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "_ninguno_\n"
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def event_supervision_markdown(context: dict, out_dir: str) -> str:
    payload: Optional[dict] = context.get("event_supervision")
    if not isinstance(payload, dict):
        return ""
    parts = ["## Labels de eventos y compuerta de supervisión (4.5)\n"]
    if payload.get("status") != "executed":
        parts.append(f"{_notice(payload)} La corrida continuó solo con IF/VAE.\n")
        return "\n".join(parts)
    gate = payload["gate"]
    parts.append(
        f"**Nivel de la compuerta: {_LEVEL_LABELS[gate['level']]}** (por conteo: "
        f"{_LEVEL_LABELS[gate['level_by_count']]}). {gate['action']}\n")
    for veto in gate.get("vetoes") or []:
        parts.append(f"- **Veto `{veto['code']}`**: {veto['message']}\n")
    if gate.get("unmet_for_next_level"):
        parts.append("Falta para el siguiente nivel: " + "; ".join(gate["unmet_for_next_level"]) + ".\n")
    parts.append("Familias autorizadas: " + ", ".join(gate.get("authorized_families") or []) +
                 (f" (sin implementar en este repositorio: {', '.join(gate['not_implemented_families'])})"
                  if gate.get("not_implemented_families") else "") + ".\n")
    parts.append(_md_table(["Métrica de suficiencia", "Valor"], _metric_rows(gate)))
    parts.append("### Desempeño frente a los labels revisados\n")
    parts.append(_md_table(_EVAL_HEADERS, _evaluation_rows(payload)))
    parts.append("Una ventana con menos de 20 episodios positivos es solo descriptiva y no decide qué "
                 "modelo gana. Las filas sin revisar y las inmaduras no se cuentan como negativas.\n")
    chal = payload.get("challengers") or {}
    parts.append("### Challengers supervisados\n")
    if chal.get("status") != "executed":
        parts.append(f"No se ejecutaron: {chal.get('reason')}\n")
    else:
        parts.append(_md_table(["Familia", "Estado", "C", "Variables", "Episodios de ajuste",
                                "Episodios por parámetro", "Avisos / motivo"], _challenger_rows(payload)))
        if chal.get("forced"):
            parts.append("**Ejecución forzada** a pesar de la compuerta: resultado exploratorio.\n")
        parts.append(f"{chal.get('promotion', '')}\n")
    for label, path in (payload.get("artifacts") or {}).items():
        rel = os.path.relpath(path, out_dir).replace(os.sep, "/")
        parts.append(f"- {label}: [`{os.path.basename(path)}`]({rel})\n")
    return "\n".join(parts)


def _html_table(headers: Sequence[str], rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return "<p class='lead'>ninguno</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def event_supervision_html(context: dict, out_dir: str) -> str:
    payload: Optional[dict] = context.get("event_supervision")
    if not isinstance(payload, dict):
        return ""
    title = "<h2 id='event-supervision'>Labels de eventos y compuerta de supervisión (4.5)</h2>"
    if payload.get("status") != "executed":
        return (f"{title}<div class='card'><p class='lead'>{html.escape(_notice(payload))} "
                "La corrida continuó solo con IF/VAE.</p></div>")
    gate = payload["gate"]
    chal = payload.get("challengers") or {}
    vetoes = "".join(f"<li><b>Veto <code>{html.escape(v['code'])}</code></b>: {html.escape(v['message'])}</li>"
                     for v in gate.get("vetoes") or [])
    unmet = ("<p>Falta para el siguiente nivel: " + html.escape("; ".join(gate["unmet_for_next_level"])) + ".</p>"
             if gate.get("unmet_for_next_level") else "")
    families = ", ".join(gate.get("authorized_families") or [])
    pending = (f" (sin implementar en este repositorio: {', '.join(gate['not_implemented_families'])})"
               if gate.get("not_implemented_families") else "")
    if chal.get("status") != "executed":
        challengers = f"<p>No se ejecutaron: {html.escape(str(chal.get('reason')))}</p>"
    else:
        challengers = (
            _html_table(["Familia", "Estado", "C", "Variables", "Episodios de ajuste",
                         "Episodios por parámetro", "Avisos / motivo"], _challenger_rows(payload))
            + ("<p><b>Ejecución forzada</b> a pesar de la compuerta: resultado exploratorio.</p>" if chal.get("forced") else "")
            + f"<p>{html.escape(str(chal.get('promotion', '')))}</p>")
    links = "".join(
        f"<li><a href='{html.escape(os.path.relpath(p, out_dir).replace(os.sep, '/'))}'>{html.escape(str(k))}</a>"
        f"<span class='fname'>{html.escape(os.path.basename(str(p)))}</span></li>"
        for k, p in (payload.get("artifacts") or {}).items())
    return (
        f"{title}<div class='card'>"
        f"<p class='lead'><b>Nivel: {html.escape(_LEVEL_LABELS[gate['level']])}</b> (por conteo: "
        f"{html.escape(_LEVEL_LABELS[gate['level_by_count']])}). {html.escape(gate['action'])}</p>"
        f"<ul>{vetoes}</ul>{unmet}<p>Familias autorizadas: {html.escape(families)}{html.escape(pending)}.</p>"
        f"{_html_table(['Métrica de suficiencia', 'Valor'], _metric_rows(gate))}"
        "<h3>Desempeño frente a los labels revisados</h3>"
        f"{_html_table(_EVAL_HEADERS, _evaluation_rows(payload))}"
        "<p class='lead'>Una ventana con menos de 20 episodios positivos es solo descriptiva y no decide "
        "qué modelo gana. Las filas sin revisar y las inmaduras no se cuentan como negativas.</p>"
        f"<h3>Challengers supervisados</h3>{challengers}"
        f"<h3>Entregables detallados</h3><ul class='oot-list'>{links}</ul></div>"
    )
