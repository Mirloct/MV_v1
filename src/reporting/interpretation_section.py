"""Renderers for the "Interpretación y recomendaciones" chapter.

Mirrors ``src/reporting/diagnostic_section.py``'s split: both formats walk
the SAME contract (``src/evaluation/ifvae_interpretation.py``) and lay it
out. Unlike the factual ficha's renderers, these are allowed to show
interpretive language -- that is the module's entire purpose -- but they
still compute nothing: every fragment, threshold check, and decision-flow
node arrives pre-built from ``ifvae_interpretation.py``. A renderer that
started comparing numbers itself would silently duplicate (and could
diverge from) the interpretation contract's own logic.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

__all__ = ["render_interpretation_html", "render_interpretation_markdown"]

#: severity -> chip class (reuses the report's own `.chip good/warning/serious`).
_SEVERITY_CHIP = {"info": "good", "attention": "warning", "caution": "serious"}
_SEVERITY_LABEL = {"info": "Informativo", "attention": "Atención", "caution": "Precaución"}


def _severity_chip_html(severity: str) -> str:
    cls = _SEVERITY_CHIP.get(severity, "good")
    label = _SEVERITY_LABEL.get(severity, severity)
    return f"<span class='chip {cls}'>{html.escape(label)}</span>"


# --------------------------------------------------------------------------- #
# HTML                                                                        #
# --------------------------------------------------------------------------- #
def _html_fragments(fragments: Sequence[dict]) -> str:
    if not fragments:
        return "<p><em>Sin lecturas para esta sección.</em></p>"
    items = "".join(
        f"<li>{_severity_chip_html(f['severity'])} {html.escape(f['text'])}"
        f"<div class='sub'>Base: {html.escape(f['basis'])}</div></li>"
        for f in fragments
    )
    return f"<ul class='oot-list'>{items}</ul>"


def _html_toolkit(toolkit: Sequence[dict]) -> str:
    parts = []
    for section in toolkit:
        parts.append(f"<h4>{html.escape(section['title'])}</h4>")
        parts.append(_html_fragments(section["reading"]))
        if section["validity"]:
            parts.append("<p class='sub'><strong>Validez del indicador:</strong></p>")
            parts.append(_html_fragments(section["validity"]))
    return "".join(parts)


def _html_indicator_validation(checks: Sequence[dict]) -> str:
    if not checks:
        return "<p><em>Sin indicadores que validar en esta corrida.</em></p>"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(c['indicator'])}</td>"
        f"<td>{_severity_chip_html('info' if c['valid'] else 'attention')}</td>"
        f"<td>{html.escape(c['detail'])}</td>"
        f"<td class='src'>{html.escape(c['basis'])}</td>"
        "</tr>"
        for c in checks
    )
    return (
        "<div class='table-wrap'><table>"
        "<tr><th scope='col'>Indicador</th><th scope='col'>Válido</th>"
        "<th scope='col'>Detalle</th><th scope='col'>Base metodológica</th></tr>"
        f"{rows}</table></div>"
    )


def _html_decision_flow(decision_flow: dict) -> str:
    nodes_html = []
    for i, node in enumerate(decision_flow["nodes"]):
        chip = _severity_chip_html(node["severity"])
        outcome = (
            f"<p>{html.escape(node['fragment']['text'])}</p>"
            if node.get("fragment") else ""
        )
        nodes_html.append(
            f"<div class='decision-node sev-{html.escape(node['severity'])}'>"
            f"<div class='decision-q'>{html.escape(node['question'])}</div>"
            f"<div class='decision-a'>{chip} {html.escape(node['answer'])}</div>"
            f"{outcome}</div>"
        )
        if i < len(decision_flow["nodes"]) - 1:
            nodes_html.append("<div class='decision-arrow' aria-hidden='true'>↓</div>")
    fragments = decision_flow["recommendation_fragments"]
    rec_items = "".join(
        f"<li>{_severity_chip_html(f['severity'])} {html.escape(f['text'])}</li>"
        for f in fragments
    ) or "<li>Sin hallazgos que priorizar en esta corrida.</li>"
    return (
        "<div class='decision-flow'>" + "".join(nodes_html) + "</div>"
        "<h4>Recomendación al analista</h4>"
        "<p class='subtitle'>Heurística, compuesta a partir de los nodos de "
        "arriba para esta corrida específica -- no es un veredicto de verdad "
        "base.</p>"
        f"<ol class='oot-list'>{rec_items}</ol>"
    )


#: Chapter-local styles for the decision-flow diagram. Reuses the report's
#: own design tokens.
INTERPRETATION_CSS = """
.decision-flow { display: flex; flex-direction: column; align-items: stretch;
  gap: 2px; margin: .6rem 0 1.2rem; max-width: 640px; }
.decision-node { border: 1px solid var(--border); border-left-width: 4px;
  border-radius: 8px; padding: .6rem .8rem; background: var(--surface-1); }
.decision-node.sev-info { border-left-color: var(--good); }
.decision-node.sev-attention { border-left-color: var(--warning); }
.decision-node.sev-caution { border-left-color: var(--serious); }
.decision-q { font-weight: 600; font-size: .88rem; color: var(--text-primary); }
.decision-a { font-size: .84rem; color: var(--text-secondary); margin: .2rem 0; }
.decision-node p { font-size: .82rem; color: var(--text-muted); margin: .2rem 0 0; }
.decision-arrow { text-align: center; color: var(--text-muted); font-size: 1rem;
  line-height: 1; }
"""


def render_interpretation_html(contract: Optional[dict]) -> str:
    if not isinstance(contract, dict) or not contract.get("toolkit"):
        return ""
    return (
        f"<h2 id='{html.escape(contract['id'])}'>{html.escape(contract['title'])}</h2>"
        "<div class='card'>"
        f"<p class='subtitle'>{html.escape(contract['subtitle'])}</p>"
        "<h3>Toolkit de interpretación por análisis</h3>"
        f"{_html_toolkit(contract['toolkit'])}"
        "<h3>Validación de indicadores</h3>"
        f"{_html_indicator_validation(contract['indicator_validation'])}"
        "<h3>Flujo de decisión de esta corrida</h3>"
        f"{_html_decision_flow(contract['decision_flow'])}"
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #
def _md_fragments(fragments: Sequence[dict]) -> str:
    if not fragments:
        return "_Sin lecturas para esta sección._\n"
    return "\n".join(
        f"- **[{f['severity'].upper()}]** {f['text']} _(base: {f['basis']})_"
        for f in fragments
    ) + "\n"


def render_interpretation_markdown(contract: Optional[dict]) -> str:
    if not isinstance(contract, dict) or not contract.get("toolkit"):
        return ""
    parts = [f"## {contract['title']}\n", f"_{contract['subtitle']}_\n"]
    parts.append("### Toolkit de interpretación por análisis\n")
    for section in contract["toolkit"]:
        parts.append(f"**{section['title']}**\n")
        parts.append(_md_fragments(section["reading"]))
        if section["validity"]:
            parts.append("_Validez del indicador:_\n")
            parts.append(_md_fragments(section["validity"]))
    parts.append("### Validación de indicadores\n")
    checks = contract["indicator_validation"]
    if checks:
        parts.append("| Indicador | Válido | Detalle | Base metodológica |")
        parts.append("| --- | --- | --- | --- |")
        for c in checks:
            parts.append(
                f"| {c['indicator']} | {'Sí' if c['valid'] else 'Atención'} | "
                f"{c['detail']} | {c['basis']} |"
            )
        parts.append("")
    else:
        parts.append("_Sin indicadores que validar en esta corrida._\n")
    parts.append("### Flujo de decisión de esta corrida\n")
    for node in contract["decision_flow"]["nodes"]:
        parts.append(f"**{node['question']}**")
        parts.append(f"- Resultado: [{node['severity'].upper()}] {node['answer']}")
        if node.get("fragment"):
            parts.append(f"- {node['fragment']['text']}")
        parts.append("")
    parts.append("**Recomendación al analista** _(heurística, compuesta para esta corrida)_\n")
    fragments = contract["decision_flow"]["recommendation_fragments"]
    if fragments:
        for i, f in enumerate(fragments, 1):
            parts.append(f"{i}. **[{f['severity'].upper()}]** {f['text']}")
    else:
        parts.append("1. Sin hallazgos que priorizar en esta corrida.")
    return "\n".join(parts)
