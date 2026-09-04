"""Renderers for the "Diagnóstico cruzado IF-VAE" chapter.

Both renderers walk the *same* structured contract
(``src/evaluation/ifvae_contract.py``) and lay out the blocks it declares.
They contain no availability rules, no severity rules, no arithmetic over the
run's numbers, and no narrative about what the numbers mean: a block that is
not in the contract is not rendered, and a value the contract marked as
non-``EXECUTED`` is rendered as its declared absence plus the reason.

Block kinds the contract may declare, and how each is laid out:

``fields``   label/value rows with provenance
``table``    column headers + rows, optionally with a client-side filter
``note``     neutral methodological text or reading aid
``scatter``  inline SVG (HTML) / counts + artifact link (Markdown)

Keeping both renderers in one module is deliberate: HTML/Markdown parity is a
tested property (``tests/test_diagnostic_section.py``), and the two
implementations drifting apart is exactly the failure that test exists to
catch.
"""

from __future__ import annotations

import html
import os
from typing import Any, Optional, Sequence

from src.evaluation.ifvae_contract import (
    NO_VALUE_TEXT,
    STATUS_EXECUTED,
    STATUS_LABELS,
)

__all__ = ["render_diagnostic_html", "render_diagnostic_markdown", "iter_contract_values"]


# --------------------------------------------------------------------------- #
# Shared value formatting (identical text in both formats, by construction)   #
# --------------------------------------------------------------------------- #
def _value_text(field: dict) -> str:
    """The visible text for one contract field, in either output format."""
    if field.get("status") != STATUS_EXECUTED:
        reason = field.get("reason") or ""
        return f"{NO_VALUE_TEXT} — {reason}" if reason else NO_VALUE_TEXT
    value = field.get("value")
    if value is None:
        return NO_VALUE_TEXT
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "(ninguno)"
    return str(value)


def _cell_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, bool):
        return "Sí" if cell else "No"
    return str(cell)


def iter_contract_values(contract: dict):
    """Yield ``(section_id, block_index, label, text)`` for every visible value.

    Used by the parity test: both renderers must show exactly what this
    enumerates, so a value can never appear in one format and not the other.
    """
    for section in contract.get("sections", []):
        for index, block in enumerate(section.get("blocks", [])):
            kind = block.get("kind")
            if kind == "fields":
                for row in block["rows"]:
                    yield (section["id"], index, row["label"], _value_text(row["field"]))
            elif kind == "table":
                # Cell by cell, not row by row: the two formats separate cells
                # differently (pipes vs. table markup), so a joined row would
                # only ever match one of them.
                for column_index, column in enumerate(block["columns"]):
                    yield (section["id"], index, f"__column_{column_index}__", column)
                for row_index, row in enumerate(block["rows"]):
                    for cell_index, cell in enumerate(row):
                        yield (section["id"], index,
                               f"__cell_{row_index}_{cell_index}__", _cell_text(cell))
                if not block["rows"]:
                    yield (section["id"], index, "__empty__", block["empty_text"])
            elif kind == "note":
                yield (section["id"], index, "__note__", block["text"])
            elif kind == "scatter":
                yield (section["id"], index, "__scatter_alt__", block["alt_text"])
                for quadrant in block["quadrants"]:
                    # Label and count separately: the SVG draws them as one
                    # string, the Markdown table as two cells -- both must
                    # carry both values, but not in the same arrangement.
                    yield (section["id"], index,
                           f"__quadrant_{quadrant['key']}_label__", quadrant["label"])
                    yield (section["id"], index,
                           f"__quadrant_{quadrant['key']}_count__", str(quadrant["count"]))


# --------------------------------------------------------------------------- #
# HTML                                                                        #
# --------------------------------------------------------------------------- #
def _html_field_rows(block: dict) -> str:
    rows = []
    for row in block["rows"]:
        field = row["field"]
        source = field.get("source") or ""
        note = field.get("note") or ""
        status = field.get("status")
        status_chip = (
            "" if status == STATUS_EXECUTED else
            f"<span class='chip warning'>{html.escape(STATUS_LABELS.get(status, status))}</span> "
        )
        rows.append(
            "<tr>"
            f"<th scope='row'>{html.escape(row['label'])}</th>"
            f"<td>{status_chip}{html.escape(_value_text(field))}"
            + (f"<div class='sub'>{html.escape(note)}</div>" if note else "")
            + "</td>"
            f"<td class='src'>{html.escape(source)}</td>"
            "</tr>"
        )
    title = f"<h4>{html.escape(block['title'])}</h4>" if block.get("title") else ""
    return (
        f"{title}<div class='table-wrap'><table>"
        "<caption class='sr-caption'>Campos, valores y su procedencia</caption>"
        "<tr><th scope='col'>Campo</th><th scope='col'>Valor</th>"
        "<th scope='col'>Fuente</th></tr>"
        + "".join(rows) + "</table></div>"
    )


def _html_table(block: dict, table_id: str) -> str:
    title = f"<h4>{html.escape(block['title'])}</h4>" if block.get("title") else ""
    caption = block.get("caption") or block.get("title") or "Tabla de diagnóstico"
    if not block["rows"]:
        return (f"{title}<p><em>{html.escape(block['empty_text'])}</em></p>"
                + (f"<p class='sub'>{html.escape(block['caption'])}</p>"
                   if block.get("caption") else ""))
    head = "".join(f"<th scope='col'>{html.escape(c)}</th>" for c in block["columns"])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell_text(c))}</td>" for c in row) + "</tr>"
        for row in block["rows"]
    )
    filter_html = ""
    if block.get("filters"):
        options = "".join(
            f"<option value='{html.escape(v)}'>{html.escape(v)}</option>"
            for v in block["filters"]["values"]
        )
        filter_html = (
            f"<label class='diag-filter' for='{table_id}-filter'>"
            f"{html.escape(block['filters']['label'])} "
            f"<select id='{table_id}-filter' data-table='{table_id}' "
            f"data-column='{html.escape(block['filters']['column'])}'>"
            "<option value='__all__'>Todas</option>" + options + "</select></label>"
        )
    return (
        f"{title}{filter_html}<div class='table-wrap'><table id='{table_id}'>"
        f"<caption class='sr-caption'>{html.escape(caption)}</caption>"
        f"<tr>{head}</tr>{body}</table></div>"
        + (f"<p class='sub'>{html.escape(block['caption'])}</p>"
           if block.get("caption") else "")
    )


def _html_scatter(block: dict) -> str:
    """Inline SVG: no library, no network, readable without colour.

    Quadrant names and counts are drawn as text inside the plot area, and the
    threshold lines are labelled, so the figure is legible in greyscale and to
    a screen reader via its ``role='img'`` label.
    """
    width, height, pad = 520, 520, 46
    points = block["points"]
    xs, ys, quads = points["x"], points["y"], points["quadrant"]
    x_threshold = block.get("x_threshold")
    y_threshold = block.get("y_threshold")

    def _px(value: float) -> float:
        return pad + float(value) * (width - 2 * pad)

    def _py(value: float) -> float:
        return height - pad - float(value) * (height - 2 * pad)

    # One marker shape per quadrant so the chart survives greyscale printing.
    shapes = {"BOTH": "▲", "IF_ONLY": "■", "VAE_ONLY": "◆", "NEITHER": "·"}
    marks = []
    for x, y, quadrant in zip(xs, ys, quads):
        marks.append(
            f"<circle cx='{_px(x):.1f}' cy='{_py(y):.1f}' r='2' "
            f"class='pt pt-{html.escape(str(quadrant))}' />"
        )
    threshold_lines = ""
    if x_threshold is not None:
        threshold_lines += (
            f"<line x1='{_px(x_threshold):.1f}' y1='{pad}' "
            f"x2='{_px(x_threshold):.1f}' y2='{height - pad}' class='thr' />"
            f"<text x='{_px(x_threshold):.1f}' y='{pad - 8}' class='thr-label' "
            f"text-anchor='middle'>umbral {x_threshold}</text>"
        )
    if y_threshold is not None:
        threshold_lines += (
            f"<line x1='{pad}' y1='{_py(y_threshold):.1f}' "
            f"x2='{width - pad}' y2='{_py(y_threshold):.1f}' class='thr' />"
        )
    labels = ""
    corners = {
        "NEITHER": (pad + 6, height - pad - 8, "start"),
        "IF_ONLY": (width - pad - 6, height - pad - 8, "end"),
        "VAE_ONLY": (pad + 6, pad + 16, "start"),
        "BOTH": (width - pad - 6, pad + 16, "end"),
    }
    for quadrant in block["quadrants"]:
        key = quadrant["key"]
        if key not in corners:
            continue
        x, y, anchor = corners[key]
        labels += (
            f"<text x='{x}' y='{y}' text-anchor='{anchor}' class='q-label'>"
            f"{html.escape(shapes.get(key, ''))} {html.escape(quadrant['label'])}: "
            f"{quadrant['count']}</text>"
        )
    caption = (f"<p class='sub'>{html.escape(block['caption'])}</p>"
               if block.get("caption") else "")
    return (
        f"<h4>{html.escape(block['title'])}</h4>"
        f"<svg class='diag-scatter' viewBox='0 0 {width} {height}' "
        f"role='img' aria-label='{html.escape(block['alt_text'])}'>"
        f"<rect x='{pad}' y='{pad}' width='{width - 2 * pad}' "
        f"height='{height - 2 * pad}' class='frame' />"
        f"{''.join(marks)}{threshold_lines}{labels}"
        f"<text x='{width / 2}' y='{height - 8}' text-anchor='middle' class='ax'>"
        f"{html.escape(block['x_label'])}</text>"
        f"<text x='14' y='{height / 2}' text-anchor='middle' class='ax' "
        f"transform='rotate(-90 14 {height / 2})'>{html.escape(block['y_label'])}</text>"
        "</svg>"
        f"<p class='sr-only'>{html.escape(block['alt_text'])}</p>{caption}"
    )


#: Chapter-local styles. Uses the report's own design tokens (`--series-1`,
#: `--text-secondary`, ...) so the chapter inherits light/dark theming instead
#: of defining a second palette.
DIAGNOSTIC_CSS = """
.diag-section { margin-bottom: 1.6rem; }
.diag-section h3 { font-size: 1.02rem; color: var(--text-primary);
  border-bottom: 1px solid var(--border); padding-bottom: .35rem; }
.diag-section h4 { font-size: .92rem; color: var(--text-secondary);
  margin: 1rem 0 .4rem; }
.diag-section td.src { color: var(--text-muted); font-size: .78rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.diag-section .sub { color: var(--text-muted); font-size: .78rem; margin-top: .2rem; }
.diag-filter { display: inline-flex; gap: .4rem; align-items: center;
  font-size: .82rem; color: var(--text-secondary); margin: .3rem 0 .5rem; }
.diag-filter select { font: inherit; padding: .2rem .35rem;
  border: 1px solid var(--border); border-radius: 5px;
  background: var(--surface-1); color: var(--text-primary); }
.diag-scatter { width: 100%; max-width: 520px; height: auto; display: block;
  margin: .4rem 0; }
.diag-scatter .frame { fill: none; stroke: var(--gridline); }
.diag-scatter .pt { fill: var(--text-muted); opacity: .5; }
.diag-scatter .pt-BOTH { fill: var(--series-1); opacity: .85; }
.diag-scatter .pt-VAE_ONLY { fill: var(--series-2); opacity: .8; }
.diag-scatter .pt-IF_ONLY { fill: var(--good); opacity: .8; }
.diag-scatter .thr { stroke: var(--serious); stroke-width: 1.2;
  stroke-dasharray: 4 3; }
.diag-scatter .thr-label, .diag-scatter .q-label, .diag-scatter .ax {
  fill: var(--text-secondary); font-size: 11px;
  font-family: ui-sans-serif, system-ui, sans-serif; }
.diag-scatter .q-label { fill: var(--text-primary); font-size: 11.5px; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0;
  margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;
  border: 0; }
.sr-caption { text-align: left; caption-side: top; color: var(--text-muted);
  font-size: .78rem; padding-bottom: .25rem; }
"""

#: Client-side filter for any table the contract marked as filterable. Pure
#: presentation: it hides rows, it does not recompute anything.
DIAGNOSTIC_JS = """
document.querySelectorAll('.diag-filter select').forEach(function (select) {
  select.addEventListener('change', function () {
    var table = document.getElementById(select.dataset.table);
    if (!table) { return; }
    var header = Array.prototype.slice.call(table.rows[0].cells)
      .map(function (c) { return c.textContent.trim(); });
    var column = header.indexOf(select.dataset.column);
    if (column < 0) { return; }
    var wanted = select.value;
    Array.prototype.slice.call(table.rows, 1).forEach(function (row) {
      row.hidden = wanted !== '__all__' &&
        row.cells[column].textContent.trim() !== wanted;
    });
  });
});
"""


def render_diagnostic_html(contract: Optional[dict]) -> str:
    """Lay the contract out as HTML. ``''`` when there is no contract."""
    if not isinstance(contract, dict) or not contract.get("sections"):
        return ""
    parts = [
        f"<h2 id='{html.escape(contract['id'])}'>{html.escape(contract['title'])}</h2>",
        "<div class='card'>",
        f"<p class='subtitle'>{html.escape(contract['subtitle'])}</p>",
    ]
    for section in contract["sections"]:
        parts.append("<section class='diag-section'>")
        parts.append(
            f"<h3 id='{html.escape(contract['id'])}-{html.escape(section['id'])}'>"
            f"{html.escape(section['title'])}</h3>"
        )
        for index, block in enumerate(section["blocks"]):
            table_id = f"diag-{section['id']}-{index}"
            kind = block["kind"]
            if kind == "fields":
                parts.append(_html_field_rows(block))
            elif kind == "table":
                parts.append(_html_table(block, table_id))
            elif kind == "note":
                parts.append(f"<p class='sub'>{html.escape(block['text'])}</p>")
            elif kind == "scatter":
                parts.append(_html_scatter(block))
        parts.append("</section>")
    parts.append("</div>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #
def _md_escape(text: str) -> str:
    """Escape only what would break a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(_md_escape(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_md_escape(_cell_text(c)) for c in row) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body]) + "\n"


def render_diagnostic_markdown(contract: Optional[dict]) -> str:
    """Lay the same contract out as Markdown. ``''`` when there is no contract."""
    if not isinstance(contract, dict) or not contract.get("sections"):
        return ""
    parts = [f"## {contract['title']}\n", f"_{contract['subtitle']}_\n"]
    for section in contract["sections"]:
        parts.append(f"### {section['title']}\n")
        for block in section["blocks"]:
            kind = block["kind"]
            if kind == "fields":
                if block.get("title"):
                    parts.append(f"**{block['title']}**\n")
                parts.append(_md_table(
                    ["Campo", "Valor", "Fuente"],
                    [[row["label"], _value_text(row["field"]),
                      row["field"].get("source") or ""]
                     for row in block["rows"]],
                ))
            elif kind == "table":
                if block.get("title"):
                    parts.append(f"**{block['title']}**\n")
                if block["rows"]:
                    parts.append(_md_table(block["columns"], block["rows"]))
                else:
                    parts.append(f"_{block['empty_text']}_\n")
                if block.get("caption"):
                    parts.append(f"_{block['caption']}_\n")
                if block.get("filters"):
                    parts.append(
                        f"_{block['filters']['label']}: "
                        f"{', '.join(block['filters']['values'])} "
                        "(filtro interactivo disponible en el reporte HTML)._\n"
                    )
            elif kind == "note":
                parts.append(f"_{block['text']}_\n")
            elif kind == "scatter":
                parts.append(f"**{block['title']}**\n")
                parts.append(_md_table(
                    ["Cuadrante", "Observaciones"],
                    [[q["label"], q["count"]] for q in block["quadrants"]],
                ))
                parts.append(f"_{block['alt_text']}_\n")
                if block.get("caption"):
                    parts.append(f"_{block['caption']}_\n")
    return "\n".join(parts)
