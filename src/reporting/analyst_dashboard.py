"""Analyst-facing "Cola de Revisión" dashboard.

Fed exclusively by two things this project already produces from the OOT
block, no other source: the table :func:`src.evaluation.oot_report.
export_oot_top_anomalies` writes to the OOT Excel deliverable, and the
per-month recurrence view :func:`src.evaluation.oot_report.
months_present_by_entity` derives from that same OOT block. No business
categorization is invented over ``top_5_variables`` (that grouping does not
exist anywhere in the real project output), and no per-(entity, period)
panel field is shown without its period (none is shown at all here, since
the export's own ``VARIABLES`` columns are not passed to this renderer).

This is the target design worked out across three review rounds of
dashboard mockups (2026-08-30, see ``CHANGELOG.md``) and the explicit
constraint from the round that asked for this integration: the dashboard
reads Modelo v0.1's output the way it would read any upstream feed --
one input, no extra computation layered on top inside this project.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional, Sequence

import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.oot_report import BAND_COL
from src.reporting.report import _HTML_CSS, _THEME_TOGGLE_JS, _stat_tile_html
from src.utils import paths
from src.utils.logging_config import setup_logging

__all__ = ["build_analyst_dashboard"]

_ACCENT_LABEL = {"iforest": "Isolation Forest", "vae": "VAE"}

# Reuses the report's own token system (`_HTML_CSS`) so this reads as part of
# the same product rather than a one-off style -- only what that stylesheet
# does not already cover (the row-click table, the month-recurrence dots,
# the profile modal) is added here, and every color still comes from a
# `--series-*`/`--good`/`--warning`/`--serious` token, never a new literal.
_EXTRA_CSS = """
.rq-table { width: 100%; border-collapse: collapse; font-size: .87rem; }
.rq-table th { position: sticky; top: 0; background: var(--surface-1); z-index: 1;
  text-align: left; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--text-muted); font-weight: 650; padding: .45rem .6rem;
  border-bottom: 1px solid var(--gridline); }
.rq-table td { padding: .5rem .6rem; border-bottom: 1px solid var(--gridline);
  vertical-align: middle; }
.rq-table tbody tr { cursor: pointer; }
.rq-table tbody tr:hover td { background: var(--surface-page); }
.rq-table .idx { color: var(--text-muted); font-variant-numeric: tabular-nums; width: 2.2rem; }
.rq-table .idc { font-weight: 650; white-space: nowrap; }
.rq-scroll { max-height: 70vh; overflow-y: auto; border: 1px solid var(--border);
  border-radius: 10px; }

.band { font-size: .72rem; font-weight: 700; padding: .12rem .5rem; border-radius: 999px;
  letter-spacing: .02em; white-space: nowrap; }
.band-p90 { color: var(--text-secondary); background: var(--surface-page); }
.band-p95 { color: var(--warning); background: var(--warning-bg); }
.band-p99 { color: var(--serious); background: var(--serious-bg); }

.scorecell { white-space: nowrap; min-width: 130px; }
.pbar { display: inline-block; width: 60px; height: 5px; border-radius: 3px;
  background: var(--surface-page); overflow: hidden; vertical-align: middle; margin-right: .5rem; }
.pbar i { display: block; height: 100%; border-radius: 3px; }
.pbar.wide { width: 100%; height: 8px; margin: .5rem 0 .3rem; }
.pval { font-variant-numeric: tabular-nums; font-weight: 600; vertical-align: middle; }

.vars5 { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .78rem;
  color: var(--text-secondary); max-width: 0; width: 280px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; cursor: help; }

.monthscell { white-space: nowrap; display: flex; align-items: center; gap: .5rem; }
.mcount { font-variant-numeric: tabular-nums; font-weight: 700; font-size: .8rem;
  padding: .05rem .4rem; border-radius: 5px; color: var(--text-secondary); }
.mcount.sev-warning { color: var(--warning); background: var(--warning-bg); }
.mcount.sev-serious { color: var(--serious); background: var(--serious-bg); }
.mdots { display: inline-flex; gap: 3px; }
.mdot { width: 7px; height: 7px; border-radius: 50%; display: inline-block;
  background: var(--surface-page); border: 1px solid var(--border); }
.mdot.on { border-color: transparent; }

.overlay { position: fixed; inset: 0; background: rgba(10,12,20,.5); backdrop-filter: blur(2px);
  display: none; align-items: center; justify-content: center; z-index: 100; padding: 1.25rem; }
.overlay.open { display: flex; }
.modal { width: 100%; max-width: 520px; max-height: 86vh; overflow-y: auto;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.5rem; position: relative; }
.modal .xclose { position: absolute; top: 1rem; right: 1rem; width: 1.9rem; height: 1.9rem;
  border-radius: 8px; border: 1px solid var(--border); background: transparent;
  color: var(--text-secondary); cursor: pointer; }
.modal .xclose:hover { background: var(--surface-page); }
.modal .m-eyebrow { font-size: .68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; color: var(--text-muted); margin-bottom: .2rem; }
.modal .m-id { font-size: 1.3rem; font-weight: 700; }
.modal .m-field-label { font-size: .68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .04em; color: var(--text-muted); margin: 1rem 0 .5rem; }
.modal .m-chip { display: inline-block; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: .72rem; padding: .35rem .6rem; border-radius: 6px; margin: 0 .35rem .35rem 0;
  background: var(--surface-page); border: 1px solid var(--border); color: var(--text-secondary); }
.modal .m-monthchip { font-weight: 700; }
.modal .m-monthchip.on { background: var(--good-bg); color: var(--good); border-color: transparent; }
"""


def _severity_for(count: int, total: int) -> str:
    """Recurrence severity -- reuses the report's own good/warning/serious
    vocabulary instead of a fourth, dashboard-only color."""
    if total <= 0 or count <= 0:
        return ""
    ratio = count / total
    if ratio >= 1.0:
        return "serious"
    if ratio >= 0.5:
        return "warning"
    return ""


def build_analyst_dashboard(
    table: pd.DataFrame,
    schema: PanelSchema,
    model_name: str,
    oot_periods: Sequence,
    percentile_by_entity: dict,
    months_present: dict,
    score_col: str = "anomaly_score",
    out_path: Optional[str] = None,
) -> str:
    """Render the analyst review-queue dashboard for one model's OOT export.

    Every value shown is read from ``table`` (the exact DataFrame
    :func:`src.evaluation.oot_report.export_oot_top_anomalies` writes to the
    OOT Excel) or from ``months_present``/``percentile_by_entity`` (both
    derived from that same OOT block, computed by the caller) -- nothing
    else. There is deliberately no business categorization of
    ``top_5_variables`` and no raw per-(entity, period) panel field.

    Args:
        table: The table ``export_oot_top_anomalies`` returned for this
            model -- already sorted by score descending; row order is
            preserved.
        schema: Panel schema (for ``entity_col``/``time_col``).
        model_name: ``"iforest"`` or ``"vae"`` -- selects the accent color,
            the same ``--series-1``/``--series-2`` tokens the rest of the
            report already uses for these two detectors.
        oot_periods: Every period in the OOT window (not only the ones a
            given entity happens to appear in), so the recurrence indicator
            always shows the same N columns for every row.
        percentile_by_entity: ``{entity_id: percentile_0_100}`` for
            ``score_col``, computed over this model's own de-duplicated OOT
            population (same population `_percentile_band_labels` grades
            ``table`` against).
        months_present: ``{entity_id: [period_str, ...]}`` from
            :func:`months_present_by_entity` -- which OOT months this
            entity's score cleared the P95 cut-off in.
        score_col: Name of the score column in ``table``.
        out_path: Destination ``.html``. Defaults to
            ``artifacts/reports/analyst_dashboard_<model_name>.html``.

    Returns:
        The absolute path written.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    resolved_out = out_path or paths.ANALYST_DASHBOARD_DEFAULT.format(model=model_name)

    all_periods = [str(p)[:10] for p in oot_periods]
    n_months = max(1, len(all_periods))
    accent = "var(--series-1)" if model_name == "iforest" else "var(--series-2)"
    accent_label = _ACCENT_LABEL.get(model_name, model_name)
    has_vars = "top_5_variables" in table.columns

    records = table.to_dict("records")
    n_reviewed = 0
    n_recurrent = 0
    row_cells: list[str] = []
    profiles: dict = {}
    for i, r in enumerate(records, start=1):
        eid = str(r[entity_col])
        band = str(r.get(BAND_COL, ""))
        if band in ("p95", "p99"):
            n_reviewed += 1
        score = float(r[score_col])
        pctl = float(percentile_by_entity.get(eid, float("nan")))
        present = sorted(months_present.get(eid, []))
        if len(present) >= 2:
            n_recurrent += 1
        sev = _severity_for(len(present), n_months)
        top5 = str(r.get("top_5_variables") or "").strip() if has_vars else ""
        band_label = band.upper() if band else "&mdash;"
        pctl_txt = f"{pctl:.1f}" if pctl == pctl else "&mdash;"  # NaN check
        bar_pct = max(2, min(100, pctl)) if pctl == pctl else 0

        dots = "".join(
            f"<span class='mdot {'on' if p in present else ''}' title='{html.escape(p)}'></span>"
            for p in all_periods
        )
        row_cells.append(f"""
        <tr onclick="openProfile('{html.escape(eid, quote=True)}')" tabindex="0"
            onkeypress="if(event.key==='Enter')openProfile('{html.escape(eid, quote=True)}')">
          <td class="idx">{i}</td>
          <td class="idc">{html.escape(eid)}</td>
          <td><span class="band band-{html.escape(band) or 'p90'}">{band_label}</span></td>
          <td class="scorecell">
            <span class="pbar"><i style="width:{bar_pct:.0f}%;background:{accent}"></i></span>
            <span class="pval">{pctl_txt}</span>
          </td>
          <td class="vars5" title="{html.escape(top5)}">{html.escape(top5) or '&mdash;'}</td>
          <td class="monthscell">
            <span class="mcount{(' sev-' + sev) if sev else ''}">{len(present)}/{n_months}</span>
            <span class="mdots">{dots}</span>
          </td>
        </tr>""")
        profiles[eid] = {
            "band": band, "score": score, "pctl": pctl,
            "months": present, "top5": top5,
        }

    # `_stat_tile_html` runs `label`/`value`/`sub` through `html.escape`, so
    # these use the literal "≥" character, not the `&ge;` entity -- an
    # entity would come back double-escaped ("&amp;ge;") and render as
    # literal text instead of the symbol.
    kpi_html = (
        _stat_tile_html(
            "En revisión (score ≥ P95)", f"{n_reviewed:,}",
            sub=f"de {len(records):,} en la cola exportada (percentil 95/99 de la tabla)",
        )
        + _stat_tile_html(
            f"Recurrentes (≥2 de {n_months} meses)", f"{n_recurrent:,}",
            sub="de los individuos en revisión, marcados en más de un mes del OOT",
        )
    )

    period_note = ", ".join(all_periods)
    oot_note = (
        f"Ventana OOT: {period_note} ({n_months} mes{'es' if n_months != 1 else ''})."
        if all_periods else ""
    )

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='es'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>Cola de Revisión &middot; {html.escape(accent_label)}</title>")
    parts.append(f"<style>{_HTML_CSS}{_EXTRA_CSS}</style>")
    parts.append("</head><body>")

    parts.append("<header class='topbar'><div class='topbar-inner'>")
    parts.append(
        f"<div class='topbar-title'>Cola de Revisión &middot; {html.escape(accent_label)}"
        f"<span class='ts'>{html.escape(oot_note)}</span></div>"
    )
    parts.append("<button id='theme-toggle' type='button'>Modo oscuro</button>")
    parts.append("</div></header>")

    parts.append("<div class='wrap'>")
    parts.append(
        "<p class='lead'>Cola de análisis para el bloque fuera de tiempo (OOT). "
        "Alimentada exclusivamente por la tabla que este pipeline exporta "
        "(<code>export_oot_top_anomalies</code>) y su vista de recurrencia "
        "mensual sobre el mismo bloque -- ninguna variable ni categoría "
        "adicional se agrega aquí.</p>"
    )
    parts.append(f"<div class='tile-row'>{kpi_html}</div>")

    parts.append("<h2>Individuos priorizados</h2>")
    parts.append(
        "<p class='chart-note'>Clic en una fila para ver el detalle. "
        "Orden: score descendente, igual que en el Excel exportado.</p>"
    )
    parts.append("<div class='rq-scroll'><table class='rq-table'><thead><tr>"
                  "<th></th><th>ID</th><th>Banda</th><th>Percentil</th>"
                  "<th>Top-5 variables (salida del modelo)</th>"
                  f"<th>Meses ({n_months})</th></tr></thead><tbody>")
    parts.append("".join(row_cells))
    parts.append("</tbody></table></div>")

    parts.append(
        "<footer class='report-footer'>Modelo v0.1 &middot; entregable OOT "
        f"[{html.escape(accent_label)}] &middot; uso interno</footer>"
    )
    parts.append("</div>")  # .wrap

    # -- profile modal -------------------------------------------------------- #
    parts.append("""
<div class="overlay" id="overlay" onclick="if(event.target===this)closeProfile()">
  <div class="modal">
    <button class="xclose" type="button" onclick="closeProfile()">&#10005;</button>
    <div class="m-eyebrow">Perfil de individuo priorizado</div>
    <div class="m-id" id="mId">&mdash;</div>
    <span class="band" id="mBand" style="margin-top:.5rem;display:inline-block"></span>
    <div class="m-field-label">Score</div>
    <div><span class="pval" id="mScore">&mdash;</span></div>
    <div class="pbar wide"><i id="mBar" style="width:0%"></i></div>
    <div id="mPctlLine" style="color:var(--text-muted);font-size:.8rem"></div>
    <div class="m-field-label">Presencia en el OOT</div>
    <div id="mMonths"></div>
    <div class="m-field-label">Top-5 variables que más empujan el score</div>
    <div id="mChips"></div>
  </div>
</div>
""")

    parts.append(f"<script>var PROFILES={json.dumps(profiles, ensure_ascii=False)};")
    parts.append(f"var ACCENT={json.dumps(accent)};")
    parts.append("""
function openProfile(id){
  var r = PROFILES[id]; if(!r) return;
  document.getElementById("mId").textContent = id;
  var bandEl = document.getElementById("mBand");
  bandEl.textContent = (r.band || "-").toUpperCase();
  bandEl.className = "band band-" + (r.band || "p90");
  document.getElementById("mScore").textContent = r.score.toFixed(4);
  var pctl = r.pctl;
  document.getElementById("mBar").style.width = (isNaN(pctl) ? 0 : pctl) + "%";
  document.getElementById("mBar").style.background = ACCENT;
  document.getElementById("mPctlLine").textContent = isNaN(pctl) ? "" : ("percentil " + pctl.toFixed(1));
  var mm = document.getElementById("mMonths"); mm.innerHTML = "";
  (r.months || []).forEach(function(m){
    mm.innerHTML += '<span class="m-chip m-monthchip on">' + m + '</span>';
  });
  if (!r.months || !r.months.length) { mm.innerHTML = "<span class='m-chip'>ninguno</span>"; }
  var mc = document.getElementById("mChips"); mc.innerHTML = "";
  (r.top5 || "").split(",").map(function(s){return s.trim();}).filter(Boolean).forEach(function(v){
    mc.innerHTML += '<span class="m-chip">' + v + '</span>';
  });
  document.getElementById("overlay").classList.add("open");
}
function closeProfile(){ document.getElementById("overlay").classList.remove("open"); }
document.addEventListener("keydown", function(e){ if(e.key==="Escape") closeProfile(); });
""")
    parts.append(f"<script>{_THEME_TOGGLE_JS}</script>")
    parts.append("</body></html>")

    parent = os.path.dirname(os.path.abspath(resolved_out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    content = "\n".join(parts)
    with open(resolved_out, "w", encoding="utf-8") as fh:
        fh.write(content)

    log.info(
        "Analyst dashboard [%s] -> %s (%d in review, %d recurrent of %d rows)",
        model_name, resolved_out, n_reviewed, n_recurrent, len(records),
    )
    return os.path.abspath(resolved_out)
