"""Analyst-facing "Cola de Revisión" dashboard.

Ports, verbatim in layout/CSS/JS, the mockup design reviewed and approved
across three rounds (2026-08-30, see `CHANGELOG.md`) -- same shell, header,
two KPI tiles, single priority table, profile modal, footer legend. Only the
data source changed: every value here comes from this project's own real
OOT output, never mock data.

Fed by exactly two things this project already produces from the OOT
block: the table :func:`src.evaluation.oot_report.export_oot_top_anomalies`
writes to the OOT Excel deliverable (identity, band, `top_5_variables`, row
order), and each detector's own de-duplicated OOT scores
(`true_oot_entity_scores`, `main.py` Phase 8) plus the per-month recurrence
view derived from the same OOT block
(:func:`src.evaluation.oot_report.months_present_by_entity`). No business
categorization is invented over `top_5_variables`.

One dashboard, not one per model: both detector percentiles are shown for
every individual in the union of both P95 queues. Three tabs make agreement
explicit: only IF, only IF+VAE, and their intersection. The detail view embeds
the selected individual's complete raw history (every available source row
and column) for an offline CSV download. Case-review state is maintained in
the browser so this self-contained HTML can be used without a backend.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional, Sequence

import pandas as pd

from src.data.loader import PanelSchema
from src.utils import paths
from src.utils.logging_config import setup_logging

__all__ = ["build_analyst_dashboard"]

_SEV_FOR_MONTHS_DEFAULT = "mute"


def _severity_for(count: int, total: int) -> str:
    """Recurrence severity band, generalized to any `total` (the mockup's
    fixed 3-month case is `total == 3`): `mute` for a one-off, `warn` for
    appearing in at least half the OOT window, `crit` for every month."""
    if total <= 0 or count <= 0:
        return "mute"
    ratio = count / total
    if ratio >= 1.0:
        return "crit"
    if ratio >= 0.5:
        return "warn"
    return "mute"


def _month_dots_html(present: list, all_periods: list, sev: str) -> str:
    present_set = set(present)
    cells = []
    for p in all_periods:
        cls = f"dot-on dot-{sev}" if p in present_set else "dot-off"
        cells.append(f"<span class='mdot {cls}'></span>")
    return "".join(cells)


def _pct_bar(pct: float, kind: str) -> str:
    w = max(2, min(100, pct)) if pct == pct else 0  # NaN check
    txt = f"{pct:.1f}" if pct == pct else "&mdash;"
    return (f"<span class='pbar pbar-{kind}'><i style='width:{w:.0f}%'></i></span>"
            f"<span class='pval'>{txt}</span>")


def build_analyst_dashboard(
    table: pd.DataFrame,
    schema: PanelSchema,
    base_model_name: str,
    oot_periods: Sequence,
    if_percentile_by_entity: dict,
    vae_percentile_by_entity: dict,
    if_score_by_entity: dict,
    vae_score_by_entity: dict,
    months_present: dict,
    n_total_oot: int,
    score_col: str = "anomaly_score",
    out_path: Optional[str] = None,
    model_tables: Optional[dict[str, pd.DataFrame]] = None,
    months_present_by_model: Optional[dict[str, dict]] = None,
    oot_records: Optional[pd.DataFrame] = None,
    entity_records: Optional[pd.DataFrame] = None,
    identity_column: str = "puesto",
) -> str:
    """Render the single, unified analyst review-queue dashboard.

    Args:
        table: The table `export_oot_top_anomalies` returned for
            ``base_model_name``. Retained as the backwards-compatible source
            of top-variable explanations when ``model_tables`` is omitted.
        schema: Panel schema (for `entity_col`).
        base_model_name: Primary exported model, used for provenance and as
            the backwards-compatible table/month fallback.
        oot_periods: Every period in the OOT window, so the recurrence
            indicator always shows the same N columns for every row.
        if_percentile_by_entity / vae_percentile_by_entity: `{entity_id:
            percentile_0_100}` for each detector's own score, computed over
            that detector's own de-duplicated OOT population -- an in-memory
            join on `entity_id`, not a second file. An entity missing from
            one dict (should not happen; both detectors see the same OOT
            rows) renders as "&mdash;" rather than raising.
        if_score_by_entity / vae_score_by_entity: the raw scores behind the
            percentiles above, for the modal's score readout.
        months_present: Legacy recurrence map for ``base_model_name``. New
            callers should also pass ``months_present_by_model``.
        n_total_oot: Total unique individuals in the OOT window (from
            `base_model_name`'s own de-duplicated population) -- the
            denominator for the "en revisión" KPI's percentage.
        score_col: Name of the score column in ``table``.
        model_tables: Optional OOT export table for each detector. Supplying
            both ``iforest`` and ``vae`` lets the three detector-agreement
            tabs retain each model's own top-variable explanation.
        months_present_by_model: Per-detector recurrence maps. When omitted,
            the legacy ``months_present`` map is assigned to the base model.
        oot_records: Deprecated backwards-compatible alias for
            ``entity_records``.
        entity_records: Complete raw panel. Every available row and source
            column for entities in the P95 review union is embedded so the
            profile download covers full history, not only OOT.
        identity_column: Source column displayed below the entity ID in the
            profile. Defaults to the configured business field ``"puesto"``.
        out_path: Destination ``.html``. Defaults to
            ``artifacts/reports/analyst_dashboard.html``.

    Returns:
        The absolute path written.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    resolved_out = out_path or paths.ANALYST_DASHBOARD_DEFAULT

    if_percentile_by_entity = {str(k): v for k, v in if_percentile_by_entity.items()}
    vae_percentile_by_entity = {str(k): v for k, v in vae_percentile_by_entity.items()}
    if_score_by_entity = {str(k): v for k, v in if_score_by_entity.items()}
    vae_score_by_entity = {str(k): v for k, v in vae_score_by_entity.items()}

    all_periods = [str(p)[:10] for p in oot_periods]
    n_months = max(1, len(all_periods))
    model_tables = dict(model_tables or {base_model_name: table})
    model_tables.setdefault(base_model_name, table)
    recurrence = dict(months_present_by_model or {base_model_name: months_present})
    recurrence.setdefault("iforest", months_present if base_model_name == "iforest" else {})
    recurrence.setdefault("vae", months_present if base_model_name == "vae" else {})

    def _rows_by_entity(frame: Optional[pd.DataFrame]) -> dict[str, dict]:
        if frame is None or frame.empty or entity_col not in frame.columns:
            return {}
        return {str(row[entity_col]): row for row in frame.to_dict("records")}

    table_rows = {name: _rows_by_entity(frame) for name, frame in model_tables.items()}
    # The review universe is deliberately detector-independent: every entity
    # at/above P95 in either score distribution appears exactly once.
    entity_ids = sorted(
        set(str(eid) for eid, pct in if_percentile_by_entity.items() if float(pct) >= 95.0)
        | set(str(eid) for eid, pct in vae_percentile_by_entity.items() if float(pct) >= 95.0),
        key=lambda eid: (
            -max(float(if_percentile_by_entity.get(eid, -1)),
                 float(vae_percentile_by_entity.get(eid, -1))),
            eid,
        ),
    )

    source_records = entity_records if entity_records is not None else oot_records
    records_by_entity: dict[str, list[dict]] = {}
    record_columns: list[str] = []
    if source_records is not None and entity_col in source_records.columns:
        record_columns = [str(c) for c in source_records.columns]
        wanted = set(entity_ids)
        raw = source_records.loc[source_records[entity_col].astype(str).isin(wanted)].copy()
        time_col = schema.time_col or "period"
        if time_col in raw.columns:
            # Stable chronological order makes both the downloaded history
            # and the "latest puesto" rule deterministic.
            raw = raw.sort_values([entity_col, time_col], kind="stable")
        # pandas' JSON encoder handles Timestamp/numpy scalars and emits null
        # for NaN, producing a safe, exact-enough browser payload.
        safe_rows = json.loads(raw.to_json(orient="records", date_format="iso", force_ascii=False))
        for original_id, safe_row in zip(raw[entity_col].astype(str), safe_rows):
            records_by_entity.setdefault(original_id, []).append(safe_row)

    def _identity_for(eid: str) -> str:
        """Return the latest non-empty configured identity value."""
        for row in reversed(records_by_entity.get(eid, [])):
            value = row.get(identity_column)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    counts = {"if_only": 0, "vae_only": 0, "intersection": 0}
    recurrent_counts = {"if_only": 0, "vae_only": 0, "intersection": 0}
    row_html_list: list[str] = []
    profiles: dict = {}

    for i, eid in enumerate(entity_ids, start=1):
        if_pctl = float(if_percentile_by_entity.get(eid, float("nan")))
        vae_pctl = float(vae_percentile_by_entity.get(eid, float("nan")))
        if_score = float(if_score_by_entity.get(eid, float("nan")))
        vae_score = float(vae_score_by_entity.get(eid, float("nan")))
        if_hit, vae_hit = if_pctl >= 95.0, vae_pctl >= 95.0
        tab = "intersection" if if_hit and vae_hit else ("if_only" if if_hit else "vae_only")
        counts[tab] += 1
        if_months = sorted(recurrence["iforest"].get(eid, []))
        vae_months = sorted(recurrence["vae"].get(eid, []))
        present = sorted(set(if_months) | set(vae_months))
        months_count = len(present)
        if months_count >= 2:
            recurrent_counts[tab] += 1
        sev = _severity_for(months_count, n_months)
        if_row = table_rows.get("iforest", {}).get(eid, {})
        vae_row = table_rows.get("vae", {}).get(eid, {})
        if_top5 = str(if_row.get("top_5_variables") or "").strip()
        vae_top5 = str(vae_row.get("top_5_variables") or "").strip()
        shown_top5 = if_top5 if tab == "if_only" else vae_top5
        if tab == "intersection" and if_top5 and vae_top5:
            shown_top5 = f"IF: {if_top5} · IF+VAE: {vae_top5}"
        elif tab == "intersection":
            shown_top5 = if_top5 or vae_top5
        band = "p99" if max(if_pctl, vae_pctl) >= 99.0 else "p95"
        band_label = "AMBOS" if tab == "intersection" else ("IF" if tab == "if_only" else "IF+VAE")
        profile_key = f"p{i}"
        safe_eid = html.escape(eid, quote=True)
        safe_top5 = html.escape(shown_top5, quote=True)

        row_html_list.append(f"""
        <tr data-id="{safe_eid}" data-tab="{tab}" data-profile="{profile_key}"
            data-case-status="sin_revision"
            onclick="openProfile('{profile_key}')" tabindex="0"
            onkeypress="if(event.key==='Enter')openProfile('{profile_key}')">
          <td class="idx">{i}</td>
          <td class="idc">{safe_eid}</td>
          <td><span class="band band-{band or 'p90'}">{band_label}</span></td>
          <td class="scorecell">{_pct_bar(if_pctl, 'if')}</td>
          <td class="scorecell">{_pct_bar(vae_pctl, 'vae')}</td>
          <td class="vars5" title="{safe_top5}">{safe_top5 or '&mdash;'}</td>
          <td class="monthscell">
            <span class="mcount mcount-{sev}">{months_count}/{n_months}</span>
            <span class="mdots">{_month_dots_html(present, all_periods, sev)}</span>
          </td>
          <td class="casecell" onclick="event.stopPropagation()">
            <select class="case-select status-sin_revision" data-profile="{profile_key}"
                    aria-label="Estado del caso {safe_eid}"
                    onkeydown="event.stopPropagation()"
                    onchange="changeCaseStatus('{profile_key}',this.value,event)">
              <option value="sin_revision">Sin revisión</option>
              <option value="en_revision">En revisión</option>
              <option value="cerrado">Cerrado</option>
            </select>
            <small class="case-date" data-case-date="{profile_key}">&mdash;</small>
          </td>
        </tr>""")

        profiles[profile_key] = {
            "id": eid, "band": band,
            "identity": _identity_for(eid),
            "tab": tab,
            "if_score": if_score, "vae_score": vae_score,
            "if_pctl": if_pctl, "vae_pctl": vae_pctl,
            "months": present, "months_count": months_count,
            "if_months": if_months, "vae_months": vae_months,
            "if_top5": if_top5, "vae_top5": vae_top5,
            "records": records_by_entity.get(eid, []),
        }

    rows_html = "\n".join(row_html_list)
    n_rows = len(entity_ids)
    oot_label = ", ".join(all_periods) if all_periods else "(sin periodos)"

    month_label = {p: p for p in all_periods}  # ISO date is already the label

    html_out = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cola de Revisión OOT</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{_CSS}
</style>
</head>
<body>
<div class="page">
  <div class="shell">
    <header class="top">
      <div class="brand">
        <div class="mark">&#8710;</div>
        <div class="brandtext">
          <div class="ttl">Motor de Anomalías &middot; Cola de Revisión</div>
          <div class="sub">Isolation Forest + VAE &middot; OOT {oot_label}</div>
        </div>
      </div>
      <div class="topright">
        <div class="livepill"><span class="pulse"></span>PIPELINE OK</div>
        <div class="clock" id="clock">&mdash;</div>
      </div>
    </header>

    <section class="kpis">
      <div class="kpi">
        <div class="kpi-label" id="queueKpiLabel">En revisión</div>
        <div class="kpi-value"><span id="queueKpiValue">0</span><span class="kpi-unit">individuos</span></div>
        <div class="kpi-sub" id="queueKpiSub">&mdash;</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Recurrentes &ge; 2 de {n_months} meses</div>
        <div class="kpi-value"><span id="recurrentKpiValue">0</span><span class="kpi-unit">de la pestaña</span></div>
        <div class="kpi-sub" id="recurrentKpiSub">&mdash;</div>
      </div>
    </section>

    <section class="main">
      <div class="col-table">
        <div class="panel-head">
          <div class="panel-title">Individuos priorizados<span class="hint">clic en una fila &rarr; perfil completo</span></div>
          <span class="badge" id="rowCountBadge">mostrando {n_rows:,} de {n_total_oot:,}</span>
        </div>
        <p class="tablenote"><b>Definición fija:</b> la cola priorizada contiene scores &ge; percentil 95 del bloque OOT completo. Las pestañas separan quién fue priorizado solo por Isolation Forest, solo por el detector IF+VAE, o por ambos. El estado operativo del caso es independiente del modelo y el perfil permite descargar todo el historial disponible del individuo, no solo OOT.</p>
        <div class="tabs" role="tablist" aria-label="Coincidencia entre detectores">
          <button class="tabbtn" role="tab" data-tab-target="if_only" onclick="setActiveTab('if_only')">Solo IF <span>{counts['if_only']:,}</span></button>
          <button class="tabbtn" role="tab" data-tab-target="vae_only" onclick="setActiveTab('vae_only')">Solo IF+VAE <span>{counts['vae_only']:,}</span></button>
          <button class="tabbtn active" role="tab" data-tab-target="intersection" onclick="setActiveTab('intersection')">Intersección <span>{counts['intersection']:,}</span></button>
          <button class="tabbtn tab-reviewed" role="tab" data-tab-target="reviewed" onclick="setActiveTab('reviewed')">Casos revisados <span id="reviewedCount">0</span></button>
          <button class="export-cases-btn" id="exportCasesBtn" onclick="exportReviewedCases()" disabled>Exportar casos revisados (.csv)</button>
        </div>
        <div class="searchbar">
          <input type="search" id="tableSearch" class="search-input"
                 placeholder="Filtrar por ID, banda o variable..." autocomplete="off"
                 oninput="filterTable(this.value)">
          <span class="search-hint" id="searchHint"></span>
        </div>
        <div class="tablewrap">
          <table id="priorityTable">
            <thead>
              <tr>
                <th></th><th>ID</th><th>Banda</th>
                <th>Percentil IF</th><th>Percentil VAE</th>
                <th>Top-5 variables (salida del modelo)</th><th>Meses ({n_months})</th><th>Estado del caso</th>
              </tr>
            </thead>
            <tbody id="priorityTableBody">{rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <footer class="bottom">
      <div class="legend">
        <span class="litem"><i class="lswatch if"></i>Isolation Forest</span>
        <span class="litem"><i class="lswatch vae"></i>VAE</span>
        <span class="litem"><i class="lswatch dot-crit"></i>{n_months}/{n_months} meses</span>
        <span class="litem"><i class="lswatch dot-warn"></i>&ge;50% de los meses</span>
        <span class="litem"><i class="lswatch dot-mute"></i>1 mes</span>
      </div>
      <div class="footnote">Modelo v0.1 &middot; entregable OOT ({base_model_name}) &middot; uso interno</div>
    </footer>
  </div>
</div>

<div class="overlay" id="overlay" onclick="if(event.target===this)closeProfile()">
  <div class="modal">
    <button class="xclose" onclick="closeProfile()">&#10005;</button>
    <div class="mhead">
      <div>
        <div class="meyebrow">Perfil de individuo priorizado</div>
        <div class="mid" id="mId">&mdash;</div>
        <div class="midentity"><span>{html.escape(identity_column)}</span><b id="mIdentity">No disponible</b></div>
      </div>
      <span class="mband" id="mBand">&mdash;</span>
    </div>
    <div class="case-panel">
      <label for="mCaseStatus">Estado del caso</label>
      <select id="mCaseStatus" onchange="changeActiveCaseStatus(this.value)">
        <option value="sin_revision">Sin revisión</option>
        <option value="en_revision">En revisión</option>
        <option value="cerrado">Cerrado</option>
      </select>
      <small id="mCaseChanged">Sin cambios registrados</small>
    </div>
    <div class="mscores">
      <div class="mscore">
        <div class="mscore-label"><i class="lswatch if"></i>Isolation Forest</div>
        <div class="mscore-val" id="mIfScore">&mdash;</div>
        <div class="pbar pbar-if wide"><i id="mIfBar" style="width:0%"></i></div>
        <div class="mscore-pctl" id="mIfPctl">&mdash;</div>
      </div>
      <div class="mscore">
        <div class="mscore-label"><i class="lswatch vae"></i>VAE</div>
        <div class="mscore-val" id="mVaeScore">&mdash;</div>
        <div class="pbar pbar-vae wide"><i id="mVaeBar" style="width:0%"></i></div>
        <div class="mscore-pctl" id="mVaePctl">&mdash;</div>
      </div>
    </div>
    <div class="mfieldlabel">Presencia en el OOT ({n_months} mes{'es' if n_months != 1 else ''})</div>
    <div class="detector-detail"><span><i class="lswatch if"></i>Isolation Forest</span><div class="mmonths" id="mIfMonths"></div></div>
    <div class="detector-detail"><span><i class="lswatch vae"></i>IF+VAE</span><div class="mmonths" id="mVaeMonths"></div></div>
    <div class="mfieldlabel">Top-5 variables por detector</div>
    <div class="detector-detail"><span><i class="lswatch if"></i>Isolation Forest</span><div class="mchips" id="mIfChips"></div></div>
    <div class="detector-detail"><span><i class="lswatch vae"></i>IF+VAE</span><div class="mchips" id="mVaeChips"></div></div>
    <div class="download-panel">
      <div><b>Historial completo de la entidad</b><small id="mDownloadMeta">&mdash;</small></div>
      <button class="download-btn" id="downloadBtn" onclick="downloadObservation()">Descargar todas las variables (.csv)</button>
    </div>
  </div>
</div>

<script>
var PROFILES = {json.dumps(profiles, ensure_ascii=False)};
var MONTH_LABEL = {json.dumps(month_label, ensure_ascii=False)};
var OOT_MONTHS = {json.dumps(all_periods, ensure_ascii=False)};
var N_MONTHS = {n_months};
var N_ROWS_TOTAL = {n_rows};
var N_TOTAL_OOT = {n_total_oot};
var COUNTS = {json.dumps(counts)};
var RECURRENT_COUNTS = {json.dumps(recurrent_counts)};
var RECORD_COLUMNS = {json.dumps(record_columns, ensure_ascii=False)};
var ACTIVE_TAB = "intersection";
var ACTIVE_PROFILE = null;
var TAB_LABELS = {{if_only:"Solo IF", vae_only:"Solo IF+VAE", intersection:"Intersección", reviewed:"Casos revisados"}};
var STATUS_LABELS = {{sin_revision:"Sin revisión", en_revision:"En revisión", cerrado:"Cerrado"}};
var CASE_STORAGE_KEY = "analyst-case-status:v1:" + window.location.pathname;
var CASES = loadCases();

function pad(n){{return n<10?"0"+n:""+n}}
var MESES=["ENE","FEB","MAR","ABR","MAY","JUN","JUL","AGO","SEP","OCT","NOV","DIC"];
function tick(){{
  try{{
    var d=new Date();
    document.getElementById("clock").textContent =
      pad(d.getDate())+" "+MESES[d.getMonth()]+" "+d.getFullYear()+" · "+
      pad(d.getHours())+":"+pad(d.getMinutes())+":"+pad(d.getSeconds());
  }}catch(e){{}}
}}
tick(); setInterval(tick,1000);

function loadCases(){{
  try{{
    var parsed=JSON.parse(localStorage.getItem(CASE_STORAGE_KEY)||"{{}}");
    return parsed && typeof parsed==="object" ? parsed : {{}};
  }}catch(e){{return {{}};}}
}}
function saveCases(){{
  try{{localStorage.setItem(CASE_STORAGE_KEY,JSON.stringify(CASES));}}catch(e){{}}
}}
function caseFor(profileKey){{
  var r=PROFILES[profileKey];
  return (r && CASES[r.id]) || {{status:"sin_revision",changed_at:null}};
}}
function formatChanged(iso){{
  if(!iso) return "Sin cambios registrados";
  var d=new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString("es-PE",{{dateStyle:"short",timeStyle:"medium"}});
}}
function syncCaseUI(profileKey){{
  var c=caseFor(profileKey);
  var row=document.querySelector('tr[data-profile="'+profileKey+'"]');
  if(row) row.dataset.caseStatus=c.status;
  var select=document.querySelector('.case-select[data-profile="'+profileKey+'"]');
  if(select){{select.value=c.status;select.className="case-select status-"+c.status;}}
  var date=document.querySelector('[data-case-date="'+profileKey+'"]');
  if(date) date.textContent=c.changed_at ? formatChanged(c.changed_at) : "—";
  if(ACTIVE_PROFILE===PROFILES[profileKey]){{
    document.getElementById("mCaseStatus").value=c.status;
    document.getElementById("mCaseChanged").textContent=c.changed_at ? "Último cambio: "+formatChanged(c.changed_at) : "Sin cambios registrados";
  }}
}}
function refreshCaseSummary(){{
  var reviewed=Object.keys(PROFILES).filter(function(k){{return caseFor(k).status!=="sin_revision";}}).length;
  document.getElementById("reviewedCount").textContent=reviewed.toLocaleString("es");
  document.getElementById("exportCasesBtn").disabled=reviewed===0;
  return reviewed;
}}
function changeCaseStatus(profileKey,status,event){{
  if(event) event.stopPropagation();
  if(!STATUS_LABELS[status] || !PROFILES[profileKey]) return;
  var eid=PROFILES[profileKey].id;
  var current=caseFor(profileKey);
  if(current.status===status) return;
  if(status==="sin_revision") delete CASES[eid];
  else CASES[eid]={{status:status,changed_at:new Date().toISOString()}};
  saveCases();syncCaseUI(profileKey);refreshCaseSummary();
  filterTable(document.getElementById("tableSearch").value);
}}
function changeActiveCaseStatus(status){{
  if(!ACTIVE_PROFILE) return;
  var key=Object.keys(PROFILES).find(function(k){{return PROFILES[k]===ACTIVE_PROFILE;}});
  if(key) changeCaseStatus(key,status,null);
}}
Object.keys(PROFILES).forEach(syncCaseUI);
refreshCaseSummary();

// Client-side filter: matches the query against every visible cell in the
// row (ID, banda, percentiles, top-5 variables, meses) -- not just the ID
// column, since an analyst may just as easily search by a variable name or
// a band. Debounced with requestAnimationFrame so typing stays smooth even
// with a few thousand rows.
var _filterPending = null;
function filterTable(query){{
  if (_filterPending) cancelAnimationFrame(_filterPending);
  _filterPending = requestAnimationFrame(function(){{
    var needle = query.trim().toLowerCase();
    var rows = document.querySelectorAll("#priorityTableBody tr[data-id]");
    var shown = 0;
    rows.forEach(function(row){{
      var inTab = ACTIVE_TAB === "reviewed"
        ? row.dataset.caseStatus !== "sin_revision"
        : row.dataset.tab === ACTIVE_TAB;
      var match = inTab && (!needle || row.textContent.toLowerCase().indexOf(needle) !== -1);
      row.hidden = !match;
      if (match) shown++;
    }});
    var emptyRow = document.getElementById("noMatchRow");
    if (needle && shown === 0){{
      if (!emptyRow){{
        emptyRow = document.createElement("tr");
        emptyRow.id = "noMatchRow";
        emptyRow.innerHTML = "<td colspan='8' class='no-match'>"
          + "Ningún individuo coincide con el filtro.</td>";
        document.getElementById("priorityTableBody").appendChild(emptyRow);
      }}
    }} else if (emptyRow) {{
      emptyRow.remove();
    }}
    var badge = document.getElementById("rowCountBadge");
    var hint = document.getElementById("searchHint");
    if (needle) {{
      badge.textContent = "mostrando " + shown + " de " + N_ROWS_TOTAL;
      hint.textContent = shown + " coincidencia" + (shown === 1 ? "" : "s");
    }} else {{
      badge.textContent = "mostrando " + shown.toLocaleString("es") + " de " + N_TOTAL_OOT.toLocaleString("es");
      hint.textContent = "";
    }}
  }});
}}

function setActiveTab(tab){{
  ACTIVE_TAB = tab;
  document.querySelectorAll(".tabbtn").forEach(function(btn){{
    var selected = btn.dataset.tabTarget === tab;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-selected", selected ? "true" : "false");
  }});
  var count = tab === "reviewed" ? refreshCaseSummary() : (COUNTS[tab] || 0);
  var recurrent = tab === "reviewed"
    ? Array.from(document.querySelectorAll('#priorityTableBody tr[data-profile]')).filter(function(row){{
        var r=PROFILES[row.dataset.profile];
        return row.dataset.caseStatus!=="sin_revision" && r && r.months_count>=2;
      }}).length
    : (RECURRENT_COUNTS[tab] || 0);
  var pct = N_TOTAL_OOT ? (100*count/N_TOTAL_OOT) : 0;
  var recurrentPct = count ? (100*recurrent/count) : 0;
  document.getElementById("queueKpiLabel").textContent = TAB_LABELS[tab] + (tab === "reviewed" ? "" : " (≥ P95)");
  document.getElementById("queueKpiValue").textContent = count.toLocaleString("es");
  document.getElementById("queueKpiSub").textContent = tab === "reviewed"
    ? pct.toFixed(1)+"% de "+N_TOTAL_OOT.toLocaleString("es")+" individuos únicos OOT con gestión iniciada"
    : pct.toFixed(1)+"% de "+N_TOTAL_OOT.toLocaleString("es")+" individuos únicos OOT";
  document.getElementById("recurrentKpiValue").textContent = recurrent.toLocaleString("es");
  document.getElementById("recurrentKpiSub").textContent = recurrentPct.toFixed(1)+"% reaparece en 2 o más meses";
  filterTable(document.getElementById("tableSearch").value);
}}

function sevFor(count, total){{
  if (total <= 0 || count <= 0) return "mute";
  var ratio = count / total;
  if (ratio >= 1.0) return "crit";
  if (ratio >= 0.5) return "warn";
  return "mute";
}}

function openProfile(id){{
  var r = PROFILES[id]; if(!r) return;
  ACTIVE_PROFILE = r;
  document.getElementById("mId").textContent = r.id;
  document.getElementById("mIdentity").textContent = r.identity || "No disponible";
  syncCaseUI(id);
  var bandEl = document.getElementById("mBand");
  bandEl.textContent = (r.band || "-").toUpperCase();
  bandEl.className = "mband mband-"+(r.band || "p90");
  document.getElementById("mIfScore").textContent = isNaN(r.if_score) ? "—" : r.if_score.toFixed(3);
  document.getElementById("mIfBar").style.width = (isNaN(r.if_pctl) ? 0 : r.if_pctl)+"%";
  document.getElementById("mIfPctl").textContent = isNaN(r.if_pctl) ? "" : ("percentil "+r.if_pctl.toFixed(1));
  document.getElementById("mVaeScore").textContent = isNaN(r.vae_score) ? "—" : r.vae_score.toFixed(3);
  document.getElementById("mVaeBar").style.width = (isNaN(r.vae_pctl) ? 0 : r.vae_pctl)+"%";
  document.getElementById("mVaePctl").textContent = isNaN(r.vae_pctl) ? "" : ("percentil "+r.vae_pctl.toFixed(1));

  renderMonths("mIfMonths", r.if_months);
  renderMonths("mVaeMonths", r.vae_months);
  renderChips("mIfChips", r.if_top5);
  renderChips("mVaeChips", r.vae_top5);
  var recordCount = (r.records || []).length;
  document.getElementById("mDownloadMeta").textContent = recordCount+" fila"+(recordCount===1?"":"s")+" de todos los periodos · "+RECORD_COLUMNS.length+" columnas originales";
  document.getElementById("downloadBtn").disabled = recordCount === 0 || RECORD_COLUMNS.length === 0;

  document.getElementById("overlay").classList.add("open");
}}
function renderMonths(targetId, present){{
  var mm=document.getElementById(targetId); mm.innerHTML="";
  var sev=sevFor((present||[]).length,N_MONTHS);
  OOT_MONTHS.forEach(function(m){{
    var on=(present||[]).indexOf(m)!==-1;
    var chip=document.createElement("span");
    chip.className="mchip "+(on ? ("mchip-on mchip-"+sev) : "mchip-off");
    chip.textContent=MONTH_LABEL[m]||m; mm.appendChild(chip);
  }});
}}
function renderChips(targetId, text){{
  var mc=document.getElementById(targetId); mc.innerHTML="";
  var values=(text||"").split(",").map(function(s){{return s.trim();}}).filter(Boolean);
  if(!values.length){{var empty=document.createElement("span");empty.className="empty-detail";empty.textContent="No disponible";mc.appendChild(empty);return;}}
  values.forEach(function(v){{var chip=document.createElement("span");chip.className="vchip";chip.textContent=v;mc.appendChild(chip);}});
}}
function csvCell(value){{
  if(value===null || value===undefined) return '""';
  var text=(typeof value === "object") ? JSON.stringify(value) : String(value);
  return '"'+text.replace(/"/g,'""')+'"';
}}
function downloadObservation(){{
  var r=ACTIVE_PROFILE; if(!r || !(r.records||[]).length) return;
  var lines=[RECORD_COLUMNS.map(csvCell).join(",")];
  r.records.forEach(function(row){{lines.push(RECORD_COLUMNS.map(function(c){{return csvCell(row[c]);}}).join(","));}});
  var blob=new Blob(["\\ufeff"+lines.join("\\r\\n")],{{type:"text/csv;charset=utf-8"}});
  var url=URL.createObjectURL(blob); var a=document.createElement("a");
  a.href=url; a.download="entidad_"+String(r.id).replace(/[^a-zA-Z0-9._-]+/g,"_")+"_historial_completo.csv";
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(function(){{URL.revokeObjectURL(url);}},1000);
}}
function exportReviewedCases(){{
  var rows=[];
  Object.keys(PROFILES).forEach(function(key){{
    var r=PROFILES[key], c=caseFor(key);
    if(c.status==="sin_revision") return;
    var d=new Date(c.changed_at);
    rows.push({{
      id:r.id,
      identity:r.identity||"",
      status:STATUS_LABELS[c.status],
      changed_at:c.changed_at||"",
      changed_date:c.changed_at && !isNaN(d.getTime()) ? d.toLocaleDateString("es-PE") : "",
      changed_time:c.changed_at && !isNaN(d.getTime()) ? d.toLocaleTimeString("es-PE") : ""
    }});
  }});
  if(!rows.length) return;
  rows.sort(function(a,b){{return String(b.changed_at).localeCompare(String(a.changed_at));}});
  var headers=[{json.dumps(entity_col, ensure_ascii=False)},{json.dumps(identity_column, ensure_ascii=False)},"estado","fecha_cambio_estado","hora_cambio_estado","timestamp_cambio_estado"];
  var lines=[headers.map(csvCell).join(",")];
  rows.forEach(function(r){{lines.push([r.id,r.identity,r.status,r.changed_date,r.changed_time,r.changed_at].map(csvCell).join(","));}});
  var blob=new Blob(["\\ufeff"+lines.join("\\r\\n")],{{type:"text/csv;charset=utf-8"}});
  var url=URL.createObjectURL(blob), a=document.createElement("a");
  a.href=url;a.download="casos_revisados.csv";document.body.appendChild(a);a.click();a.remove();
  setTimeout(function(){{URL.revokeObjectURL(url);}},1000);
}}
function closeProfile(){{ document.getElementById("overlay").classList.remove("open"); }}
document.addEventListener("keydown", function(e){{ if(e.key==="Escape") closeProfile(); }});
setActiveTab(ACTIVE_TAB);
</script>
</body>
</html>
"""

    parent = os.path.dirname(os.path.abspath(resolved_out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved_out, "w", encoding="utf-8") as fh:
        fh.write(html_out)

    log.info(
        "Analyst dashboard [base=%s] -> %s (%d in P95 union of %d unique OOT "
        "individuals; tabs=%s; recurrent>=2=%s; %d source columns downloadable)",
        base_model_name, resolved_out, n_rows, n_total_oot, counts,
        recurrent_counts, len(record_columns),
    )
    return os.path.abspath(resolved_out)


_CSS = r"""
*,::before,::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0}
table{border-collapse:collapse;width:100%}
button{font:inherit;background:none;border:0;cursor:pointer}

:root{
  --paper:#EAEEF3; --surface:#FFFFFF; --surface-2:#F4F6F9; --border:#DCE2EA;
  --ink:#12141C; --ink-soft:#4A5163; --ink-mute:#8891A3;
  --if:#2a78d6; --if-soft:#E7F0FC;
  --vae:#c4571f; --vae-soft:#FBEBE1;
  --crit:#A8322A; --crit-soft:#F7E4E2;
  --warn:#8A6608; --warn-soft:#F6EDD8;
  --good:#1F8F63; --good-soft:#E2F3EC;
  --mute:#8891A3; --mute-soft:#EEF1F5;
  --shadow:0 24px 60px rgba(20,26,40,.10), 0 2px 10px rgba(20,26,40,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0E1016; --surface:#171A22; --surface-2:#1D2029; --border:#2A2E3A;
    --ink:#EDEFF3; --ink-soft:#AEB4C2; --ink-mute:#767E90;
    --if:#3987e5; --if-soft:#1B2A3E;
    --vae:#d97a3f; --vae-soft:#2E2013;
    --crit:#E0526C; --crit-soft:#3A1E24;
    --warn:#D0A030; --warn-soft:#332912;
    --good:#2FA88C; --good-soft:#123028;
    --mute:#767E90; --mute-soft:#1E212B;
    --shadow:0 24px 60px rgba(0,0,0,.45), 0 2px 10px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --paper:#0E1016; --surface:#171A22; --surface-2:#1D2029; --border:#2A2E3A;
  --ink:#EDEFF3; --ink-soft:#AEB4C2; --ink-mute:#767E90;
  --if:#3987e5; --if-soft:#1B2A3E;
  --vae:#d97a3f; --vae-soft:#2E2013;
  --crit:#E0526C; --crit-soft:#3A1E24;
  --warn:#D0A030; --warn-soft:#332912;
  --good:#2FA88C; --good-soft:#123028;
  --mute:#767E90; --mute-soft:#1E212B;
  --shadow:0 24px 60px rgba(0,0,0,.45), 0 2px 10px rgba(0,0,0,.3);
}

body{background:var(--paper);color:var(--ink);
  font-family:"Public Sans",system-ui,-apple-system,sans-serif}
.page{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.shell{width:100%;max-width:1760px;height:min(94vh,980px);background:var(--surface);
  border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);
  display:flex;flex-direction:column;overflow:hidden}

.top{display:flex;align-items:center;justify-content:space-between;
  padding:16px 28px;border-bottom:1px solid var(--border);flex-shrink:0}
.brand{display:flex;align-items:center;gap:14px}
.mark{width:38px;height:38px;border-radius:10px;background:linear-gradient(160deg,var(--if),#1f4f8c);
  color:#fff;display:flex;align-items:center;justify-content:center;font-size:19px;font-weight:700;flex-shrink:0}
.ttl{font-family:"Archivo",sans-serif;font-weight:800;font-size:18px;letter-spacing:-.01em}
.sub{font-size:11px;color:var(--ink-mute);font-weight:500;margin-top:2px}
.topright{display:flex;align-items:center;gap:20px}
.livepill{display:flex;align-items:center;gap:7px;font-family:"IBM Plex Mono",monospace;
  font-size:11px;font-weight:600;color:var(--good);letter-spacing:.04em}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--good);position:relative}
.pulse::after{content:"";position:absolute;inset:-4px;border-radius:50%;background:var(--good);
  opacity:.35;animation:p 1.8s ease-out infinite}
@keyframes p{0%{transform:scale(.6);opacity:.5}100%{transform:scale(2.4);opacity:0}}
.clock{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-soft);
  font-variant-numeric:tabular-nums}

.kpis{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
.kpi{flex:1;padding:16px 28px;border-left:1px solid var(--border)}
.kpi:first-child{border-left:0}
.kpi-label{font-size:11px;font-weight:600;color:var(--ink-mute);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:6px}
.kpi-value{font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:600;
  font-variant-numeric:tabular-nums;display:flex;align-items:baseline;gap:8px}
.kpi-unit{font-family:"Public Sans",sans-serif;font-size:11px;font-weight:600;color:var(--ink-mute);text-transform:uppercase}
.kpi-sub{font-size:11.5px;color:var(--ink-mute);margin-top:4px}

.main{flex:1;display:flex;overflow:hidden}
.col-table{flex:1;min-width:0;display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;align-items:baseline;justify-content:space-between;padding:16px 24px 10px;flex-shrink:0}
.panel-title{font-family:"Archivo",sans-serif;font-weight:700;font-size:14.5px;display:flex;
  align-items:baseline;gap:10px}
.hint{font-family:"Public Sans",sans-serif;font-weight:500;font-size:10.5px;color:var(--ink-mute)}
.badge{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:500;color:var(--ink-soft);
  background:var(--surface-2);border:1px solid var(--border);padding:3px 9px;border-radius:20px}
.tablenote{margin:0 24px 12px;padding:11px 14px;background:var(--if-soft);border-left:3px solid var(--if);
  border-radius:6px;font-size:11.5px;line-height:1.55;color:var(--ink-soft);flex-shrink:0}
.tablenote b{color:var(--ink)}

.tabs{margin:0 24px 10px;display:flex;gap:7px;flex-wrap:wrap;flex-shrink:0}
.tabbtn{border:1px solid var(--border);border-radius:8px;padding:7px 11px;color:var(--ink-soft);
  background:var(--surface);font-size:11.5px;font-weight:700;display:flex;align-items:center;gap:7px}
.tabbtn span{font-family:"IBM Plex Mono",monospace;font-size:10px;padding:1px 6px;border-radius:10px;
  background:var(--surface-2);color:var(--ink-mute)}
.tabbtn:hover{background:var(--surface-2)}
.tabbtn.active{border-color:var(--if);background:var(--if-soft);color:var(--ink)}
.tabbtn.active span{background:var(--if);color:#fff}
.tabbtn.tab-reviewed.active{border-color:var(--good);background:var(--good-soft)}
.tabbtn.tab-reviewed.active span{background:var(--good);color:#fff}
.export-cases-btn{margin-left:auto;border:1px solid var(--good);border-radius:8px;padding:7px 11px;
  color:var(--good);background:var(--surface);font-size:11.5px;font-weight:700}
.export-cases-btn:hover{background:var(--good-soft)}
.export-cases-btn:disabled{opacity:.4;cursor:not-allowed;background:var(--surface)}

.searchbar{margin:0 24px 10px;display:flex;align-items:center;gap:10px;flex-shrink:0}
.search-input{flex:1;max-width:360px;font-family:"Public Sans",sans-serif;font-size:12.5px;
  padding:8px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface-2);
  color:var(--ink)}
.search-input:focus{outline:2px solid var(--if);outline-offset:1px;background:var(--surface)}
.search-input::placeholder{color:var(--ink-mute)}
.search-hint{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-mute)}

/* The table scrolls inside its own box on a narrow viewport or with a wide
   ID column, instead of the whole page gaining horizontal scroll -- the
   page body must never scroll sideways (see the project's artifact/report
   layout rule). Real entity IDs can run much longer than this project's
   own synthetic "CUST_000123" format, so the ID cell also wraps rather
   than forcing the table wider indefinitely. */
.tablewrap{flex:1;overflow:auto;padding:0 24px 16px}
#priorityTable{min-width:980px}
thead th{position:sticky;top:0;background:var(--surface);z-index:2;text-align:left;
  font-size:10px;font-weight:700;letter-spacing:.06em;color:var(--ink-mute);text-transform:uppercase;
  padding:8px 10px;border-bottom:1px solid var(--border)}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--surface-2)}
tbody tr[hidden]{display:none}
tbody td{padding:9px 10px;border-bottom:1px solid var(--border);font-size:12.5px;vertical-align:middle}
.idx{width:30px;color:var(--ink-mute);font-family:"IBM Plex Mono",monospace;font-size:11px}
.idc{font-family:"IBM Plex Mono",monospace;font-weight:600;overflow-wrap:anywhere;min-width:120px}
.no-match{text-align:center;color:var(--ink-mute);font-size:12.5px;padding:22px 10px !important;
  cursor:default}

.band{font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:700;padding:2px 8px;
  border-radius:5px;letter-spacing:.03em}
.band-p90{color:var(--ink-soft);background:var(--surface-2)}
.band-p95{background:var(--warn-soft);color:var(--warn)}
.band-p99{background:var(--crit-soft);color:var(--crit)}

.scorecell{white-space:nowrap;min-width:120px}
.pbar{display:inline-block;width:56px;height:5px;border-radius:3px;background:var(--surface-2);
  overflow:hidden;vertical-align:middle;margin-right:7px}
.pbar.wide{width:100%;height:7px;margin:8px 0 4px}
.pbar i{display:block;height:100%;border-radius:3px}
.pbar-if i{background:var(--if)}
.pbar-vae i{background:var(--vae)}
.pval{font-family:"IBM Plex Mono",monospace;font-size:11.5px;font-weight:600;
  font-variant-numeric:tabular-nums;vertical-align:middle}

.vars5{font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--ink-soft);
  max-width:0;width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:help}

.monthscell{white-space:nowrap;display:flex;align-items:center;gap:8px}
.mcount{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:2px 6px;border-radius:5px}
.mcount-crit{background:var(--crit-soft);color:var(--crit)}
.mcount-warn{background:var(--warn-soft);color:var(--warn)}
.mcount-mute{background:var(--mute-soft);color:var(--ink-mute)}
.mdots{display:inline-flex;gap:3px}
.mdot{width:7px;height:7px;border-radius:50%;display:inline-block}
.dot-off{background:var(--surface-2);border:1px solid var(--border)}
.dot-on.dot-crit{background:var(--crit)}
.dot-on.dot-warn{background:var(--warn)}
.dot-on.dot-mute{background:var(--ink-mute)}
.casecell{min-width:150px;cursor:default}
.case-select{width:100%;border:1px solid var(--border);border-radius:7px;padding:5px 8px;
  background:var(--surface);color:var(--ink);font-size:11px;font-weight:700}
.case-select.status-en_revision{border-color:var(--warn);background:var(--warn-soft);color:var(--warn)}
.case-select.status-cerrado{border-color:var(--good);background:var(--good-soft);color:var(--good)}
.case-date{display:block;margin-top:4px;color:var(--ink-mute);font-family:"IBM Plex Mono",monospace;font-size:9px}

.bottom{display:flex;align-items:center;justify-content:space-between;padding:12px 28px;
  border-top:1px solid var(--border);flex-shrink:0}
.legend{display:flex;gap:18px;flex-wrap:wrap}
.litem{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--ink-soft);font-weight:600}
.lswatch{width:9px;height:9px;border-radius:3px;display:inline-block}
.lswatch.if{background:var(--if)}.lswatch.vae{background:var(--vae)}
.lswatch.dot-crit{background:var(--crit);border-radius:50%}
.lswatch.dot-warn{background:var(--warn);border-radius:50%}
.lswatch.dot-mute{background:var(--ink-mute);border-radius:50%}
.footnote{font-size:10.5px;color:var(--ink-mute)}

.overlay{position:fixed;inset:0;background:rgba(10,12,20,.55);backdrop-filter:blur(2px);
  display:none;align-items:center;justify-content:center;z-index:100;padding:20px}
.overlay.open{display:flex}
.modal{width:100%;max-width:560px;max-height:88vh;overflow-y:auto;background:var(--surface);
  border-radius:14px;box-shadow:var(--shadow);padding:26px;position:relative}
.xclose{position:absolute;top:18px;right:18px;width:30px;height:30px;border-radius:8px;
  border:1px solid var(--border);color:var(--ink-soft);display:flex;align-items:center;
  justify-content:center;font-size:14px}
.xclose:hover{background:var(--surface-2)}
.mhead{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}
.meyebrow{font-size:10px;font-weight:700;color:var(--ink-mute);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}
.mid{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:700}
.midentity{margin-top:6px;display:flex;gap:7px;align-items:baseline;color:var(--ink-soft);font-size:12px}
.midentity span{color:var(--ink-mute);text-transform:capitalize}.midentity b{font-weight:700}
.mband{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:5px 12px;
  border-radius:20px;text-transform:uppercase;margin-right:36px}
.mband-p90{background:var(--surface-2);color:var(--ink-soft)}
.mband-p95{background:var(--warn-soft);color:var(--warn)}
.mband-p99{background:var(--crit-soft);color:var(--crit)}
.case-panel{display:grid;grid-template-columns:auto minmax(150px,220px) 1fr;gap:10px;align-items:center;
  margin:-4px 0 18px;padding:11px 13px;border:1px solid var(--border);border-radius:10px;background:var(--surface-2)}
.case-panel label{font-size:11px;font-weight:700;color:var(--ink-soft)}
.case-panel select{border:1px solid var(--border);border-radius:7px;padding:6px 8px;background:var(--surface);color:var(--ink)}
.case-panel small{color:var(--ink-mute);font-size:10px;text-align:right}
.mscores{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.mscore{background:var(--surface-2);border-radius:10px;padding:12px 14px}
.mscore-label{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:var(--ink-soft);margin-bottom:6px}
.mscore-val{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:700}
.mscore-pctl{font-size:10.5px;color:var(--ink-mute);margin-top:4px}
.mfieldlabel{font-size:10px;font-weight:700;color:var(--ink-mute);text-transform:uppercase;
  letter-spacing:.06em;margin:16px 0 8px}
.mmonths{display:flex;gap:8px;flex-wrap:wrap}
.mchip{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:6px 14px;
  border-radius:8px;background:var(--surface-2);color:var(--ink-mute)}
.mchip-on.mchip-crit{background:var(--crit-soft);color:var(--crit)}
.mchip-on.mchip-warn{background:var(--warn-soft);color:var(--warn)}
.mchip-on.mchip-mute{background:var(--mute-soft);color:var(--ink-soft)}
.mchips{display:flex;flex-wrap:wrap;gap:7px}
.vchip{font-family:"IBM Plex Mono",monospace;font-size:11px;background:var(--surface-2);
  border:1px solid var(--border);padding:4px 9px;border-radius:6px;color:var(--ink-soft)}
.detector-detail{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:start;
  padding:7px 0;border-bottom:1px solid var(--border)}
.detector-detail>span{display:flex;align-items:center;gap:6px;font-size:10.5px;font-weight:700;color:var(--ink-soft)}
.empty-detail{font-size:11px;color:var(--ink-mute)}
.download-panel{margin-top:18px;padding:13px 14px;border:1px solid var(--border);border-radius:10px;
  background:var(--surface-2);display:flex;align-items:center;justify-content:space-between;gap:16px}
.download-panel b{font-size:11.5px;display:block}.download-panel small{display:block;color:var(--ink-mute);margin-top:3px}
.download-btn{background:var(--ink);color:var(--surface);padding:8px 12px;border-radius:8px;font-size:11px;font-weight:700}
.download-btn:hover{opacity:.86}.download-btn:disabled{opacity:.4;cursor:not-allowed}

@media (max-width:980px){
  .shell{height:auto;max-height:none}
  .kpis{flex-direction:column}
  .kpi{border-left:0;border-top:1px solid var(--border)}
  .kpi:first-child{border-top:0}
  .search-input{max-width:none}
  .export-cases-btn{margin-left:0}
  .case-panel{grid-template-columns:1fr}.case-panel small{text-align:left}
  .download-panel{align-items:stretch;flex-direction:column}.download-btn{width:100%}
}
"""
