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
categorization is invented over `top_5_variables`, and no per-(entity,
period) panel field is shown without its period (none is shown at all).

One dashboard, not one per model: both Isolation Forest and VAE percentiles
are shown for every individual, sourced from each detector's own
`true_oot_entity_scores` -- an in-memory join, not a second exported file,
so this holds regardless of `--stack-iforest-into-vae`.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.oot_report import BAND_COL
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
) -> str:
    """Render the single, unified analyst review-queue dashboard.

    Args:
        table: The table `export_oot_top_anomalies` returned for
            ``base_model_name`` -- already sorted by score descending; row
            order, selection (who is in the queue at all), and the `band`/
            `top_5_variables` columns all come from this one model's export.
        schema: Panel schema (for `entity_col`).
        base_model_name: Which model's export drives selection/order/band/
            `top_5_variables` -- `"iforest"` or `"vae"` (in practice always
            the last entry of `PipelineConfig.deliverable_models`, i.e. VAE).
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
        months_present: `{entity_id: [period_str, ...]}` from
            `months_present_by_entity`, keyed to `base_model_name`'s own P95
            cut-off -- which OOT months this entity's score cleared it in.
        n_total_oot: Total unique individuals in the OOT window (from
            `base_model_name`'s own de-duplicated population) -- the
            denominator for the "en revisión" KPI's percentage.
        score_col: Name of the score column in ``table``.
        out_path: Destination ``.html``. Defaults to
            ``artifacts/reports/analyst_dashboard.html``.

    Returns:
        The absolute path written.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    resolved_out = out_path or paths.ANALYST_DASHBOARD_DEFAULT

    all_periods = [str(p)[:10] for p in oot_periods]
    n_months = max(1, len(all_periods))
    has_vars = "top_5_variables" in table.columns

    records = table.to_dict("records")
    n_reviewed = 0
    n_recurrent = 0
    n_recurrent_full = 0
    row_html_list: list[str] = []
    profiles: dict = {}

    for i, r in enumerate(records, start=1):
        eid = str(r[entity_col])
        band = str(r.get(BAND_COL, ""))
        if band in ("p95", "p99"):
            n_reviewed += 1
        if_pctl = float(if_percentile_by_entity.get(eid, float("nan")))
        vae_pctl = float(vae_percentile_by_entity.get(eid, float("nan")))
        if_score = float(if_score_by_entity.get(eid, float("nan")))
        vae_score = float(vae_score_by_entity.get(eid, float("nan")))
        present = sorted(months_present.get(eid, []))
        months_count = len(present)
        if months_count >= 2:
            n_recurrent += 1
        if months_count >= n_months:
            n_recurrent_full += 1
        sev = _severity_for(months_count, n_months)
        top5 = str(r.get("top_5_variables") or "").strip() if has_vars else ""
        band_label = "P99" if band == "p99" else ("P95" if band == "p95" else band.upper() or "&mdash;")

        row_html_list.append(f"""
        <tr data-id="{eid}" onclick="openProfile('{eid}')" tabindex="0"
            onkeypress="if(event.key==='Enter')openProfile('{eid}')">
          <td class="idx">{i}</td>
          <td class="idc">{eid}</td>
          <td><span class="band band-{band or 'p90'}">{band_label}</span></td>
          <td class="scorecell">{_pct_bar(if_pctl, 'if')}</td>
          <td class="scorecell">{_pct_bar(vae_pctl, 'vae')}</td>
          <td class="vars5" title="{top5}">{top5 or '&mdash;'}</td>
          <td class="monthscell">
            <span class="mcount mcount-{sev}">{months_count}/{n_months}</span>
            <span class="mdots">{_month_dots_html(present, all_periods, sev)}</span>
          </td>
        </tr>""")

        profiles[eid] = {
            "id": eid, "band": band,
            "if_score": if_score, "vae_score": vae_score,
            "if_pctl": if_pctl, "vae_pctl": vae_pctl,
            "months": present, "months_count": months_count,
            "top5": top5,
        }

    rows_html = "\n".join(row_html_list)
    n_rows = len(records)
    p95_pct = (100.0 * n_reviewed / n_total_oot) if n_total_oot else 0.0
    recurrent_rate = (100.0 * n_recurrent / n_rows) if n_rows else 0.0
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
        <div class="kpi-label">En revisión (&ge; P95 del score)</div>
        <div class="kpi-value">{n_reviewed:,}<span class="kpi-unit">individuos</span></div>
        <div class="kpi-sub">{p95_pct:.1f}% de {n_total_oot:,} individuos únicos en el OOT ({n_months} mes{'es' if n_months != 1 else ''})</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Recurrentes &ge; 2 de {n_months} meses</div>
        <div class="kpi-value">{n_recurrent:,}<span class="kpi-unit">de la cola</span></div>
        <div class="kpi-sub">{recurrent_rate:.1f}% reaparece; {n_recurrent_full:,} marcados los {n_months} meses</div>
      </div>
    </section>

    <section class="main">
      <div class="col-table">
        <div class="panel-head">
          <div class="panel-title">Individuos priorizados<span class="hint">clic en una fila &rarr; perfil completo</span></div>
          <span class="badge">mostrando {n_rows:,} de {n_total_oot:,}</span>
        </div>
        <p class="tablenote"><b>Definición fija:</b> "en revisión" = score &ge; percentil 95 del bloque OOT completo (no un umbral calibrado que pueda caer a cero); el conteo son individuos únicos, no filas mes-a-mes. Todo lo mostrado abajo es una columna que Modelo v0.1 produce directamente (ID, scores, percentiles, banda, top-5 variables, presencia mensual) &mdash; sin categorización de negocio añadida.</p>
        <div class="tablewrap">
          <table>
            <thead>
              <tr>
                <th></th><th>ID</th><th>Banda</th>
                <th>Percentil IF</th><th>Percentil VAE</th>
                <th>Top-5 variables (salida del modelo)</th><th>Meses ({n_months})</th>
              </tr>
            </thead>
            <tbody>{rows_html}
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
      </div>
      <span class="mband" id="mBand">&mdash;</span>
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
    <div class="mmonths" id="mMonths"></div>
    <div class="mfieldlabel">Top-5 variables que más empujan el score</div>
    <div class="mchips" id="mChips"></div>
  </div>
</div>

<script>
var PROFILES = {json.dumps(profiles, ensure_ascii=False)};
var MONTH_LABEL = {json.dumps(month_label, ensure_ascii=False)};
var OOT_MONTHS = {json.dumps(all_periods, ensure_ascii=False)};
var N_MONTHS = {n_months};

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

function sevFor(count, total){{
  if (total <= 0 || count <= 0) return "mute";
  var ratio = count / total;
  if (ratio >= 1.0) return "crit";
  if (ratio >= 0.5) return "warn";
  return "mute";
}}

function openProfile(id){{
  var r = PROFILES[id]; if(!r) return;
  document.getElementById("mId").textContent = id;
  var bandEl = document.getElementById("mBand");
  bandEl.textContent = (r.band || "-").toUpperCase();
  bandEl.className = "mband mband-"+(r.band || "p90");
  document.getElementById("mIfScore").textContent = isNaN(r.if_score) ? "—" : r.if_score.toFixed(3);
  document.getElementById("mIfBar").style.width = (isNaN(r.if_pctl) ? 0 : r.if_pctl)+"%";
  document.getElementById("mIfPctl").textContent = isNaN(r.if_pctl) ? "" : ("percentil "+r.if_pctl.toFixed(1));
  document.getElementById("mVaeScore").textContent = isNaN(r.vae_score) ? "—" : r.vae_score.toFixed(3);
  document.getElementById("mVaeBar").style.width = (isNaN(r.vae_pctl) ? 0 : r.vae_pctl)+"%";
  document.getElementById("mVaePctl").textContent = isNaN(r.vae_pctl) ? "" : ("percentil "+r.vae_pctl.toFixed(1));

  var sev = sevFor(r.months_count, N_MONTHS);
  var mm = document.getElementById("mMonths"); mm.innerHTML = "";
  OOT_MONTHS.forEach(function(m){{
    var on = r.months.indexOf(m) !== -1;
    var sevClass = on ? ("mchip-on mchip-"+sev) : "mchip-off";
    mm.innerHTML += '<span class="mchip '+sevClass+'">'+(MONTH_LABEL[m]||m)+'</span>';
  }});

  var mc = document.getElementById("mChips"); mc.innerHTML = "";
  (r.top5 ? r.top5.split(",").map(function(s){{return s.trim();}}).filter(Boolean) : []).forEach(function(v){{
    mc.innerHTML += '<span class="vchip">'+v+'</span>';
  }});

  document.getElementById("overlay").classList.add("open");
}}
function closeProfile(){{ document.getElementById("overlay").classList.remove("open"); }}
document.addEventListener("keydown", function(e){{ if(e.key==="Escape") closeProfile(); }});
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
        "Analyst dashboard [base=%s] -> %s (%d in review of %d unique OOT individuals, "
        "%d recurrent >=2 of %d months)",
        base_model_name, resolved_out, n_reviewed, n_total_oot, n_recurrent, n_months,
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

.tablewrap{flex:1;overflow-y:auto;padding:0 24px 16px}
thead th{position:sticky;top:0;background:var(--surface);z-index:2;text-align:left;
  font-size:10px;font-weight:700;letter-spacing:.06em;color:var(--ink-mute);text-transform:uppercase;
  padding:8px 10px;border-bottom:1px solid var(--border)}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--surface-2)}
tbody td{padding:9px 10px;border-bottom:1px solid var(--border);font-size:12.5px;vertical-align:middle}
.idx{width:30px;color:var(--ink-mute);font-family:"IBM Plex Mono",monospace;font-size:11px}
.idc{font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap}

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
.mband{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:700;padding:5px 12px;
  border-radius:20px;text-transform:uppercase}
.mband-p90{background:var(--surface-2);color:var(--ink-soft)}
.mband-p95{background:var(--warn-soft);color:var(--warn)}
.mband-p99{background:var(--crit-soft);color:var(--crit)}
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

@media (max-width:980px){
  .shell{height:auto;max-height:none}
  .kpis{flex-direction:column}
  .kpi{border-left:0;border-top:1px solid var(--border)}
  .kpi:first-child{border-top:0}
}
"""
