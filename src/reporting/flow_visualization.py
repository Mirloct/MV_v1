"""n8n/Databricks-style flow visualization, built from real execution events.

Reads the JSON Lines stream `src.utils.observability` writes
(`artifacts/logs/run_events.jsonl`) and renders one self-contained HTML file
(inline CSS + vanilla JS, no CDN/network -- same offline-artifact convention
as `src.reporting.report`) showing the pipeline's top-level phases as
connected nodes, colored by real status, with a replay control that steps
through the actual recorded event order.

**Nothing here invents state.** Every node's status, duration, and ordering
comes directly from `phase_started`/`phase_completed`/`phase_failed` events
already written by `log_phase` (`src/utils/logging_config.py`) for every
phase in the codebase, not just `main.py`'s. The only thing that is not
"real" is playback *pacing* during the animated replay (a fixed per-step
delay, like a video scrubber) -- the sequence and every duration/metric shown
is exactly what happened.

Grouping rule: a "node" is any phase whose name matches `^Phase \\d+` (the
convention every `main.py` top-level phase uses, e.g. "Phase 6: Isolation
Forest"). Every other logged phase (`iforest.fit`, `preprocessing.
fit_transform_panel`, ...) is *nested* -- attributed, via file order (this
JSONL is strictly append-only and single-process, so line order is true
emission order), to whichever top-level node was open when it fired. A
`[modelname]` suffix in a phase name (e.g. "Phase 8: evaluation [vae]") is
parsed into a `model` tag so the per-model phases 8/8b/9/10 render distinctly
without needing a hardcoded phase list -- this file has no knowledge of
`main.py`'s phase names beyond that one regex convention, so it keeps working
if phases are added, renamed, or reordered.

Two distinct deliverables, both built from the same `_read_events`/
`_select_run`/`_build_nodes` pipeline:

* :func:`build_flow_visualization` -- a static HTML file written *after* a
  run ends, with a manual replay control (Play/Step/Reset) over the
  now-complete event history. For later review/sharing.
* :func:`start_live_view` -- a background HTTP server on `127.0.0.1` only,
  started *before* the run's first phase, whose single page polls `/state`
  every second and shows nodes appearing/completing as they actually happen.
  For watching a run live, locally, in a normal browser tab -- never
  published anywhere (no claude.ai Artifact, no non-loopback network
  exposure).
"""

from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from src.utils import paths
from src.utils.logging_config import setup_logging

__all__ = ["build_flow_visualization", "start_live_view"]

_PHASE_RE = re.compile(r"^Phase\s+\d+")
_MODEL_TAG_RE = re.compile(r"\[(\w+)\]\s*$")


def _read_events(events_path: str) -> list[dict]:
    events = []
    if not os.path.isfile(events_path):
        return events
    with open(events_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a mid-write crash is expected, not fatal
    return events


def _select_run(events: list[dict], run_id: Optional[str]) -> list[dict]:
    if not events:
        return []
    target = run_id or events[-1].get("run_id")
    return [e for e in events if e.get("run_id") == target]


def _model_tag(phase_name: str) -> tuple[str, Optional[str]]:
    m = _MODEL_TAG_RE.search(phase_name)
    if not m:
        return phase_name, None
    return phase_name[: m.start()].rstrip(), m.group(1)


def _build_nodes(events: list[dict]) -> dict:
    """Single linear scan over `events` (already in true emission order)."""
    nodes: list[dict] = []
    by_name: dict[str, dict] = {}
    current: Optional[dict] = None
    run_meta: dict[str, Any] = {}
    all_health_checks: list[dict] = []
    dataset_fp: Optional[dict] = None

    for ev in events:
        kind = ev.get("event")

        if kind == "run_started":
            run_meta.update({
                "run_id": ev.get("run_id"), "config": ev.get("config"),
                "config_hash": ev.get("config_hash"), "seed": ev.get("seed"),
                "pipeline_version": ev.get("pipeline_version"),
                "python_version": ev.get("python_version"), "platform": ev.get("platform"),
                "started_at": ev.get("ts"), "status": "running",
            })
            continue
        if kind == "run_ended":
            run_meta["status"] = ev.get("status")
            run_meta["ended_at"] = ev.get("ts")
            run_meta["summary"] = ev.get("summary")
            run_meta["error"] = ev.get("error")
            continue
        if kind == "dataset_fingerprint":
            dataset_fp = {k: v for k, v in ev.items() if k not in ("event", "run_id")}
            continue
        if kind == "health_check":
            all_health_checks.append(ev)
            continue
        if kind == "tuning_trial":
            # Attributed to whichever top-level node was open when it fired
            # (Phase 6/7 for this project's `tune_iforest`/`tune_vae`), the
            # same file-order rule used for nested sub_events -- not matched
            # by model-name text, which would silently miss unsuffixed nodes.
            if current is not None:
                current.setdefault("tuning_trials", []).append(ev)
            continue

        phase = ev.get("phase")
        if not phase:
            continue
        is_top = bool(_PHASE_RE.match(phase))

        if is_top:
            if kind == "phase_started":
                display_name, model = _model_tag(phase)
                node = {
                    "name": phase, "display_name": display_name, "model": model,
                    "status": "running", "started_at": ev.get("ts"),
                    "duration_s": None, "sub_events": [], "tuning_trials": [],
                    "order": len(nodes),
                }
                nodes.append(node)
                by_name[phase] = node
                current = node
            elif kind in ("phase_completed", "phase_failed"):
                node = by_name.get(phase)
                if node is not None:
                    node["status"] = "completed" if kind == "phase_completed" else "failed"
                    node["duration_s"] = ev.get("duration_s")
                if current is node:
                    current = None
        else:
            target = current
            if target is not None:
                target["sub_events"].append({
                    "phase": phase, "event": kind,
                    "duration_s": ev.get("duration_s"), "ts": ev.get("ts"),
                })

    # A node still "running" once the run itself has ended never completed --
    # the run was cancelled or killed mid-phase. Leaving it as "running" in a
    # *static, after-the-fact* diagram would assert something false about a
    # finished run, so relabel it. (During a live run `run_meta["status"]` is
    # "running" and this correctly does nothing.)
    if run_meta.get("status") in ("cancelled", "failed"):
        for node in nodes:
            if node["status"] == "running":
                node["status"] = "interrupted"

    return {
        "run": run_meta,
        "dataset": dataset_fp,
        "nodes": nodes,
        "health_checks": all_health_checks,
    }


def _render_html(data: dict) -> str:
    run = data["run"]
    payload = json.dumps(data, default=str)
    # A stable product name, not a caption -- the run_id/status live in the
    # visible header instead, per the artifact-design convention that a
    # <title> should read like a name in a gallery, not an identifier.
    title = "Detection Pipeline Flow"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
/* Light (default / un-stamped): cool slate-navy family, not warm cream --
   matches the dark theme's hue rather than switching temperature. */
:root {{
  --bg: #f3f5fa; --panel: #ffffff; --border: #d8dcea; --text: #161d2e;
  --muted: #5b6479; --accent: #a8680f; --accent-ink: #ffffff;
  --ok: #2f8f5b; --bad: #c8453f; --ok-bg: #2f8f5b1a; --bad-bg: #c8453f1a;
}}
/* System dark preference wins only when no explicit choice was stamped. */
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #10141f; --panel: #171e2d; --border: #2a3348; --text: #e7ebf5;
    --muted: #8b93ab; --accent: #e8a33d; --accent-ink: #171e2d;
    --ok: #4caf7d; --bad: #dd685f; --ok-bg: #4caf7d26; --bad-bg: #dd685f26;
  }}
}}
/* Explicit dark choice always wins, regardless of OS preference. */
:root[data-theme="dark"] {{
  --bg: #10141f; --panel: #171e2d; --border: #2a3348; --text: #e7ebf5;
  --muted: #8b93ab; --accent: #e8a33d; --accent-ink: #171e2d;
  --ok: #4caf7d; --bad: #dd685f; --ok-bg: #4caf7d26; --bad-bg: #dd685f26;
}}
* {{ box-sizing: border-box; }}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }}
}}
body {{
  margin: 0; padding: clamp(16px, 3vw, 32px); background: var(--bg); color: var(--text);
  font: 14px/1.55 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}
code, .mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "Consolas", monospace; }}
h1 {{ font-size: 19px; margin: 0 0 4px; font-weight: 650; text-wrap: balance; }}
h1 code {{ font-size: 15px; font-weight: 500; color: var(--accent); }}
.sub {{ color: var(--muted); font-size: 12.5px; margin-bottom: 22px; }}
.panel {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px; margin-bottom: 16px;
}}
.panel h3 {{ margin: 0 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 600; }}
.controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 18px; flex-wrap: wrap; }}
button {{
  background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 7px; padding: 7px 14px; cursor: pointer; font-size: 13px; font-weight: 500;
}}
button:hover {{ border-color: var(--accent); color: var(--accent); }}
button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
button:disabled {{ opacity: 0.4; cursor: default; }}
.progress-bar {{ flex: 1; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; min-width: 120px; }}
.progress-fill {{ height: 100%; background: var(--accent); width: 0%; transition: width 0.35s ease; }}
.meta {{ color: var(--muted); font-size: 12px; }}
.meta, .node-status, td, th {{ font-variant-numeric: tabular-nums; }}
.flow {{ display: flex; flex-wrap: wrap; gap: 0; align-items: stretch; }}
.node-wrap {{ display: flex; align-items: center; }}
.node {{
  border: 1px solid var(--border); border-left: 3px solid var(--border); border-radius: 9px;
  padding: 10px 14px; min-width: 172px; max-width: 220px; background: var(--panel);
  opacity: 0.32; transition: opacity 0.35s ease, border-color 0.25s ease, box-shadow 0.25s ease, transform 0.15s ease;
  cursor: pointer;
}}
.node:hover {{ transform: translateY(-1px); border-color: var(--accent); }}
.node:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.node.revealed {{ opacity: 1; }}
.node.active {{ border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); }}
.node.st-completed {{ border-left-color: var(--ok); }}
.node.st-failed, .node.st-interrupted {{ border-left-color: var(--bad); }}
.node.st-running {{ border-left-color: var(--accent); }}
.node-title {{ font-weight: 600; font-size: 12.5px; margin-bottom: 5px; }}
.node-model {{
  display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 9px;
  margin-left: 6px; background: var(--border); color: var(--text); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.03em;
}}
.node-status {{ display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted); }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--muted); }}
.dot-completed {{ background: var(--ok); }}
.dot-failed, .dot-interrupted {{ background: var(--bad); }}
.dot-running {{ background: var(--accent); }}
.arrow {{ color: var(--muted); font-size: 18px; padding: 0 6px; flex: none; }}
.detail {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-top: 16px; display: none; }}
.detail.shown {{ display: block; }}
.detail h3 {{ margin: 0 0 8px; font-size: 15px; text-transform: none; letter-spacing: 0; color: var(--text); }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--muted); font-weight: 500; }}
td.mono, th.mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "Consolas", monospace; }}
.badge {{ display: inline-block; padding: 1px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; }}
.badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
.badge.bad {{ background: var(--bad-bg); color: var(--bad); }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 700px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Detection pipeline flow &nbsp;<code id="run-id-title"></code></h1>
<div class="sub" id="run-sub"></div>

<div class="panel">
  <h3>Phase timeline</h3>
  <div class="controls">
    <button id="btn-reset">Reset</button>
    <button id="btn-step-back">&laquo; Step</button>
    <button id="btn-play">Play</button>
    <button id="btn-step">Step &raquo;</button>
    <button id="btn-end">Show all</button>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <span class="meta" id="progress-label">0 / 0</span>
    <span class="meta" id="elapsed-label"></span>
  </div>
  <div class="flow" id="flow"></div>
</div>

<div class="detail" id="detail"></div>

<div class="grid2">
  <div class="panel">
    <h3>Run summary</h3>
    <div id="run-summary"></div>
  </div>
  <div class="panel">
    <h3>Health checks</h3>
    <div id="health-summary"></div>
  </div>
</div>

<script>
const DATA = {payload};

const flowEl = document.getElementById("flow");
const detailEl = document.getElementById("detail");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const elapsedLabel = document.getElementById("elapsed-label");

function esc(s) {{
  const d = document.createElement("div");
  d.textContent = s === null || s === undefined ? "" : String(s);
  return d.innerHTML;
}}

document.getElementById("run-id-title").textContent = DATA.run.run_id || "(unknown)";
document.getElementById("run-sub").textContent =
  "config_hash=" + (DATA.run.config_hash || "?") +
  " | seed=" + DATA.run.seed +
  " | " + (DATA.run.python_version || "") + " on " + (DATA.run.platform || "") +
  " | status=" + (DATA.run.status || "?");

let revealCount = 0;
let playTimer = null;
const nodes = DATA.nodes;

function statusDot(status) {{
  return '<span class="dot dot-' + status + '"></span>';
}}

function fmtDur(s) {{
  if (s === null || s === undefined) return "--";
  return s < 1 ? (s * 1000).toFixed(0) + "ms" : s.toFixed(2) + "s";
}}

function renderFlow() {{
  flowEl.innerHTML = "";
  nodes.forEach((n, i) => {{
    if (i > 0) {{
      const arrow = document.createElement("div");
      arrow.className = "arrow";
      arrow.textContent = "\\u2192";
      flowEl.appendChild(arrow);
    }}
    const wrap = document.createElement("div");
    wrap.className = "node-wrap";
    const displayStatus = i < revealCount ? n.status : "pending";
    const el = document.createElement("div");
    el.className = "node st-" + displayStatus +
      (i < revealCount ? " revealed" : "") + (i === revealCount - 1 ? " active" : "");
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.innerHTML =
      '<div class="node-title">' + esc(n.display_name) +
      (n.model ? '<span class="node-model">' + esc(n.model) + '</span>' : '') + '</div>' +
      '<div class="node-status">' + statusDot(displayStatus) + displayStatus +
      (n.duration_s !== null && i < revealCount ? ' &middot; ' + fmtDur(n.duration_s) : '') + '</div>';
    el.addEventListener("click", () => showDetail(n, displayStatus));
    el.addEventListener("keydown", (e) => {{
      if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); showDetail(n, displayStatus); }}
    }});
    wrap.appendChild(el);
    flowEl.appendChild(wrap);
  }});
  const pct = nodes.length ? Math.round(100 * revealCount / nodes.length) : 0;
  progressFill.style.width = pct + "%";
  progressLabel.textContent = revealCount + " / " + nodes.length + " phases";
  const elapsed = nodes.slice(0, revealCount).reduce((a, n) => a + (n.duration_s || 0), 0);
  elapsedLabel.textContent = "cumulative real duration shown: " + elapsed.toFixed(2) + "s";
}}

function showDetail(node, displayStatus) {{
  detailEl.classList.add("shown");
  const statusBadge = displayStatus === "completed" ? '<span class="badge ok">completed</span>'
    : displayStatus === "failed" ? '<span class="badge bad">failed</span>'
    : '<span class="badge">' + displayStatus + '</span>';
  let rows = node.sub_events.map(se =>
    '<tr><td class="mono">' + esc(se.phase) + '</td><td>' + esc(se.event) + '</td><td class="mono">' + fmtDur(se.duration_s) + '</td></tr>'
  ).join("");
  let trials = "";
  const t = node.tuning_trials || [];
  if (t.length) {{
    trials = "<h3>Optuna trials logged during this phase (" + t.length + ")</h3><div class=\\"table-wrap\\"><table><tr><th>#</th><th>value</th><th>best_value</th><th>no-improve streak</th></tr>" +
      t.map(x => '<tr><td class="mono">' + x.trial_number + '</td><td class="mono">' + (x.value !== null && x.value !== undefined ? x.value.toFixed(6) : "--") + '</td><td class="mono">' + (x.best_value !== null && x.best_value !== undefined ? x.best_value.toFixed(6) : "--") + '</td><td class="mono">' + x.trials_since_improvement + '</td></tr>').join("") +
      "</table></div>";
  }}
  detailEl.innerHTML =
    "<h3>" + esc(node.display_name) + (node.model ? '<span class="node-model">' + esc(node.model) + '</span>' : "") + "</h3>" +
    "<p class=\\"meta\\">" + statusBadge + " &middot; duration=" + fmtDur(node.duration_s) + " &middot; started_at=<span class=\\"mono\\">" + esc(node.started_at) + "</span></p>" +
    (rows ? '<div class="table-wrap"><table><tr><th>nested phase</th><th>event</th><th>duration</th></tr>' + rows + "</table></div>" : "<p class=\\"meta\\">No nested phases logged under this node.</p>") +
    trials;
}}

function renderSummary() {{
  const s = DATA.run.summary || {{}};
  const statusBadge = DATA.run.status === "success"
    ? '<span class="badge ok">success</span>'
    : '<span class="badge bad">' + (DATA.run.status || "unknown") + '</span>';
  const rows = [
    ["Status", statusBadge],
    ["Total health checks", s.total_checks ?? "?"],
    ["Failed checks", (s.failed_checks || []).length ? (s.failed_checks || []).join(", ") : "none"],
    ["Peak Python memory", s.peak_python_memory_mb !== undefined ? s.peak_python_memory_mb + " MB" : "?"],
    ["Dataset", DATA.dataset ? (DATA.dataset.n_rows + " rows x " + DATA.dataset.n_cols + " cols") : "?"],
  ];
  if (DATA.run.error) rows.push(["Error", '<span class="mono">' + esc(DATA.run.error) + '</span>']);
  document.getElementById("run-summary").innerHTML = '<div class="table-wrap"><table>' +
    rows.map(r => "<tr><th>" + r[0] + "</th><td>" + r[1] + "</td></tr>").join("") + "</table></div>";

  const byCat = {{}};
  (DATA.health_checks || []).forEach(h => {{
    byCat[h.category] = byCat[h.category] || {{pass: 0, fail: 0}};
    byCat[h.category][h.passed ? "pass" : "fail"] += 1;
  }});
  let tbl = "<table><tr><th>category</th><th>passed</th><th>failed</th></tr>";
  Object.keys(byCat).sort().forEach(cat => {{
    tbl += "<tr><td>" + cat + "</td><td>" + byCat[cat].pass + "</td><td>" +
      (byCat[cat].fail ? '<span class="badge bad">' + byCat[cat].fail + '</span>' : '<span class="badge ok">0</span>') + "</td></tr>";
  }});
  tbl += "</table>";
  document.getElementById("health-summary").innerHTML = '<div class="table-wrap">' + tbl + '</div>';
}}

function step(delta) {{
  revealCount = Math.max(0, Math.min(nodes.length, revealCount + delta));
  renderFlow();
}}

document.getElementById("btn-step").addEventListener("click", () => step(1));
document.getElementById("btn-step-back").addEventListener("click", () => step(-1));
document.getElementById("btn-reset").addEventListener("click", () => {{ stopPlay(); revealCount = 0; renderFlow(); }});
document.getElementById("btn-end").addEventListener("click", () => {{ stopPlay(); revealCount = nodes.length; renderFlow(); }});

function stopPlay() {{
  if (playTimer) {{ clearInterval(playTimer); playTimer = null; }}
  document.getElementById("btn-play").textContent = "Play";
}}
document.getElementById("btn-play").addEventListener("click", () => {{
  if (playTimer) {{ stopPlay(); return; }}
  document.getElementById("btn-play").textContent = "Pause";
  playTimer = setInterval(() => {{
    if (revealCount >= nodes.length) {{ stopPlay(); return; }}
    step(1);
  }}, 700);
}});

renderFlow();
renderSummary();
</script>
</body>
</html>
"""


def build_flow_visualization(
    events_path: str = paths.RUN_EVENTS_LOG,
    out_path: str = paths.FLOW_VISUALIZATION_DEFAULT,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """Render the n8n-style flow visualization for one run and write it as HTML.

    Args:
        events_path: JSONL event stream written by `src.utils.observability`.
        out_path: Destination `.html`.
        run_id: Which run to render. `None` (default) renders the most recent
            run found in the file (the one the last line belongs to).

    Returns:
        The absolute path written, or `None` if `events_path` has no events
        yet (nothing to render -- not an error, e.g. a run that failed before
        `start_run` was ever called).
    """
    log = setup_logging()
    events = _read_events(events_path)
    events = _select_run(events, run_id)
    if not events:
        log.warning(
            "No events found in %s for run_id=%r; skipping flow visualization.",
            events_path, run_id,
        )
        return None

    data = _build_nodes(events)
    if not data["nodes"]:
        log.warning(
            "Run %s has events but no top-level 'Phase N: ...' nodes; "
            "skipping flow visualization.", data["run"].get("run_id"),
        )
        return None

    out_html = _render_html(data)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out_html)

    log.info(
        "Flow visualization: %d node(s), run=%s, status=%s -> %s",
        len(data["nodes"]), data["run"].get("run_id"), data["run"].get("status"), out_path,
    )
    return os.path.abspath(out_path)


# --------------------------------------------------------------------------- #
# Live local view: a background HTTP server on localhost only, polled by a   #
# small page while the pipeline is still running -- not a claude.ai Artifact #
# publish (nothing here ever leaves the machine) and not the static replay   #
# file above (that one is written once, after the run ends).                 #
# --------------------------------------------------------------------------- #
_LIVE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Detection Pipeline Flow -- Live</title>
<style>
:root {
  --bg: #f3f5fa; --panel: #ffffff; --border: #d8dcea; --text: #161d2e;
  --muted: #5b6479; --accent: #a8680f;
  --ok: #2f8f5b; --bad: #c8453f; --ok-bg: #2f8f5b1a; --bad-bg: #c8453f1a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #10141f; --panel: #171e2d; --border: #2a3348; --text: #e7ebf5;
    --muted: #8b93ab; --accent: #e8a33d;
    --ok: #4caf7d; --bad: #dd685f; --ok-bg: #4caf7d26; --bad-bg: #dd685f26;
  }
}
:root[data-theme="dark"] {
  --bg: #10141f; --panel: #171e2d; --border: #2a3348; --text: #e7ebf5;
  --muted: #8b93ab; --accent: #e8a33d;
  --ok: #4caf7d; --bad: #dd685f; --ok-bg: #4caf7d26; --bad-bg: #dd685f26;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: clamp(16px, 3vw, 32px); background: var(--bg); color: var(--text);
  font: 14px/1.55 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
code, .mono { font-family: ui-monospace, "SF Mono", "Cascadia Code", "Consolas", monospace; }
h1 { font-size: 19px; margin: 0 0 4px; font-weight: 650; display: flex; align-items: center; gap: 10px; }
.live-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); animation: pulse 1.4s ease-in-out infinite; }
.live-dot.stopped { animation: none; background: var(--muted); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
@media (prefers-reduced-motion: reduce) { .live-dot { animation: none; } }
.sub { color: var(--muted); font-size: 12.5px; margin-bottom: 22px; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 16px; }
.panel h3 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); font-weight: 600; }
.flow { display: flex; flex-wrap: wrap; gap: 0; align-items: stretch; }
.node-wrap { display: flex; align-items: center; }
.node {
  border: 1px solid var(--border); border-left: 3px solid var(--border); border-radius: 9px;
  padding: 10px 14px; min-width: 172px; max-width: 220px; background: var(--panel);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.node.st-completed { border-left-color: var(--ok); }
.node.st-failed { border-left-color: var(--bad); }
.node.st-running { border-left-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); }
.node-title { font-weight: 600; font-size: 12.5px; margin-bottom: 5px; }
.node-model {
  display: inline-block; font-size: 10px; padding: 1px 7px; border-radius: 9px; margin-left: 6px;
  background: var(--border); color: var(--text); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em;
}
.node-status { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--muted); }
.dot-completed { background: var(--ok); }
.dot-failed { background: var(--bad); }
.dot-running { background: var(--accent); }
.arrow { color: var(--muted); font-size: 18px; padding: 0 6px; flex: none; }
.meta { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 1px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; }
.badge.ok { background: var(--ok-bg); color: var(--ok); }
.badge.bad { background: var(--bad-bg); color: var(--bad); }
.empty { color: var(--muted); font-style: italic; padding: 8px 0; }

/* -- progress header ------------------------------------------------------ */
.progress-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; }
.progress-bar {
  flex: 1; min-width: 180px; height: 8px; background: var(--border);
  border-radius: 5px; overflow: hidden;
}
.progress-fill {
  height: 100%; width: 0%; background: var(--accent);
  transition: width .45s cubic-bezier(.4,0,.2,1);
}
.progress-pct { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; min-width: 62px; }
.now-running {
  display: flex; align-items: center; gap: 9px; font-size: 13px;
  color: var(--text); padding: 11px 14px; border-radius: 9px;
  background: var(--panel); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); margin-bottom: 16px;
}
.now-running .spinner {
  width: 13px; height: 13px; flex: none; border-radius: 50%;
  border: 2px solid var(--border); border-top-color: var(--accent);
  animation: spin .8s linear infinite;
}
.now-running.done .spinner { animation: none; border-top-color: var(--ok); border-color: var(--ok); }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
  .now-running .spinner { animation: none; }
  .progress-fill { transition: none; }
}
.now-label { color: var(--muted); font-size: 11px; text-transform: uppercase;
  letter-spacing: .05em; font-weight: 650; }
.elapsed { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<h1><span class="live-dot" id="live-dot"></span> Detection pipeline flow -- live</h1>
<div class="sub" id="run-sub">Connecting...</div>

<div class="panel">
  <div class="progress-row">
    <span class="progress-pct" id="progress-pct">0%</span>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <span class="elapsed" id="elapsed">--</span>
  </div>
  <div class="now-running" id="now-running">
    <span class="spinner"></span>
    <span><span class="now-label">Current phase</span><br><span id="now-phase">waiting…</span></span>
  </div>
  <h3>Phase timeline (auto-refreshing)</h3>
  <div class="flow" id="flow"><div class="empty">Waiting for the first event...</div></div>
</div>
<div class="panel">
  <h3>Run status</h3>
  <div id="run-summary" class="meta">--</div>
</div>
<script>
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s === null || s === undefined ? "" : String(s);
  return d.innerHTML;
}
function fmtDur(s) {
  if (s === null || s === undefined) return "--";
  return s < 1 ? (s * 1000).toFixed(0) + "ms" : s.toFixed(2) + "s";
}
function statusDot(status) { return '<span class="dot dot-' + status + '"></span>'; }

let stopped = false;
let sawTerminalStatus = false;
let failedPolls = 0;
const TERMINAL = ["success", "failed", "cancelled"];

function render(data) {
  const nodes = data.nodes || [];
  const flowEl = document.getElementById("flow");
  if (!nodes.length) {
    flowEl.innerHTML = '<div class="empty">Waiting for the first phase to start...</div>';
  } else {
    flowEl.innerHTML = "";
    nodes.forEach((n, i) => {
      if (i > 0) {
        const arrow = document.createElement("div");
        arrow.className = "arrow";
        arrow.textContent = "\\u2192";
        flowEl.appendChild(arrow);
      }
      const wrap = document.createElement("div");
      wrap.className = "node-wrap";
      const el = document.createElement("div");
      el.className = "node st-" + n.status;
      el.innerHTML =
        '<div class="node-title">' + esc(n.display_name) +
        (n.model ? '<span class="node-model">' + esc(n.model) + '</span>' : '') + '</div>' +
        '<div class="node-status">' + statusDot(n.status) + n.status +
        (n.duration_s !== null ? ' &middot; ' + fmtDur(n.duration_s) : '') + '</div>';
      wrap.appendChild(el);
      flowEl.appendChild(wrap);
    });
  }

  const run = data.run || {};

  /* Progress. The pipeline's phase count is not known ahead of time (phases
     are discovered from events as they fire), so completed/seen is the only
     honest ratio available while a run is in flight -- it is labelled as
     "phases done" rather than presented as overall completion. */
  const done = nodes.filter(n => n.status === "completed").length;
  const pct = nodes.length ? Math.round(100 * done / nodes.length) : 0;
  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-pct").textContent = pct + "%";
  const totalSecs = nodes.reduce((a, n) => a + (n.duration_s || 0), 0);
  document.getElementById("elapsed").textContent =
    done + "/" + nodes.length + " phases done · " + totalSecs.toFixed(1) + "s of work";

  const running = nodes.find(n => n.status === "running");
  const nowEl = document.getElementById("now-running");
  const phaseEl = document.getElementById("now-phase");
  if (running) {
    nowEl.classList.remove("done");
    phaseEl.textContent = running.display_name + (running.model ? " [" + running.model + "]" : "");
  } else if (nodes.length) {
    nowEl.classList.add("done");
    const last = nodes[nodes.length - 1];
    phaseEl.textContent = TERMINAL.indexOf(run.status) !== -1
      ? "finished — " + run.status
      : "between phases (last: " + last.display_name + ")";
  }

  document.getElementById("run-sub").innerHTML =
    "run <code>" + esc(run.run_id || "?") + "</code> &middot; config_hash=" + esc(run.config_hash || "?") +
    " &middot; seed=" + esc(run.seed);

  const s = run.summary || {};
  const statusBadge = run.status === "success" ? '<span class="badge ok">success</span>'
    : run.status === "failed" ? '<span class="badge bad">failed</span>'
    : run.status === "cancelled" ? '<span class="badge bad">cancelled</span>'
    : '<span class="badge">running</span>';
  let summary = statusBadge + " &middot; " + nodes.length + " phase(s) logged so far";
  if (s.total_checks !== undefined) summary += " &middot; " + s.total_checks + " health check(s), " + (s.failed_checks || []).length + " failed";
  if (run.error) summary += "<br><span class=\\"mono\\">" + esc(run.error) + "</span>";
  document.getElementById("run-summary").innerHTML = summary;

  if (TERMINAL.indexOf(run.status) !== -1) {
    sawTerminalStatus = true;
    stopped = true;
    document.getElementById("live-dot").classList.add("stopped");
  }
}

/* The pipeline process owns this server, so when the process ends (Ctrl+C,
   a kill, or simply finishing) the socket drops. If that happens *before*
   any terminal status was read, the run was interrupted: freeze the view and
   say so, rather than leaving the last phase pulsing "running" forever --
   which is exactly what a stale snapshot used to look like. */
function markInterrupted() {
  stopped = true;
  document.getElementById("live-dot").classList.add("stopped");
  const nowEl = document.getElementById("now-running");
  if (nowEl) nowEl.classList.add("done");
  const phaseEl = document.getElementById("now-phase");
  if (phaseEl) phaseEl.textContent = "interrupted before completing";
  document.querySelectorAll(".node.st-running").forEach(el => {
    el.classList.remove("st-running");
    el.classList.add("st-failed");
    const st = el.querySelector(".node-status");
    if (st) st.innerHTML = '<span class="dot dot-failed"></span>interrupted';
  });
  document.getElementById("run-sub").innerHTML =
    '<span class="badge bad">interrupted</span> The pipeline process ended before this phase completed ' +
    '(cancelled from the terminal, or killed). Phases already marked completed above did finish.';
}

async function poll() {
  try {
    const res = await fetch("/state", {cache: "no-store"});
    const data = await res.json();
    failedPolls = 0;
    render(data);
  } catch (e) {
    failedPolls += 1;
    /* One miss can be a transient hiccup; two consecutive misses means the
       server is really gone. */
    if (failedPolls >= 2) {
      if (!sawTerminalStatus) markInterrupted();
      else { stopped = true; document.getElementById("live-dot").classList.add("stopped"); }
    }
  }
  if (!stopped) setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>
"""


class _LiveFlowHandler(BaseHTTPRequestHandler):
    """Two routes only: `/` (the polling page above) and `/state` (fresh JSON
    rebuilt from the events file on every request). `events_path`/`run_id`
    are set per-instance via `functools.partial` in `start_live_view`."""

    events_path: str = ""
    run_id: Optional[str] = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # the pipeline's own logger is the record; silence stderr access logging

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _LIVE_HTML.encode("utf-8"))
        elif self.path == "/state":
            events = _select_run(_read_events(self.events_path), self.run_id)
            data = _build_nodes(events) if events else {
                "run": {}, "dataset": None, "nodes": [], "health_checks": [],
            }
            self._send(200, "application/json", json.dumps(data, default=str).encode("utf-8"))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")


def start_live_view(
    events_path: str = paths.RUN_EVENTS_LOG,
    run_id: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> str:
    """Start a background local HTTP server showing this run's flow live.

    Bound to `host` (default `127.0.0.1`, i.e. never reachable from outside
    this machine) on `port` (default `0` = the OS picks a free ephemeral
    port). Read-only: the two routes it serves never accept a body or modify
    anything. Runs `serve_forever()` in a daemon thread, so it needs no
    explicit shutdown -- it dies with the process.

    Args:
        events_path: JSONL event stream to poll (rebuilt from disk on every
            `/state` request, so it reflects events written after the server
            started).
        run_id: Which run to show. `None` follows whichever run_id the most
            recent line in the file belongs to at request time -- correct
            for the common case (this call happens right after `start_run`,
            before any events exist yet) without needing the id in advance.
        host: Bind address. Never change this to `0.0.0.0`/a real interface
            without adding authentication -- this server has none.
        port: Bind port, `0` for an OS-assigned free port.

    Returns:
        The URL to open, e.g. `"http://127.0.0.1:54231/"`.
    """
    handler = type(
        "_BoundLiveFlowHandler", (_LiveFlowHandler,),
        {"events_path": events_path, "run_id": run_id},
    )
    httpd = ThreadingHTTPServer((host, port), handler)
    actual_port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, name="flow-live-view", daemon=True)
    thread.start()
    return f"http://{host}:{actual_port}/"
