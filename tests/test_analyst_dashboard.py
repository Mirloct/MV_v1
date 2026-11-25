"""Focused contract tests for the analyst dashboard's P99 tabs/download.

Data sources / inputs: synthetic detector tables, percentiles, and entity history.
Created: 2026-08-30
Last modified: 2026-11-25
Changelog:
- 2026-11-22: Rebased the official review-union assertions from P95 to P99.
- 2026-11-25: Exercise raw-score quantiles and executable browser case-state regressions.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.reporting.analyst_dashboard import build_analyst_dashboard


class AnalystDashboardTests(unittest.TestCase):
    def _build(self, history_text=None) -> str:
        schema = PanelSchema(time_col="period", entity_col="customer_id", target_col=None)
        if_table = pd.DataFrame({
            # Explanation input is intentionally broader than IF's own export:
            # ``only-vae`` is below IF P90 but still needs IF variables in its
            # profile because the dashboard union selected it through VAE.
            "customer_id": ["only-if", "only-vae", "both"],
            "period": ["2026-08", "2026-09", "2026-09"],
            "anomaly_score": [0.9, 0.2, 0.8],
            "percentil": ["p99", "", "p95"],
            "top_5_variables": ["balance, transfers", "if-low-feature", "income"],
        })
        vae_table = pd.DataFrame({
            "customer_id": ["only-vae", "both"],
            "period": ["2026-09", "2026-09"],
            "anomaly_score": [8.0, 7.0],
            "percentil": ["p99", "p95"],
            "top_5_variables": ["age, products", "transactions"],
        })
        raw = pd.DataFrame({
            "customer_id": ["only-if", "only-if", "only-if", "only-vae", "both"],
            "period": ["2026-07", "2026-08", "2026-09", "2026-09", "2026-09"],
            "balance": [90, 100, 150, 200, 300],
            "free_text": ["before OOT", "first", "second", "third", "kept verbatim"],
            "puesto": ["Analista", "Analista", "Gerente", "Auditor", "Director"],
        })
        if history_text is not None:
            raw.loc[0, "free_text"] = history_text
            raw[history_text] = "source column"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.html"
            build_analyst_dashboard(
                vae_table, schema, "vae", ["2026-08", "2026-09"],
                {"only-if": 99.1, "only-vae": 50.0, "both": 99.5, "control-only": 98.0},
                {"only-if": 40.0, "only-vae": 99.2, "both": 99.6, "control-only": 97.0},
                {**{f"low-{i}": 0.0 for i in range(100)},
                 "only-if": 0.9, "only-vae": 0.2, "both": 0.8, "control-only": 0.4},
                {**{f"low-{i}": 0.0 for i in range(100)},
                 "only-if": 1.0, "only-vae": 8.0, "both": 7.0, "control-only": 2.0},
                {"only-vae": ["2026-09"], "both": ["2026-09"]},
                n_total_oot=104,
                out_path=str(out),
                model_tables={"iforest": if_table, "vae": vae_table},
                months_present_by_model={
                    "iforest": {"only-if": ["2026-08", "2026-09"], "both": ["2026-09"]},
                    "vae": {"only-vae": ["2026-09"], "both": ["2026-09"]},
                },
                entity_records=raw,
                identity_columns=["puesto"],
            )
            return out.read_text(encoding="utf-8")

    def test_three_tabs_partition_p99_union(self):
        html = self._build()
        self.assertIn("Solo IF <span>1</span>", html)
        self.assertIn("Solo IF+VAE <span>1</span>", html)
        self.assertIn("Intersección <span>1</span>", html)
        self.assertEqual(html.count('data-tab="if_only"'), 1)
        self.assertEqual(html.count('data-tab="vae_only"'), 1)
        self.assertEqual(html.count('data-tab="intersection"'), 1)
        self.assertIn("≥ P99", html)
        self.assertNotIn('data-id="control-only"', html)

    def test_profile_payload_keeps_every_oot_row_and_source_column(self):
        html = self._build()
        columns_match = re.search(r"var RECORD_COLUMNS = (.*?);", html)
        self.assertIsNotNone(columns_match)
        self.assertEqual(
            json.loads(columns_match.group(1)),
            ["customer_id", "period", "balance", "free_text", "puesto"],
        )
        self.assertIn('"period": "2026-07"', html)
        self.assertIn('"free_text": "before OOT"', html)
        self.assertIn('"free_text": "first"', html)
        self.assertIn('"free_text": "second"', html)
        self.assertIn("Descargar todas las variables (.csv)", html)
        self.assertIn("downloadObservation()", html)
        self.assertIn("_historial_completo.csv", html)

    def test_detector_specific_explanations_and_months_are_embedded(self):
        html = self._build()
        self.assertIn('"if_top5": "income"', html)
        self.assertIn('"vae_top5": "transactions"', html)
        self.assertIn('"if_top5": "if-low-feature"', html)
        self.assertIn('"if_months": ["2026-08", "2026-09"]', html)
        self.assertIn('"vae_months": ["2026-09"]', html)

    def test_case_workflow_identity_and_reviewed_export_are_present(self):
        html = self._build()
        self.assertIn('"identities": [{"label": "puesto", "value": "Gerente"}]', html)
        self.assertIn('id="mIdentity_0"', html)
        self.assertIn("Sin revisión", html)
        self.assertIn("En revisión", html)
        self.assertIn("Cerrado", html)
        self.assertIn("Casos revisados", html)
        self.assertIn("localStorage.setItem(CASE_STORAGE_KEY", html)
        self.assertIn("exportReviewedCases()", html)
        self.assertIn("fecha_cambio_estado", html)
        self.assertIn("hora_cambio_estado", html)

    def test_identity_value_renders_inline_next_to_the_id_in_the_row(self):
        """Explicit user request: the identity field (e.g. `despuestocolaborador`)
        must show beside the ID in each case row, not only inside the profile
        modal. `only-if`'s latest `puesto` is `"Gerente"` (2026-09 row)."""
        html = self._build()
        row_match = re.search(r'<tr data-id="only-if"[^>]*>.*?</tr>', html, re.S)
        self.assertIsNotNone(row_match, "row for only-if not found")
        row = row_match.group(0)
        self.assertIn('class="identitycell"', row)
        self.assertIn(">Gerente<", row)
        # The header must name the configured field so the column is self-explanatory.
        self.assertIn("<th>puesto</th>", html)


class MultipleIdentityColumnsTests(unittest.TestCase):
    """Explicit user request: the case card must be able to show SEVERAL
    additional fields, not only one -- each resolved to its own latest
    non-empty value, independently of the others."""

    def _build(self, identity_columns) -> str:
        schema = PanelSchema(time_col="period", entity_col="customer_id", target_col=None)
        vae_table = pd.DataFrame({
            "customer_id": ["only-if"],
            "period": ["2026-09"],
            "anomaly_score": [8.0],
            "percentil": ["p99"],
            "top_5_variables": ["age"],
        })
        raw = pd.DataFrame({
            "customer_id": ["only-if", "only-if", "only-if"],
            "period": ["2026-07", "2026-08", "2026-09"],
            "balance": [90, 100, 150],
            # puesto's latest non-empty is the 2026-09 row ("Gerente").
            "puesto": ["Analista", "Analista", "Gerente"],
            # area is BLANK in that same latest row -- its own latest
            # non-empty must still resolve independently, to the 2026-08
            # value, never masked by puesto's own latest row being filled.
            "area": ["Comercial", "Riesgos", ""],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.html"
            build_analyst_dashboard(
                vae_table, schema, "vae", ["2026-09"],
                {"only-if": 50.0}, {"only-if": 99.2},
                {"only-if": 0.2}, {"only-if": 8.0},
                {"only-if": ["2026-09"]},
                n_total_oot=1,
                out_path=str(out),
                model_tables={"vae": vae_table},
                months_present_by_model={"vae": {"only-if": ["2026-09"]}},
                entity_records=raw,
                identity_columns=identity_columns,
            )
            return out.read_text(encoding="utf-8")

    def test_each_configured_field_renders_its_own_latest_value(self):
        html = self._build(["puesto", "area"])
        self.assertIn(
            '"identities": [{"label": "puesto", "value": "Gerente"}, '
            '{"label": "area", "value": "Riesgos"}]',
            html,
        )
        self.assertIn('id="mIdentity_0"', html)
        self.assertIn('id="mIdentity_1"', html)
        self.assertIn("<span>puesto</span>", html)
        self.assertIn("<span>area</span>", html)

    def test_an_empty_list_renders_no_identity_row_at_all(self):
        html = self._build([])
        self.assertIn('"identities": []', html)
        self.assertNotIn('id="mIdentity_0"', html)
        self.assertNotIn('<div class="midentity">', html)

    def test_multiple_fields_render_inline_joined_in_one_cell(self):
        """Several configured fields show in ONE inline cell next to the ID,
        joined (not one column per field) -- `area`'s own latest non-empty
        value ("Riesgos", from 2026-08) independently of `puesto`'s."""
        html = self._build(["puesto", "area"])
        row_match = re.search(r'<tr data-id="only-if"[^>]*>.*?</tr>', html, re.S)
        row = row_match.group(0)
        self.assertIn('class="identitycell"', row)
        self.assertIn("Gerente", row)
        self.assertIn("Riesgos", row)
        self.assertIn("<th>puesto / area</th>", html)

    def test_an_empty_list_renders_no_identity_column_in_the_row_either(self):
        """Same contract as the modal's own empty-list test above, but for the
        inline row/header added next to the ID -- an empty
        `dashboard.identity_columns` must not add a dangling header or an
        empty/broken column, and must not change the "no results" colspan."""
        html = self._build([])
        self.assertNotIn('class="identitycell"', html)
        self.assertNotIn("<th>puesto</th>", html)
        # Header cell count for the ID/band/score/.../case-status row is
        # unchanged (8) when no identity column is configured.
        self.assertIn("colspan='8'", html)


class ResponsiveLayoutTests(unittest.TestCase):
    """`.main`/`.col-table`/`.tablewrap` use `flex:1` (`flex-basis:0`) to
    size against `.shell`'s fixed desktop height. Explicit user report: once
    `.shell` drops that fixed height under `@media (max-width:980px)`,
    `flex:1` has nothing to grow into and collapses to ~0, leaving the case
    table invisible and unscrollable -- only the header/KPI text above it
    stayed visible. Regression guard: the mobile breakpoint must size these
    three by content (`flex:none`) instead of leaving them flexed against an
    undefined ancestor height."""

    def test_narrow_viewport_lets_the_table_area_size_to_content(self):
        from src.reporting.analyst_dashboard import _CSS

        media_block = re.search(
            r"@media \(max-width:980px\)\{(.*?)\n\}", _CSS, re.S,
        )
        self.assertIsNotNone(media_block, "mobile breakpoint not found in _CSS")
        block = media_block.group(1)
        for selector in (".main", ".col-table", ".tablewrap"):
            self.assertIn(f"{selector}{{flex:none", block,
                          f"{selector} must not keep flex:1 once .shell's height is auto")

    def test_identity_column_has_no_fixed_width_that_would_break_narrow_viewports(self):
        """The inline identity cell added next to the ID must size by content
        (overflow-wrap, no hardcoded width wider than a phone viewport) --
        same contract as `.idc` beside it -- so it cannot reintroduce the
        table-invisible-on-mobile regression these other tests guard."""
        from src.reporting.analyst_dashboard import _CSS

        match = re.search(r"\.identitycell\{([^}]*)\}", _CSS)
        self.assertIsNotNone(match, ".identitycell rule not found in _CSS")
        rule = match.group(1)
        self.assertIn("overflow-wrap:anywhere", rule)
        self.assertNotRegex(rule, r"width:\s*[2-9]\d\d+px",
                             ".identitycell must not hardcode a wide fixed width")

    def test_the_wide_table_keeps_its_own_horizontal_scrollbar(self):
        # Regression: a first fix set .tablewrap to plain `overflow:visible`,
        # which also killed its horizontal scroll -- #priorityTable has
        # min-width:980px (wider than any viewport this breakpoint applies
        # to), so without overflow-x:auto that overflow bleeds up into
        # .shell's overflow:hidden (unchanged, desktop-only concern) and the
        # right-hand columns become unreachable with no scrollbar anywhere.
        from src.reporting.analyst_dashboard import _CSS

        media_block = re.search(r"@media \(max-width:980px\)\{(.*?)\n\}", _CSS, re.S)
        block = media_block.group(1)
        self.assertIn(".tablewrap{flex:none;overflow-x:auto", block)
        self.assertNotIn(".tablewrap{flex:none;overflow:visible", block)


class CaseStateTests(unittest.TestCase):
    def _run(self, scenario, stored=None, html=None, storage_error=False):
        node = shutil.which("node")
        if node is None:
            self.fail("Node.js is required to execute dashboard case-state tests")
        html = html or AnalystDashboardTests()._build()
        script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
        rows = [dict(id=eid, tab=tab, profile=key, caseStatus="sin_revision")
                for eid, tab, key in re.findall(
                    r'<tr data-id="([^"]+)" data-tab="([^"]+)" data-profile="([^"]+)"', html)]
        harness = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const elements = new Map(), frames = new Map(), listeners = {};
let nextFrame = 0, stored = input.stored, downloaded = null;
function element() {
  return {textContent:'', value:'', style:{}, dataset:{}, disabled:false,
    classList:{toggle(){}, add(){}, remove(){}}, setAttribute(){},
    appendChild(){}, click(){}, remove(){}};
}
function get(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }
const rows = input.rows.map(dataset => Object.assign(element(), {dataset, textContent:dataset.id}));
const document = {
  getElementById:get, addEventListener(){}, createElement:element, body:element(),
  querySelector(selector) {
    const match = selector.match(/data-(?:profile|case-date)="([^"]+)"/);
    if (!match) throw Error('Unhandled selector: ' + selector);
    if (selector.startsWith('tr')) return rows.find(r => r.dataset.profile === match[1]);
    return get(selector);
  },
  querySelectorAll(selector) {
    if (selector === '.tabbtn') return [];
    if (selector.startsWith('#priorityTableBody')) return rows;
    throw Error('Unhandled selector: ' + selector);
  }
};
function flush() { const pending = [...frames.values()]; frames.clear(); pending.forEach(f => f()); }
const context = vm.createContext({document, window:{location:{pathname:'/dashboard.html'},
  addEventListener(name,fn){listeners[name]=fn;}},
  localStorage:{getItem(){if(input.storage_error) throw Error('blocked'); return stored;},
    setItem(key,value){if(input.storage_error) throw Error('blocked'); stored=value;}},
  setInterval(){}, setTimeout(){}, requestAnimationFrame(f){frames.set(++nextFrame,f); return nextFrame;},
  cancelAnimationFrame(id){frames.delete(id);},
  Blob:class {constructor(parts){downloaded=parts.join('');}},
  URL:{createObjectURL(){return 'blob:test';}, revokeObjectURL(){}},
  get, rows, flush, check:require('assert').strict,
  saved:()=>stored, download:()=>downloaded,
  external(value,key){stored=value; listeners.storage({key:key===undefined?context.CASE_STORAGE_KEY:key});}});
vm.runInContext(input.script, context); flush();
vm.runInContext(input.scenario, context); flush();
process.stdout.write('ok');
"""
        result = subprocess.run(
            [node, "-e", harness], input=json.dumps(dict(
                script=script, scenario=scenario, rows=rows, stored=stored,
                storage_error=storage_error)),
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ok")

    def test_invalid_saved_entries_are_not_reviewed_or_exported(self):
        for entry in [None, [], "cerrado", {}, {"status": "unknown"},
                      {"status": "toString"}, {"status": "cerrado", "changed_at": {}}]:
            with self.subTest(entry=entry):
                self._run("""
                    check.equal(refreshCaseSummary(), 0);
                    check.equal(get('exportCasesBtn').disabled, true);
                    Object.keys(PROFILES).forEach(k => check.equal(caseFor(k).status, 'sin_revision'));
                    exportReviewedCases(); check.equal(download(), null);
                """, json.dumps({"only-if": entry}))

    def test_reviewed_kpis_update_when_resetting_a_case_in_reviewed_tab(self):
        self._run("""
            const key = Object.keys(PROFILES).find(k => PROFILES[k].id === 'only-if');
            changeCaseStatus(key, 'en_revision'); setActiveTab('reviewed'); flush();
            check.equal(String(get('queueKpiValue').textContent), '1');
            check.equal(String(get('recurrentKpiValue').textContent), '1');
            changeCaseStatus(key, 'sin_revision'); flush();
            check.equal(String(get('queueKpiValue').textContent), '0');
            check.equal(String(get('recurrentKpiValue').textContent), '0');
            check.equal(get('exportCasesBtn').disabled, true);
            check.equal(rows.find(r => r.dataset.profile === key).hidden, true);
        """)

    def test_corrupt_top_level_storage_recovers_and_persists_changes(self):
        for stored in ['[]', 'null', '12', '"text"', '{broken']:
            with self.subTest(stored=stored):
                self._run("""
                    const key=Object.keys(PROFILES)[0];
                    check.equal(refreshCaseSummary(), 0);
                    changeCaseStatus(key,'cerrado');
                    CASES=loadCases();
                    check.equal(caseFor(key).status,'cerrado');
                    check.equal(JSON.parse(saved())[PROFILES[key].id].status,'cerrado');
                """, stored)

    def test_storage_failure_is_visible_and_session_state_remains_exportable(self):
        self._run("""
            check.match(get('caseStorageStatus').textContent,/cargar/);
            changeCaseStatus(Object.keys(PROFILES)[0],'cerrado');
            check.equal(refreshCaseSummary(),1);
            check.match(get('caseStorageStatus').textContent,/No se pudo guardar/);
            exportReviewedCases(); check.ok(download().includes('Cerrado'));
        """, storage_error=True)

    def test_external_storage_updates_sync_modal_rows_counts_and_reset(self):
        self._run("""
            const key=Object.keys(PROFILES).find(k=>PROFILES[k].id==='only-if');
            openProfile(key);setActiveTab('reviewed');
            external(JSON.stringify({'only-if':{status:'cerrado',changed_at:'2026-11-25T12:00:00Z'}}));
            flush();check.equal(caseFor(key).status,'cerrado');
            check.equal(get('mCaseStatus').value,'cerrado');
            check.equal(String(get('queueKpiValue').textContent),'1');
            check.equal(rows.find(r=>r.dataset.profile===key).hidden,false);
            external('{}','unrelated');check.equal(caseFor(key).status,'cerrado');
            external(null,null);flush();
            check.equal(get('mCaseStatus').value,'sin_revision');
            check.equal(String(get('queueKpiValue').textContent),'0');
        """)

    def test_prototype_named_entity_ids_and_invalid_statuses(self):
        scores={eid:1.0 for eid in ['__proto__','constructor','toString']}
        html=OfficialScoreSelectionTests()._build(scores)
        self._run("""
            Object.keys(PROFILES).forEach(k=>changeCaseStatus(k,'cerrado'));
            CASES=loadCases();check.equal(refreshCaseSummary(),3);
            Object.keys(PROFILES).forEach(k=>{
                changeCaseStatus(k,'toString');check.equal(caseFor(k).status,'cerrado');
                changeCaseStatus(k,'sin_revision');
            });
            check.equal(refreshCaseSummary(),0);
        """, html=html)

    def test_malicious_history_stays_inside_single_script_and_round_trips(self):
        malicious='</script><script>window.injected=true</script><!-- & \\"'
        html=AnalystDashboardTests()._build(history_text=malicious)
        self.assertEqual(html.count('<script>'),1)
        self.assertEqual(html.count('</script>'),1)
        self.assertNotIn(malicious,html)
        self._run("""
            const key=Object.keys(PROFILES).find(k=>PROFILES[k].id==='only-if');
            check.equal(typeof window.injected,'undefined');
            check.equal(PROFILES[key].records[0].free_text,"""+json.dumps(malicious)+""");
            check.ok(RECORD_COLUMNS.includes(PROFILES[key].records[0].free_text));
        """, html=html)


class OfficialScoreSelectionTests(unittest.TestCase):
    def _build(self, if_scores, vae_scores=None, if_pcts=None, vae_pcts=None,
               review_percentile=99.0):
        schema = PanelSchema(time_col="period", entity_col="customer_id", target_col=None)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.html"
            build_analyst_dashboard(
                pd.DataFrame(), schema, "iforest", ["2026-09"],
                if_pcts or {}, vae_pcts or {}, if_scores, vae_scores or {}, {},
                n_total_oot=len(if_scores), out_path=str(out),
                review_percentile=review_percentile,
            )
            return out.read_text(encoding="utf-8")

    def _rows(self, html):
        return dict(re.findall(r'<tr data-id="([^"]+)" data-tab="([^"]+)"', html))

    def test_500_unique_scores_select_exactly_five_despite_rank_percentiles(self):
        scores = {str(i): float(i) for i in range(500)}
        pcts = {str(i): 100 * (i + 1) / 500 for i in range(500)}
        html = self._build(scores, if_pcts=pcts)
        self.assertEqual(self._rows(html), {str(i): "if_only" for i in range(495, 500)})
        self.assertIn('"if_pctl": 99.2', html)
        self.assertIn("<span class='pval'>99.2</span>", html)

    def test_all_equal_scores_include_every_tie_even_with_low_display_percentiles(self):
        scores = {str(i): 7.0 for i in range(500)}
        pcts = {eid: 50.1 for eid in scores}
        html = self._build(scores, scores, pcts, pcts)
        self.assertEqual(self._rows(html), {eid: "intersection" for eid in scores})

    def test_exact_boundary_and_ties_are_included_but_next_float_below_is_not(self):
        scores = {f"low-{i}": 0.0 for i in range(98)}
        scores.update(below=float(np.nextafter(1.0, 0.0)), boundary=1.0, tied=1.0)
        self.assertEqual(np.percentile(list(scores.values()), 99), 1.0)
        html = self._build(scores, if_pcts={eid: 100.0 for eid in scores})
        self.assertEqual(self._rows(html), {"boundary": "if_only", "tied": "if_only"})

    def test_percentile_fallback_is_per_detector_when_score_map_is_empty(self):
        html = self._build({"raw": 1.0}, {}, {"raw": 1.0, "phantom": 100.0},
                           {"raw": 99.0, "fallback": 99.0, "below": 98.999})
        self.assertEqual(self._rows(html), {"raw": "intersection", "fallback": "vae_only"})

    def test_custom_percentile_uses_full_population_not_explanation_table(self):
        scores = {str(i): float(i) for i in range(101)}
        html = self._build(scores, review_percentile=95.0)
        self.assertEqual(self._rows(html), {str(i): "if_only" for i in range(95, 101)})

    def test_empty_populations_and_non_finite_inputs(self):
        self.assertEqual(self._rows(self._build({})), {})
        for invalid in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(score=invalid), self.assertRaisesRegex(ValueError, 'finite'):
                self._build({'invalid':invalid})
        for invalid in [0, 100, -1, float('nan'), float('inf')]:
            with self.subTest(percentile=invalid), self.assertRaises(ValueError):
                self._build({'one':1.0}, review_percentile=invalid)


class BrowserDownloadTests(unittest.TestCase):
    def test_real_browser_download_bytes_and_malicious_history(self):
        node=shutil.which('node')
        self.assertIsNotNone(node, 'Node.js is required for browser download verification')
        malicious='</script><script>window.injected=true</script>'
        html=AnalystDashboardTests()._build(history_text=malicious)
        browser_test=r"""
const fs=require('fs'), assert=require('assert').strict;
const {pathToFileURL}=require('url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/Marco/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
(async()=>{
  const browser=await chromium.launch({headless:true,
    executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try {
    const context=await browser.newContext({acceptDownloads:true});
    const page=await context.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(input.path).href);
    assert.equal(await page.evaluate(()=>typeof window.injected),'undefined');
    const key=await page.evaluate(()=>Object.keys(PROFILES).find(k=>PROFILES[k].id==='only-if'));
    await page.evaluate(k=>openProfile(k),key);
    async function bytes(click){
      const pending=page.waitForEvent('download');await click();
      return fs.readFileSync(await (await pending).path());
    }
    const history=await bytes(()=>page.locator('#downloadBtn').click());
    assert.equal(history.subarray(0,3).toString('hex'),'efbbbf');
    const text=history.toString('utf8');
    assert.equal(text.split('\r\n').length,4);
    assert.ok(text.includes(input.malicious));
    assert.ok(!text.includes('\\ufeff'));assert.ok(!text.includes('\\r\\n'));
    await page.keyboard.press('Escape');
    await page.evaluate(k=>changeCaseStatus(k,'cerrado'),key);
    const reviewed=await bytes(()=>page.locator('#exportCasesBtn').click());
    assert.equal(reviewed.subarray(0,3).toString('hex'),'efbbbf');
    assert.equal(reviewed.toString('utf8').split('\r\n').length,2);
    assert.ok(reviewed.toString('utf8').includes('Cerrado'));
    await page.reload();
    assert.equal(await page.evaluate(k=>caseFor(k).status,key),'cerrado');
    const second=await page.context().newPage();
    await second.goto(pathToFileURL(input.path).href);
    await second.evaluate(k=>changeCaseStatus(k,'sin_revision'),key);
    await page.waitForFunction(k=>caseFor(k).status==='sin_revision',key);
    assert.equal(await page.locator('#reviewedCount').textContent(),'0');
    await second.close();
    assert.deepEqual(errors,[]);
    process.stdout.write('ok');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'dashboard.html'
            out.write_text(html,encoding='utf-8')
            result=subprocess.run([node,'-e',browser_test],
                input=json.dumps(dict(path=str(out),malicious=malicious)),
                capture_output=True,text=True,encoding='utf-8',timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'ok')


if __name__ == "__main__":
    unittest.main()
