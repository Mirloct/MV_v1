# CHANGELOG.md — Dated project history

This file holds the dated history that used to live inline in `CONTEXT.md`:
measurements, verification notes, and the reasoning behind changes as they
were made. `CONTEXT.md` holds only the resulting current contracts/defaults —
read that first; come here when you need the "why", the exact numbers behind
a decision, or how something was verified at the time.

Entries are in chronological order. Each keeps the wording and numbers from
when it was written; where a later entry supersedes an earlier measurement,
the later one says so, but earlier numbers are not deleted.

---

## 2026-08-01 — Baseline and open problems (measured)

Two runs, both `python main.py --quick`, seed 42. **They are not directly
comparable** — the panel depth, the split geometry and the feature set all
changed — so the table records what each configuration was, not a clean A/B.

| | iForest | VAE | configuration |
|---|---|---|---|
| OOT ROC-AUC | 0.656 | 0.825 | 800×6, 2-way in/OOT split, 1 test month, lag1 only |
| OOT PR-AUC | 0.094 | 0.414 | " |
| OOT ROC-AUC | **0.799** | 0.816 | 500×12, 3-way chronological, **3** test months, horizons (1,3,6) |
| OOT PR-AUC | **0.137** | 0.140 | " |

The **iForest improves substantially** (ROC-AUC 0.656 → 0.799) once the
multi-horizon contrast features exist — consistent with the Phase 1 argument
that a single-period lag under-reports a shock.

The **VAE's PR-AUC drops** (0.414 → 0.140) and this is *not* yet explained.
Confounded by all three changes at once; the honest next step is a controlled
one-factor-at-a-time sweep before drawing any conclusion. Candidates worth
testing in order: the test block growing from 1 to 3 months (it now contains
"plateau" repeats of the same event, which changes the positive mix), the
narrowed search space (`beta` ≤ 2.0, `dropout` ≥ 0.1), and the 129-column
feature matrix being harder to reconstruct. Do not tune this away without
measuring which factor it is. (Note, added later: the VAE's loss scaling was
found and fixed on 2026-08-22 — see `CONTEXT.md` "Known open problems" — so
any future re-measurement of this should be done against the corrected loss,
not this baseline.)

An older `artifacts/tuning/best_params_iforest.yaml` reported
`best_value: 0.0833`. That number is **not comparable to anything**: it was an
*in-sample* PR-AUC from a study that had pooled 60 trials across different
datasets and preprocessing configs.

### OOT recall by anomaly type — the point of the instrumentation

`anomaly_type` had always been in the ground truth and *nothing* read it
(`labels.py` only loaded `is_anomaly`), so an aggregate PR-AUC could not say
which geometry was failing. It now can:

| type | iForest recall@10 % | mean score pctile |
|---|---|---|
| global | 0.600 | 0.855 |
| contextual | 0.600 | 0.648 |
| **local** | **0.000** | **0.457** |

**`local` is the open problem.** The forest ranks a local anomaly *below the
median row*. By construction these values are drawn from inside the population
`[p2, p95]` band and are anomalous only against the entity's own history, so
no marginal geometry separates them; `_own_z` is the intended instrument and is
not sufficient. Next step is here, not in more hyperparameter search. (Still
true as of 2026-08-23 — see `CONTEXT.md` "Known open problems".)

### Measured conflict: the two models want opposite numeric transforms

Controlled sweep (fixed data, fixed hyperparameters, only `numeric_transform`
varies) — see `docs/models_isolation_forest.md` for the full per-anomaly-type
table:

| transform | max\|x\| | IF PR-AUC | VAE PR-AUC |
|---|---|---|---|
| `robust` / `standard` (affine, shape-preserving) | 5.3×10⁵ | **0.272** | **NaN — total failure** |
| `yeo-johnson` (default) | 47 | 0.117 | **0.414** |
| `log1p` | — | 0.089 | — |
| `auto` (per-column, min `abs_skewness`) | — | 0.057 | — |

The IF wants the heavy tail intact (extremes isolate in few splits; compressing
the right tail is compressing exactly where `global` anomalies live). The VAE
cannot survive it — MSE gradients on a scaled value of 5×10⁵ overflow to NaN.

**Decisions taken:** the shared default stays `yeo-johnson` (one side degrades
gracefully, the other fails outright). `auto` is implemented and tested but is
**not** the default — the sweep shows minimising skewness is the wrong criterion
for this task. Genuinely resolving this needs a per-model shape transform (two
matrices), which is an architecture change, not a default change.

Note this is the *correct* version of a rejected proposal to "bifurcate the
scaling per model": scaling is provably a no-op for the IF and the pipeline
already delivers scaled data to the VAE, so that bifurcation was empty. The
real conflict is over **shape**.

---

## 2026-08-16 — IF → VAE stacking (measured)

`--stack-iforest-into-vae` (default on) appends the Isolation Forest's score
to the VAE's input matrix, and the VAE then ships the only Excel queue.
Implementation in `src/models/stacking.py`; full three-arm measurement in
`docs/leakage_free_pipeline.md` (Appendix).

**The measurement does not support the architecture on this data:**

| Arm | PR-AUC | ROC-AUC | Recovers the forest's top-50 |
| --- | ---: | ---: | ---: |
| VAE on raw X (parallel) | **0.1416** | **0.8045** | 16/50 |
| VAE on X + IF score (stacked) | 0.1251 | 0.7576 | 16/50 |

The appended column is **0.736% of the VAE's reconstruction MSE** (uniform share
= 0.769%), i.e. the VAE treats it as one ordinary feature among 130. Stacking
therefore *does not* transfer the forest's ranking: the stacked VAE's queue
covers the forest's picks exactly as poorly as the un-stacked one. Out-of-fold
scores changed nothing, for the same reason.

**Implication for the deliverable.** The premise "the forest's score is already
included, so one VAE queue suffices" is not borne out — the two rankings still
share only ~33% of their members. If a single queue is wanted, the evidence
points to **score-level rank combination**, not feature-level stacking. Flip
`--no-stack-iforest-into-vae` to restore the parallel arrangement (and both
Excels) at any time.

---

## 2026-08-17 — Panel features default OFF in main.py

`PipelineConfig.panel_features` defaults to `False`, and `main.py`'s call to
`fit_transform_panel` passes `add_panel_features=config.panel_features`
accordingly — flip it back on with `--panel-features`.

**Why:** this pipeline's real-data usage computes within-entity
lag/diff/ratio/own-z + seasonality features in a separate upstream flow
outside this project, so generating them again here would duplicate/conflict
with that. The underlying `fit_transform_panel`/`PanelFeatureEngineer`
function still defaults `add_panel_features=True` when called directly (used
that way by the synthetic-data workflow in the README) — only `main.py`'s
CLI-level default changed.

**Consequence.** With panel features off, the `local`/`contextual` anomaly
definitions lose their intended instrument (own-history lag/diff/z-score) —
see "OOT recall by anomaly type" above, measured with panel features on. A
run with `--no-panel-features` (the new default) has not yet been
re-measured against that table; treat those recall numbers as stale for the
new default until someone reruns the comparison.

---

## 2026-08-19 — Observability, assumption gate, and tuning early-stopping

Three additive layers, built in this order because each is a prerequisite
for the next (a flow visualization needs real events to draw from; the
assumption gate's failures need a structured trace; tuning early-stopping is
told apart from the VAE's *existing* per-epoch stopping by the same event
schema). Current-state summary lives in `CONTEXT.md`; this entry keeps the
verification detail.

**A. `src/utils/observability.py`.** A second, JSON Lines event channel
(`artifacts/logs/run_events.jsonl`) alongside the existing text logger
(`src/utils/logging_config.py`), which is unchanged for any caller that never
calls `observability.start_run()`. `log_phase` was extended (local import
inside the function, no new module-level dependency) to emit
`phase_started`/`phase_completed`/`phase_failed` for every existing call site
across the whole codebase, not just `main.py`'s 12 phases. `main.py` calls
`start_run`/`end_run` around the whole pipeline (failure path closes the run
with `status="failed"` from `main()`'s except-block, without needing to
re-indent `run_pipeline`'s existing linear body). Verified 2026-08-19 with a
real `--quick --no-tune` run (81 JSONL events, `run_ended` status="success",
`peak_python_memory_mb` via stdlib `tracemalloc` — no `psutil` added) and a
real forced failure (`--numeric-transform bogus_value_xyz`), which correctly
nested `phase_failed` and closed with `status="failed"` + the exception text.

**B. `src/utils/assumptions.py`.** Typed exception hierarchy
(`SchemaAssumptionError`, `DataQualityAssumptionError`, `LeakageAssumptionError`,
`TemporalSplitAssumptionError`, `IsolationForestAssumptionError`,
`VAEAssumptionError`, `ArtifactGenerationError`) — every raise also records a
failed `category="assumption"` health check via `observability.check(...)` in
the same call, so a blocking stop still leaves a structured trace. Wired into
`main.py` as a new **Phase 3b** (blocking: duplicate `(entity_id, period)`
keys, unparseable periods, infinite values — before Phase 4 preprocessing,
which is undefined over either) plus non-blocking diagnostics (null rates,
constant features, full-row duplicates) and a **person-overlap measurement**
in Phase 3a (diagnostic only, never raises — see the theoretical contrast
below). Phase 5's pre-existing bare `RuntimeError` for split/mask disagreement
is now `LeakageAssumptionError`. Phase 6/7 validate the feature matrix is
finite immediately before each model's `.fit()`, and Phase 6 sanity-checks
`contamination`. Verified with the real `--quick` run (Phase 3b passes in
0.14s) and by calling all four blocking checks directly with deliberately
broken inputs (duplicate keys, `inf`, `contamination=0.9`, a `NaN` matrix) —
each raised the correct typed exception naming the exact failing values.

**Measured person overlap (2026-08-19, `--quick`, this synthetic panel):**
train/test=100%, val/test=100%, 0 test entities never seen in train/val. This
is the **expected** reading for this project's balanced synthetic panel
(every entity observed in every period, by the Data contract in `CONTEXT.md`)
— not evidence of a defect by itself. The theoretical contrast: under a pure
rescoring/forecasting objective on a closed, stationary population, 100%
overlap is a designed property; it becomes a real leakage-adjacent risk claim
only once churn/attrition/new-entity-inflow are part of the evaluated
population, which this synthetic panel does not model. Real-data runs should
re-measure this (`assumptions.measure_person_overlap`, logged every run) and
revisit the claim against what the real portfolio actually does over time.

**C. `src/models/_tuning_stop.py` — `TrialPatienceStopper`.** Trial-level
early stopping for both `tune_iforest` and `tune_vae`'s Optuna studies,
distinct from the VAE's existing *per-epoch* early stopping inside one fit
(`VAEDetector.early_stopping_patience`, unrelated and untouched). An Optuna
trial is an independent draw from the search space, not one more step of the
same optimization, so there is no epoch-style "loss stopped decreasing" to
test; the rule instead is economic — stop once `patience` consecutive trials
(default 10) show no `min_delta`-relative improvement (default 0.5%) in
`study.best_value`, never before `min_trials` (default 10) trials complete.
Verified against the installed `optuna==4.9.0` API (`Study.stop()`) before
writing, and empirically: a 40-trial request on a saturating objective
stopped at trial 15 (best found at trial 9, skipped 25 trials), with the
`{model}.tuning_early_stopped` health check recording the exact counts.
Defaults are opt-out (`early_stopping_patience=None` disables it) and
`min_trials=10` guarantees zero behavior change for every test that existed
at the time, which all used `n_trials` in [2, 4].

**D/E. `src/evaluation/oot_report.py::export_p95_checkpoint`.** A new **Phase
6c**, between Phase 6 (Isolation Forest fit) and Phase 6b (IF -> VAE
stacking)/Phase 7 (VAE) in `main.py`, not wrapped in a lenient try/except on
purpose: `export_p95_checkpoint` raises `ArtifactGenerationError` (defined in
`assumptions.py`) on any validation failure, and letting that propagate is
what makes "the VAE does not start without a validated IF export" true rather
than aspirational. This is a distinct artifact from the OOT top-N/percentile
review-queue deliverable (`export_oot_top_anomalies`) — different purpose (an
immediate post-IF checkpoint of the whole panel vs. the final,
threshold-calibrated, per-model OOT review queue), different selection rule,
same underlying `openpyxl` export mechanism.

**P95 rule, decided explicitly (not left implicit):**
- Score = `detector.score_samples(X)` (project convention: higher = more
  anomalous).
- Threshold = the 95th percentile of scores **restricted to `in_mask`**
  (train+val, the rows the detector was allowed to see) — never OOT/test,
  mirroring how `calibrate_threshold` is fitted on validation only elsewhere
  in this pipeline.
- That fixed threshold is then **applied to every row of the panel**,
  in-time and OOT alike (scoring is not fitting, so this does not leak).
- Ties at the threshold are **included** (`score >= threshold`), so the
  exported set is `>= 5%`, not exactly 5%.
- Columns: every original column of `df` (never overwritten/subsetted) plus
  `isolation_forest_score`, `p95_threshold`, `run_id`, `split`.
  `isolation_forest_decision_function`/`_prediction` were considered and
  deliberately left out — both would need the preprocessed matrix `X`
  threaded through a function whose contract is otherwise "a df and its
  row-aligned scores."
- Validation before returning: file exists, non-empty, a full re-read via
  `pd.read_excel` reproduces the in-memory table's exact shape, plus a
  SHA-256 checksum, row/column counts, threshold, and timestamp all recorded
  as an `artifact`-category health check.

Verified 2026-08-19 both ways: a real `--quick` run selected 328-343/6000
rows (varies slightly run to run — expected, contamination/tuning shift the
forest, not a bug) with a passing health check and Phase 6b starting only
after Phase 6c logged complete; and a direct call with mismatched
`scores`/`df` lengths raised `ArtifactGenerationError` naming the exact
mismatched counts.

**F. Parameter reference tables** added to `docs/models_isolation_forest.md`
(section "2b", verified against installed `scikit-learn==1.7.2` via
`inspect.signature` — not recalled from memory, since sklearn has changed
`contamination`'s and `max_samples`'s defaults across versions before) and
`docs/models_vae.md` (section "2b", verified against installed
`torch==2.9.1+cpu`, this project's own architecture so no external default to
check but every value re-read from `VAEDetector.__init__`, `vae.py:315`).
Each row: default, meaning, valid alternatives (with the actual `tune_iforest`
/`tune_vae` Optuna search-space bounds where one is searched), and the
trade-off driving the choice.

**Trial-budget honesty note on `TrialPatienceStopper`.** A full `--quick`
pipeline run with `--iforest-trials 20 --vae-trials 12` completed 2026-08-19
without early-stopping firing for either model (0 of 13 health checks
failed, run healthy) — with `min_trials=10` and `patience=10` both at their
defaults, a stop is only reachable once at least `min_trials + patience` =
20 trials have run with the plateau starting right at trial `min_trials`, so
a 20-trial budget has essentially no margin to actually trigger it. The
40-trial isolated test above is the real evidence the mechanism *works*;
whether it fires in a given production run is a separate, expected function
of how large `n_trials` is relative to `patience`.

---

## 2026-08-19 — Parameter centralization in `main.py`

Audited every place `run_pipeline` calls into a model/tuning/export function
and classified each parameter as already threaded through `PipelineConfig`
vs. silently defaulted at the callee. Design rule adopted: **each module
keeps its own sensible defaults** (`IsolationForestDetector`, `VAEDetector`,
`tune_iforest`, `tune_vae` all remain fully usable standalone, per the
README's "drive one of them directly" workflow) — but `PipelineConfig` now has
an explicit field for every one of them, defaulting to the exact same value
the callee would use on its own, so **changing a full pipeline run's behavior
never requires editing anything outside `main.py`**.

**New `PipelineConfig` fields**, all wired into their call sites: `p95_percentile`
(was hardcoded `95.0` in Phase 6c), `iforest_holdout_frac`,
`iforest_tuning_early_stopping` / `vae_tuning_early_stopping` (dicts of
`patience`/`min_delta`/`min_trials` — the trial-level stopping added earlier
today was wired into `tune_iforest`/`tune_vae`'s function signatures but not
threaded from `main.py` until now), and `vae_params` (a new dict covering
every `VAEDetector` architecture/training arg used by the untuned fallback
fit: `latent_dim`, `hidden_dim`, `n_layers`, `dropout`, `beta`, `lr`,
`optimizer`, `batch_size`, `weight_decay`, `activation`, `kl_anneal_epochs`,
`early_stopping_patience`).

**Real bug the audit found and fixed, not just a tidiness pass:**
`iforest_params["contamination"]` (`0.02`) was only ever passed to the
*untuned* fallback `IsolationForestDetector(...)`. The *tuned* path's call to
`tune_iforest(...)` never passed `contamination=`, so it silently used that
function's own internal default (`_DEFAULT_CONTAMINATION = 0.10`,
`src/models/iforest.py:73`) instead — meaning **the effective contamination
changed depending on whether `--tune`/`--no-tune` was passed, with zero log
warning either way.** `tune_iforest`'s call now explicitly passes
`contamination=config.iforest_params.get("contamination", 0.02)`, so both
paths agree by construction. Verified 2026-08-19: a tuned run's log now reads
`contamination=0.02 (fixed)` (previously would have read `0.10`), and the
untuned fallback's log confirms the same `0.02`.

**New CLI flags**: `--contamination` (overrides `iforest_params["contamination"]`
directly, since this is the one users are most likely to want to sweep
per-run) and `--p95-percentile`. The tuning early-stopping dicts and
`vae_params` are deliberately **not** individually exposed as ~15 more CLI
flags — they are still fully centralized (one field each, in
`PipelineConfig`, editable directly in `main.py`), but a CLI flag per
VAE-architecture key was judged more surface area than value for parameters
that are normally swept *by the tuner*, not hand-set per run.

---

## 2026-08-19 — Flow visualization

`src/reporting/flow_visualization.py::build_flow_visualization` renders
`artifacts/reports/flow_visualization.html` — an n8n/Databricks-style diagram
of one run's phases, built **only** from `run_events.jsonl` (Option A). No
state is invented: node status/duration/ordering come directly from
`phase_started`/`phase_completed`/`phase_failed` events; the only
non-literal part is the replay's per-step pacing (a fixed 700ms, like a
video scrubber), not the sequence or numbers themselves.

**Grouping rule, self-updating by design.** A "node" is any phase name
matching `^Phase \d+` (the convention every `main.py` top-level phase already
follows); everything else (`iforest.fit`, `preprocessing.fit_transform_panel`,
...) is a nested sub-event, attributed to whichever top-level phase was open
via file order in the JSONL (strictly append-only, single-process, so line
order is true emission order — no timestamp-precision issues even though
`observability._now()` only has second resolution). A trailing `[modelname]`
in a phase name becomes a `model` tag, so the per-model branch in Phases
8/8b/9/10 renders as distinct nodes without this file knowing `main.py`'s
phase names beyond that one regex. This module has zero hardcoded phase list
— verified 2026-08-19 that it correctly rendered **18 nodes** from a real run
including the true conditional branching ("Phase 9: top-N Excel deliverable"
appearing only for `model=vae`, never `model=iforest`, matching
`config.deliverable_models` under stacking).

Wired into `main.py` on **both** the success path (Phase 12b, after
`end_run()` so the diagram's own summary panel reflects final status) and the
failure path (`main()`'s except-block, so a crash still produces a diagram
showing exactly which phase stopped) — both best-effort, wrapped in
try/except, since a visualization bug must never mask (or be mistaken for) a
real pipeline failure. Verified both: a clean run produced 18 completed
nodes; a forced failure (`--numeric-transform bogus_xyz`) produced 4 nodes
ending in `status: failed`.

**Design pass.** Built as a UI/dashboard treatment (scanned, not read):
status encoded as color *and* text label *and* a left-border stripe on each
node (never color alone), semantic ok/bad colors kept separate from the
amber accent, tabular-nums on every numeric column, `prefers-reduced-motion`
respected, keyboard-operable nodes (`tabindex` + Enter/Space). Deliberately
**no Google Fonts / CDN** — system font stacks only (sans for labels, mono
for data/run-ids/durations) — this file inherits the project's existing,
explicitly-stated convention for its other HTML artifact (`report.py`:
"self-contained... no external CDNs/network, opens offline anywhere"), which
takes precedence over generic webfont advice for a file the pipeline itself
writes to disk on every run. Palette is a deliberately-chosen cool
slate-navy + amber (not the near-black-plus-blue-accent look most dev-tool
dashboards default to), correctly implemented as three CSS states (bare
`:root` light default, `prefers-color-scheme: dark` guarded by
`:not([data-theme="light"])`, and an explicit `[data-theme="dark"]` override)
so it also renders correctly if published as a claude.ai Artifact, which it
was for visual verification (this repo has no browser/screenshot tool of its
own, so publishing was the only way to actually confirm the interactive
replay renders/behaves as coded, not only that it parses).

**Superseded 2026-08-19 (same day, later): live local view added.** The
static replay above only appears *after* a run ends. `start_live_view`
(`src/reporting/flow_visualization.py`) starts a background `127.0.0.1`-only
HTTP server (stdlib `http.server`, daemon thread, no new dependency) the
moment `run_pipeline` starts, and `main.py` opens it in the default browser
immediately via `webbrowser.open`. Its one page polls `/state` every second;
`/state` rebuilds the same node/health-check structure fresh from
`run_events.jsonl` on every request, so the diagram grows in real time as
phases actually complete. Never a claude.ai Artifact publish, never reachable
off the local machine (binds `127.0.0.1`, not `0.0.0.0`). Controlled by
`PipelineConfig.live_view` (default `True`) / `--live-view`/`--no-live-view`.
Verified 2026-08-19 by running the pipeline as a background process and
polling `/state` twice, 15s apart, from outside that process: went from 4
nodes (`Phase 4` shown `running`) to 7 nodes (`Phase 4` now `completed`,
`duration_s=29.491`, `Phase 6` now `running`) -- genuinely live, not a
snapshot. The static post-run file (`build_flow_visualization`) still runs
too, for later review; the two are complementary, not redundant.

---

## 2026-08-19 — Strategy default: unsupervised unless requested

`PipelineConfig.supervised` (default `False`) / `--supervised`/`--no-supervised`.
Previously `supervised = n_pos > 0` in Phase 3 -- the run's strategy silently
depended on whether a ground-truth file happened to exist, not on an
explicit choice, so the exact same code could evaluate supervised or
unsupervised depending only on what data happened to be sitting in
`artifacts/data/`. Ground truth is still always loaded (diagnostics need it
regardless), but it now only feeds the tuning objective and supervised
metrics when `--supervised` is passed; `n_pos > 0` remains a hard
requirement even then (an explicit request against a label-free dataset logs
a warning and falls back to unsupervised rather than raising or silently
computing degenerate PR-AUC). Verified 2026-08-19: a real run against ground
truth with 111 real positive labels (1.85% rate) still logged `->
UNSUPERVISED evaluation (strategy=default)`, confirming the presence of
usable labels no longer flips the strategy on its own.

---

## 2026-08-19 — SHAP progress heartbeats

`shap_summary_iforest` (`src/interpretability/iforest_explain.py`) previously
went silent between `Starting interpretability.shap_iforest` and the next log
line, with a numpy `FutureWarning` in between that looks like a stray error
but is actually incidental evidence of *progress* (it only fires once
execution reaches the `shap.summary_plot(...)` call, i.e. after SHAP-value
computation already finished) -- easy to misread as a hang. Added explicit
start/finish log lines around both the `TreeExplainer` computation and the
beeswarm-plot rendering, and wrapped `_permutation_importance`'s per-feature
loop (path 3, the label-free fallback) in `tqdm`. Verified 2026-08-19 against
a real run: `Computing SHAP values via shap.TreeExplainer...` /
`shap.TreeExplainer finished.` / `Rendering SHAP beeswarm plot...` /
`SHAP beeswarm plot saved.`, each pair ~10s apart -- no more silent gap.

---

## 2026-08-19 — Windows atomic-write retry

Hit in production: `PermissionError: [WinError 5] Access denied` on
`os.replace(tmp, path)` while checkpointing VAE tuning trial 29's
`best_model.pth` (`src/models/vae.py::_save_state`). Not a logic bug --
Windows (unlike POSIX) can transiently deny a rename/replace onto a
just-written file while antivirus/the search indexer briefly holds it open;
it happened on 1 of ~30 trials, not reproducibly. `src/utils/atomic_io.py::atomic_replace`
wraps every `os.replace(tmp, ...)` in this codebase (VAE checkpoints, VAE and
Isolation Forest best-params YAML -- all three call sites updated) with up to
5 retries, exponential backoff (0.1s/0.2s/0.4s/0.8s), **only** for
`PermissionError` -- any other exception (e.g. a genuinely missing source
file) still propagates on the first attempt, so this cannot mask an actual
bug the way a blanket retry would. Verified with a 4-case test: succeeds
immediately with no contention, recovers after simulated transient failures,
re-raises after exhausting retries against a simulated permanent lock, and
never retries a non-`PermissionError` exception.

---

## 2026-08-19 — Compact yyyyMM period parsing

`pd.to_datetime(["202401"])` raises (`month must be in 1..12`) rather than
reading `2024-01-01` -- a bare 6-digit string is ambiguous without a format
hint, so pandas' dateutil fallback reads it as a 4-digit year plus a 2-digit
"month" of `01`...`41` extracted some other way and rejects the impossible
values. Real banking panels ship the period this way often enough that it
needed handling, not just a documented workaround.

`src/data/loader.py::detect_period_format` inspects a column's values (after
normalizing numeric dtypes through nullable `Int64` first, so `202401` and
`202401.0` both match) against two explicit compact patterns --
`^\d{6}$` -> `"%Y%m"`, `^\d{8}$` -> `"%Y%m%d"` -- and `parse_period_column`
passes the detected format straight to `pd.to_datetime`, so parsing is
deterministic instead of left to dateutil's inference. Everything else
(`2024-01-01`, already-`datetime64`) still goes through plain
`pd.to_datetime(series)` unchanged. A genuine parse failure is logged and the
column is returned **unmodified**, not coerced with `errors="coerce"` --
silently turning every period into `NaT` would let `chronological_split` and
the panel feature engineer run over missing timestamps without ever raising;
`assumptions.validate_panel`'s `temporal.parseable` check is what turns an
unparsed period column into an actual stop. Verified against 6 cases (yyyyMM
string, yyyyMM int, yyyyMMdd, ISO strings, already-datetime, and genuine
garbage) -- all six parse or fail exactly as intended.

---

## 2026-08-19 — Cancellation is now a detected, terminal state

Two independent gaps, found by actually sending real signals to a running
process rather than reasoning about the code:

**1. `except Exception` in `main()` never saw Ctrl+C.** `KeyboardInterrupt`
inherits from `BaseException`, not `Exception` -- confirmed the bug by
reading the exception hierarchy, not assumed. The run's event stream was
left mid-phase with no terminating `run_ended`, so the live view (added
earlier the same day) kept showing the last phase as "running" forever,
exactly the complaint that prompted this fix. `main()` now has a dedicated
`except KeyboardInterrupt:` above the existing `except Exception:`, sharing
a new `_close_run_as(status, error, live_view)` helper that both paths call.

**2. Ctrl+Break (Windows `SIGBREAK`) was not, and cannot by default, become
a `KeyboardInterrupt` at all.** Discovered empirically, not assumed: a bare
`time.sleep(30)` script with **no application code** hard-kills on
`CTRL_BREAK_EVENT` with `STATUS_CONTROL_C_EXIT`, before any Python `except`
clause runs, every time. `signal.getsignal(signal.SIGBREAK)` reads `0`
(`SIG_DFL`) by default -- unlike `SIGINT` (real Ctrl+C), which CPython wires
to `default_int_handler` out of the box. `main._install_sigbreak_handler`
now registers a handler that raises `KeyboardInterrupt` on `SIGBREAK` too
(Windows-only; a no-op everywhere else via `hasattr(signal, "SIGBREAK")`,
since plain Ctrl+C already works correctly on every platform without this).

**Fundamental limit, not fixed and not fixable without a much larger
redesign:** if the interrupt arrives while execution is inside a long-running
C-extension call (matplotlib rendering, some torch/scipy internals) rather
than in Python bytecode, CPython cannot act on a pending signal until that
call returns control -- confirmed by reproducing a hang-until-forced-kill
sending `CTRL_BREAK_EVENT` ~20s in, mid-`plot_transform_diagnostics`. A
`--quick` run interrupted at t=2s (pure pandas/Python) closed cleanly with
`run_ended`/`status: "cancelled"` and exit code 130 every time tested; one
landing inside a plotting call did not, in the time tested. Running each
phase in a killable subprocess would close this fully but is a materially
different architecture, not attempted here.

**Live view now shows "interrupted" even when the process dies before
writing anything.** `flow_visualization._LIVE_HTML`'s poll loop tolerates one
missed `/state` request (a transient hiccup) but on two consecutive misses
concludes the server -- and therefore the pipeline process that owns it --
is gone, and freezes the UI: any node still shown "running" is relabeled
"interrupted" client-side, without needing a final event from the Python
side at all. This is the layer that actually covers the C-extension-block
case above: even when `_close_run_as` never gets to run, a live view that
was open at the time still reflects the interruption instead of a stale
"running" snapshot. `build_flow_visualization` (the static, post-run file)
gained the equivalent server-side relabeling: any node whose `run_ended`
status is `"cancelled"`/`"failed"` but whose own phase status is still
`"running"` is written out as `"interrupted"`, for review after the fact,
in the (common) case the Python-side cleanup did run.

---

## 2026-08-20 — Report redesign: Plotly, explanations, statistical rationale

`src/reporting/report_content.py` (new) owns the report's *meaning* --
indicator glossary, statistical-reliability rationale, parameter glossary,
ML-vs-econometrics positioning, and the interactive charts -- while
`report.py` keeps owning document assembly. The split exists because the
explanatory text is long and is edited for entirely different reasons than
the HTML/MD scaffolding.

**Interactive charts (Plotly 6.8.0, already a declared dependency).** Five
figures: ROC+PR (both models, OOT), one score-distribution histogram **per
model**, headline metric comparison, and recall by injected anomaly type.
Two rules shaped them:

* *Never a dual axis.* The two detectors' scores live on different scales
  (isolation depth vs reconstruction error), so score distributions are one
  chart **per model**, never overlaid. ROC and PR *are* overlaid across
  models because both axes are rates in [0,1]; they sit in separate subplots
  from each other because they answer different questions.
* *Honest baselines drawn on the chart.* The PR panel draws the panel's own
  anomaly rate as its dashed baseline (not 0.5), annotated with the value,
  because PR-AUC on a ~2% positive rate is otherwise read against the wrong
  reference.

Offline-safety is preserved: `plotly.js` is inlined once (~4.8MB; subsequent
figures cost ~7KB each), so the HTML is ~6.8MB and loads with **zero remote
requests** -- verified by scanning every `<script>/<link>/<img>/<iframe>` in
the output (23 tags, 0 pointing at a remote URL; the 16 `https://` strings
that *do* appear are map-tile attribution literals inside the plotly bundle,
which this report never renders).

**Colour.** Series use categorical slots 1-2 of the project's validated
reference palette, iForest always blue / VAE always orange, never cycled.
The palette validator was run against **this report's own surfaces**, not
the skill defaults: light `#ffffff` and dark `#171e2d` both pass all six
checks (worst adjacent pair ΔE 24.7 protan / 33.6 normal-vision light; 26.8 /
31.8 dark). A single fixed colour set was tried first and **FAILS** -- slot-2
orange `#eb6834` sits at OKLCH L 0.671, outside the dark band's 0.48-0.67 --
which is why per-theme hexes exist and why `THEME_RESTYLE_JS` recolours every
trace on load and on each theme toggle (Plotly cannot read CSS custom
properties, so figures cannot inherit the page's tokens the way everything
else does). Backgrounds are transparent so the panel shows through in both
themes. Identity never rests on colour alone: every chart legends its series.

**Anti-overlap.** Every figure reserves `margin.t=56` with the title pinned
at `y=0.97`, and bar charts set `cliponaxis=false` so outside value labels
are not clipped -- verified present in all five emitted figures. Chart
containers carry `min-width:0` because a flex/grid child defaults to
min-content width, which is what lets a wide SVG push a page into horizontal
scroll.

**Figures actually removed, not just reordered.** Five static PNGs are now
dropped from the report per run (logged as such):
score-distribution and reconstruction-error PNGs and both ROC/PR PNGs are
strictly less informative than their interactive replacements (which also
draw the calibrated alert threshold); PCA embeddings are dropped where a UMAP
embedding of the same matrix is present. A genuine duplicate was found and
fixed: **"VAE latent space" was being emitted twice by two different
modules** (`models.vae.plot_latent_space` and
`interpretability.vae_explain.latent_space_plot`), both reaching the report.
The ten raw-feature histograms are kept but demoted to a collapsed group
placed last -- they answer "is the input sane", which the reliability section
already asserts.

**Figure grouping was silently broken and is fixed.** `_group_figures` keyed
off the `reports/figures/<module>/` parent folder, but that layout was
deliberately flattened long ago (`CONTEXT.md` "HARD RULE -- Figures"), so
every figure's parent became the single word "figures" and *all* groups
collapsed into one bucket titled "Figures" nested under the "## Figures"
heading -- visible in the Markdown output as a duplicated heading. It now
groups by filename prefix, which is the convention the flattening was
justified by.

**New sections, ordered for a reader rather than for the code:** results
(charts) -> per-model detail -> how to read each indicator -> why the results
are statistically trustworthy -> parameters and what each value means ->
ML-vs-econometrics -> diagnostic figures. The reliability section documents
eight real gates (chronological split, train-only preprocessing fits, causal
panel features, key integrity, finite matrix, validation-calibrated
threshold, person-overlap measurement, rank stability), each with what it
verifies, why it matters, and what a failure would mean. The parameter
tables mark each value <span>this run</span> (selected by the tuner) or
<span>default</span>, so the report says what the run actually did rather
than only what the defaults are.

**On the ML-vs-econometrics question:** the report states the verdict as
machine learning (unsupervised anomaly detection), and answers "which
parameter drives the classification" explicitly -- it is **the calibrated
threshold**, not any model coefficient, since neither detector exposes an
interpretable coefficient at all. It also names the two parameters most
easily mistaken for it: the Isolation Forest's `contamination` (moves only
`predict()`'s internal boundary, not the ranking this pipeline uses) and the
top-N queue size (an operational capacity decision).

---

## 2026-08-20 — Ruteo de features por dtype: iForest numérico, VAE completo

**Qué cambió.** Los dos detectores ya no reciben la misma matriz — ver el
contrato vigente en `CONTEXT.md` ("Feature routing by dtype"). Esta entrada
conserva la medición.

**Medido 2026-08-20** (`--quick`): 67 features totales, 44 derivadas de
categóricas → el Isolation Forest ve 23 features numéricas, el VAE ve las 67.
Cubierto en su momento por `TestPerModelFeatureRouting` (3 tests, ya
eliminados junto con el resto de la suite el 2026-08-22), incluyendo uno que
agregaba una columna de texto nueva y una numérica con nombre "categórico"
para confirmar que el ruteo seguía el dtype y **no** el nombre.

---

## 2026-08-20 — Interpretabilidad movida después de los Excel

La Fase 10 salió del bucle por modelo y ahora corre después de que **todos**
los deliverables Excel están en disco (contrato vigente en `CONTEXT.md`).
Verificado 2026-08-20: Fase 9 (Excel) termina 15:50:49, Fase 10 arranca
15:50:50.

---

## 2026-08-20 — El reporte respeta la estrategia no supervisada

**Corrección de un error propio introducido el día anterior.** Al agregar los
gráficos Plotly se calcularon métricas supervisadas (ROC/PR, comparación de
métricas, recall por tipo) *sin importar* el valor de `config.supervised`,
con el argumento de que evaluar contra ground truth offline es su propósito
documentado. Eso contradice el default explícito del proyecto: la estrategia
por defecto es **no supervisada**, y un reporte que muestra curvas ROC está
afirmando una evaluación que la corrida no realizó.

Ahora `chart_data` adjunta etiquetas **sólo** si `supervised` es verdadero, y
los tres gráficos con etiquetas están condicionados a eso. El glosario de
indicadores también se filtra: sólo lista los indicadores que la corrida
realmente produjo. Verificado: 0 menciones de ROC-AUC / PR-AUC /
Precision@10 / Recall@10 en el cuerpo del reporte de una corrida por defecto.

**Gráfico nuevo que sí aporta sin etiquetas — acuerdo entre detectores.**
Scatter de percentil de score bajo cada detector, con rho de Spearman y
solapamiento del top-5% anotados (ver `CONTEXT.md` "Reporting"). Es el
diagnóstico central cuando no hay ground truth: los dos modelos parten de
principios distintos (geometría de aislamiento vs error de reconstrucción) y
además ven conjuntos de features distintos, así que un individuo que ambos
rankean alto está corroborado por evidencia independiente.

**Leyendas fuera del área de título.** Estaban en `y=1.0` (orientación
horizontal arriba), que es la misma banda donde vive el título — de ahí la
superposición. Ahora van abajo (`y=-0.22`, ancladas al centro) con
`margin.b=86` reservando esa franja.

---

## 2026-08-20 — Pantalla de carga con progreso real

La vista en vivo mostraba sólo el diagrama de nodos. Ahora encabeza con
porcentaje, barra de progreso, fase actual con spinner, y conteo
`hechas/vistas` + segundos de trabajo acumulados. El porcentaje se calcula
como completadas/vistas y se **etiqueta como tal** ("phases done"), no como
avance total: el pipeline descubre sus fases desde los eventos conforme
ocurren, así que no existe un total conocido de antemano y presentarlo como
"% del run" sería inventar una cifra.

---

## 2026-08-22 — Test suite removed

The project carried a pytest suite (peaked at 319 tests across 11 files:
data, preprocessing, both models, evaluation, interpretability/reporting,
tuning budget, linear scaling, Optuna storage) used throughout development
to verify every fix logged in this file. Removed by explicit decision
(2026-08-22): not needed to run the project, so not shipped. `pytest` and
`pytest.ini` were removed alongside it. Several `docs/*.md` files still name
specific test files/classes as the historical evidence for a claim in this
changelog and in `docs/` — those names are kept as a record of what was
checked at the time, even though the files no longer exist on disk.

---

## 2026-08-22/23 — Synthetic-data provenance marker: introduced, then fixed

A run over generated data was writing `best_params_*.yaml`, model
checkpoints, and Optuna studies to the **same path** a real-data run uses,
with nothing in the artifact recording which kind of data produced it.

**First fix (2026-08-22):** `generate_synthetic_panel` writes a provenance
marker (`.synthetic.json`) next to the CSV; `PanelSchema.is_synthetic`
propagates it; `main.py` redirects model/params/study under `_dev/` for a
non-official run. Verified: after a `--quick` run on synthetic data,
`artifacts/tuning/` and `artifacts/models/` stayed empty and everything
tunable landed under `_dev/`.

**Regression found the same fix introduced (2026-08-22), fixed 2026-08-23:**
the marker was written as `.synthetic.json` **in the panel's directory**, and
read back by existence alone — so *any* CSV in that folder, including real
data placed in the same default `artifacts/data/` location, was classified as
synthetic, and an official run's tuned parameters ended up under `_dev/`
instead of their real location. Fixed with two independent checks: the
marker is now named after its panel (`data.csv` → `data.csv.synthetic.json`)
and its `panel` field must also name that same file (covering a marker
copied or renamed alongside a different CSV). An unreadable marker resolves
to **real data**, the conservative direction — it protects official
artifacts from being silently redirected, and it is noisy when wrong, so it
gets noticed. Verified: generated panel → synthetic; reload of the same →
synthetic; real panel in the same folder → real; a marker copied next to
another panel → ignored.

**Related, same week:** a ground-truth file left behind by an earlier
synthetic run could be picked up by a later real-data run (`_discover_
ground_truth` finds `ground_truth.parquet`/`.csv` by naming convention, not
content) — its key columns (`entity_id`, `period`) don't match a real panel's
(e.g. `id`, `codmes`), so the existing column-mismatch check already falls
back to unsupervised safely; the only change was making the warning message
name the likely cause and the fix (delete the stale file, or rename the real
one to match) instead of just listing mismatched columns.

---

## 2026-08-23 — Four deliverable-format changes

User-requested changes to deliverable format, not diagnostic findings —
recorded here because they change the pipeline's default behavior. Current
contracts are in `CONTEXT.md`; kept here is the reasoning.

**1. Numeric imputation defaults to zero.**
`build_preprocessing_pipeline(impute_numeric=...)` accepts `"zero"` (new
default) alongside `"median"`/`"mean"`/`"most_frequent"`; `--no-zero-impute`
restores `"median"`. Zero maps to `SimpleImputer(strategy="constant",
fill_value=0.0)` — it estimates nothing from the data, so unlike `"median"`
it cannot leak between train/test and does not change if the fit window
changes. The cost: on a column where 0 already means something, an imputed
zero is indistinguishable from a real one, which is why
`add_missing_indicators=True` staying on matters — the pair (zero-impute +
indicator) doesn't lose information even though zero-impute alone would.

**2. OOT deliverable: percentile with band, period included.**
`export_oot_top_anomalies` changed its default selection from a fixed top-50
headcount to every individual at or above P90 of the OOT block, with a
`percentil`/band column (`p90`/`p95`/`p99`) grading each row by the highest
band it reaches. Rationale: a fixed top-N has no distributional meaning
("50" is 2.5% of a 2,000-entity portfolio and 0.05% of a 100,000-entity one),
while a percentile scales with portfolio size and keeps its interpretation
regardless. Only three bands, because more stops being actionable — nobody
triages ten urgency levels. Cut-offs are computed over the **full** OOT
population before filtering, never over the already-exported subset (the top
5% of the top 5% is not P99). The period column was added because the export
already deduplicated to one row per individual (their highest-scoring month)
but didn't expose *which* month that was — without it an analyst can't
locate the observation in time to investigate it. (Superseded same day, see
below — the default moved from a first documented P95 cut to the current
P90/p90/p95/p99 scheme once the genuine OOT/test split landed.)

**3. Full per-variable attribution workbook.**
`src/interpretability/attribution_export.py::export_attribution_workbook`
writes `feature_attribution.xlsx`, one sheet per model, with every variable
— not just the 20 a beeswarm/bar chart can legibly show. The chart's top-20
crop is correct for a figure; cropping the *deliverable* to the same 20 would
silently drop anything outside it, which someone auditing the model or
monitoring feature drift needs. Two different methodologies per sheet
(`mean_abs_shap` for the forest, `mean_reconstruction_error` for the VAE,
since there is no SHAP analogue for a VAE in this pipeline) — each sheet
documents its own methodology in row 1 so the numbers are not assumed
comparable across sheets.

**4. Genuine OOT/test split and P90/P95/P99 bands** (folds into #2 above —
this is the change that landed the *current* default described in
`CONTEXT.md`). Superseded the same-day P95-only cut with the current
P90-with-p90/p95/p99-band scheme, and made the OOT block genuinely distinct
from the test block used for model metrics (see the Phase 2 split table in
`docs/leakage_free_pipeline.md`). Any doc or note elsewhere in this project
that still says "P95 default" or `oot_p95_<model>.xlsx` predates this and
describes the version before this change.

---

## 2026-08-26 — Interpretability hangs at ~150-200 features: found, measured, fixed

**Reported symptom:** the pipeline "crashes" during the interpretability
phase on a real dataset whose preprocessed feature count is ~150-200 —
several times more than this project's synthetic default (22 raw columns,
~23-67 after preprocessing depending on `--panel-features`), which is why it
had not surfaced before.

**Root cause, found by measuring rather than guessing.** Every call site in
`main.py` Phase 10 already wraps its interpretability call in
`try/except Exception: ... continuing` (main.py:1092-1130), so a normal
Python exception was never going to be the explanation — whatever was
happening had to be bypassing that. `shap_summary_iforest`
(`src/interpretability/iforest_explain.py`) tries three tiers in order:
`shap.TreeExplainer` (fast, native), a model-agnostic `shap.Explainer` over
`score_samples` (a documented, explicitly-flagged-as-risky fallback), and
manual permutation importance (the last resort). Reproduced directly against
this project's own `IsolationForestDetector` class (600 trees, 190 features,
4000 rows — a realistic real-data scale) with `shap.TreeExplainer` forced to
fail, to isolate each fallback's cost in seconds rather than reasoning about
it:

| Path | Cost at 190 features (measured) | Extrapolated total (default settings) |
| --- | --- | --- |
| 1. `shap.TreeExplainer` (happy path) | 8-23s regardless of tree count tried (200-600 trees) | fine, unaffected |
| 2. `shap.Explainer` (model-agnostic, `PermutationExplainer`) | **~24.5s/row** (worse than the ~4.8s/row measured at 180 random-normal features in an isolated repro — real forest, more trees) | **~2.7-13.6 hours** to explain the default 2000 rows |
| 3. manual permutation importance | ~1.27s per `(feature x repeat)` call at 600 trees/5000 rows | ~762s (~12.7 min) for 200 features x 3 repeats before this fix |

Path 2's cost is `O(n_features)` per explained row (each SHAP evaluation
re-scores the whole forest ~`2*n_features+1` times), and path 3's is
`O(n_features x n_repeats)` full-dataset `score_samples()` calls — both scale
with the feature count, which is exactly the variable the report named. At
20-70 features neither fallback was ever slow enough to notice; at 150-200
it is the difference between seconds and hours, which reads exactly like a
crash to someone watching a pipeline that took minutes end-to-end before
this phase.

**Fix** (`src/interpretability/iforest_explain.py`): both fallbacks are now
budgeted instead of unbounded.

* The model-agnostic path times a small calibration batch
  (`_MODEL_AGNOSTIC_CALIBRATION_ROWS = 2` rows) against the real explainer,
  extrapolates a per-row cost, and explains only as many additional rows as
  fit inside `_MODEL_AGNOSTIC_TIME_BUDGET_S = 60.0` seconds — down to just
  the calibration rows themselves if even those project past budget. A
  static feature-count cutoff was considered and rejected: it would not
  generalize across tree count, hardware, or shap version, whereas timing
  the actual call on the actual model self-corrects for all three.
* The permutation-importance fallback subsamples rows to
  `_PERM_IMPORTANCE_MAX_SAMPLES = 1000` (fewer than the SHAP paths need,
  since an aggregate importance ranking needs less data than a per-row
  attribution) and caps `n_repeats` so total `score_samples()` calls never
  exceed `_PERM_IMPORTANCE_CALL_BUDGET = 150` — e.g. at 200 features the
  requested 3 repeats become 1, logged explicitly.
* When the model-agnostic path returns fewer rows than it was handed, the
  beeswarm plot's feature matrix is re-sliced to match (`Xd =
  Xd[:shap_values.shape[0]]`) so the figure still renders instead of
  silently degrading to the bar-chart fallback on every bounded run.

**Verified two ways.** (1) A stress test against the real project classes
(`IsolationForestDetector`, `VAEDetector`, not a synthetic stand-in) at 190
features / 4000 rows / 600 trees, forcing each fallback path in turn via
monkeypatching `shap.TreeExplainer`/`shap.Explainer`: path 2 (previously
projected at 48,959s = ~13.6 hours for the full row count) now completes in
**49.4s**; path 3 completes in **26.8s** with `n_repeats` auto-reduced to 1;
the untouched happy path (path 1) and the VAE's `latent_space_plot`/
`reconstruction_error_by_feature` (never part of the problem — they scale
with latent dimension and batch size, not raw feature count, confirmed at
the same 190-feature scale) are unaffected. (2) A full `python main.py
--quick --no-tune` run end-to-end on this project's existing synthetic
`artifacts/data/` (22 features, so the happy path — this only confirms no
regression at normal scale, not the fix itself): completed successfully,
51/51 observability health checks passed, `run_ended status=success`.

**Not changed, and why:** `shap.TreeExplainer` itself (path 1) was not
touched — it did not fail in any configuration tried here (`max_features`
from 0.3-1.0, `bootstrap` True/False, up to 600 trees, up to 200 features all
succeeded in 2-23s), so whatever forces a real run onto the slow fallbacks is
environment/version-specific and outside what could be reproduced directly;
the fix makes the fallback *safe to fall back to* regardless of why path 1
was unavailable, rather than chasing why path 1 might fail in one specific
environment.

> **Superseded two days later (2026-08-28): path 1 needed the same fix.** A
> real run hung on the exact log line `"Computing SHAP values via
> shap.TreeExplainer..."` — i.e. inside path 1 itself, not a fallback — for
> 6+ minutes. The "not changed" reasoning above turned out to be right that
> path 1 never *failed* in any configuration tried, but wrong that it was
> therefore safe: it can still be *slow* without failing. See the entry
> below for the actual mechanism (found this time, not left unexplained) and
> the fix, which finally closes this gap symmetrically across all three paths.

---

## 2026-08-28 — TreeExplainer itself needed budgeting too; root cause found

**What happened.** A production run hung for 6+ minutes with the last log
line being `"Computing SHAP values via shap.TreeExplainer over ... row(s) x
... feature(s)..."` — inside path 1, the one path the 2026-08-26 fix
deliberately left unbounded because it never failed in any test run against
it. This time the cause was found and measured directly, not left as "some
version/environment difference."

**Root cause: `max_samples` as a float fraction, on a training block that
spans several months.** `IsolationForest`'s `max_samples` this project tunes
over `{"auto", int in {64,128,256}, float in [0.3, 1.0]}`
(`src/models/iforest.py`). `"auto"` means `min(256, n)` rows per tree —
**independent of total dataset size** — but a float means *that fraction of
whatever training set is passed in*. `main.py` fits the Isolation Forest on
the full train block, which spans several months, not one. A controlled
sweep (`IsolationForestDetector`, 125 features, 100 trees) isolated this
cleanly:

| Training rows | `max_samples` | Tree depth | Leaves/tree | Projected cost (2000 rows, unbounded) |
| ---: | --- | ---: | ---: | ---: |
| 20,000 (1 month) | `"auto"` | 8 | 64 | 1.8s |
| 20,000 (1 month) | `0.5` | 14 | 756 | 63.6s |
| 200,000 (10 months) | `"auto"` | 8 | 56 | 1.8s |
| **200,000 (10 months)** | **`0.5`** | **17** | **2,761** | **341s (5.7 min)** |

`"auto"` stays at depth 8 regardless of total rows (confirming it is immune
to dataset size); a float fraction on a large training block is what
produces the deep, high-leaf-count trees `TreeExplainer`'s cost scales with.
At 300 trees (still within this project's Optuna range, which goes to 600)
the same 200,000-row/`max_samples=0.5` configuration measured **0.532s/row**,
i.e. **~17.7 minutes unbounded for 2000 rows** — closely matching the "6+
minutes and still running" symptom reported in production.

**Two other hypotheses tested and ruled out** (same sweep): low-cardinality
"numeric but effectively categorical" columns mixed into the Isolation
Forest's numeric-only feature block (e.g. integer-coded regions/segments —
these still route to the Isolation Forest under the dtype-based split in
`CONTEXT.md`'s "Feature routing by dtype", since routing is by dtype, not
semantic meaning) showed no measurable effect on tree depth or `Tree
Explainer` cost at `max_samples="auto"` in this sweep; and feature count
alone (125) at the default `max_samples="auto"` cost 1.8s regardless of
whether the training set was 20,000 or 200,000 rows. Feature count still
matters for paths 2/3 (2026-08-26 entry above) — it just is not what made
path 1 slow here.

**Fix.** `shap.TreeExplainer` (path 1, `src/interpretability/
iforest_explain.py`) now gets the identical calibrate-then-bound treatment
already applied to paths 2/3: times `_TREE_EXPLAINER_CALIBRATION_ROWS = 5`
rows against the real model, extrapolates a per-row cost, and explains only
as many additional rows as fit `_TREE_EXPLAINER_TIME_BUDGET_S = 90.0`
seconds (a larger budget than path 2/3's 60s, since path 1 is legitimately
expected to be the fast path in the common case and a real but moderate
slowdown should not trigger premature bailout). Verified against the exact
reconstructed scenario (200,000 rows, 125 features, 300 trees,
`max_samples=0.5`): the full `shap_summary_iforest` call, which would have
cost on the order of 17+ minutes for the TreeExplainer call alone, now
completes in **110.2s** — calibration correctly measured 0.532s/row and
capped the explanation to 169 of 2000 rows.

**Stated residual limitation (not fixed, by design):** this bounds cost that
scales with the number of rows explained *after* calibration returns. If
even the `_TREE_EXPLAINER_CALIBRATION_ROWS` (or `_MODEL_AGNOSTIC_
CALIBRATION_ROWS`) calibration rows themselves do not return — a genuine
per-call pathology (e.g. an actual infinite loop or C-level defect) rather
than "slow proportional to rows" — no in-process timing can preempt that; a
hard kill would require running the call in a subprocess, which was
considered and not implemented (added latency/complexity for every call,
not just the pathological case, and no evidence yet that the "even
calibration hangs" case is real rather than theoretical).

**New: fine-grained health checkpoints, for exactly the "where did it stop"
question.** Both `iforest_explain.py` and `vae_explain.py` now call a
module-local `_checkpoint(name, **observed)` at every meaningful sub-step
(calibration started/measured for all three paths, full-explain started/
done, beeswarm render started/done/failed, UMAP started/done/failed,
permutation-importance progress every ~25% of features, and a final
`completed` per function) — each an always-passing `observability.check(...)`
under `interpretability.iforest_shap.*` / `interpretability.vae_explain.*`.
Verified end-to-end with a real `python main.py --quick --no-tune` run: 16
checkpoints recorded in `artifacts/logs/run_events.jsonl` in the correct
order with timestamps and per-step data (e.g. `tree_explainer_calibrated`
carrying `calib_seconds`, `seconds_per_row`, `planned_rows`, `bounded`), and
mirrored live to the console dashboard's existing "Supuestos (IF/VAE)" panel
via `observability.add_check_observer` with zero changes needed there. The
diagnostic use: whichever checkpoint name is *last* in the log when a run
stalls is the exact sub-step in flight — a `tree_explainer_calibration_
started` with no following `tree_explainer_calibrated` means the calibration
call itself is the one hanging (the residual gap above), as opposed to a
`tree_explainer_explain_started` with no `tree_explainer_done`, which means
the budget-bounded continuation is unexpectedly slow (a calibration
under-estimate, not a hang). **This residual gap is closed by the entry
below, the same day.**

---

## 2026-08-28 (later the same day) — real hard-kill: a production run still hung 3+ hours

**What happened.** The fix above (calibrate-then-bound) was deployed, and a
real run still hung for **3+ hours with no forward progress**, past every
soft budget defined. This is exactly the residual gap the previous entry
named explicitly: a soft time budget only protects cost that scales with the
number of rows explained *after* calibration returns -- it cannot do
anything if the *calibration call itself* (or the subsequent bounded explain
call) simply does not return within a reasonable time, because Python cannot
preempt a blocked call into a native C/Cython library (shap's internals)
from a timer, a thread, or a signal handler running in the same process. The
requirement stated directly: implement an actual, enforced kill.

**Design.** `shap.TreeExplainer` and the model-agnostic `shap.Explainer`
(paths 1 and 2) now each run inside a freshly spawned child process
(`multiprocessing`, `"spawn"` context -- required on Windows, and confirmed
safe here since `main.py`'s entry logic already lives behind `if __name__ ==
"__main__":`, so re-importing `__main__` in the child cannot re-trigger the
pipeline). The parent polls the child (`_run_with_hard_kill`,
`_HARD_KILL_POLL_S = 1` second) up to a hard ceiling
(`_TREE_EXPLAINER_HARD_KILL_S = 180`, `_MODEL_AGNOSTIC_HARD_KILL_S = 150`);
if the child has not sent a final result by then, it is forcibly terminated
(`Process.terminate()`, then `Process.kill()` after a `_HARD_KILL_GRACE_S =
5`-second grace period) and that path is treated as failed -- the next path
is tried, exactly as if it had raised an exception. Both ceilings, the grace
period, and the poll interval are plain integer seconds; no tunable in this
change is a fraction, since none of them needed to be.

**The child still reports progress before the risk point.** Rather than
having the child do everything silently and only report a final result
(which would mean a kill erases the calibration measurement too, undoing the
2026-08-26 checkpoint work), each child sends a `"calibrated"` message
*before* attempting the bounded full explain. The parent's polling loop
processes that message immediately (`on_progress` callback) and emits the
normal `*_calibrated` / `*_explain_started` checkpoints from it -- so a kill
during the full-explain phase still leaves the calibration numbers in
`run_events.jsonl`, only the final `*_done` checkpoint is missing, replaced
by a new `*_hard_killed` one recording the ceiling that fired.

**Why a subprocess and not a thread.** A `threading.Thread` with
`join(timeout=...)` can only stop *waiting*; the thread itself keeps running
in the background (Python has no API to forcibly stop a thread), consuming
CPU and holding whatever memory it allocated, for as long as the blocked
call takes to return on its own -- which, per the 3+ hour report, may be
never within a practical session. Only a separate OS process can be
unconditionally terminated regardless of what it is doing.

**Verified three ways:**

1. **The kill mechanism itself**, isolated from shap entirely: a worker that
   sleeps 2s under a 30s ceiling returns normally; a worker that would sleep
   300s under a 5s ceiling is confirmed killed at ~5.0s (not left running to
   300s); `multiprocessing.active_children()` shows zero lingering processes
   afterward.
2. **The exact reconstructed real-world scenario** (200,000 rows, 125
   features, `max_samples=0.5`, 300 trees -- the 2026-08-28 entry above)
   still completes correctly through the now-isolated path: **112.9s**,
   consistent with the 110.2s measured before subprocess isolation (the
   ~2.7s difference is process-spawn/pickling overhead, not a regression).
3. **A real (not simulated) forced kill**, using a tight ceiling
   (`_TREE_EXPLAINER_HARD_KILL_S = 2`) against a genuinely expensive
   real model (600 trees, 190 features): path 1 was killed mid-explain,
   correctly fell through to path 2, which was *also* killed (tight
   `_MODEL_AGNOSTIC_HARD_KILL_S = 30`), correctly fell through to path 3
   (permutation importance, already budget-capped since 2026-08-26), and the
   whole call still returned a valid, non-empty importance dict in 42.8s
   total -- proving the fallback chain survives two consecutive hard kills,
   not just one.

**Numeric-argument convention applied to this change** (and worth stating
as a going-forward rule for this module): every new timing/count constant is
a plain integer -- `_TREE_EXPLAINER_HARD_KILL_S`, `_MODEL_AGNOSTIC_HARD_
KILL_S`, `_HARD_KILL_GRACE_S`, `_HARD_KILL_POLL_S`, `_MODEL_AGNOSTIC_
BACKGROUND_ROWS` -- and every value logged for a human (`calib_seconds`,
`seconds_per_row`) is rounded to at most 2 decimal places. The one
pre-existing constant that did not follow this (`_MODEL_AGNOSTIC_TIME_
BUDGET_S`, previously `60.0`) was tightened to `60` in the same pass.

**Still not, and will not be, fully closed:** if a hard-killed process
leaves the OS itself in a bad state (not observed, but not provable absent),
or if `multiprocessing.Process.kill()` fails on a given platform/Python
build, this degrades to the previous behavior for that one call. This is
believed sufficient -- `Process.kill()` sends `SIGKILL`-equivalent
termination, which the OS itself, not this process, enforces -- but it is
stated rather than assumed given how wrong the "path 1 never needs this"
assumption from 2026-08-26 turned out to be.

---

## 2026-08-28 (same day, later still) — checkpoint coverage audit: 3 gaps found and closed

**Trigger.** The user reported the anomaly report *does* get generated,
which is real evidence: if Phase 11 (report) runs, Phase 10 (interpretability)
did not hang forever on this particular run -- it more likely raised/failed
through the fallback chain (already caught by `main.py`'s per-call
try/except) rather than blocking indefinitely. That distinction matters for
diagnosis, and the checkpoint system added earlier the same day was not
actually complete enough to tell the two apart everywhere interpretability
runs -- it only covered `shap_summary_iforest`.

**Audit, three gaps found and closed:**

1. **`shap_summary_iforest` path 2 (model-agnostic Explainer) was missing a
   `_done` checkpoint** — path 1 (`tree_explainer_done`) had one right after
   a successful result, path 2 did not. Added `model_agnostic_done`,
   mirroring path 1 exactly.
2. **`path_length_analysis` had zero checkpoints** — it was judged low-risk
   (no SHAP, a closed-form calculation, historically sub-second) when the
   original checkpoint pass was scoped to the SHAP paths specifically. Added
   `path_length_started` / `path_length_completed`.
3. **`attribution_export.py::export_attribution_workbook` (Phase 10b — the
   per-model Excel-sheet writer that runs between Phase 10 and the report in
   Phase 11) had zero checkpoints at all**, since it is a separate module the
   original pass never touched. Added `started` (records which models are
   present), one `sheet_written` per model, and `completed`. This is
   directly relevant to the trigger above: Phase 10b sits *between*
   interpretability and the report the user confirmed runs, so if something
   in the pipeline were silently slow or degraded there rather than in
   Phase 10 proper, there was previously no way to tell from the checkpoint
   log alone.

**Also renamed the shared Isolation Forest checkpoint namespace** from
`interpretability.iforest_shap.*` to `interpretability.iforest.*`, since it
is used by both `shap_summary_iforest` *and* (now) `path_length_analysis` --
the old name implied SHAP-only coverage that was never quite accurate and
is now actively wrong.

**Verified with a full `python main.py --quick --no-tune` run:** 22
interpretability checkpoints recorded across all three modules (10 from
`iforest_explain.py`, 8 from `vae_explain.py`, 4 from
`attribution_export.py`), 39/39 total health checks passed, `run_ended
status=success`. Every function in all three interpretability modules that
`main.py` calls now has at least a `started`/`completed` pair, so a stall
anywhere in Phase 10 or 10b -- not just inside the SHAP paths -- is now
diagnosable from `run_events.jsonl` alone.

---

## 2026-08-28 (same day, later still) — VAE attribution: categorical granularity measured and fixed

**Report:** "mis campos strings que van al VAE son tan granulares que ocupan
casi todo el [share] del score de anomalía" -- a real mechanism, not a bug in
anything shipped earlier today. One-hot encoding turns one string column
into one column *per category*; the VAE's score (`score_samples`) and its
per-feature reconstruction-error attribution are both **sums over columns**,
so a high-cardinality categorical (many one-hot slices) can out-weigh a
single numeric column in the ranking, and -- if granular enough -- in the
score itself, purely by column count, not by being more informative. This is
the flip side of a deliberate, documented design choice (`CONTEXT.md`
"Feature routing by dtype"): the VAE keeps categorical-derived columns
specifically so category *identity* is available for the "contextual"
anomaly definition; the failure mode here is that identity's cardinality was
never checked against how much it ends up weighing.

**Two things were genuinely unclear before fixing anything: whether this was
a reporting artifact (A) or a real score-level skew (B).** Both turn out to
be answerable from the *same* measurement, because the per-column
reconstruction error already sums directly into `score_samples` -- there is
no separate "score-level" quantity to check independently.

**Added, all additive / opt-in (no existing caller's behavior changes):**

1. **`src/preprocessing/pipeline.py::group_name_by_source` /
   `aggregate_attribution_by_source`** -- maps a transformed feature name
   (e.g. `cat__region_North`) back to its original source column (`region`),
   longest-match-first against the known original categorical column names
   so e.g. `region_type` is never mis-grouped under `region`, and sums a
   `{feature: value}` attribution dict's one-hot-derived entries back
   together under that source column. Unit-tested including the ambiguous
   `region` vs. `region_type` case, and a conservation check (grouped total
   == ungrouped total -- summing cannot lose or invent error mass).
2. **`reconstruction_error_by_feature(..., categorical_columns=[...])`**
   (new optional parameter, default `None` -- every existing caller is
   unaffected) -- when given the original categorical column names: logs and
   records a `categorical_contribution` checkpoint with the categorical
   block's share of total *columns* vs. share of total *reconstruction
   error* (over-represented / roughly proportional / under-represented), and
   the bar chart ranks/labels by the grouped-by-source values instead of raw
   one-hot slices. The **returned dict is unchanged** (still the full,
   ungrouped per-column detail) -- grouping only touches the chart and the
   diagnostic, never silently the data a caller stores.
3. **`export_attribution_workbook(..., categorical_columns=[...])`** -- when
   a `"vae"` entry is present, writes an additional `vae_by_source` sheet
   (grouped) alongside the existing, still fully granular `vae` sheet.
4. **`main.py` always passes `categorical_columns`** (`df.select_dtypes(
   include=["object", "category"]).columns.tolist()`) to both call sites
   above -- the diagnostic and the grouped views are on by default for every
   run, not something a user has to remember to request.
5. **`--rare-min-frequency`** (new CLI flag / `PipelineConfig.
   rare_min_frequency`, default `0.001` -- `fit_transform_panel`'s own
   pre-existing default, previously only reachable by calling the
   preprocessing module directly) -- exposed as the lever to pull *if* the
   diagnostic in (2) shows real over-representation: raising it collapses
   more low-frequency categories into one "infrequent" bucket before
   one-hot, shrinking column count per categorical while keeping identity
   for the common ones. `--categorical-encoding frequency`/`ordinal`
   (already available, no code change) is the more drastic option --
   collapses each categorical to one column regardless of cardinality, at
   the cost of losing category identity for the "contextual" anomaly
   definition, so it is documented as a deliberate trade-off requiring a
   re-tune, not pushed as the default fix.

**Verified two ways.** (1) A synthetic scenario deliberately more granular
than this project's own synthetic panel (three string columns with 40/60/25
categories vs. this project's 10/4/6/4/6/4) against a real `VAEDetector`:
grouping correctly reduced 140 columns to 18 report rows, all three source
columns present with zero raw one-hot slices remaining, and the grouped
ranking's top entry (`branch_id`, aggregated) differed from the ungrouped
ranking's top entry (`num__income`) -- demonstrating the exact effect being
fixed. (2) A full `python main.py --quick --no-tune` run against this
project's own (much less granular) synthetic panel: categorical columns are
64.7% of features and 67.0% of reconstruction error -- correctly diagnosed
as "roughly proportional," i.e. **not** a real problem on this project's own
data, confirming the diagnostic does not cry wolf when the effect is not
actually present. 41/41 health checks passed.
