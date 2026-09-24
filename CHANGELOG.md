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

---

## 2026-08-28 (same day, later still) — per-row "why is this one flagged" explanation

**Ask:** for each record in the OOT alert queue, the top 5 variables driving
*that specific row's* score, as a column in the Excel a reviewer already
opens -- not the population-level ranking `shap_summary_iforest`/
`reconstruction_error_by_feature` already produce, which answers "what
matters overall," not "why did this one person get flagged."

**Added, both new functions, no existing signature changed:**

- **`explain_rows_iforest(detector, X, feature_names=None, top_k=5, ...)`**
  (`src/interpretability/iforest_explain.py`) -- explains *exactly* the rows
  given (not a subsample), reusing `shap.TreeExplainer` through the same
  isolated-child-process, hard-kill-guarded path (`_run_with_hard_kill`,
  `_tree_explainer_child`) that `shap_summary_iforest`'s path 1 already uses
  -- the same tree-depth-driven slowness (2026-08-28 earlier entry) applies
  here too, just against a small fixed row set instead of a 2000-row
  subsample. Returns one comma-joined string of the `top_k` feature names
  per row (by `|SHAP value|`), or `None` for a row that could not be
  explained -- never raises, since this feeds a business deliverable.
- **`explain_rows_vae(detector, X, feature_names=None, top_k=5,
  categorical_columns=None, ...)`** (`vae_explain.py`) -- always exactly
  computable (no fallback needed): per-row squared reconstruction error,
  ranked per row. When `categorical_columns` is given, one-hot columns are
  summed back under their source variable **per row** first
  (`group_name_by_source`, same mechanism as the aggregate diagnostic two
  entries above), so the column reports `region`, never `cat__region_North`.
- **`main.py` Phase 9** computes this for exactly the OOT rows (via
  `oot_period`, the same population `export_oot_top_anomalies` selects from
  -- not the whole panel, which would be wasteful at real-data scale) right
  after `build_scored_frame`, and attaches it as a new `top_5_variables`
  column before the Excel export. Wrapped in its own try/except: a failure
  here logs a warning and the export proceeds without the column, rather
  than losing the headline deliverable over an interpretability add-on.

**Verified two ways.** (1) Unit-level: deliberately injecting an extreme
value into one known column of a handful of rows makes `explain_rows_
iforest` and `explain_rows_vae` both name that exact column as the top
explanation for that exact row; a categorical-grouping check confirms the
per-row output never leaks a raw one-hot slice name. (2) A full `python
main.py --quick --no-tune` run: log confirms "500 OOT row(s)" explained, and
the real `oot_p90_vae.xlsx` output carries a `top_5_variables` column with
**0 nulls across all 50 exported rows**, genuinely differentiated per
individual (e.g. one row's top driver is a missing-value indicator, another's
is the stacked Isolation Forest score, another's is `region`) -- not a
placeholder or a repeated global ranking. 41/41 health checks passed.

**Known limitation, not fixed here:** the column lands as the *last* column
in the exported table (after every raw feature column), since
`export_oot_top_anomalies` preserves `scored_df`'s column order and
`top_5_variables` is appended after the raw columns already there --
promoting it next to SCORE/BAND for visibility would need a small change to
`export_oot_top_anomalies`'s column ordering, not attempted here since it
touches the headline deliverable's layout and was not asked for.

---

## 2026-08-28 (same day, later still still) — model-agreement chart rebuilt as a genuine-OOT density heatmap

**Ask, three parts:** (1) confirm the per-row `top_5_variables` column from
the entry above lands in *both* the IF P90 export and the VAE export, not
just whichever one happens to be the deliverable; (2) confirm the alert
queue is deduplicated to unique OOT `entity_id`s, matching what the anomaly
report describes; (3) replace the IF-vs-VAE agreement scatterplot with a
hexbin/2D-density heatmap: Viridis colour by count, the top-5%-x-top-5%
quadrant outlined, Spearman rho + top-5% overlap kept as an annotation, a
y=x reference line, and a side-panel 4x4 quartile confusion matrix.

**(1) and (2) were already correct, verified rather than changed:** ran the
full pipeline with `--no-stack-iforest-into-vae` (parallel mode, the only
config that produces both `oot_p90_iforest.xlsx` and `oot_p90_vae.xlsx` in
one run) -- both files carry `top_5_variables` with 0 nulls across their 50
rows and genuinely different content per model (IF's queue and VAE's queue
rank different individuals, as documented). `export_oot_top_anomalies`
already deduplicates to one row per `entity_id` (keeps each entity's
highest-scoring OOT month) before selecting the P90/P95/P99 bands, so the
"unique entity_id" property already held.

**(3) is a real fix, not just a redraw.** The old "Model agreement"
scatterplot (`report_content.py`, `fig-agreement`) was titled "(OOT)" but
actually plotted `oot_scores`, which is `scores[eval_mask]` -- `eval_mask`
is the **test** block (`main.py`'s Phase 3a comment on why `eval_mask` and
`oot_mask` are kept distinct), not the genuine held-out OOT window Phase 9's
Excel draws from. With `n_oot_periods > 1` it was also not deduplicated by
entity, so the same person could plot as two separate points. Both defects
meant this chart could describe a different population than the one the
alert queue and this very changelog entry's point (2) are about.

**Fixed by adding a real entity-level field, not by patching the chart:**
`main.py`'s Phase 8 loop now also computes `true_oot_entity_scores` --
`{entity_id: max score across the genuine oot_mask window}`, the exact same
dedup rule `export_oot_top_anomalies` uses -- for every model, stored in
`chart_data["models"][name]`. Verified directly: running with
`--n-oot-periods 2` (which puts 2 OOT months x 500 entities = 1000 rows into
`oot_mask`), the export log reports "500 individuals in the OOT block" and
the new chart's own title reports the identical "500 individuos" -- the
chart and the deliverable now provably describe the same population.

**The chart itself** (`build_plotly_figures`'s section 2 in
`report_content.py`): the two models' `true_oot_entity_scores` are inner-
joined on `entity_id`, each converted to a percentile rank via
`scipy.stats.rankdata` over that model's own OOT population, then plotted
as a `go.Histogram2d` (Plotly has no Cartesian hexbin binning -- only a
mapbox one -- so a 2D histogram heatmap is the direct equivalent, and was
explicitly accepted as an alternative) with the Viridis colorscale, a y=x
`go.Scatter` reference line, and a `fig.add_shape` rectangle outlining the
[95,100]x[95,100] quadrant. Spearman rho and the top-5%-x-top-5% overlap
count (now defined by the same >=95th-percentile threshold the rectangle
draws, rather than the old fixed-k rank intersection) are kept as an
annotation. A second `go.Heatmap` panel bins the same population into a 4x4
quartile confusion matrix (counts and % of total per cell, via
`np.add.at`), built with `make_subplots` beside the density panel. Skips
(with a logged warning, chart omitted rather than the report failing) if
fewer than 4 entities are common to both models' OOT populations.

**Verified with three full pipeline runs** (`python main.py --quick
--no-tune`, once default-stacked, once `--no-stack-iforest-into-vae`, once
`--n-oot-periods 2`): all three produced the `fig-agreement` chart with the
new `Histogram2d`/`Heatmap` traces, the `Spearman rho` annotation, and the
"OOT genuino, N individuos" title present in the rendered HTML; health
checks passed in every run (42/42 parallel-mode, 41/41 stacked-mode); no
"Model-agreement chart skipped" warning was logged in any run.

---

## 2026-08-28 (same day, later still still still) — one-hot category counts (diagnostic answer, no code change), then closing the last checkpoint-coverage gap and giving interpretability its own live progress line

**Ask 1 (diagnostic, no code change):** "¿tienes regulares datos por cada
categoría one-hot encoded?" -- measured directly on the current synthetic
panel (6,000 rows) by running `fit_transform_panel` and summing each
resulting `cat__*` column. Most categories are well populated (hundreds to
thousands of rows), but a real tail exists: `transaction_channel_legacy_
terminal` (6 rows, exactly at the `rare_min_frequency=0.001` boundary) and
`product_type_student_account` (12 rows) survive as their own one-hot column
despite having almost no support, while `crypto_gateway` (4 rows, just under
the threshold) correctly gets folded into sklearn's `infrequent` bucket --
confirming the boundary is a strict `<`, not `<=`. Answered directly with the
numbers and the existing `--rare-min-frequency` lever (raising it to e.g.
`0.01` would fold both of the thin survivors in too); no code changed for
this part.

**Ask 2:** "Actualiza la documentación..., limpia el contexto..., y
asegúrate que la vista de avance contemple los checks de interpretabilidad
[...] indica un nivel más de detalle al avance." Investigating what the live
console dashboard (`src/utils/console_ui.py`) actually did with
interpretability's checkpoints turned up two real gaps, not just a
documentation/cosmetics task:

1. **`explain_rows_iforest` and `explain_rows_vae` (the per-row `top_5_
   variables` functions added earlier today) emitted *zero* checkpoints.**
   `explain_rows_iforest` in particular reuses the exact same isolated,
   hard-kill-guarded `shap.TreeExplainer` subprocess path as `shap_summary_
   iforest` -- the one path this project spent most of today's earlier
   session hardening specifically *because* it can hang -- yet had no
   `_checkpoint(...)` calls at all, unlike every other function in that
   module. Fixed by adding the identical checkpoint shape used elsewhere
   in `iforest_explain.py`: `explain_rows_started` ->
   `explain_rows_calibration_started` -> `explain_rows_calibrated` ->
   `explain_rows_explain_started` -> `explain_rows_done` /
   `explain_rows_hard_killed` / `explain_rows_failed` ->
   `explain_rows_completed`, reusing the same `_on_progress` callback
   pattern as `shap_summary_iforest`'s path 1. `explain_rows_vae`
   (`vae_explain.py`) got the lighter-weight equivalent appropriate to its
   always-fast batched-forward-pass implementation: `explain_rows_started`,
   `explain_rows_progress` every ~25% of batches, `explain_rows_completed`.
   Verified: a `--no-stack-iforest-into-vae` run's `run_events.jsonl` shows
   all six `interpretability.iforest.explain_rows_*` names and all four
   `interpretability.vae_explain.explain_rows_*` names firing in order;
   52/52 health checks passed.

2. **The console dashboard mixed interpretability's routine progress pings
   into the "Supuestos (IF / VAE)" panel**, alongside genuine IF/VAE
   assumption gates -- a panel titled for assumption checks, fed by both a
   handful of rare, meaningful pass/fail gates *and* a few dozen
   always-passing `interpretability.*` pings that Phase 10 alone can fire in
   quick succession, which could flush every real assumption result out of
   the 8-slot deque. `ConsoleUI._on_check` now routes anything named
   `interpretability.*` to its own separate 3-slot deque
   (`_interp_checks`) instead of `_checks`, so "Supuestos" is
   assumption-gates-only again, and a **new sub-step line** renders directly
   under the current-phase readout: `↳ interpretabilidad  <last up-to-3
   checkpoint names> <time on the latest> (<n> checkpoints)` -- the extra
   level of detail asked for. A phase-level progress bar can only say "Phase
   10 is running"; this line says exactly which internal checkpoint it last
   reached and for how long, which is what actually distinguishes "still
   working" from "stuck" without opening `execution.log`. The line persists
   (frozen on its last value) after Phase 10 finishes, matching how the
   phase checklist keeps its boxes lit rather than resetting.
   Verified by directly driving `ConsoleUI` with synthetic phase/check
   events and rendering it to a captured buffer (avoids a Windows-console
   Unicode encoding issue unrelated to the change itself): the
   interpretability checkpoints appear only in the new sub-step line with a
   correct running trail and count, and a real passing/failing assumption
   check still lands in "Supuestos" as before, untouched.

**`CONTEXT.md` cleanup, done alongside the above:** two passages had gone
stale relative to the code as it now stands and were rewritten rather than
left to contradict it -- the interpretability-checkpoint-coverage bullet
still said "no changes needed there" about the console dashboard hook (now
false: `console_ui.py` changed, described above) and still listed only the
pre-existing checkpoint-bearing functions (missing the two `explain_rows_*`
additions from earlier today); and the Reporting section's one-line
dashboard description still called the panel "Assumptions (IF/VAE)" fed by
"the same `observability.check(...)` calls" without qualification, which is
no longer accurate now that interpretability's checks are routed elsewhere.
Both are corrected in place rather than appended as a second, conflicting
note.

---

## 2026-08-30 — Downstream analyst dashboard: three mockups, two rounds of correction

**Ask:** design mockups of an analyst-facing dashboard "like" a reference
fraud-dashboard HTML file the user attached, fed *only* by this project's
own output columns (no data from elsewhere), covering: the top variables
driving each individual's score plus their IF **and** VAE score together,
and a count of which months (out of the last 3, since OOT should widen to
3 months) each individual was flagged in. Explicitly deferred: no real
pipeline run/validation yet, no `main.py`/`oot_report.py` code changes —
mockups first.

**Round 1 — three designs, one shared mock dataset.** Built a Python
generator (`gen_mock_data.py`, not committed to the repo — lives in the
session scratchpad) producing ~84 illustrative individuals with this
project's real schema vocabulary (`CUST_NNNNNN` ids, `num__`/`cat__`/
`missing__`-prefixed variable names, the actual synthetic-panel columns)
rather than the reference file's fictional fraud fields, plus three
distinct HTML layouts sharing one token system (IF-blue/VAE-orange taken
directly from `report_content.py`'s already-validated `SERIES_COLORS`; a
separately validated red/amber/green status triad,
`node scripts/validate_palette.js` from the dataviz skill run against both
light and dark surfaces until it passed): a queue-style table ("Cola de
Revisión"), an executive/aggregate view ("Panorama Ejecutivo"), and a
recurrence-kanban ("Mesa de Persistencia"). Published as three Claude
Artifacts for review. A concurrent, unrelated fix in this same session
shrank the model-agreement chart's quartile-panel font sizes in
`report_content.py` (title/ticks/colorbar, ~9-11px instead of 12-14px) so
they stop overlapping in the narrow 34%-width panel.

**Round 2 — the grouping was fabricated, cut entirely.** The first round
grouped `top_5_variables` into five named business categories
("Comportamiento transaccional", "Perfil financiero", ...) for a
"driver principal/secundario" pair of table columns and a "share by
category" sidebar chart. Correctly flagged: that taxonomy exists nowhere
in the real project output — `explain_rows_iforest`/`explain_rows_vae`
return raw variable names only. Removed both the two driver columns (now
one column showing the literal comma-joined `top_5_variables` string,
exactly as the real Excel export would) and the sidebar chart built on
top of the same invented grouping, from the chosen "Cola de Revisión"
design and from its underlying embedded data model (not just hidden in
the view — the fields were dropped from the JSON payload entirely).

**Round 3 — the flat per-entity profile was time-ambiguous, cut entirely.**
The modal's per-entity "profile" (region, segment, age, income, account
balance, transaction counts, ...) was flagged with a sharp, correct
question: *which month does this refer to?* Those are real input-CSV
columns, but they are genuinely time-varying in the panel (one value per
`(entity_id, period)`, not per `entity_id`), and an entity can be flagged
in more than one OOT month — a flat, undated value is ambiguous. Cut
entirely, along with the two other pieces the user identified as
computed extras beyond a single raw feed: the "Cobertura mensual OOT"
sidebar bar chart (a population-level aggregate) and the "Concordancia
IF↔VAE (ρ Spearman)" KPI tile (a statistic over the whole shown
population, not a per-row field). The modal now shows, at most: identity,
IF/VAE score + percentile + band, months present, and the raw top-5
variable list — every one of those is a genuine per-row model output, not
a derived or externally-sourced concept.

**End-to-end validation of the corrected mockup** (static, since there is
no browser in this environment): verified with an HTML parser that every
tag closes correctly; verified every `document.getElementById(...)` call
in the page's JS resolves to an id that actually exists in the markup
(catches a dangling reference from a removed section); grepped the
embedded `PROFILES` JSON payload sent to the browser and confirmed it
contains exactly the 9 intended keys (`id`, `if_score`, `vae_score`,
`if_pctl`, `vae_pctl`, `band`, `months`, `months_count`, `top5`) and none
of the removed ones; confirmed every string used elsewhere in the earlier
rounds ("Concordancia IF", "Cobertura mensual", `primary_cat`, `mFields`,
...) no longer appears anywhere in the output file.

**Target architecture, now written down** (`CONTEXT.md` "Downstream
analyst dashboard"): this dashboard is meant to be a separate consumer
outside this repo, fed by one output table "similar to an API" — no
categorization layer added inside Modelo v0.1, no undated per-period
field shown flat. The single-feed contract the chosen mockup validates,
and what is still missing to produce it for real
(`n_oot_periods` still defaults to 1, the review/alert count still uses
the POT-calibrated `threshold` that can degenerate to zero rather than a
fixed P95-of-unique-individuals rule, and per-entity month-recurrence has
no query in the real pipeline yet since `export_oot_top_anomalies`
deliberately collapses to one row per entity), are recorded there rather
than implemented blind — each needs its own validation run before it
becomes real pipeline behavior, which was explicitly deferred throughout
this exercise.

---

## 2026-08-31 — the deferred items land: real 0-anomaly fix, explicit 3-month OOT, and the analyst dashboard actually integrated

**Ask:** three items, no longer deferred. (1) The real pipeline still showed
"0 anomalías marcadas para revisión" on a real run -- fix it so the count
comes from the OOT export's own P95/P99 rows, as specified two days ago.
(2) Make the OOT window explicitly 3 months in `main.py`. (3) Integrate
"Cola de Revisión" for real, fed *exclusively* by the resulting CSV/table
-- not the mockup's fabricated data.

**Root cause of the 0, confirmed by reading the code, not guessed:**
`src/reporting/report.py::_hero_html` reads `dataset["n_anomalies"]`, which
`main.py` set to `n_pos = int(labels.sum())` -- a **ground-truth positive
count** (Phase 3). Real production data has no ground-truth file, so
`labels` is all zero and `n_pos` is always 0 there, regardless of what the
model actually flagged. The hero was answering "how many rows are truly
anomalous" (unknowable in production) when its own label says "anomalías
marcadas para revisión" (an alert-queue question). This was latent since
whenever the hero code was written; the synthetic panel's own ground truth
masked it in every test run to date.

**Fix:** `main.py` Phase 9 now computes, right after `export_oot_top_
anomalies` writes the file, `oot_review_count = int(_table[BAND_COL].isin(
("p95", "p99")).sum())` -- read directly off the just-written export, so
the number is always exactly what a reviewer would get by opening that
file and counting P95+ rows. `context["dataset"]["n_anomalies"]`/
`"anomaly_rate"` now prefer this over `n_pos`/`anomaly_rate`, falling back
to the ground-truth pair only if no deliverable ran. `chart_data["anomaly_
rate"]` (the PR-curve baseline, a genuinely different, still-ground-truth
quantity) is untouched -- verified by grepping for the two separate
dict-key assignments after the change. `src/evaluation/__init__.py` now
re-exports `BAND_COL`/`PERCENTILE_BANDS` from `oot_report.py` so `main.py`
can import them without reaching into the submodule directly.

**`n_oot_periods` default: 1 -> 3, explicit**, in both `PipelineConfig`
(`main.py`) and the `--n-oot-periods` CLI default -- comments and the
`_QUICK` preset's period-budget note updated to the new actual split
(5/2/3/3 train/val/test/OOT under `--quick`'s 13 periods, still comfortably
above the `n_periods >= n_oot_periods + 3` floor `chronological_split`
enforces).

**The analyst dashboard, integrated for real**
(`src/reporting/analyst_dashboard.py::build_analyst_dashboard`, wired into
`main.py` Phase 9, one file per deliverable model at `artifacts/reports/
analyst_dashboard_<model>.html`). Two design decisions made explicit,
directly from the "exclusively from the resulting csv" constraint:

- **No cross-model score join.** The mockup showed IF and VAE scores side
  by side per entity; that join does not exist as a real, single output
  table (each model's OOT export carries only its own score), so building
  it would mean fabricating data outside "the resulting csv." Dropped for
  the real integration -- each dashboard shows exactly one model's own
  score/percentile, built strictly from that model's own export. Under
  default stacking this is a non-issue (VAE is the only deliverable
  anyway); under `--no-stack-iforest-into-vae` there are simply two
  dashboards, one per model, never merged.
- **Month-recurrence needed one small, additive real function**, since
  `export_oot_top_anomalies` deliberately discards every OOT month but an
  entity's best one. Added `src/evaluation/oot_report.py::
  months_present_by_entity(scored_df, schema, oot_periods, cutoff,
  score_col)`: given the *undeduplicated* OOT block and a score cutoff,
  returns `{entity_id: [period_str, ...]}` -- which months that entity's
  score cleared the cutoff in. Does not touch `export_oot_top_anomalies`'s
  own dedup or return contract; a second view over the same data, nothing
  replaced. The cutoff passed is the same P95 value the review-count KPI
  uses (`np.percentile` of `true_oot_entity_scores.values()`, the exact
  same de-duplicated population `_percentile_band_labels` grades against),
  so "present in this month" and "counted as in review" mean the same
  threshold everywhere.
- Percentile-per-row (needed for the score bar, not present in the
  exported table as a column) is recomputed via `rankdata` over
  `true_oot_entity_scores` -- the same population, same method
  `report_content.py`'s model-agreement chart already uses; not persisted,
  cheap enough to rebuild per call.
- Visually reuses `report.py`'s own design system verbatim
  (`_HTML_CSS`/`_THEME_TOGGLE_JS`/`_stat_tile_html` imported directly, not
  copied) rather than the mockup's separate Google-Fonts identity, so the
  shipped artifact reads as part of this project's actual report rather
  than a one-off style. Wrapped in its own try/except in `main.py`; a
  failure here logs a warning and never blocks the OOT Excel deliverable.

**Verified, both `python main.py --quick --no-tune` (default, stacked) and
`--no-stack-iforest-into-vae` (parallel, two dashboards):**
- OOT export log confirms 3 distinct months selected (e.g.
  `period(s)=['2026-04-01...', '2026-05-01...', '2026-06-01...']`).
- The report's hero figure (grepped from the rendered
  `anomaly_report.html`) exactly equals `df["percentil"].isin(["p95",
  "p99"]).sum()` computed independently from the just-written
  `oot_p90_<model>.xlsx` -- 25 in every run tried, both modes.
- The analyst dashboard's embedded per-entity JSON has exactly one entry
  per exported row, entity-id-for-entity-id identical to the Excel's own
  `entity_id` column; its "en revisión"/"recurrentes" KPI tiles match an
  independent count from that same JSON; HTML parses with zero unclosed
  tags in both the IF and VAE dashboards.
- A found-and-fixed bug during this same verification pass: the two stat
  tiles' labels used the `&ge;` HTML entity inside a string later passed
  through `html.escape()` (`_stat_tile_html`), which escaped the `&` a
  second time into literal `&amp;ge;` text in the browser. Fixed by using
  the literal `≥` character instead (escaping a real Unicode symbol is a
  no-op); re-verified in the regenerated output.
- Health checks: 50/50 (stacked) and 58/58 (parallel).

`CONTEXT.md` "Downstream analyst dashboard" section rewritten from "target
architecture, not yet built" to describe what is now actually shipped and
verified, including the two constraints above and how each is enforced in
the code (not just asserted in prose).

---

## 2026-08-31 (same day, later) — the dashboard integration corrected: right format, one file with both scores

**Ask, verbatim in spirit:** "this is wrong -- I didn't ask for this
analyst dashboard, it's not in the 'cola de revisión' format we saw
yesterday, fix that. Also there shouldn't be 2, one for IF and one for
VAE -- you already have the VAE score for that, integrate the
information." Two real mistakes in the integration entry above, both
fixed here.

**Mistake 1: wrong visual format.** The integration re-derived the
dashboard's HTML/CSS from `report.py`'s own design system (`_HTML_CSS`,
`_stat_tile_html`, `.hero`/`.tile` classes) instead of reusing the actual
approved mockup. That produced a plainer, differently-structured page --
not what was reviewed and signed off on 2026-08-30. Fixed by throwing that
version away and porting the approved mockup's HTML/CSS/JS **verbatim**:
same shell, sticky header with brand mark/live clock/"PIPELINE OK" pill,
the same two KPI tiles, the same single priority table (idx/ID/Banda/
Percentil IF/Percentil VAE/Top-5 variables/Meses), the same profile modal,
the same footer legend, same Archivo + Public Sans + IBM Plex Mono type
system loaded from Google Fonts, same `--if`/`--vae`/severity color
tokens. Only the data-population logic is new; the page itself is the
approved design, not a reinterpretation of it.

**Mistake 2: split into two dashboards.** The integration built one
`analyst_dashboard_<model>.html` per deliverable model and, worse, dropped
the side-by-side IF+VAE score view entirely -- misreading "fed exclusively
by the resulting csv" as "never join two models' scores," when the actual
ask (several turns earlier: "adjuntale también cual fue su score IF y
VAE") was always for both scores together. The fix does not need a second
exported file to satisfy this: `main.py` Phase 8 already computes
`true_oot_entity_scores` for **both** detectors every run, regardless of
`--stack-iforest-into-vae` (the same in-memory join `report_content.py`'s
model-agreement chart already relies on) -- an in-memory join on
`entity_id`, not a new data source.

**What changed, concretely:**
- `src/reporting/analyst_dashboard.py` -- rewritten. `build_analyst_
  dashboard` now takes the *primary* deliverable's export table (selection/
  order/`band`/`top_5_variables` source -- `config.deliverable_models[-1]`,
  always `"vae"`) plus **both** detectors' `{entity_id: score}` /
  `{entity_id: percentile}` dicts, and renders one page with both
  "Percentil IF" and "Percentil VAE" columns per row, and both score bars
  in the modal.
- `main.py` -- the per-model Phase 9 loop no longer builds a dashboard
  per iteration; it only records each deliverable's table into
  `deliverable_tables[name]`. A new step, "Phase 9b: analyst dashboard",
  runs once after the whole per-model loop: picks
  `primary_name = config.deliverable_models[-1]`, pulls both models'
  `true_oot_entity_scores` from `chart_data["models"]`, computes
  `months_present_by_entity` against the primary's own P95 cut-off, and
  calls the renderer exactly once. `analyst_dashboards: dict` (one entry
  per model) replaced with `analyst_dashboard_path: Optional[str]` (one
  path, or `None`).
- `src/utils/paths.py` -- `ANALYST_DASHBOARD_DEFAULT` is now a fixed
  `artifacts/reports/analyst_dashboard.html`, not a `{model}`-templated
  name.

**Verified, both `python main.py --quick --no-tune` (stacked, one Excel)
and `--no-stack-iforest-into-vae` (parallel, two Excels):**
- Exactly one `analyst_dashboard.html` produced in both modes -- confirmed
  by directory listing, not just by reading the code. (Two stale
  `analyst_dashboard_<model>.html` files left over from the previous,
  now-superseded code were found and deleted from the local `artifacts/`
  during this same check -- not something the corrected code produces.)
- HTML parses with balanced tags; the embedded per-entity JSON has a
  non-NaN `if_score` **and** a non-NaN `vae_score` for all 50 rows in both
  modes -- the cross-model join is real, not a placeholder.
- In parallel mode specifically: the dashboard's `if_score` for every one
  of the 35 entities that appear in *both* separately-exported Excels
  matches `oot_p90_iforest.xlsx`'s own score column exactly (0
  mismatches), while row selection/order/`band` still come from
  `oot_p90_vae.xlsx` -- proving the join pulls the real IF export's
  numbers, not a re-derived or fabricated value.
- "En revisión" KPI (25 in both modes) matches an independent count of
  `df["percentil"].isin(["p95","p99"])` from the just-written primary
  Excel, same check as the previous entry, re-run after the rewrite.
- Health checks: 50/50 (stacked), 57/57 (parallel).

`CONTEXT.md` "Downstream analyst dashboard" rewritten again to describe
the corrected, unified design -- explicitly naming both mistakes and how
each is now prevented in the code, not just narrating the current state
as if it had always been this way.

---

## 2026-09-03 — IF-VAE Diagnostic Suite vendored, adapted, debugged, and run against real project data; dashboard filter + responsive fix

**Ask:** vendor a third-party "IF-VAE Diagnostic Suite" package into the
project, apply whichever of its capabilities fit an unsupervised model
(official runs carry no target), adapt this project's own scripts to feed
it real data, run it, root-cause and fix whatever breaks, and interpret
the results -- plus, separately, add a search filter and responsive
layout to `analyst_dashboard.html`'s priority table. A full backup was
taken first (`git archive` zip + a local, unpushed tag
`backup-pre-diagnostic-suite-2026-09-03`) since this is explicitly a
before-you-evaluate-it change: **not pushed to GitHub this round**, by
explicit instruction.

**Suite vendored** into `tools/if_vae_diagnostic_suite/` (copied from the
user-provided package, not a submodule) and installed editable. Read all
~1,100 lines of its `src/ifvae_diag/` before writing any adapter code
(`cli.py`, `config.py`, `contracts.py`, `pipeline.py`, `modeling.py`,
`scoring.py`, `diagnostics.py`, `metrics.py`, `data_quality.py`,
`stability.py`, `reporting.py`, `simulation.py`) -- confirmed its own
26-test suite and `simulate` demo both passed clean before touching
anything, so every later finding is attributable to the real-data
integration, not a pre-broken package.

**`tools/export_diagnostic_suite_inputs.py`** (new): mirrors `main.py`'s
Phases 2→3a→4→6→6b→7 to fit a real IF+VAE pair (same functions, same seed,
stacking on) and export the suite's `reference.csv`/`scored.csv` contract
-- see `CONTEXT.md` "IF-VAE Diagnostic Suite integration" for the exact
mapping (id/time/label/segment/family columns, why `scored.csv` is one row
per (entity, period) rather than deduplicated like the OOT Excel). One bug
caught and fixed immediately: the VAE's `encoder`/`decoder` `Linear` layers
are float32 (`src/models/vae.py::_densify`), and my first draft densified
the export matrix to float64, producing `RuntimeError: mat1 and mat2 must
have the same dtype, but got Double and Float` on the very first real run.
Also hit and fixed pandas' `PerformanceWarning: DataFrame is highly
fragmented` (from ~140 individual `frame[col] = ...` assignments building
the mu/logvar/recon columns one at a time) by building each frame via a
single `pd.concat` of column blocks instead.

**Two real, root-caused bugs found and fixed in the vendored suite itself**
(full detail, code, and verification in `CONTEXT.md` -- summarized here):
1. `scripts/mutation_probe.py` hardcoded a POSIX `PYTHONPATH` separator
   (`:`), which is broken on Windows (`os.pathsep` is `;`, and a
   drive-letter path already contains a `:`) -- every mutation test
   silently ran against the real, installed package instead of the
   mutated copy, so `killed=False` for all 4 mutants regardless of test
   quality. Found because I ran `make chore-lint` on this Windows machine
   as part of "hasta que funcione correctamente," not because anything in
   my own adapter script triggered it.
2. No label-free mode: the data contract hard-required a binary label
   column that this project's real, unsupervised production runs do not
   have. Added `label_col: str | None` end to end
   (`config.py`/`contracts.py`/`pipeline.py`/`reporting.py`), following
   the suite's own Red→Green discipline (`AGENTS.md`): wrote
   `tests/test_pipeline_unsupervised.py` first, watched it fail for the
   right reason (`DataContractError: scored is missing columns: [None]`),
   then implemented.

Both fixes are covered by new tests (`tests/test_mutation_probe.py`,
`tests/test_pipeline_unsupervised.py`); the suite's own quality gate
(`PYTHONPATH=src python scripts/quality_gate.py`) went from
`mutations=False` (broken) to fully green: `compile=True tests=True
mutations=True`, 31/31 tests. Verified end to end against this project's
own real export, twice -- once with the project's synthetic ground truth
attached (to prove the numbers are right) and once with every label/
segment/family column stripped and `label_col: null` (to prove the
label-free path is real, not just unit-tested): both runs produce
identical disagreement quadrant counts (77 BOTH / 115 IF_ONLY / 132
VAE_ONLY / 1176 NEITHER), confirming quadrant assignment never touches
the label column either way.

**Findings from running it against a real, freshly-fitted IF+VAE** (full
numbers in `CONTEXT.md`): VAE strongly dominates IF on `global` anomalies
(AP 0.82 vs. 0.08) but both are near-random on `local` and `contextual`
(AP ≈ 0.01–0.02) -- an independent, differently-coded confirmation of the
project's own long-standing "`local`-type anomalies are unrecovered"
finding, now extended to `contextual` too; IF contributed zero unique hits
to the top-K queue beyond what VAE already found, and a naive mean
ensemble was worse than VAE alone; no posterior collapse (independent
confirmation the 2026-08-22/23 VAE loss-scaling fix is holding); the
drift report's top entries are dominated by calendar/lag-feature artifacts
of the chronological split itself, not genuine concerning drift; and the
`collective` anomaly family had zero known positives in this particular
3-month OOT sample, a reminder that a family-level breakdown needs a large
enough OOT window to be trusted.

**Per-layer performance chart added to `report.md`** (follow-up ask: "de
ser necesario, al nuevo reporte agregale los gráficos que sirven para
monitorear el desempeño de los modelos en cada capa"). The report only had
`disagreement.png`, a percentile-agreement scatter that shows where IF and
VAE *agree*, not which one actually *performs*. Added
`metrics_bar_plot()` (`reporting.py`): a precision@k bar chart, one group
per alert budget (k=10/25/50), one bar per score candidate
(`if_percentile`/`vae_percentile`/`ensemble_max`/`ensemble_mean`) -- the
visual counterpart to `metrics.csv`. Same Red→Green discipline: wrote
`tests/test_reporting_metrics_plot.py` first (Red --
`ImportError: cannot import name 'metrics_bar_plot'`), implemented,
green. Wired into `pipeline.py::_write_outputs`, which now passes a
`has_metrics_plot` flag through to `write_markdown_report` so the report
embeds `![...](metrics.png)` when it exists and a plain sentence in its
place when it does not -- this chart needs known positives, so it is
correctly and silently skipped (not a broken image link) on the label-free
path. Re-verified against this project's real export in both modes:
labeled `report/report.md` embeds `metrics.png`; label-free
`report_unsupervised/report.md` has neither the file nor a dangling
reference. Full suite re-run after this change: 31/31 tests,
`compile=True tests=True mutations=True`.

**`analyst_dashboard.html`: search filter + responsive table**
(`src/reporting/analyst_dashboard.py`). Added `#tableSearch`, a
client-side filter (`filterTable()`, debounced via
`requestAnimationFrame`) matching the query against every visible cell of
a row -- ID, band, both percentiles, top-5 variables, meses -- not just
the ID column, with a "no matches" row and a live-updating count badge.
Made the table responsive for entity IDs longer than this project's own
synthetic `CUST_000123` format: `.tablewrap` scrolls both axes on its own
(`overflow:auto`) instead of the page gaining horizontal scroll, and the
ID cell wraps (`overflow-wrap:anywhere`) instead of forcing the whole
table wider for one long value. Verified against a real
`python main.py --quick --no-tune` run: HTML parses with balanced tags,
every row carries the `data-id` attribute the filter's selector expects,
50/50 health checks passed.

**Full-pipeline integrity check** (both stacked and parallel modes,
`python main.py --quick --no-tune`, run after every change in this entry)
-- see the final validation note at the end of this file for the exact
health-check counts. No regression found in the core pipeline; all
changes in this entry are additive (`tools/`, dashboard JS/CSS) or
confined to the vendored suite.

**Not pushed.** Local commit only, per explicit instruction to evaluate
first.

---

## 2026-09-04 — IF-VAE Diagnostic Suite wired into the pipeline itself; new "Diagnóstico cruzado IF-VAE" report chapter

**Ask:** run the diagnostic suite and index its results into
`anomaly_report` as a new chapter, written pyramid-style (most actionable
conclusion first), pulling every number directly from the code with no
hardcoded values, stubs, or placeholders -- iterated twice, checking both
the methodology and the reading flow.

Until now the suite only ran as a manual, standalone exercise
(`tools/export_diagnostic_suite_inputs.py` + its own CLI, see 2026-09-03
above). This entry wires it into `main.py` itself as an optional phase.

**New: `src/evaluation/ifvae_diagnostic.py`**
(`run_ifvae_diagnostic_suite`). Builds the suite's `reference`
(`train_mask`)/`scored` (`oot_mask`) frame contract directly from this
run's own already-fitted `models["iforest"]`/`models["vae"]` tuples --
no refit, no CSV round-trip, no dependency on the standalone export
script. VAE reconstruction/mu/logvar come from a batched
`detector._check_fitted()` forward pass (same private-model access
pattern `VAEDetector.score_samples`/`latent_diagnostics` already use
internally); the IF score reuses `models["iforest"][1]`
(`if_detector.score_samples(X_if)`, already computed in Phase 6) as-is.
Always label-free (`label_col=None`): official runs on this project carry
no target, so this is the only mode the pipeline itself exercises; a
labeled run stays the standalone script's job for validating the suite.

**`main.py`**: new `--run-diagnostic-suite` flag (`BooleanOptionalAction`,
default off -- the vendored package is a dev/analyst dependency, not a
core one). When on, "Phase 9c" (after the per-model OOT export loop,
before interpretability) calls the new bridge function inside a
try/except that logs and continues on any failure -- same
never-block-the-deliverable contract as the analyst dashboard's Phase 9b.
Output goes to `artifacts/reports/ifvae_diagnostics/` (already covered by
the existing blanket `artifacts/` gitignore rule, no new entry needed).
The result feeds
`context["diagnostic_suite"]` for Phase 11 and gets a line in the final
summary and the `model_documentation.md` artifact catalog.

**`src/reporting/report.py`**: new chapter, "Diagnóstico cruzado IF-VAE"
(`_diagnostic_suite_section_md`/`_diagnostic_suite_section_html`,
`_diagnostic_suite_quadrant_verdict`), placed right after the interactive
"Explicabilidad" charts and before the per-model detail cards, in both
the Markdown and HTML reports. Written pyramid-first:
1. A **verdict sentence computed fresh from this run's own quadrant
   counts** (never a fixed narrative) -- which detector (IF or VAE)
   contributed more *exclusive* hits on this OOT window.
2. Quadrant-count KPI tiles/table (BOTH/VAE_ONLY/IF_ONLY/NEITHER, counts
   and percentages).
3. The suite's own `disagreement.png`, embedded inline as base64
   (`_img_data_uri`, HTML) or linked (Markdown).
4. An independent VAE latent-health recheck (active units, collapsed
   fraction, mean KL) -- computed by the external suite over the
   reference/train block, a second opinion on the pipeline's own
   `latent_health` check.
5. Top population drift (reference vs. scored), with the same
   calendar/lag-artifact caveat already documented for the standalone run.
6. Any risk warnings the suite raised.
7. A methodology/scope footer (label-free, row counts, percentile
   threshold) and a link out to the suite's own full `report.md`.

The chapter renders as `''` -- no placeholder, no stub -- when the flag is
off, the package isn't installed, or the phase failed; same contract the
rest of this report already uses for optional sections (`_hero_html`,
the analyst dashboard, etc.).

**Iteration 1 → 2, methodology check:** while validating the chapter I
found that its "ambos detectores" (BOTH) count is *not* the same
measurement as the top-5% overlap already annotated on the existing
in-house "Concordancia entre detectores" chart shown just above it
(`report_content.py::build_plotly_figures`) -- that chart deduplicates the
OOT window to one row per entity (max score) and ranks each score against
that same OOT population; this chapter keeps one row per (entity, month)
and ranks each score against the **training** distribution
(`ifvae_diag.scoring.anomaly_percentile`, genuinely out-of-sample). The two
numbers landing in the same ballpark (2.0% vs. 2.3% on the `--quick`
stacked run) is a reassuring cross-check, not a coincidence to hide -- but
a reader who notices two "similar" numbers differ deserves the reason
spelled out rather than left to guess, so iteration 2 added an explicit
methodology note to both the HTML and Markdown chapter (worded differently
per format, since the comparison chart itself only exists in the
interactive HTML report, not in Markdown).

**Verified end to end**, `python main.py --quick --no-tune`, both
`--stack-iforest-into-vae` (default) and `--no-stack-iforest-into-vae`:
- Stacked: 51/51 health checks, quadrants `NEITHER=1327 VAE_ONLY=73
  IF_ONLY=65 BOTH=35` (1,500 OOT rows vs. 2,000 reference rows).
- Parallel: 58/58 health checks, quadrants `NEITHER=1091 VAE_ONLY=309
  BOTH=59 IF_ONLY=41`.
- Flag off: chapter cleanly absent from both HTML and Markdown (only the
  nav anchor remains, same as every other optional section), 50/50 health
  checks, no regression.
- Cross-check confirming the stacking plumbing is genuinely reflected, not
  hardcoded: the parallel run's drift table has no `iforest_score` row
  (the stacked run's does) -- only stacking mode injects that column into
  the VAE's own feature space, and the chapter's drift table correctly
  follows whichever `vae_feature_names` this run actually used.
- **Cross-checked every number in the rendered chapter against the raw
  `summary.json`/`drift.csv`/`warnings.json` the suite itself wrote** --
  exact match, including the manually-recomputed `mean_kl` average and the
  percentage roundings. Parsed the full `anomaly_report.html` with
  `html.parser` end to end: zero unclosed/mismatched tags. Verified the
  embedded `disagreement.png` base64 payload decodes to a valid PNG
  (1360x1190) and that both the image and the linked suite `report.md`
  resolve to real files on disk.
- **Superseded the same day** by the structural rewrite below: the verdict
  sentence, the priority-labelled tiles and the "confirma o contradice"
  latent wording documented above no longer exist. See the 2026-09-04
  entry "Capítulo reestructurado como ficha de diagnóstico no supervisado".
- **Found and fixed one real (if currently unreached) correctness gap**
  during this validation: the latent-diagnostics callout guarded on
  `latent.get("latent_dimensions")` truthy, but the suite returns a
  *smaller* dict -- `{"latent_dimensions": N, "kl_available": False}`, no
  `active_units`/`collapsed_fraction`/`mean_kl_by_unit` -- when the frames
  carry `mu__*` columns but no `logvar__*`. The old guard would have
  defaulted those missing keys to `0`/`0.0` and rendered "0 de N
  dimensiones activas (fracción colapsada 0.0%, KL media nan)" -- reading
  as "total posterior collapse" when the true state is "unknown, no
  logvar available". This project's own bridge
  (`run_ifvae_diagnostic_suite`) always supplies both `mu` and `logvar`,
  so the bug could not fire through today's integration -- caught by
  directly unit-testing the two render functions with a crafted
  `kl_available: False` payload, not by any real run. Fixed by checking
  `"active_units" in latent` instead of the `latent_dimensions` truthy
  check, in both the Markdown and HTML renderers; re-tested with the
  crafted payload (callout now correctly omitted, no "nan" leaks into
  either format) and re-verified end to end that the real, always-has-both
  path still renders identically (51/51 health checks, same numbers).

---

## 2026-09-04 — Capítulo reestructurado como ficha de diagnóstico no supervisado

**Ask:** reorganizar y completar *únicamente la estructura documental* del
apartado "Diagnóstico cruzado IF-VAE", para que funcione como una ficha
estructurada de diagnóstico no supervisado y no emita interpretaciones.
Restricciones explícitas: no interpretar resultados, no determinar qué modelo
es mejor, no afirmar que un detector aporta más señal, no asignar prioridad
operativa a ningún cuadrante, no convertir concordancia/latente/drift/
estabilidad en evidencia de desempeño, no inventar nada, no usar lenguaje
causal ("confirma", "demuestra", "valida el desempeño"), y no hardcodear
cifras, umbrales, nombres de features ni conclusiones. Los renderers HTML y
Markdown deben consumir el mismo objeto estructurado y no contener lógica.

**Qué se eliminó.** El capítulo escrito esa misma mañana (entrada anterior)
violaba varias de estas restricciones y se retiró por completo: la frase de
veredicto ("el VAE aporta más señal exclusiva que el Isolation Forest"), el
subtítulo de prioridad operativa en la tarjeta de cuadrante BOTH ("máxima
prioridad de revisión"), el texto "Confirma — o contradice —" del bloque
latente, y `_diagnostic_suite_quadrant_verdict()` como función. Una prueba
verifica que ese helper ya no exista en el módulo, no sólo que no se use.

**Arquitectura nueva (tres capas).**
1. `src/evaluation/ifvae_contract.py` — contrato versionado y serializable
   (`CONTRACT_VERSION = "1.0.0"`), única capa que decide disponibilidad,
   severidad, procedencia y redacción. Cada valor es un
   `field(value, source=…, status=…, reason=…)`: `field()` lanza si falta la
   fuente y lanza si un estado distinto de `EXECUTED` no trae motivo, de modo
   que un valor de respaldo silencioso no se puede escribir por descuido.
   Cinco estados estructurados: `EXECUTED`, `NOT_APPLICABLE`, `UNAVAILABLE`,
   `FAILED`, `NOT_REQUESTED`.
2. `src/evaluation/ifvae_diagnostic.py` — reúne lo que el contrato necesita
   leyendo los artefactos que la suite ya escribió: aritmética de conjuntos
   sobre los cuadrantes, Spearman sobre las columnas de percentil,
   comparación de candidatos VAE *descubiertos desde las columnas presentes*
   (no desde una lista fija), barrido de la malla configurada, lectura
   latente, autopsias de alertas por cuadrante reutilizando
   `residual_contributions`/`build_autopsy` de la propia suite con un
   selector declarado (orden por percentil) en vez de etiquetas, drift con
   columna de familia para filtrar, estabilidad, cortes por periodo y
   segmento, y verificación de existencia y tamaño de los 13 artefactos.
   Se extrajo `diagnose_frames()` para que todo ese camino sea ejecutable sin
   un VAE entrenado en el proceso — es la función que usan las pruebas.
3. `src/reporting/diagnostic_section.py` — los dos renderers y nada más.
   Recorren el mismo contrato y maquetan cuatro tipos de bloque (`fields`,
   `table`, `note`, `scatter`). El scatter es un SVG en línea, sin librerías,
   con líneas de umbral etiquetadas, forma de marcador distinta por cuadrante
   y conteos impresos, para que se lea en escala de grises y por lector de
   pantalla. `src/reporting/report.py` sólo delega.

**Las 15 secciones** quedaron: alcance, configuración efectiva, matriz de
disponibilidad (21 diagnósticos), unidad de análisis, concordancia y
desacuerdo, comparación de candidatos VAE, sensibilidad, latente,
reconstrucción y autopsias de alertas, calidad y desplazamiento, estabilidad,
evolución temporal y segmentación, bloques condicionados a verdad base,
matriz de experimentos, y riesgos/limitaciones/procedencia.

**Perillas nuevas, todas apagadas o vacías por defecto** para que nada se
barra ni se muestre sin pedirlo: `--diagnostic-sensitivity-grid P [P …]`
(malla de umbrales de §7; sin ella la sección reporta `NOT_REQUESTED` en vez
de inventar percentiles), `--diagnostic-autopsy-rows N` (presupuesto de
despliegue de §9, no de alertas) y `--diagnostic-entity-view` (agrega la
vista por entidad a §4 sólo si además hay regla de agregación declarada).

**Pruebas** — `tests/test_diagnostic_section.py`, 28 pruebas, primer
directorio `tests/` del proyecto. Cubren los criterios pedidos: paridad
HTML/Markdown (celda por celda, no fila por fila — las dos maquetaciones
separan celdas distinto), trazabilidad de cada campo a su fuente, ausencia de
valores de respaldo, actualización al cambiar umbral / candidato VAE / regla
de agregación, arquitectura apilada / paralela / desconocida, separación
observación vs. entidad, estados con y sin etiquetas, `UNAVAILABLE` cuando no
hubo reajustes de estabilidad, invariantes aritméticos de cuadrantes y
porcentajes, artefactos ausentes o de cero bytes, y los casos degenerados
(cero alertas, empates, `logvar` ausente, IDs duplicados, infinitos,
faltantes, solapamiento temporal). Dos pruebas más escanean el código de los
renderers —con los docstrings removidos vía AST— para verificar que no
contengan lenguaje de veredicto ni constantes de corrida (`0.95`, `P95`,
`recon_topk`, nombres de modelo), y que no hagan aritmética sobre las cifras
de la corrida.

**Dos defectos encontrados y corregidos durante las pruebas:** la prueba de
paridad comparaba filas unidas con `" | "`, que no sobrevive al despojado de
etiquetas HTML (se corrigió el enumerador del contrato para emitir celda por
celda); y la prueba de "cero alertas" asumía que un umbral de 0.999999 dejaba
todo por debajo, cuando `anomaly_percentile` puede devolver exactamente 1.0
— se reescribió construyendo una población evaluada íntegramente por debajo
de la referencia, que es el escenario real de cero alertas.

**Comandos de validación ejecutados.**
- `python -m pytest tests/ -q` → **28 passed**.
- `python -m pytest tools/if_vae_diagnostic_suite/tests -q` → **31 passed**
  (las pruebas metodológicas de la suite original quedan intactas).
- `PYTHONPATH=src python scripts/quality_gate.py` (suite) →
  `compile=True tests=True mutations=True complexity_limit=10`, 4/4 mutantes
  eliminados.
- `python tools/render_diagnostic_example.py` → ejemplo sintético en
  `artifacts/reports/ifvae_diagnostics_example/` (HTML + Markdown), con
  banner que lo identifica como datos sintéticos; HTML sin etiquetas
  desbalanceadas.
- Dos ejecuciones completas consecutivas del pipeline, sin fallos:
  1. `python main.py --quick --no-tune --run-diagnostic-suite
     --diagnostic-sensitivity-grid 0.90 0.95 0.99` → 51/51 health checks,
     modo apilado, §7 con la malla barrida.
  2. `python main.py --quick --no-tune --no-stack-iforest-into-vae
     --run-diagnostic-suite --diagnostic-entity-view` → 58/58 health checks,
     §1 reporta "Paralelo", §4 muestra la vista por entidad, y §7 vuelve a
     `NOT_REQUESTED` al no declararse malla — el capítulo sigue la
     configuración, no valores fijos.

**Limitaciones pendientes** (declaradas en el propio capítulo, no ocultas):
estabilidad IF sale `UNAVAILABLE` porque el puntaje se pasa precalculado y la
suite no reajusta con varias semillas; estabilidad VAE no existe en la suite;
el análisis por segmento queda `NOT_APPLICABLE` mientras no se declare
columna de segmentación; la matriz de experimentos queda `NOT_REQUESTED`
porque el pipeline no lleva registro por variante; y todos los bloques
supervisados quedan `NOT_APPLICABLE` por ausencia de verdad base.

---

## 2026-09-05 — Ficha reestructurada (9 secciones), estabilidad IF/VAE implementada de verdad, segmentación real, y nuevo capítulo de interpretación + flujo de decisión + recomendaciones

**Ask** (llega inmediatamente después de la entrada anterior, que había
prohibido explícitamente toda interpretación en la ficha): validar
indicadores, construir un toolkit de interpretación por gráfico/análisis, un
flujo estilo diagrama de decisión para la corrida específica, una
recomendación al analista, identificar e implementar correctamente lo que
"no pudo correr", depurar segmentos que hablan de corridas no realizadas
(el modelo es no supervisado), eliminar explícitamente las secciones 3
(disponibilidad), 4 (unidad de análisis), 9 (autopsias), 10 (calidad/
desplazamiento), 13 (bloques condicionados a verdad base) y 15 (riesgos/
procedencia), y finalmente fusionar con el proyecto original como la
corrida oficial -- con la aclaración explícita de que el cambio es solo del
reporte, no del resto del código. Mandato explícito de rigor: investigar
qué haría un experto del área antes de fijar cualquier umbral, y declarar
cada trade-off en vez de absorberlo silenciosamente.

**Tensión resuelta, no promediada.** La entrada anterior pedía "sin
interpretación, sin veredicto, sin prioridad operativa" en la ficha; esta
pide un toolkit de interpretación y una recomendación. Ambas se mantienen
en pie: la ficha (`ifvae_contract.py`) sigue sin interpretar nada, y un
capítulo NUEVO y separado, "Interpretación y recomendaciones"
(`ifvae_interpretation.py` + `interpretation_section.py`), lee los mismos
números y sí interpreta, con cada afirmación etiquetada con su base y su
severidad, y un banner explícito de que es heurística, no verdad base.

**Investigación previa a implementar** (WebSearch, antes de fijar ningún
umbral nuevo): estabilidad multisemilla de modelos tipo autoencoder --
"Evaluating the Stability of Deep Learning Latent Feature Spaces"
(arXiv:2402.11404, 2024) reporta disimilitud de Jaccard >0.6 (moda ≈0.86)
entre espacios latentes de autoencoders entrenados independientemente, i.e.
los VAE son, en la evidencia publicada, bastante menos estables entre
semillas que los ensambles de árboles por defecto. Consecuencia directa:
**no se fijó un umbral universal de "estable"** para el Jaccard del VAE --
un experto que revisara ese corte lo rechazaría por no estar respaldado por
evidencia. Solo se marca el caso degenerado (Jaccard < 0.05, solapamiento
casi nulo), no un punto de corte intermedio inventado.

**Seis secciones eliminadas de la ficha** (`src/evaluation/ifvae_contract.py`,
`CONTRACT_VERSION` 1.0.0 → 2.0.0), por pedido editorial explícito, no
porque no pudieran ejecutarse: disponibilidad de diagnósticos, unidad de
análisis (la nota de unidad de observación se movió a §1 y a la tabla de
concordancia), reconstrucción/autopsias de alertas, calidad y
desplazamiento de datos (tabla cruda), bloques condicionados a verdad base,
riesgos/limitaciones/procedencia. Las 9 restantes se renumeraron 1-9. Drift
y autopsias no se descartaron: siguen alimentando el capítulo de
interpretación como señales resumidas (drift con corrección FDR), solo ya
no aparecen como tablas crudas fila-por-fila en la ficha.

**Dos diagnósticos permanentemente `UNAVAILABLE` ahora se ejecutan de
verdad** (`src/evaluation/ifvae_diagnostic.py`):
- *Estabilidad IF* estaba `UNAVAILABLE` porque el puntaje de producción es
  precalculado (`if_score_col`), así que la suite nunca reajustaba nada
  para medir estabilidad. Arreglado reajustando
  `IsolationForestDetector` (clase propia del proyecto, mismos
  hiperparámetros leídos de los atributos públicos del detector ya
  ajustado, no de un `best_params` posiblemente parcial) con semillas
  distintas y aplicando el mismo `ifvae_diag.stability.top_k_stability`
  que la suite ya usa para IF -- no una métrica nueva, la misma, aplicada a
  un detector que la integración no reajusta por su cuenta.
- *Estabilidad VAE* no existía en absoluto (la suite no tiene reajuste
  multisemilla para VAE). Implementado igual: reajustar `VAEDetector`
  (clase propia) con la misma arquitectura leída del detector ya ajustado,
  mismo `top_k_stability` sobre los puntajes de los reajustes.
  **Trade-off explícito**: los reajustes del VAE son entrenamientos
  completos, no solo puntuación -- es la parte más cara de la Fase 9c.
  `diagnostic_stability_refits` (nuevo, default 3) controla cuántos;
  `--diagnostic-stability-refits 0` lo desactiva (`UNAVAILABLE` con motivo
  declarado). Las semillas se derivan de `base_seed` (`config.seed +
  1000·i`), no están hardcodeadas.

**Segmentación ahora se ejecuta de verdad.** El panel ya trae una columna
`segment` real (retail/corporate/...) que la integración simplemente nunca
conectaba. `main.py` ahora pasa `df["segment"]` al puente diagnóstico; §8
puebla su tabla por segmento en vez de quedar en `NOT_APPLICABLE`.

**Sensibilidad y vista por entidad, encendidas por defecto.**
`diagnostic_sensitivity_grid` pasa de `()` a `(0.90, 0.95, 0.99)` (los
mismos puntos operativos P90/P95/P99 que ya usa el resto del proyecto, no
una elección arbitraria); `diagnostic_entity_view` pasa de `False` a
`True`. Ambas estaban en `NOT_REQUESTED`/apagadas solo por ser opt-in, no
por no poder ejecutarse. `§9 Experimentos diagnósticos` se deja
explícitamente `NOT_REQUESTED`: implementarlo de verdad exigiría un
harness de seguimiento por variante (contaminación/capacidad/β/
preprocesamiento/ablación/ensembles/backtests) equivalente a reconstruir
gran parte de `evidence/EXPERIMENT_MATRIX.md` de la suite -- fuera de
alcance para un cambio "solo de reporte"; la sección lo declara en su
propio texto en vez de ocultar el vacío.

**`--run-diagnostic-suite` pasa a ON por defecto** ("que esta sea la nueva
corrida oficial"). **Trade-off explícito**: toda corrida oficial ahora
requiere el paquete vendorizado instalado (`pip install -e
tools/if_vae_diagnostic_suite`) y paga su costo de tiempo --
`--no-run-diagnostic-suite` para desactivarlo si no es aceptable en una
máquina dada.

**Nuevo capítulo "Interpretación y recomendaciones"**
(`src/evaluation/ifvae_interpretation.py` +
`src/reporting/interpretation_section.py`): un `build_interpretation_contract`
separado que produce `{toolkit, indicator_validation, decision_flow,
methodology_notes}` a partir de los mismos números ya calculados --
ninguna medición nueva, solo lectura y regla determinista. Cada afirmación
lleva `basis` (de dónde sale) y `severity` (info/attention/caution, nunca
un veredicto pasa/no-pasa sobre la corrida). El flujo de decisión evalúa 5
nodos con los números reales de la corrida (¿BOTH > 0? ¿espacio latente
activo ≥⅓, mismo umbral de Burda et al. 2016 que ya usa
`collapse_verdict`? ¿drift FDR-significativo en variables de negocio,
excluyendo calendario/panel por el motivo estructural ya documentado?
¿sensibilidad al umbral ≥3×?) y la recomendación se arma con los
fragmentos que los nodos de ESTA corrida realmente produjeron, ordenados
por severidad -- nunca una cadena fija por escenario.

**Validación de indicadores**, la petición explícita de este turno: tamaño
de muestra para Spearman/Jaccard (n<30 se marca), cantidad de reajustes
para estabilidad, rango válido de la malla de sensibilidad, y -- la mejora
estadística concreta de este pase -- **corrección Benjamini-Hochberg (FDR)**
sobre los p-valores de KS del drift, reemplazando el hueco que la entrada
anterior había dejado explícitamente señalado ("no se declaró una regla de
severidad") por una regla principiada: probar docenas de features a la vez
sin corregir produce varios falsos positivos por construcción.

**Pruebas** (`tests/test_diagnostic_section.py`, 40 pruebas): se
reescribieron las que dependían de las 6 secciones eliminadas, se agregó
paridad HTML/Markdown también para el contrato de interpretación, pruebas
de selección de rama del flujo de decisión con escenarios construidos
(BOTH=0, latente colapsado, drift de negocio vs. calendario), y --lo que
antes no era comprobable-- **pruebas de estabilidad con reajustes reales**,
ajustando instancias diminutas de `IsolationForestDetector`/`VAEDetector`
de verdad y verificando un Jaccard válido en `[0,1]`. Se corrigieron dos
bugs propios durante la escritura de pruebas: dos aserciones quedaron fuera
de su bloque `with tempfile.TemporaryDirectory()`, leyendo un archivo que
ya no existía (movidas adentro).

**Comandos de validación ejecutados:**
- `python -m pytest tests/ -q` → **40 passed**.
- `python -m pytest tools/if_vae_diagnostic_suite/tests -q` → **31 passed**
  (sin cambios en la suite vendorizada).
- Dos ejecuciones completas consecutivas, con el diagnóstico ya encendido
  por defecto:
  1. `python main.py --quick --no-tune` (apilado) → 51/51 health checks;
     Jaccard real IF≈0.80, VAE=1.00 (número real de esta corrida a escala
     `--quick`, no una afirmación de "bueno" en ningún lugar de la ficha);
     tabla de segmento poblada con datos reales; malla de sensibilidad
     barrida por defecto; flujo de decisión y recomendación con los
     números de esta corrida en ambos formatos.
  2. `python main.py --quick --no-tune --no-stack-iforest-into-vae`
     (paralelo) → 58/58 health checks.
- HTML parseado end-to-end (`html.parser`): cero etiquetas sin cerrar en
  ninguno de los dos capítulos.

**Alcance respetado**: solo se tocaron `main.py` (metadatos de la Fase 9c y
CLI del capítulo diagnóstico), `src/evaluation/ifvae_contract.py`,
`src/evaluation/ifvae_diagnostic.py`, `src/evaluation/ifvae_interpretation.py`
(nuevo), `src/reporting/report.py` (hooks del capítulo),
`src/reporting/diagnostic_section.py`,
`src/reporting/interpretation_section.py` (nuevo), y
`tests/test_diagnostic_section.py` -- ningún modelo, preprocesamiento, ni
lógica de negocio existente fuera del reporte se modificó.

**Limitaciones pendientes, declaradas y no absorbidas:** `§9 Experimentos
diagnósticos` sigue sin ejecutarse (falta un registro de corridas por
variante); estabilidad VAE añade costo real de entrenamiento cada corrida
oficial (mitigable con `--diagnostic-stability-refits 0`); el paquete
vendorizado es ahora una dependencia de facto de la corrida oficial, no
solo de un modo opcional.

---

## 2026-09-07 — Export OOT del IF antes del stacking (validación), tiempo transcurrido en minutos/horas, y confirmación visual con captura de pantalla real

**Ask.** Validar que el stacking IF→VAE no esté "perdiendo" del top de OOT
individuos que sí interesan: exportar el Isolation Forest evaluado en la
OOT a Excel, igual que ya se hace con el checkpoint P95, sin dejar de
exportar el P95. Agregar ese paso al flujo de ejecución visual. Mostrar el
tiempo transcurrido en minutos (no segundos), y en horas y minutos pasados
los 60 minutos. Al terminar, al menos 3 iteraciones de prueba, subir a
GitHub, depurar contexto y actualizar documentación. Un pedido adicional a
mitad de turno pidió además una doble validación explícita de que el
capítulo de diagnóstico/interpretación de la sesión anterior sí aparece en
el reporte real, con capturas de pantalla como evidencia -- cubierto en
detalle más abajo.

**Nueva "Phase 6d: IF OOT export (validation)"** (`main.py`, justo después
de "Phase 6c: IF P95 checkpoint export" y antes de "Phase 6b: IF -> VAE
stacking" -- es decir, con el forest ya ajustado pero su puntaje todavía
sin entrar a la matriz del VAE). Calibra su propio umbral sobre validación
(igual que la Fase 8b hace después por modelo, pero adelantado y usando
solo el puntaje del forest), calcula el top-5 de variables por fila para
las observaciones OOT (`explain_rows_iforest`, igual patrón que la Fase 9),
y exporta con la misma función que ya usa el entregable oficial
(`export_oot_top_anomalies`) -- mismo layout ID-PERIODO-PUNTAJE-BANDA-
VARIABLES, mismas bandas P90/P95/P99 -- a
`artifacts/reports/oot_p90_iforest.xlsx`. **Solo corre cuando el stacking
está activado** (`if config.stack_iforest_into_vae:`): en modo paralelo el
forest ya recibe este mismo export por la vía normal (Fase 9, por modelo),
así que la Fase 6d ahí se saltaría y recalcularía un archivo idéntico --
en vez de eso, no corre en absoluto, sin duplicar trabajo. El checkpoint
P95 (Fase 6c) sigue exportándose exactamente igual, sin cambios.

**Resultado real, `--quick` sintético**: de las 50 filas del top-P90 del
Isolation Forest, solo 29 coinciden con el top-P90 de la VAE apilada -- 21
individuos que el forest solo habría marcado no aparecen en la cola
apilada. Esto es justo la comparación operativa que pedía el usuario, y es
consistente con -- pero más granular que -- el hallazgo ya documentado del
2026-08-16 (`CONTEXT.md` "IF → VAE stacking"): que el stacking a nivel de
feature no transfiere el ranking del forest de forma agregada. La
diferencia es que ese hallazgo fue una medición histórica de una corrida;
esta exportación deja la comparación disponible en cada corrida oficial.

**Aparece en el flujo visual sin código adicional.** Tanto la vista en vivo
como el diagrama estático (`src/reporting/flow_visualization.py`) derivan
sus nodos directamente de `run_events.jsonl`, tratando cualquier nombre de
fase que calce con `^Phase \d+[a-z]?` como un nodo nuevo -- envolver el
bloque en `with log_phase("Phase 6d: ..."):` bastó por sí solo. Confirmado
visualmente: captura de pantalla del diagrama muestra "Phase 6d: IF OOT
export (validation)" en su posición correcta, entre "Phase 6c" y "Phase
6b". La única pieza que sí necesitó una entrada manual fue el checklist del
dashboard de terminal (`_PHASE_PLAN` en `src/utils/console_ui.py`), que es
una lista fija usada solo para dibujar las filas pendientes por adelantado
y ponderar el porcentaje de avance.

**Tiempo transcurrido: minutos, y horas+minutos pasados los 60 minutos --
solo para los contadores de tiempo TOTAL, nunca por fase.** Dos lugares
mostraban el tiempo total transcurrido en `H:MM:SS` o en segundos crudos:
el encabezado del dashboard de consola (`ConsoleUI._fmt_elapsed`) y la
línea acumulada "`N/M fases · ...s of work`" de la vista en vivo del
navegador (`flow_visualization.py`, plantilla `_LIVE_HTML`). Se agregó un
formateador nuevo en cada archivo (`_fmt_elapsed_minutes` en Python,
`fmtElapsedMinutes` en JS) que da `"45m"` por debajo de 60 minutos y
`"1h 5m"` en adelante, y se usó **únicamente** en esos dos contadores de
total. Los contadores POR FASE (el spinner de la fase en curso, la línea de
checkpoints de interpretabilidad, la duración de cada nodo en ambos
diagramas) se dejaron exactamente como estaban, en segundos/milisegundos:
la mayoría de las fases terminan en unos pocos segundos, y redondear a
minutos ahí mostraría "0m" durante toda una fase corta sin decir si sigue
progresando o está colgada. Verificado con valores de borde
(`0s→"0m"`, `59s→"0m"`, `60s→"1m"`, `3599s→"59m"`, `3600s→"1h 0m"`,
`3661s→"1h 1m"`).

**Tres iteraciones de prueba ejecutadas:**
1. Modo apilado (`--quick --no-tune`): "Phase 6d" corre, exporta
   `oot_p90_iforest.xlsx` (50 filas), ambos Excel OOT aparecen en el resumen
   final, 0 chequeos de salud fallidos, comparación real de 21/50
   individuos divergentes confirmada leyendo ambos `.xlsx` con pandas.
2. Modo paralelo (`--quick --no-tune --no-stack-iforest-into-vae`): "Phase
   6d" NO aparece en el log (correctamente omitida), el forest sigue
   recibiendo su export por la Fase 9 de siempre, sin duplicar archivo ni
   trabajo, 0 chequeos de salud fallidos.
3. Formato de tiempo + integración visual: prueba unitaria de
   `_fmt_elapsed_minutes` con valores de borde: correcto. Captura de
   pantalla real (Playwright/Chromium headless) del diagrama de flujo
   confirmando "Phase 6d" en su posición correcta, sin errores de
   JavaScript ni de página.

**Validación adicional pedida a mitad de turno: ¿el capítulo de
diagnóstico/interpretación (entrada 2026-09-05) realmente aparece en el
reporte?** El usuario reportó no verlo. Se hicieron 5 pasadas de
validación real (no solo revisión de código): (1) borrar el
`anomaly_report.html` existente y correr el pipeline desde cero: (2)
revisar el log completo por advertencias de "Phase 9c" -- ninguna; (3)
verificación de texto de cada `id`/encabezado esperado de ambos capítulos
en el HTML resultante; (4) carga real en Chromium headless vía Playwright,
clic en ambos enlaces de navegación, verificación de que ambas secciones
son visibles en el DOM sin errores de consola; (5) diez capturas de
pantalla reales del contenido renderizado (ficha, toolkit de
interpretación, flujo de decisión con sus 4 nodos, recomendación). Todo
confirmó que el código sí funciona -- la causa real era que el usuario
tenía abierto un `anomaly_report.html` generado antes del commit `2c13c0f`
(el reporte no se versiona en git; vive en `artifacts/`, que está en
`.gitignore`, y se regenera localmente en cada corrida). Se subió esa
evidencia (reporte fresco + las 10 capturas) a una rama de GitHub aparte
(`reporte-validacion-2026-09-07`) para que el usuario pudiera verificarlo
sin volver a correr el pipeline, y se documentó en su propio `README.md`
qué revisar si a alguien localmente no le aparece el capítulo (paquete
`ifvae_diag` no instalado, o copia vieja del archivo).

**Alcance**: `main.py` (Fase 6d), `src/utils/console_ui.py` (formateador +
entrada en `_PHASE_PLAN`), `src/reporting/flow_visualization.py`
(formateador JS + call site). Ningún modelo ni lógica de preprocesamiento
se tocó.

---

## 2026-09-13/14 — Diagnóstico ejecutable, incidentes, segmentación configurable y dashboard por consenso

**La suite dejó de exigir instalación manual.**
`src/evaluation/ifvae_diagnostic.py::ensure_suite_installed` comprueba primero
si `ifvae_diag` es importable. Si falta, ejecuta `pip install -e` exclusivamente
sobre `tools/if_vae_diagnostic_suite` y añade su `src/` vendorizado a
`sys.path`, porque `importlib.invalidate_caches()` no vuelve a procesar por sí
solo los `.pth` creados durante el proceso actual. Se probaron los estados
`already_installed`, `installed_now` y `not_attempted`; el nuevo
`--no-auto-install-suite` permite validar sin instalar.

**El apartado 8 acepta cualquier columna de segmentación.**
`--diagnostic-segment-column NAME` reemplaza el antiguo `df["segment"]`
hardcodeado. Cadena vacía desactiva la tabla; una columna ausente registra un
warning y produce `NOT_APPLICABLE` sin ocultar la causa.

**El apartado 9 ahora ejecuta experimentos reales.** Ensembles máximo/promedio
y variantes de reconstrucción reutilizan resultados de la corrida;
contaminación IF reajusta un forest por punto (`0.01 0.02 0.05` por defecto).
Las mallas `--diagnostic-experiment-capacity-grid` y
`--diagnostic-experiment-beta-grid` son opt-in porque cada punto reentrena el
VAE. Las cinco familias que necesitan cambios de arquitectura,
preprocesamiento o varias ventanas conservadas quedan `NOT_REQUESTED` con un
motivo específico.

**El reporte hace visibles los fallos, sin ensuciarse con warnings.**
`main.py` importa `warnings` y ejecuta `warnings.filterwarnings("ignore")` para
silenciar advertencias de Python/librerías. `IncidentCollector` captura solo
ERROR/CRITICAL y los renderers filtran defensivamente cualquier WARNING que
reciban. “Qué no se ejecutó o falló” aparece únicamente con fallos reales; los
estados omitidos/no solicitados siguen explicados en §9.

**Dashboard por relación entre detectores.** La cola ya no depende únicamente
del export primario. Usa la unión P95 real y la divide en “Solo IF”, “Solo
IF+VAE” e “Intersección”; percentiles, explicaciones y recurrencia se conservan
por detector. El perfil incorpora un botón que genera un CSV UTF-8 con todas
las filas OOT del individuo y todas las columnas originales, no solo las cinco
variables explicativas.

**Documentación depurada y orientada a públicos mixtos.**
`docs/guia_practica.md` explica IF, VAE, stacking y percentiles OOT con ejemplos
y diagramas Mermaid; documenta auto-instalación, segmentación y las mallas. El
generador consolidado renderiza fences Mermaid mediante el cliente oficial CDN
y deja la fuente legible sin red. `geeksforgeeks_notes.md` se redujo de una
segunda explicación completa a un índice de fuentes, eliminando la redundancia
con las guías temáticas.

**Pruebas automatizadas:** 51/51 pasan. Son las 46 existentes, tres contratos
del dashboard para la partición de pestañas, la retención de todas las
filas/columnas OOT y los detalles por detector, y dos pruebas que aseguran que
WARNING no llega a HTML/Markdown pero ERROR sí.

**Tres iteraciones visuales Playwright (Chrome real, 1440×1000):** dashboard
(16 Solo IF / 16 Solo IF+VAE / 10 Intersección, descarga CSV real de 22
columnas), reporte (6 filas `EXECUTED`, sin sección de warnings) y documentación
(3 Mermaid convertidos a SVG), todas con 0 errores JavaScript. La primera pasada
descubrió y corrigió dos defectos que las pruebas textuales no veían: `\r\n`
quedaba como un salto literal que invalidaba el script del dashboard, y el modal
mostraba la clave interna `p2` en lugar del `entity_id`. Evidencia versionada en
`docs/validation/2026-09-14/`.

---

## 2026-09-17 — Gestión de casos, historial completo y sensibilidad post-entrenamiento

**Dashboard del analista.** La descarga individual dejó de recibir solo el
bloque OOT: ahora incorpora todas las filas y columnas disponibles de la
entidad. Se agregó el identificador configurable `puesto` bajo el ID, estados
persistentes `Sin revisión` / `En revisión` / `Cerrado`, la vista dinámica
“Casos revisados” y su CSV con fecha/hora del último cambio. Las explicaciones
IF ya no se leen de la exportación P90 filtrada sino del frame OOT explicado
completo; así, una entidad seleccionada por VAE con IF <P90 conserva sus
variables IF en el perfil.

**Sensibilidad post-entrenamiento (Fase 9d).** Nuevo módulo
`src/evaluation/sensitivity.py`, activado por defecto. Reutiliza el
preprocesador y los detectores ya ajustados y prueba, sin refit: ablación
neutral por variable, null, cero cuando corresponde, pares de variables,
niveles aleatorios de pérdida de información y poda acumulativa. Compara
variación porcentual/dirección del score, correlación de rangos, Jaccard top
10%, cambio de alertas y métricas post-label. La matriz/ranking clasifica
variables indispensables, intermedias y marginales.

Los registros con ≥90% de entradas cero/faltantes se analizan por score, CDF,
alerta, label, variables críticas, prueba KS y performance antes/después de
excluirlos. Las reglas explícitas producen una recomendación de conservar en
train/test, reservar para prueba/monitoreo o excluir en futuras evaluaciones.
Se generan HTML interactivo, Excel multih hoja, cuatro CSV y JSON, todos
enlazados desde el reporte principal.

**Verificación:** compilación de los módulos modificados; 5/5 pruebas focales
(`test_analyst_dashboard` + `test_sensitivity`) pasan; JavaScript generado del
dashboard validado con Node (`new Function`) sin errores. La ejecución focal
de sensibilidad confirmó 15 escenarios × 2 modelos, el libro/CSV/JSON/HTML y
el caso sintético con 100% de información cero.

## 2026-09-20 — Filtro de filas en cero exacto (Fase 2)

`src/data/loader.py::drop_exact_zero_rows`, invocado en la Fase 2 de
`main.py` justo después de cargar el panel y **antes** de cualquier split,
ajuste, umbral, exportación, dashboard o reporte. Elimina toda fila con al
menos `--exact-zero-row-cutoff` (default 0.90) de sus columnas numéricas/
booleanas en 0 exacto. Los faltantes (`NaN`) no cuentan como cero, y las
columnas categóricas, las llaves y el target quedan fuera del denominador
(con 8 de 20 entradas categóricas, ninguna fila podría superar 60%).

Se registra en `execution.log`: `Filtro de filas en cero exacto: N/M filas
(x%) ... quedan K filas`, seguido de la línea `Panel: K rows ...`, más la
estadística "Filas excluidas (cero exacto)" en la consola. Es una exclusión
real; el análisis de la Fase 9d (`--sensitivity-high-zero-cutoff`) sigue
siendo solo una medición post-entrenamiento que no retira filas.

El reporte (MD y HTML) incluye la sección "Filtro de filas en cero exacto" con
registros cargados, excluidos y que entran al flujo.

**Verificación:** 8 pruebas (`tests/test_zero_row_filter.py`) y corrida
`--quick` sobre el panel sintético con 60 filas inyectadas en cero (50 en
100%, 10 en 12/13 columnas) más 10 en 11/13 que debían sobrevivir:
`60/6000 filas (1.0%) ... quedan 5940 filas`.

---

## 2026-09-23 — Seguimiento en vivo de la suite diagnóstica (tqdm, función en ejecución, cronómetros) y depuración de documentación

**Pedido:** ver cada prueba de la suite diagnóstica avanzar con barras tqdm, y ver
en el flujo en vivo qué función se ejecuta y cuánto lleva, para seguir la Fase 9c
(minutos dentro de un solo bucle: reajustes de estabilidad, mallas de experimentos).

**Suite (`tools/if_vae_diagnostic_suite`).** Nuevo `ifvae_diag/progress.py`
(autónomo, sin importar nada de `src/`): `step(name)` (una función con inicio,
fin/fallo y duración medida), `track(iterable, desc=, unit=, label=)` (barra
tqdm; un ítem cuenta como hecho solo cuando el llamador terminó con él) y
`stages(desc, total=)` (barra sobre una secuencia fija de pruebas). Publica
eventos a observadores (`add_observer`), con las actualizaciones limitadas a una
por 0.5 s por barra (inicio y fin siempre). tqdm es opcional (extra `progress`).
Instrumentado: los 12 pasos de `run_diagnostic` (`_percentile_frame` extraído
para mantener la complejidad ≤10) y los bucles `isolation_forest[seeds]`,
`compare_populations[features]` y `write_outputs[files]` (13 archivos). No cambia
ningún valor calculado.

**Puente (`src/evaluation/ifvae_diagnostic.py`).** `_suite_progress` (reentrante)
reenvía los eventos de la suite a `logging_config.report_phase_event` (nueva:
línea de log + `run_events.jsonl` + observadores de fase) y a
`observability.progress_event` (nueva: evento `progress` + observadores). Se
instrumentaron `_vae_forward[batches]`, las 11 pruebas de `diagnose_frames`
(barra `ifvae_diagnostic`), cada reajuste de estabilidad
(`stability_refit[<Detector>]`, paso `refit[<Detector> seed=N]`) y cada punto de
las mallas de experimentos (`experiment[...]`).

**tqdm vs. dashboard.** Un dashboard `rich` y las redibujadas `\r` de tqdm sobre
stderr se destruyen mutuamente, así que los tqdm reales solo se dibujan cuando
`console_ui.is_live()` es falso (`--no-console-ui`, CI). Con el dashboard activo,
las mismas barras se muestran dentro del panel con `tqdm.format_meter` (formato
idéntico, sin escribir en la terminal). Las barras antiguas (`vae[epochs]`,
`optuna[...]`) no se tocaron.

**Dashboard de consola.** Nueva línea `↳ función` (funciones anidadas en
ejecución, la más interna resaltada, cada una con su tiempo) y una barra por
prueba en curso; el cronómetro de cada barra sigue corriendo entre eventos.
Corrección de un defecto previo: "n/total fases" contaba también las funciones
anidadas (`iforest.fit`, ...) y se inflaba más allá de las fases reales; ahora
cuenta solo entradas `Phase N`. La línea "ahora" nombra la fase del plan y las
funciones anidadas tienen la suya.

**Vista web en vivo.** Los eventos llevan ahora `t` (epoch con milisegundos) junto
a `ts`. `/state` devuelve `live.running_functions`, `live.progress`,
`live.recent_functions` y `server_now`; la página añade los paneles *Function
running* (con cronómetro), *Progress of running tests* (n/total, %, tiempo, ETA,
ítem en proceso) y *Last finished functions*, y los cronómetros avanzan cada
250 ms entre sondeos. `_build_nodes` sigue siendo función pura del archivo de
eventos; el reloj solo entra en `_annotate_live`. Terminada la corrida no se
muestra nada como "todavía corriendo".

**Verificación.** 82 pruebas del proyecto (21 nuevas en
`tests/test_live_progress.py`: puente sobre `diagnose_frames` real, refits reales
de un Isolation Forest, `/state` sobre un servidor local real, casos límite del
estado del flujo, render del dashboard) y 47 de la suite (16 nuevas en
`tests/test_progress.py`), con el quality gate de la suite en dos rondas
consecutivas verdes (4/4 mutantes, complejidad ≤10). El JavaScript de la página
se validó con `node --check`. Se renderizó el dashboard durante refits reales
(IF y VAE) confirmando que las barras aparecen dentro del panel y que stderr
queda vacío. **No** se ejecutó `main.py` completo: reescribiría `artifacts/` (la
corrida oficial); la verificación de extremo a extremo se hizo en directorios
temporales.

**Seguimiento en todas las fases (misma fecha, segunda parte).** Nuevo
`src/utils/progress.py` (`Bar`, `track`, `show_best_trial`): mismo contrato que el
de la suite pero del lado del pipeline (eventos `progress` + tqdm solo sin
dashboard vivo). Sustituye los seis `tqdm` sueltos (épocas del VAE, Optuna IF y
VAE con el mejor valor como postfix, permutation importance, grupos colectivos
del generador), que antes rompían el dashboard y no llegaban al flow. Nuevas
barras/pasos con nombre: diagnósticos y figuras de transformación (Fase 4),
`silhouette`/`calinski_harabasz`/`rank_stability` y el reductor 2D/UMAP (Fase 8),
explicación por fila y por lotes del VAE (Fases 9-10), los cuatro bloques de
escenarios de la Fase 9d y la escritura de sus salidas, y cada formato del
reporte (Fase 11). En el HTML del flow, cada tarjeta de fase lleva las barras que
corrieron dentro de ella (mini-barras en la página en vivo; tabla con n/total,
tiempo y último ítem en el detalle del HTML estático). Verificación: 104 pruebas
del proyecto (nuevas: `test_progress_bars.py`, `test_phase_progress.py`, casos
de `test_live_progress.py`), gate de la suite verde, JS de ambas páginas con
`node --check`. Corregido de paso el texto de ayuda de `--n-periods` (decía 8;
el default es 16).

**Depuración de documentación.** `CONTEXT.md` (1198 → ~1060 líneas): sección de
la suite reducida al contrato vigente (la narrativa fechada ya estaba aquí, en
las entradas de 2026-09-03/04/05); corregido "no hay suite de pruebas / `tests/`
fue eliminado" (existe `tests/` y el quality gate de la suite), "12 fases",
"checklist de 15 filas" (son 19), periodos por defecto (16, `--quick` 13,
partición 8/2/3/3) y "no existe medición de estabilidad entre semillas"
(la Fase 9c la mide para IF y VAE). `README.md`: mismos valores por defecto, teclas
reales del dashboard (`v`/`o`/`p`, no `d`), estructura con `tests/` y `tools/`,
sección de pruebas y descripción del seguimiento en vivo. También
`docs/guia_practica.md` (cómo seguir la suite), `docs/decisiones_de_modelado.md`
§4.3, `docs/validacion_no_supervisada.md` §4, `docs/leakage_free_pipeline.md` y
el README, `TRADEOFFS.md` y `EVIDENCE_PACKAGE.md` de la suite.

**Riesgo preexistente detectado (no corregido):** las pruebas que ajustan un
`VAEDetector` real (`tests/test_diagnostic_section.py`) no pasan `checkpoint_dir`,
así que usan `artifacts/models/vae/` por defecto. En esta máquina restauraron el
checkpoint existente sin modificarlo (mtime intacto), pero en una máquina sin
checkpoint compatible lo crearían ahí.
