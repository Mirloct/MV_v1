# The leakage-free pipeline — 7 phases

This document is the audit trail for the project's anti-leakage design: what
each phase does, **where it lives in the code**, and the reasoning behind the
choice. It doubles as the checklist to re-run whenever the pipeline changes.

Every claim below was, at the time it was written, backed by a project test
suite naming the exact assertion. That suite was removed 2026-08-22 (not
needed to run the project — see `CHANGELOG.md`), so the `Test*`/`test_*.py`
names below no longer resolve to files on disk; they are kept as a record of
what was checked, and as a guide for what a reintroduced test should cover.

---

## Phase 1 — Feature engineering: contrast against history

**Where:** `src/preprocessing/pipeline.py` → `PanelFeatureEngineer`,
`_DEFAULT_LAG_HORIZONS = (1, 3, 6)`, `_DEFAULT_RATIO_FEATURES`.

For every monetary column and every horizon `h`, three features are emitted:

| Feature | Meaning |
| --- | --- |
| `{col}_lag{h}` | the value `h` periods ago |
| `{col}_diff{h}` | `VAR_T − VAR_T−h` — the shock in the variable's own units |
| `{col}_ratio{h}` | `VAR_T / VAR_T−h` — the same shock, scale-free (2.0 = doubled) |

plus `{col}_own_z` (causal expanding z-score) and four cross-variable ratios
(`txn_amount_to_income`, `withdrawal_to_balance`, `balance_to_income`,
`avg_txn_to_income`).

> **Why both a difference and a ratio.** They answer different questions. The
> difference is the magnitude a reviewer acts on ("balance jumped 40,000"); the
> ratio is what makes a small and a large customer comparable to a detector that
> splits on absolute thresholds ("balance tripled"). A model given only
> differences ranks rich customers as permanently anomalous.

> **Why long horizons matter.** When the inputs are themselves rolling
> averages/accumulations — the normal shape of a banking feature mart — a real
> shock is spread across several months. Its one-month difference is a fraction
> of its true size, so the anomaly surfaces late or not at all. Contrasting
> against `T−3` and `T−6` recovers the amplitude the smoothing hid.

**Neutral fill.** Rows without `h` periods of history get `0.0` for
levels/differences and **`1.0` for ratios** — a ratio of 0 would read as
"collapsed to nothing" and manufacture an anomaly out of missing history.

**Horizons are resolved against the *fit window*, not the panel.** A horizon is
only kept when the training block is deep enough to contain it. See Phase 3 for
what goes wrong otherwise.

---

## Phase 2 — Strictly chronological split

**Where:** `src/evaluation/splits.py` → `chronological_split` →
`ChronologicalSplit`.

| Block | Periods (16-month default) | Job |
| --- | --- | --- |
| train | 1–8 | fits preprocessing statistics and the models |
| validation | 9–10 | selects hyperparameters **and** calibrates the threshold |
| test | 11–13 | read exactly once, at the end, to report model metrics (ROC-AUC/PR-AUC/threshold diagnostics) |
| oot | 14–16 | reserved exclusively for the `export_oot_top_anomalies` Excel deliverable (and the analyst dashboard built from it) |

> Each block is spent on a different decision, and mixing them is the classic
> leak. Validation is *consumed* by model selection and threshold fitting, so it
> can no longer be an unbiased estimate of anything — which is precisely why a
> separate untouched test block has to exist. The `oot` block goes one step
> further: it is untouched even by *reporting*. Aliasing it to `test` (the
> historical behaviour, `n_oot_periods=0`) made the "OOT Excel" describe the
> exact same rows the test-set metrics were computed from — `n_oot_periods > 0`
> carves it off as its own, strictly later, never-reported-on block.
> **Default is 3** (the last 3 months) as of 2026-08-31 — wide enough that an
> entity's OOT presence can be tracked month-to-month (recurring vs. a
> one-off), not just its single best-scoring month; `--n-oot-periods`
> overrides.

There is no random splitting anywhere in the pipeline. Short panels shrink the
train/val/test blocks rather than failing (a 6-period panel with no OOT block
yields 3/1/2); `n_oot_periods > 0` is never shrunk -- a panel too short to
honour it raises rather than silently falling back to `oot == test`. Panels
need at least 3 distinct periods with no OOT block, or 4 with one.

`chronological_split` guarantees the masks partition every row exactly once,
that the blocks are strictly ordered in time, and that no period appears in
two blocks.

---

## Phase 3 — Leak-free preprocessing

**Where:** `src/preprocessing/pipeline.py` → `fit_transform_panel(fit_mask=...)`.

The pipeline has two stages and **only one of them can leak**:

1. `PanelFeatureEngineer` estimates *nothing* — lags and diffs look strictly
   backwards, `own_z` uses only strictly-prior periods, ratios are within-row,
   the month encoding is a function of the timestamp. It runs over the **whole
   panel**.
2. The `ColumnTransformer` estimates everything that can leak — imputation
   medians, scaler moments, Yeo-Johnson exponents, one-hot categories,
   frequencies, the `"auto"` per-column transform choice. Only this stage is
   fitted on `df[fit_mask]`, i.e. the train block.

> **The trap this avoids.** The tempting shortcut — fit on train, then call
> `transform` on the test rows alone — silently destroys the panel features.
> `shift(1)` finds no history inside a single-period subset, so all lag/diff/
> own-z features become NaN, get filled with `0.0`, and end up identically zero
> on exactly the rows being evaluated. Splitting by *stage* rather than by
> *call* fixes the leak without touching the time-series dependencies.
> (formerly guarded by a regression test that reproduced the failure mode
> directly; see `CHANGELOG.md` 2026-08-22 on the test suite's removal).

### Two zero-variance hazards, both fixed

A short training window can make a column near-constant *in the fit block*. Any
fitted scaler then divides unseen values by a ~0 variance and amplifies them
without bound.

* **Cyclical seasonality.** `month_sin`/`month_cos` are already normalised by
  construction (bounded in `[-1, 1]` with a meaningful scale), so they bypass
  the scaler entirely via a `passthrough` branch. Before this fix a 3-month
  training window (Nov/Dec/Jan) drove `month_cos` to **4.9e18** on the June test
  rows, which blew up the VAE's MSE to `1.8e35`.
* **Contrast horizons.** With a 3-month training block, `lag6`/`diff6` are
  constant at the neutral fill in train and explosive at test. Horizons are
  therefore validated against the fit window (`fit_window_mask`), so a 3-month
  window keeps only `h=1`.

A **blow-up guard** (`_warn_on_extreme_magnitudes`) logs any feature exceeding
`1e6` after transformation and names it, so this class of bug can never again be
silent.

### On `RobustScaler`

`numeric_transform="robust"` is available and, measured on this data, is the
**best choice for the Isolation Forest** (OOT PR-AUC 0.272 vs 0.117 for the
`yeo-johnson` default). It is *not* the shared default, because the VAE produces
**100% NaN scores** under it: the untouched heavy tail reaches ~5e5 in scaled
units and the MSE gradients overflow. The two models want opposite treatment of
distribution *shape* and share one matrix. See `CONTEXT.md`.

---

## Phase 4 — Optuna without random folds

**Where:** `src/models/iforest.py::tune_iforest`, `src/models/vae.py::tune_vae`,
both taking `valid_mask`.

No K-fold, no shuffling. Every trial fits on the train block and is scored on
the **validation months**, passed down from `main.py` as `valid_mask`.

> `valid_mask` takes priority over the entity-blocked fallback because it is the
> split that matches deployment: the model will be applied to *future* periods,
> so selecting hyperparameters on future rows is the only honest measurement.
> Entity blocking (`groups`) defends a different leak — rows of one customer
> share a latent level — and remains the fallback when no time split exists.

### Isolation Forest

| Parameter | Range |
| --- | --- |
| `n_estimators` | 100–600, step 50 |
| `max_samples` | `"auto"`, a fraction in [0.3, 1.0], **or a fixed 64 / 128 / 256** |
| `max_features` | 0.3–1.0 |
| `bootstrap` | {True, False} |

> **Why fixed absolute `max_samples`.** It is the anti-swamping knob. Isolation
> Forest was designed around *small* sub-samples: with too many points the
> normal mass crowds the anomalies and every path lengthens. That is exactly the
> regime of an autocorrelated panel, where each entity contributes many
> near-duplicate rows.

`contamination` is **not searched** — `score_samples` ignores `offset_`, so it
cannot move any rank-based objective.

**Label-free objectives** (`objective_metric=`):

* `"tail_separation"` — `(p95 − p50) / IQR` of the validation scores, maximised.
  Both cut points are **fixed constants**, which is what makes it safe: the
  retired `_separation_margin` cut the tail at the trial's own `contamination`
  while the scores were invariant to it, so shrinking that knob mechanically
  inflated the metric and the search optimised the metric's own parameter
  instead of the forest.
* `"rank_agreement"` (default) — refit on two disjoint halves, score a common
  held-out block with both, return `max(spearman, 0) × jaccard(top deciles)`.
  A constant score vector returns `0.0`.

### VAE

| Parameter | Range |
| --- | --- |
| `latent_dim` | 4–32 (use 4–8 for smoothed inputs) |
| `hidden_dim` | 32 / 64 / 128 |
| `dropout` | 0.1–0.4 |
| `lr` | 1e-4 – 1e-3 (log) |
| `beta` | 0.1–2.0 |

* **KL annealing** — the KL weight ramps linearly `0 → beta` over the first 10
  epochs (`kl_anneal_epochs`).

  > **Posterior collapse.** The ELBO has a trivial optimum where the encoder
  > ignores the input and returns the prior: KL = 0, the decoder emits the
  > dataset mean, and the loss looks respectable while the latent code carries
  > no information. Every row then reconstructs equally badly and the anomaly
  > score is noise. The KL term pulls hardest at the start, when the decoder is
  > still useless, so ramping it in buys the decoder time to become worth
  > keeping.

* **Early stopping** — `early_stopping_patience=10`, held off until the ramp
  finishes (while `beta` is still climbing the loss is measured against a moving
  objective, so a run of "no improvement" says nothing about convergence).
* **`objective_metric="recon_p50"`** — the median validation reconstruction
  error, minimised.

  > Optimising the *mean* error would let a model win by shrinking the error on
  > the outliers too — the opposite of what is wanted, since that erases the
  > signal. The median is dominated by the normal bulk and insensitive to the
  > tail, so minimising it sharpens the contrast the score relies on.

**Study fingerprinting.** The effective study name hashes
`(X.shape, feature_names, objective mode, direction)`, so trials scored on
PR-AUC are never pooled with trials scored on a separation proxy.

---

## Phase 5 — Final training

Both `tune_*` functions refit the winning configuration on the full model-facing
block (train + validation) before saving. The split exists to make model
*selection* honest, not to throw away data once selected.

---

## Phase 6 — Threshold calibration

**Where:** `src/evaluation/thresholds.py` → `calibrate_threshold`.

**The threshold is computed on validation scores and only ever applied to
test.** Choosing the cut-off on the rows you then report on is the
threshold-fitting form of leakage — it reports the best cut-off in hindsight,
which no deployed system can have.

* `method="percentile"` — flag the top `100 − percentile` %. A business rule:
  transparent, and it fixes the alert *volume*.
* `method="pot"` (default) — Peaks-Over-Threshold. Fit a Generalized Pareto to
  the exceedances above p95, then invert for a target exceedance probability.

> **Why POT.** Pickands–Balkema–de Haan says the conditional excess
> `X − u | X > u` converges to a Generalized Pareto as `u` grows. So instead of
> assuming the score distribution's shape everywhere, POT models only its
> *tail* — the only part a threshold interacts with. That buys two things a raw
> percentile cannot: **extrapolation past the data** (a p99 from 1,000 rows is
> just the 10th largest observation; there is no p99.9 at all), and a
> **false-alarm budget** instead of an alert budget — you state "1 in 1,000" and
> get the cut-off, rather than stating "flag 1 %" and discovering the risk
> afterwards.

Degenerate cases (fewer than 30 exceedances, a failed fit, a non-finite result)
fall back to the percentile and record `fallback_reason` — a silently wrong
threshold is worse than an honest simple one.

---

## Phase 7 — Test evaluation and the deliverable

**Where:** `src/evaluation/oot_report.py` → `export_oot_top_anomalies`.

The headline output is a risk-ranked queue:

```
artifacts/reports/oot_p90_iforest.xlsx
artifacts/reports/oot_p90_vae.xlsx
```

* **`min_percentile=90.0` by default, fully parameterisable.** Every
  individual at or above the 90th percentile of the OOT score, so the queue
  scales with the portfolio instead of fixing a headcount — "the riskiest
  10%" holds whether the panel has 2,000 or 200,000 customers. Each row also
  carries a `p90`/`p95`/`p99` band, the highest it reaches, computed over the
  full OOT population before any filtering. `top_n=N` (`--top-n`) switches to
  a fixed headcount instead, for a team that works N cases a month regardless
  of portfolio size; `top_fraction` is the fallback when both are `None`.
* **One row per individual.** With several OOT months an entity appears more
  than once; the export keeps that entity's highest-scoring month, and that
  month is part of the output.
* Layout is **ID – PERIOD – SCORE – BAND – VARIABLES** (raw, human-readable
  features), plus an `alert` column flagging rows above the calibrated
  threshold when one is supplied — so the file shows both the selection
  *and* which of those cases the calibrated rule would independently have
  raised.

### The plateau effect

When the inputs are rolling averages, an entity flagged in month 13 usually
stays high in months 14–15: the spike is still inside its own moving window.
**This is not a model error** — it is the feature definition showing through,
and operationally it is an advantage, since it gives a 2–3 month window to act
before the event ages out of the history. Read the per-type recall breakdown
(`metrics_by_anomaly_type`) with that in mind: consecutive alerts on one entity
are usually one event, not three.

---

---

## Appendix — IF → VAE stacking, and what it measured

`src/models/stacking.py` implements the stacked arrangement: the Isolation
Forest runs first and its per-row score is appended to the matrix the VAE trains
on (`--stack-iforest-into-vae`, on by default; `--no-stack-iforest-into-vae`
restores the parallel arrangement). The forest used for the feature is fitted on
**train only**, and the `StandardScaler` on the augmented matrix is fitted on
train only, so the column order and the statistics are identical across blocks.

Three arms, same data, same split, same VAE hyperparameters and seed, only the
input matrix varying (800 entities × 15 periods, test block):

| Arm | Cols | PR-AUC | ROC-AUC | Recall@10% | Recovers the forest's top-50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A — raw `X` (parallel) | 129 | **0.1416** | **0.8045** | 0.4348 | 16 / 50 |
| B — `X` + in-sample IF score | 130 | 0.1251 | 0.7576 | 0.3696 | 16 / 50 |
| C — `X` + out-of-fold IF score | 130 | 0.1251 | 0.7576 | 0.3696 | 16 / 50 |
| iForest alone | — | 0.1412 | 0.7310 | 0.4348 | 50 / 50 |

Three findings, all measured:

1. **Stacking costs accuracy here** — PR-AUC 0.1416 → 0.1251 (−12%), ROC-AUC
   0.8045 → 0.7576.
2. **The appended column is nearly invisible to the VAE.** It accounts for
   **0.736%** of the total reconstruction MSE, against a uniform share of
   1/130 = 0.769%, and ranks 39th of 130 columns by error. The VAE treats it as
   one ordinary feature, because that is what it is.
3. **Stacking does not transfer the forest's ranking.** The stacked VAE's top-50
   overlaps the forest's top-50 by 16/50 — *exactly the same as the parallel
   VAE*. Adding the column changed the queue's coverage not at all.

> **Why (3) happens.** The VAE's anomaly score is a reconstruction error summed
> over all 130 columns. A row the forest finds extremely isolated has one
> unusual value out of 130, contributing under 1% of that sum. The architecture
> has no way to express "this row is anomalous *because the forest says so*" —
> only "this row's `iforest_score` column was somewhat hard to reconstruct".
> Feature-level stacking propagates a *little* information; it does not
> propagate a *ranking*. To keep both signals in one queue you need score-level
> combination (rank average / rank max), not feature-level stacking.

**Out-of-fold made no difference** (arm C ≡ arm B) for the same reason: a column
worth 0.7% of the loss cannot change the outcome regardless of how it is built.
`score_shift_report` still measures the in-sample shift — **+2.0 sd** between
the train block and validation/test on this data — and warns above 1 sd, because
on a pipeline where the column *did* carry weight that shift would inflate the
reconstruction error of every held-out row rather than only the anomalous ones.

---

## Anti-leakage checklist

| # | Control | Mechanism |
| --- | --- | --- |
| 1 | Train/val/test split is strictly chronological | `chronological_split` |
| 2 | Contrast features (`diff{h}`, `ratio{h}`) fight the smoothing | `_DEFAULT_LAG_HORIZONS` |
| 3 | Scalers/encoders `.fit()` on train only, `.transform()` elsewhere | `fit_transform_panel(fit_mask=...)` |
| 4 | Panel features still computed over the full panel (no zeroed lags) | `PanelFeatureEngineer` runs before the train-only `ColumnTransformer` fit, not after |
| 5 | Optuna uses a static temporal split, never random K-fold | `tune_*(valid_mask=...)` |
| 6 | Objective is scored on held-out rows, never in-sample | `_blocked_split` (IF) / temporal `val_fraction` split (VAE) |
| 7 | Threshold calibrated on validation, applied to test | `calibrate_threshold` |
| 8 | VAE uses KL annealing + early stopping | `kl_anneal_epochs`, `early_stopping_patience` |
| 9 | VAE validation is temporal, not shuffled | `valid_mask` passed from `main.py`, both tuned and untuned paths |
| 10 | No zero-variance column can silently explode | `_warn_on_extreme_magnitudes` |
| 11 | Deliverable is distinct individuals, parameterisable | `export_oot_top_anomalies` (percentile default; `--top-n` for a fixed headcount) |
