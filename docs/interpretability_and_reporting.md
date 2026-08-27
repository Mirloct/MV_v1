# Interpretability & Reporting

This document covers two modules:

- `src/interpretability/` — feature attribution and diagnostic figures for the
  Isolation Forest and the VAE detectors.
- `src/reporting/` — assembling a run's metrics, figures, and the OOT Excel
  deliverable into self-contained HTML / Markdown reports. No PDF output.

It explains the concepts, the public API of both modules, where figures and
reports land on disk, and gives a runnable end-to-end snippet.

> Interpretability concepts (feature attribution, autoencoder reconstruction
> error) are adapted from GeeksforGeeks; see
> `geeksforgeeks_notes.md` (section 2 for Isolation Forest path
> length, section 3 for autoencoder/VAE reconstruction). Source URLs are cited
> in [Section 1](#1-concepts). All explanations are paraphrased in our own words.

---

## 1. Concepts

### Feature attribution (why a point scored high)

A detector returns a single anomaly score per row (project convention:
**higher = more anomalous**). Interpretability answers the follow-up question:
*which features drove that score?* Two complementary views are used.

- **SHAP (SHapley Additive exPlanations)** distributes a model's output across
  its input features using a game-theoretic attribution: each feature gets a
  signed contribution, and the contributions sum to the model output. Averaging
  the absolute contributions over many rows (`mean|SHAP|`) gives a global
  ranking of which features matter most to the anomaly score. For tree models a
  fast exact variant exists (`TreeExplainer`); otherwise a model-agnostic
  explainer perturbs inputs and observes the score, and a permutation-importance
  fallback measures how much shuffling each feature moves the score.

- **Path length** ties the Isolation Forest score back to its isolation
  mechanism: anomalies are isolated with *shorter* average path lengths, so
  score and normalized path length are inversely related. This is a
  sanity/diagnostic view rather than a per-feature attribution.

### Autoencoder reconstruction error (which features the VAE cannot reproduce)

An autoencoder trained on (mostly) normal data reconstructs normal inputs well
and unfamiliar/anomalous inputs poorly, so the **reconstruction error** is the
anomaly signal. Decomposing that error *per input feature* — `mean_i (x_ij -
x_recon_ij)^2` for each column `j` — reveals which features the VAE reconstructs
worst and therefore which features drive its anomaly score. The latent-space
plot is the companion view: encode rows to latent means and project to 2D to see
whether anomalies separate from the normal bulk.

Sources (paraphrased in `geeksforgeeks_notes.md`):

- Isolation Forest / path length —
  https://www.geeksforgeeks.org/machine-learning/what-is-isolation-forest/ and
  https://www.geeksforgeeks.org/machine-learning/anomaly-detection-using-isolation-forest/
- Autoencoders / VAE reconstruction —
  https://www.geeksforgeeks.org/machine-learning/variational-autoencoders/ and
  https://www.geeksforgeeks.org/numpy/types-of-autoencoders/

---

## 2. Interpretability API (`src/interpretability`)

All four entry points write their figure(s) under
`artifacts/reports/figures/` (the project-wide figures rule) and return
plain Python `dict`s / paths so the reporting module can consume them directly.
All accept a fitted detector plus the preprocessed feature matrix `X` (dense
ndarray or scipy sparse; densified internally).

### Isolation Forest — `iforest_explain.py`

`shap_summary_iforest(detector, X, feature_names=None, out_dir=..., max_samples=2000, ...) -> dict`

Feature attribution for the Isolation Forest. Returns
`{feature: mean_abs_contribution}` sorted descending, and saves a SHAP beeswarm
(or a bar chart on the permutation fallback). It tries three tiers in order,
falling through on failure, and logs which tier fired:

1. **`shap.TreeExplainer`** on `detector.model_` — native, exact tree SHAP.
   Whether the installed `shap` version accepts a scikit-learn
   `IsolationForest` is version-dependent, so this may raise and trigger the
   fallback.
2. **Model-agnostic `shap.Explainer`** over `detector.score_samples` with a
   small subsample masker — used when tier 1 raises. Cost scales with the
   **feature count** (each evaluation re-scores the forest ~`2*n_features+1`
   times), so this tier times a couple of calibration rows first and only
   explains as many more as fit inside a ~60s budget rather than running to
   completion unbounded — see `CONTEXT.md` and `CHANGELOG.md` 2026-08-26 for
   why (measured at ~2.7+ hours unbounded at ~180-190 features).
3. **Permutation importance** on `detector.score_samples` — the last resort;
   each feature column is shuffled and the mean absolute change in the anomaly
   score is recorded. Also budgeted: rows are subsampled and `n_repeats` is
   reduced so total `score_samples()` calls stay under a fixed cap regardless
   of feature count.

`path_length_analysis(detector, X, out_dir=..., max_samples=20000, ...) -> dict`

Relates the anomaly score to the normalized average path length. scikit-learn's
score is `s_sklearn = 2 ** (-E[h] / c(n))`; the project detector exposes
`score = -sklearn.score_samples`, so `score = 2 ** (-nd)` with `nd = E[h]/c(n)`
the normalized average path length. Inverting gives the exact closed-form
`nd = -log2(score)` per sample (no private-API access). Saves a two-panel figure
(score histogram + path-length-vs-score scatter) and returns summary stats
including `score_pathlen_corr` (the score↔path-length correlation, expected
negative), score/path-length min/mean/max, `n_samples`, and `figure_path`.

### VAE — `vae_explain.py`

`latent_space_plot(detector, X, out_dir=..., y=None, ...) -> str`

Encodes `X` to latent means, reduces to 2D (**UMAP** when `umap-learn` is
importable, else **PCA**; a 2D latent is plotted directly, a 1D latent is padded)
and scatters. Colored by labels `y` (0/1) when given, otherwise by the
reconstruction-error anomaly score. Returns the absolute PNG path.

`reconstruction_error_by_feature(detector, X, feature_names=None, out_dir=..., max_samples=2000, top_n=30, ...) -> dict`

Per-feature mean squared reconstruction error. Runs the torch model in
eval / no-grad in batches using the deterministic encoder mean (exactly as
`VAEDetector.score_samples` does per row), computes `mean_i (x_ij - x_recon_ij)^2`
per column, and returns `{feature: mean_recon_error}` sorted descending. Saves a
bar chart of the `top_n` worst-reconstructed features.

---

## 3. Reporting API (`src/reporting`)

`build_report(context, out_dir="reports", basename="anomaly_report", formats=("html", "md", "model_doc")) -> dict`

Assembles a run's artifacts into up to three self-contained deliverables under
`out_dir` (default `artifacts/reports/`). Returns
`{"html": ..., "md": ..., "model_doc": ...}` (each a path or `None`). **No PDF
is generated** -- a PDF renderer existed here previously and was removed by
explicit decision; see `docs/decisiones_de_modelado.md`.

> **Language.** Every reader-facing string in the report and its documentation
> is in **Spanish**. Source comments and docstrings stay in English for the dev
> audience. See `src/reporting/report_content.py`.

The `context` is a plain dict; every key is optional and missing pieces are
silently omitted:

| Key             | Meaning |
| --------------- | ------- |
| `title`         | Report title. |
| `generated_at`  | ISO timestamp shown in the HTML header. |
| `dataset`       | Dataset summary fields (KPI tiles + table). |
| `models`        | `{model_name: {"best_params": {...}, "metrics": {...}, "threshold": {...}}}`. |
| `figures`       | List of `{"title": str, "path": str}`; missing files are skipped. |
| `chart_data`    | Raw scores/labels/thresholds driving the interactive Plotly charts; `chart_data["static"]` additionally carries the arrays behind the former-PNG charts (SHAP importances, path length, embeddings, latent space). |
| `oot_excel`     | Path, or `{model: path}`, to the risk-ranked Excel deliverable(s). |
| `preprocessing` | Flat dict of pipeline settings — goes to the technical doc only. |
| `notes`         | Free-text notes block. |

### The three outputs — and the split between them

The deliverables are split by **audience**, which is the point of the design:
the two business-facing formats never show a hyperparameter dump or the VAE
math, and the technical file never tries to be a narrative.

- **HTML** (`.html`) — the primary business artifact. A single **offline**
  file: CSS inlined, **every chart an interactive Plotly figure** (no CDN, no
  network dependency, no embedded raster images). Leads with a **hero figure**
  (the anomaly count), then grouped KPI tiles, the interactive results charts,
  a per-model explainability section (SHAP importances, path length, 2D
  embeddings, VAE latent space — all Plotly, none of them PNGs), per-model
  metric cards, the indicator reading guide, the statistical-trust section and
  the parameter glossary. Light/dark theme toggle included. Deliberately does
  **not** include a chart per raw input column: a real feature mart runs
  50+ columns, and one histogram each would saturate the page for a section
  that is diagnostic, not a modelling result (that diagnostic is still
  available -- see `model_documentation.md` below).
- **Markdown** (`.md`) — GitHub-friendly mirror of the same narrative. Figures
  linked relative to `out_dir`; the OOT deliverable explained inline.
- **`model_documentation.md`** — the technical companion, always written under
  that fixed name (independent of `basename`). Holds exactly what the business
  report deliberately omits: every model's **exact hyperparameters**, the full
  **threshold-calibration record** (GPD shape/scale, exceedances, target FAR),
  the **VAE math** in LaTeX, the **preprocessing settings**, a **catalog of
  every artifact path** the run wrote, and the preprocessing-diagnostic
  histograms as *relative links* to their static PNGs (not embedded -- this is
  where the per-column diagnostics the HTML skips are still reachable). For
  MLOps and auditors.

> Both Isolation Forest and the VAE are documented side by side. The pipeline
> runs and reports **every** model rather than nominating a single "winner", so
> the technical doc keeps both parameter sets for traceability.

### Chart & KPI conventions

Applied uniformly so the figures read as one system (see
`src/reporting/report_content.py`):

| Element | Spec |
| --- | --- |
| Column / bar | 4px rounded data-end (`marker.cornerradius`), `bargap=0.28` leaving air in every band |
| Line | 2px |
| Scatter mark | 7px, semi-transparent — density is the message on the agreement plot |
| Gridlines | hairline, one step off surface, recessive |
| Legend | present for ≥2 series; **omitted for a single series** (the title already names it) |
| Series colour | fixed per model, never cycled — iForest blue, VAE orange; CVD-validated |
| Status colour | never alone — every badge ships a visible text chip (`Bueno`/`Alerta`/`Crítico`) |
| Hero figure | exactly one per view, ≥48px, proportional figures |
| KPI tile | label · value · optional `sub` line carrying the denominator |
| Tables | wrapped in `.table-wrap { overflow-x: auto }` — scrolls on its own on a narrow screen instead of widening the page |

### What the report embeds

- **Metrics & best params** per model (from `context["models"]`).
- **Figures** — including the interpretability figures produced in Section 2 and
  any evaluation/model figures, all under `artifacts/reports/figures/`.
- **The OOT Excel deliverable** — linked and explained: individuals at or
  above the calibrated percentile of the OOT block, ranked by descending
  anomaly score, in ID – PERIOD – SCORE – BAND – VARIABLES format.
- **VAE methodology math** — the ELBO and the closed-form Gaussian KL:

  $$\mathrm{ELBO}(x) = \mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big] - \mathrm{KL}\big(q(z\mid x)\,\|\,p(z)\big)$$

  $$\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,I)\big) = -\tfrac{1}{2}\sum_{j=1}^{d}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$$

---

## 4. Where things land on disk

| Artifact | Location |
| -------- | -------- |
| Interpretability figures | `artifacts/reports/figures/` |
| Report HTML / MD         | `artifacts/reports/` (`<basename>.html`, `.md`) |
| Technical documentation  | `artifacts/reports/model_documentation.md` (fixed name) |

The interpretability functions and `build_report` both respect the project-wide
hard rule that **all figures go under `artifacts/reports/figures/`**.

---

## 5. End-to-end snippet

Fit both detectors, produce the interpretability figures, then build the report.

```python
from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import IsolationForestDetector, VAEDetector
from src.interpretability import (
    shap_summary_iforest,
    path_length_analysis,
    latent_space_plot,
    reconstruction_error_by_feature,
)
from src.reporting import build_report

# 1. Data -> feature matrix (small scale for a quick run).
df, schema = load_or_generate_panel(data_path="artifacts/data/data.csv",
                                    n_individuals=1_000, n_periods=10, seed=42)
X, keys, feature_names = fit_transform_panel(df, schema)

# 2. Fit the detectors.
iforest = IsolationForestDetector().fit(X)
vae = VAEDetector(latent_dim=8, epochs=10).fit(X, checkpoint_dir="artifacts/models/vae")

# 3. Interpretability figures (all saved under artifacts/reports/figures/).
if_importance = shap_summary_iforest(iforest, X, feature_names)
if_paths = path_length_analysis(iforest, X)
vae_latent_png = latent_space_plot(vae, X)                       # -> PNG path
vae_recon = reconstruction_error_by_feature(vae, X, feature_names)

# 4. Assemble the report (HTML + Markdown under artifacts/reports/).
context = {
    "title": "Banking Anomaly Detection — Run Report",
    "generated_at": "2026-07-24",
    "dataset": {"rows": X.shape[0], "features": X.shape[1]},
    "models": {
        "IsolationForest": {"best_params": {}, "metrics": {}},
        "VAE": {"best_params": {}, "metrics": {}},
    },
    "figures": [
        {"title": "IF SHAP summary",
         "path": "artifacts/reports/figures_shap_summary.png"},
        {"title": "IF path length",
         "path": if_paths["figure_path"]},
        {"title": "VAE latent space", "path": vae_latent_png},
        {"title": "VAE recon by feature",
         "path": "artifacts/reports/figures_recon_by_feature.png"},
    ],
    # From the evaluation module's risk-ranked export, when available.
    "oot_excel": "artifacts/reports/oot_p90_iforest.xlsx",
    "notes": "Higher score = more anomalous for both detectors.",
}
out = build_report(context)
print(out["html"], out["md"])   # absolute paths
```

See `docs/models_isolation_forest.md`, `docs/models_vae.md`, and
`docs/evaluation.md` for the detectors, the score convention, and the OOT
deliverable that this report links to.
