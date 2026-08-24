"""Run-report builder: assemble a run's artifacts into HTML, Markdown and a
separate technical documentation file.

The orchestrator fills a plain ``context`` dict (metrics, figure paths, the OOT
Excel path, model configs) and :func:`build_report` renders it into three
self-contained deliverables under ``reports/``:

* **Markdown** (always): GitHub-friendly, LaTeX via ``$$...$$``, figures linked
  with relative paths.
* **HTML** (always): single self-contained file with inline CSS and every
  chart rendered as an interactive Plotly figure (no external CDNs / network,
  no embedded raster images), so it opens offline anywhere. The VAE math is
  shown as preformatted text (no network MathJax dependency).
* **model_documentation.md** (best-effort): the technical companion --
  every model's exact hyperparameters and full threshold-calibration record,
  the VAE's mathematical formulation, the preprocessing pipeline settings,
  and a catalog of every artifact path the run wrote. For MLOps / auditors,
  not the first-pass business reader.

No PDF output: the pipeline deliberately generates none (a past PDF renderer
here was removed by explicit decision -- see `docs/decisiones_de_modelado.md`).

Everything is robust to missing pieces: absent model metrics, missing figure
files, or ``oot_excel=None`` are simply omitted.
"""

from __future__ import annotations

import base64
import html
import os
import textwrap
from datetime import datetime
from typing import Iterable, Optional, Sequence

from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = ["build_report"]

# VAE math shown in the Methodology section (LaTeX for MD; plain text for HTML).
_ELBO_LATEX = (
    r"\mathrm{ELBO}(x) = \mathbb{E}_{q(z\mid x)}\big[\log p(x\mid z)\big] "
    r"- \mathrm{KL}\big(q(z\mid x)\,\|\,p(z)\big)"
)
_KL_LATEX = (
    r"\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,I)\big) "
    r"= -\tfrac{1}{2}\sum_{j=1}^{d}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)"
)
_ELBO_TEXT = (
    "ELBO(x) = E_q(z|x)[ log p(x|z) ] - KL( q(z|x) || p(z) )"
)
_KL_TEXT = (
    "KL( N(mu, sigma^2) || N(0, I) ) "
    "= -1/2 * sum_j ( 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 )"
)

_OOT_EXPLANATION = (
    "El entregable fuera de tiempo (OOT) clasifica a los individuos del último "
    "período del panel (reservado para evaluación) por puntaje de anomalía "
    "descendente y exporta el decil superior en el formato de tabla "
    "ID - SCORE - VARIABLES (el identificador de la entidad, su puntaje de "
    "anomalía y los valores de los features subyacentes que lo produjeron)."
)

# -- Headline metric presentation: display label + higher-is-better badge bins - #
# (good_min, warning_min): value >= good_min -> "good"; >= warning_min -> "warning";
# else "serious". Chosen for anomaly-detection metrics on an imbalanced (~2-5%
# positive rate) panel, where a PR-AUC or lift only a few multiples of the base
# rate is already a meaningful signal -- these are not generic ML thresholds.
_METRIC_META = {
    "roc_auc": ("ROC-AUC", (0.75, 0.60)),
    "pr_auc": ("PR-AUC", (0.30, 0.10)),
    "best_f1": ("Mejor F1", (0.30, 0.15)),
    "best_f2": ("Mejor F2", (0.30, 0.15)),
    "mcc": ("MCC", (0.30, 0.10)),
    "precision_at_10pct": ("Precisión@10%", (0.30, 0.10)),
    "recall_at_10pct": ("Recall@10%", (0.50, 0.25)),
    "lift_at_10pct": ("Lift@10%", (3.0, 1.5)),
    "silhouette": ("Silueta", (0.40, 0.15)),
    "calinski_harabasz": ("Calinski-Harabasz", (50.0, 10.0)),
    "rank_stability": ("Estabilidad de ranking", (0.90, 0.70)),
    "n_flagged": ("Marcados (n)", None),
}
# Headline tiles shown at the top of each model card, in order, per mode.
_HEADLINE_SUPERVISED = ["roc_auc", "pr_auc", "best_f1", "lift_at_10pct"]
_HEADLINE_UNSUPERVISED = ["silhouette", "calinski_harabasz", "rank_stability", "n_flagged"]

# Fixed categorical accent per model -- assigned in order, never cycled, so a
# given model always reads the same color across a run (palette slots 1 & 2).
_MODEL_ACCENT = {"iforest": "var(--series-1)", "vae": "var(--series-2)"}
_MODEL_TITLES = {
    "iforest": ("Isolation Forest",
                "Basado en árboles - puntaje de anomalía a partir de la longitud "
                "del camino de aislamiento"),
    "vae": ("Variational Autoencoder",
            "Generativo profundo - puntaje de anomalía a partir del error de "
            "reconstrucción"),
}


# Nested per-anomaly-type breakdowns. They are dicts-of-dicts, so they get
# their own table rather than being flattened into the key/value metric lists.
_BY_TYPE_KEYS: tuple[str, ...] = ("oot_by_type", "by_type")
_BY_TYPE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("n_positive", "Positivos"),
    ("recall_at_1pct", "Recall@1%"),
    ("recall_at_5pct", "Recall@5%"),
    ("recall_at_10pct", "Recall@10%"),
    ("mean_score_percentile", "Percentil medio"),
)


def _flat_metrics(metrics: Optional[dict]) -> dict:
    """The metrics dict without the nested per-type breakdowns."""
    return {k: v for k, v in (metrics or {}).items() if k not in _BY_TYPE_KEYS}


def _by_type_rows(metrics: Optional[dict]) -> tuple[str, list[tuple[str, ...]]]:
    """``(scope label, rows)`` for the per-anomaly-type recall table.

    Prefers the OOT breakdown -- that is the period the headline deliverable
    reports on -- and falls back to the overall one.
    """
    for key in _BY_TYPE_KEYS:
        block = (metrics or {}).get(key)
        if isinstance(block, dict) and block:
            scope = "fuera de tiempo" if key.startswith("oot") else "todos los períodos"
            names = sorted(k for k in block if k != "__overall__")
            if "__overall__" in block:
                names.append("__overall__")
            rows = [
                tuple(
                    [name.replace("__overall__", "ALL")]
                    + [_fmt_value(block[name].get(col)) for col, _ in _BY_TYPE_COLUMNS]
                )
                for name in names
            ]
            return scope, rows
    return "", []


def _metric_prefix_split(metrics: Optional[dict]) -> dict:
    """Split a flat ``{'oot_roc_auc': .., 'overall_pr_auc': .., 'unsup_x': ..}``
    metrics dict into groups keyed by prefix (``oot``, ``overall``, ``unsup``),
    each holding the metric with its prefix stripped. Unprefixed keys land in
    ``other``. Nested per-type breakdowns are excluded.
    """
    groups: dict = {"oot": {}, "overall": {}, "unsup": {}, "other": {}}
    for k, v in _flat_metrics(metrics).items():
        for prefix in ("oot_", "overall_", "unsup_"):
            if k.startswith(prefix):
                groups[prefix[:-1]][k[len(prefix):]] = v
                break
        else:
            groups["other"][k] = v
    return groups


def _badge_for(metric_base: str, value) -> Optional[str]:
    """Return ``'good' | 'warning' | 'serious'`` for a headline metric value, or
    ``None`` when the metric has no defined bins or the value isn't a usable
    number (missing/NaN metrics are common in the unlabeled/unsupervised path).
    """
    meta = _METRIC_META.get(metric_base)
    if not meta or meta[1] is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    good_min, warn_min = meta[1]
    if v >= good_min:
        return "good"
    if v >= warn_min:
        return "warning"
    return "serious"


def _metric_label(metric_base: str) -> str:
    meta = _METRIC_META.get(metric_base)
    return meta[0] if meta else metric_base.replace("_", " ")


def _oot_excel_items(oot_excel) -> list[tuple[str, str]]:
    """Normalize ``context['oot_excel']`` (a single path, or ``{model: path}``)
    into an ordered ``[(model_name, path), ...]`` list. A bare path yields a
    single item with an empty model name so a caller with only one model still
    renders sensibly.
    """
    if not oot_excel:
        return []
    if isinstance(oot_excel, dict):
        return [(str(k), str(v)) for k, v in oot_excel.items() if v]
    return [("", str(oot_excel))]


# Figure path -> display group, by the parent directory the pipeline writes
# into (see CONTEXT.md's ``reports/figures/<module>/`` convention). Order here
# is the display order; anything unrecognized falls into "Other".
#: Figure grouping, matched against the figure's **filename prefix**.
#:
#: It used to key off the ``reports/figures/<module>/`` parent folder, but that
#: layout no longer exists -- the directory was deliberately flattened (see
#: CONTEXT.md "HARD RULE -- Figures"), so every figure's parent folder became
#: the single word "figures" and *every* group collapsed into one bucket
#: literally titled "Figures", nested under the "## Figures" heading. The flat
#: layout is intentional precisely because each filename already carries its
#: producer as a prefix (``iforest_*``, ``vae_*``, ``embedding_*``), so that
#: prefix is what this matches now. Order is the display order.
_FIGURE_GROUP_ORDER: list[tuple[tuple[str, ...], str]] = [
    (("iforest_", "embedding_iforest"), "Isolation Forest"),
    (("vae_", "embedding_vae"), "Variational Autoencoder"),
    (("roc_pr_",), "Evaluación"),
]
#: Everything not matched above is a raw-feature preprocessing diagnostic
#: (``age_hist.png``, ``income_hist.png``, ...).
_FIGURE_GROUP_FALLBACK = "Diagnósticos de preprocesamiento"


def _group_figures(figures: list[dict]) -> "dict[str, list[dict]]":
    """Bucket figures by filename prefix, in :data:`_FIGURE_GROUP_ORDER` order.

    Groups are emitted only when non-empty, so a run that skipped a model does
    not leave an empty section behind.
    """
    buckets: dict[str, list[dict]] = {}
    for fig in figures:
        base = os.path.basename(fig.get("path", "")).lower()
        label = _FIGURE_GROUP_FALLBACK
        for prefixes, group_label in _FIGURE_GROUP_ORDER:
            if any(base.startswith(p) for p in prefixes):
                label = group_label
                break
        buckets.setdefault(label, []).append(fig)

    ordered: dict[str, list[dict]] = {}
    for _, label in _FIGURE_GROUP_ORDER:
        if label in buckets:
            ordered[label] = buckets.pop(label)
    # The fallback group goes last: it is context, not a modelling result.
    for label, figs in buckets.items():
        ordered[label] = figs
    return ordered


# --------------------------------------------------------------------------- #
# Value formatting                                                            #
# --------------------------------------------------------------------------- #
def _fmt_value(v) -> str:
    """Compact, stable string form of a metric / param value."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v == 0:
            return "0"
        av = abs(v)
        if av < 1e-3 or av >= 1e5:
            return f"{v:.4e}"
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt_value(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt_value(val)}" for k, val in v.items())
    return str(v)


def _kv_rows(d: Optional[dict]) -> list[tuple[str, str]]:
    if not d:
        return []
    return [(str(k), _fmt_value(v)) for k, v in d.items()]


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #
def _md_kv_table(rows: list[tuple[str, str]], headers=("Clave", "Valor")) -> str:
    if not rows:
        return "_none_\n"
    out = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def _md_table(headers: Sequence[str], rows: list[tuple[str, ...]]) -> str:
    """Generic n-column markdown table."""
    if not rows:
        return "_none_\n"
    out = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _build_markdown(context: dict, out_dir: str) -> str:
    title = context.get("title", "Reporte de Detección de Anomalías")
    generated_at = context.get("generated_at", "")
    parts: list[str] = []

    parts.append(f"# {title}\n")
    if generated_at:
        parts.append(f"_Generado: {generated_at}_\n")

    # -- dataset summary ----------------------------------------------------- #
    dataset = context.get("dataset") or {}
    parts.append("## Resumen del dataset\n")
    parts.append(_md_kv_table(_kv_rows(dataset), headers=("Campo", "Valor")))

    # -- OOT deliverable (headline business output, shown early) ------------- #
    oot_paths = _oot_excel_items(context.get("oot_excel"))
    if oot_paths:
        parts.append("## Entregable OOT\n")
        parts.append(f"{_OOT_EXPLANATION}\n")
        for name, path in oot_paths:
            rel = os.path.relpath(path, out_dir).replace(os.sep, "/") \
                if os.path.isabs(path) or os.path.exists(path) else path
            label = f"Exportación Excel ({name})" if name else "Exportación Excel"
            parts.append(f"- {label}: [`{os.path.basename(str(path))}`]({rel})\n")

    # -- per-model sections -------------------------------------------------- #
    models = context.get("models") or {}
    if models:
        parts.append("## Modelos\n")
        for name, spec in models.items():
            if not spec:
                continue
            title, subtitle = _MODEL_TITLES.get(name, (name, ""))
            parts.append(f"### {title}\n")
            if subtitle:
                parts.append(f"_{subtitle}_\n")
            groups = _metric_prefix_split((spec or {}).get("metrics"))
            headline_keys = _HEADLINE_SUPERVISED if groups["oot"] else _HEADLINE_UNSUPERVISED
            headline_group = groups["oot"] if groups["oot"] else groups["unsup"]
            headline_rows = [
                (_metric_label(k), _fmt_value(headline_group.get(k)))
                for k in headline_keys if k in headline_group
            ]
            if headline_rows:
                parts.append("**Métricas principales**\n")
                parts.append(_md_kv_table(headline_rows, headers=("Métrica", "Valor")))
            best_params = (spec or {}).get("best_params")
            parts.append("**Mejores parámetros**\n")
            parts.append(_md_kv_table(_kv_rows(best_params),
                                      headers=("Parámetro", "Valor")))
            metrics = (spec or {}).get("metrics")
            scope, by_type_rows = _by_type_rows(metrics)
            if by_type_rows:
                parts.append(f"**Recall por tipo de anomalía** ({scope})\n")
                parts.append(
                    "El recall se cuenta contra el top-k *global*: el "
                    "presupuesto de alertas es una única cola compartida, por "
                    "lo que lo que importa es cuántas anomalías de cada tipo "
                    "sobreviven la competencia con el resto de las filas.\n"
                )
                parts.append(
                    _md_table(
                        ["Tipo"] + [label for _, label in _BY_TYPE_COLUMNS],
                        by_type_rows,
                    )
                )
            parts.append("**Todas las métricas**\n")
            parts.append(_md_kv_table(_kv_rows(_flat_metrics(metrics)),
                                      headers=("Métrica", "Valor")))

    # -- figures gallery ----------------------------------------------------- #
    figures = context.get("figures") or []
    existing = [f for f in figures if f.get("path") and os.path.isfile(f["path"])]
    if existing:
        parts.append("## Figuras\n")
        for group_name, group_figs in _group_figures(existing).items():
            parts.append(f"### {group_name}\n")
            for fig in group_figs:
                ftitle = fig.get("title", os.path.basename(fig["path"]))
                rel = os.path.relpath(fig["path"], out_dir).replace(os.sep, "/")
                parts.append(f"**{ftitle}**\n")
                parts.append(f"![{ftitle}]({rel})\n")

    # -- methodology --------------------------------------------------------- #
    parts.append("## Metodología\n")
    parts.append(
        "Se utilizan dos detectores complementarios: un **Isolation Forest** "
        "(las anomalías se aíslan con longitudes de camino promedio más "
        "cortas) y un **Variational Autoencoder** (las anomalías presentan un "
        "error de reconstrucción alto). Ambos siguen la convención del "
        "proyecto *mayor puntaje = más anómalo*.\n"
    )
    parts.append(
        "El VAE maximiza la cota inferior de evidencia (ELBO), lo que "
        "equivale a minimizar el error de reconstrucción más un "
        "regularizador KL:\n"
    )
    parts.append(f"$$\n{_ELBO_LATEX}\n$$\n")
    parts.append("con la KL gaussiana disponible en forma cerrada:\n")
    parts.append(f"$$\n{_KL_LATEX}\n$$\n")

    notes = context.get("notes")
    if notes:
        parts.append("## Notas\n")
        parts.append(f"{notes}\n")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# HTML                                                                        #
# --------------------------------------------------------------------------- #
_HTML_CSS = """
:root {
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --surface-page:   #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --border:         rgba(11,11,11,0.10);
  --series-1:       #2a78d6;  /* iForest accent */
  --series-2:       #eb6834;  /* VAE accent */
  --good:           #0ca30c;
  --warning:        #a66a00;  /* darkened for AA text-on-chip contrast */
  --serious:        #d03b3b;
  --good-bg:        #e6f6e6;
  --warning-bg:     #fdf1da;
  --serious-bg:     #fbe7e7;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --surface-page:   #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --series-2:       #d95926;
    --good:           #0ca30c;
    --warning:        #eac054;
    --serious:        #e66767;
    --good-bg:        #113311;
    --warning-bg:     #362a11;
    --serious-bg:     #3a1717;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --surface-page:   #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted:     #898781;
  --gridline:       #2c2c2a;
  --baseline:       #383835;
  --border:         rgba(255,255,255,0.10);
  --series-1:       #3987e5;
  --series-2:       #d95926;
  --good:           #0ca30c;
  --warning:        #eac054;
  --serious:        #e66767;
  --good-bg:        #113311;
  --warning-bg:     #362a11;
  --serious-bg:     #3a1717;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0; color: var(--text-primary); line-height: 1.55;
  background: var(--surface-page);
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 1.25rem 3rem; }
a { color: var(--series-1); }

/* -- sticky header / nav -------------------------------------------------- */
header.topbar {
  position: sticky; top: 0; z-index: 10; background: var(--surface-1);
  border-bottom: 1px solid var(--border); backdrop-filter: blur(6px);
}
.topbar-inner {
  max-width: 1080px; margin: 0 auto; padding: .8rem 1.25rem;
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
}
.topbar-title { font-weight: 600; font-size: 1.02rem; margin-right: auto; }
.topbar-title .ts { display: block; font-weight: 400; font-size: .78rem;
  color: var(--text-muted); }
nav.jump { display: flex; gap: 1rem; flex-wrap: wrap; }
nav.jump a { color: var(--text-secondary); text-decoration: none; font-size: .86rem;
  font-weight: 500; }
nav.jump a:hover { color: var(--series-1); }
#theme-toggle {
  border: 1px solid var(--border); background: transparent; color: var(--text-secondary);
  border-radius: 6px; padding: .3rem .6rem; font-size: .8rem; cursor: pointer;
}
#theme-toggle:hover { color: var(--text-primary); border-color: var(--series-1); }

[id] { scroll-margin-top: 4.5rem; }
h1 { font-size: 1.5rem; margin: 1.8rem 0 .3rem; }
h2 { font-size: 1.15rem; margin: 2.4rem 0 .8rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--gridline); }
h3 { font-size: 1rem; color: var(--text-secondary); margin: 1.2rem 0 .5rem; }
p { color: var(--text-secondary); }
.timestamp { color: var(--text-muted); font-style: italic; font-size: .85rem; }

/* -- cards ----------------------------------------------------------------- */
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.1rem 1.3rem; margin: 0 0 1.2rem;
}
.card.accent { border-left: 4px solid var(--series-1); }

/* -- hero figure ------------------------------------------------------------*/
/* Exactly one per view: the single number the report leads with. Same sans as
   the rest (never a display face) and proportional figures -- `tabular-nums`
   gives every digit a `0`'s width, which reads loose at 48px+. */
.hero {
  display: flex; align-items: baseline; gap: 1.15rem; flex-wrap: wrap;
  padding: 1.2rem 1.35rem; margin: 1rem 0 1.4rem;
  background: var(--surface-1); border: 1px solid var(--border);
  border-left: 4px solid var(--series-1); border-radius: 12px;
}
.hero .figure {
  font-size: 3rem; line-height: 1; font-weight: 650; color: var(--text-primary);
  font-variant-numeric: proportional-nums; letter-spacing: -.01em;
}
.hero .hero-text { display: flex; flex-direction: column; gap: .2rem; }
.hero .hero-label { font-size: 1rem; font-weight: 600; color: var(--text-primary); }
.hero .hero-sub { font-size: .84rem; color: var(--text-secondary); max-width: 60ch;
  line-height: 1.5; }

/* -- stat tiles -------------------------------------------------------------*/
.tile-group-label {
  font-size: .7rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-muted); font-weight: 650; margin: 1.35rem 0 .45rem;
}
.tile-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: .75rem; margin: 0 0 1.1rem; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: .85rem 1rem;
}
.tile .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .03em;
  color: var(--text-muted); margin-bottom: .3rem; }
.tile .value { font-size: 1.5rem; font-weight: 600; color: var(--text-primary);
  font-variant-numeric: proportional-nums; }
/* Secondary context under a value ("de 6,000 filas"): the denominator a bare
   count needs to mean anything, kept in muted ink so it never competes. */
.tile .sub { font-size: .75rem; color: var(--text-muted); margin-top: .25rem;
  line-height: 1.4; }
.tile.badge-good .value { color: var(--good); }
.tile.badge-warning .value { color: var(--warning); }
.tile.badge-serious .value { color: var(--serious); }

/* -- status chip ------------------------------------------------------------*/
.chip {
  display: inline-block; font-size: .72rem; font-weight: 600; padding: .12rem .5rem;
  border-radius: 999px; margin-left: .4rem; vertical-align: middle;
}
.chip.good { color: var(--good); background: var(--good-bg); }
.chip.warning { color: var(--warning); background: var(--warning-bg); }
.chip.serious { color: var(--serious); background: var(--serious-bg); }

/* -- model cards ------------------------------------------------------------*/
.model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1rem; }
.model-card { border-top: 4px solid var(--accent, var(--series-1)); }
.model-card h3.model-name { color: var(--text-primary); font-size: 1.05rem; margin: 0 0 .1rem; }
.model-card .subtitle { font-size: .82rem; color: var(--text-muted); margin: 0 0 .8rem; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 0 1.2rem; }
@media (max-width: 640px) { .two-col { grid-template-columns: 1fr; } }
details.metrics-more summary {
  cursor: pointer; color: var(--series-1); font-size: .85rem; font-weight: 500;
  margin-top: .4rem;
}

/* -- tables -----------------------------------------------------------------*/
table { border-collapse: collapse; margin: .5rem 0 1rem; width: 100%; font-size: .87rem; }
th, td { border: 1px solid var(--gridline); padding: .32rem .55rem; text-align: left; }
th { background: var(--surface-page); color: var(--text-secondary); font-weight: 600;
  font-size: .78rem; text-transform: uppercase; letter-spacing: .02em; }
tr:nth-child(even) td { background: color-mix(in srgb, var(--surface-page) 55%, transparent); }
td:nth-child(2) { font-variant-numeric: tabular-nums; }

/* -- OOT deliverable callout --------------------------------------------------*/
.oot-list { list-style: none; padding: 0; margin: .6rem 0 0; }
.oot-list li {
  display: flex; align-items: center; gap: .6rem; padding: .5rem .7rem;
  border: 1px solid var(--border); border-radius: 8px; margin-bottom: .5rem;
  background: var(--surface-page);
}
.oot-list .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.oot-list a { font-weight: 600; text-decoration: none; }
.oot-list a:hover { text-decoration: underline; }
.oot-list .fname { color: var(--text-muted); font-size: .82rem; font-weight: 400; }

/* -- figure gallery ----------------------------------------------------------*/
details.fig-group { margin: 0 0 .8rem; border: 1px solid var(--border); border-radius: 10px;
  background: var(--surface-1); overflow: hidden; }
details.fig-group summary {
  cursor: pointer; padding: .7rem 1rem; font-weight: 600; font-size: .92rem;
  list-style: none; display: flex; align-items: center; justify-content: space-between;
}
details.fig-group summary::-webkit-details-marker { display: none; }
details.fig-group summary .count { color: var(--text-muted); font-weight: 400; font-size: .8rem; }
details.fig-group summary::after { content: '+'; color: var(--text-muted); font-size: 1.1rem; }
details.fig-group[open] summary::after { content: '\\2212'; }
.fig-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem; padding: 0 1rem 1rem; }
figure.fig-item { margin: 0; }
figure.fig-item figcaption { font-size: .82rem; color: var(--text-secondary);
  margin-bottom: .35rem; }
img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px;
  display: block; }

/* -- methodology / notes -----------------------------------------------------*/
pre {
  background: var(--surface-page); border: 1px solid var(--border); border-radius: 8px;
  padding: .8rem 1rem; overflow-x: auto; font-size: .84rem; line-height: 1.6;
}
code { background: var(--surface-page); padding: .1rem .35rem; border-radius: 4px;
  font-size: .88em; }
.note { background: var(--warning-bg); border-left: 4px solid var(--warning);
  padding: .7rem 1rem; border-radius: 0 8px 8px 0; color: var(--text-primary); }
footer.report-footer { color: var(--text-muted); font-size: .8rem; text-align: center;
  margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--gridline); }

/* -- interactive charts ------------------------------------------------------*/
/* `min-width:0` is load-bearing: a grid/flex child defaults to min-content
   width, which lets a wide Plotly SVG push the page into horizontal scroll
   instead of shrinking. */
figure.chart { margin: 0 0 1.75rem; min-width: 0; }
figure.chart:last-child { margin-bottom: 0; }
/* Two-up grid for the small diagnostic histograms. `minmax(0, 1fr)` rather
   than `1fr`: a grid child defaults to min-content width, which lets a wide
   Plotly SVG push the row into horizontal scroll instead of shrinking. */
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem 1.25rem; padding: 0 1rem 1rem; }
.chart-grid > figure.chart { margin: 0; min-width: 0; }
@media (max-width: 700px) { .chart-grid { grid-template-columns: minmax(0, 1fr); } }
.chart-note {
  color: var(--text-secondary); font-size: .85rem; line-height: 1.6;
  margin: 0 0 .5rem; max-width: 78ch;
}
figure.chart .plotly-graph-div { width: 100% !important; }

/* -- lead paragraph + reliability checks -------------------------------------*/
.lead { color: var(--text-secondary); font-size: .9rem; line-height: 1.65;
  max-width: 78ch; margin: 0 0 1.25rem; }
.check {
  display: grid; grid-template-columns: minmax(160px, 220px) 1fr; gap: 1rem;
  padding: .9rem 0; border-top: 1px solid var(--gridline);
}
.check:first-of-type { border-top: none; padding-top: .25rem; }
.check-name { font-weight: 650; font-size: .9rem; color: var(--text-primary); }
.check-body p { margin: 0 0 .5rem; font-size: .86rem; line-height: 1.6;
  color: var(--text-secondary); max-width: 78ch; }
.check-body p:last-child { margin-bottom: 0; }
.check-body .lbl {
  display: block; font-size: .7rem; text-transform: uppercase;
  letter-spacing: .05em; color: var(--text-muted); font-weight: 650;
  margin-bottom: .1rem;
}
@media (max-width: 720px) { .check { grid-template-columns: 1fr; gap: .35rem; } }

/* -- approach / verdict ------------------------------------------------------*/
.verdict {
  font-size: 1rem; font-weight: 650; color: var(--text-primary);
  padding: .6rem .9rem; border-left: 4px solid var(--series-1);
  background: var(--surface-page); border-radius: 0 8px 8px 0; margin-bottom: 1rem;
}
.card p { font-size: .88rem; line-height: 1.7; color: var(--text-secondary);
  max-width: 78ch; }
.callout-inline {
  margin-top: 1.25rem; padding: .9rem 1rem; border: 1px solid var(--border);
  border-radius: 10px; background: var(--surface-page);
}
.callout-title { font-weight: 650; font-size: .85rem; color: var(--text-primary);
  margin-bottom: .35rem; }
.callout-inline p { margin: 0; }
/* Latent-health gate: a coloured left edge carries severity at a glance, but
   the visible status chip in the title is what actually communicates it --
   colour is never the only channel. */
.latent-health { border-left: 3px solid var(--border); }
.latent-health.good { border-left-color: var(--good); }
.latent-health.warning { border-left-color: var(--warning); }
.latent-health.serious { border-left-color: var(--serious); }
.latent-health p + p { margin-top: .5rem; }
.latent-health .subtitle { font-size: .8rem; color: var(--text-muted); }

/* -- glossary + parameter tables ---------------------------------------------*/
.table-wrap { overflow-x: auto; }
table.glossary, table.param-table {
  width: 100%; border-collapse: collapse; font-size: .84rem; margin-bottom: 1.25rem;
}
table.glossary th, table.param-table th {
  text-align: left; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--text-muted); font-weight: 650;
  padding: .45rem .6rem; border-bottom: 1px solid var(--baseline); white-space: nowrap;
}
table.glossary td, table.param-table td {
  padding: .55rem .6rem; border-bottom: 1px solid var(--gridline);
  vertical-align: top; line-height: 1.55; color: var(--text-secondary);
}
table.glossary .m-name, table.param-table .p-name {
  font-weight: 650; color: var(--text-primary); white-space: nowrap;
}
table.glossary .m-read, table.param-table .p-why { color: var(--text-secondary); }
table.param-table .p-val { white-space: nowrap; }
.tag {
  display: inline-block; margin-left: .4rem; padding: .05rem .45rem;
  border-radius: 9px; font-size: .65rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: .03em; vertical-align: middle;
}
.tag.tuned { background: var(--good-bg); color: var(--good); }
.tag.default { background: var(--surface-page); color: var(--text-muted);
  border: 1px solid var(--border); }

/* -- figure notes ------------------------------------------------------------*/
.fig-note {
  color: var(--text-secondary); font-size: .8rem; line-height: 1.55;
  margin: .15rem 0 .6rem; max-width: 70ch;
}
"""

_THEME_TOGGLE_JS = """
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('report-theme'); } catch (e) {}
  if (saved) { root.setAttribute('data-theme', saved); }
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  function current() {
    return root.getAttribute('data-theme')
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  btn.textContent = current() === 'dark' ? 'Modo claro' : 'Modo oscuro';
  btn.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    btn.textContent = next === 'dark' ? 'Modo claro' : 'Modo oscuro';
    try { localStorage.setItem('report-theme', next); } catch (e) {}
    /* Plotly cannot read CSS custom properties, so its figures are recoloured
       explicitly on every theme change (no-op when no figures are present). */
    if (typeof applyPlotlyTheme === 'function') { applyPlotlyTheme(); }
  });
  if (typeof applyPlotlyTheme === 'function') { applyPlotlyTheme(); }
})();
"""


def _html_kv_table(rows: list[tuple[str, str]], headers=("Clave", "Valor")) -> str:
    if not rows:
        return "<p><em>ninguno</em></p>"
    out = ["<div class='table-wrap'><table>", f"<tr><th>{html.escape(headers[0])}</th>"
           f"<th>{html.escape(headers[1])}</th></tr>"]
    for k, v in rows:
        out.append(f"<tr><td>{html.escape(str(k))}</td>"
                   f"<td>{html.escape(str(v))}</td></tr>")
    out.append("</table></div>")
    return "\n".join(out)


def _html_table(headers: Sequence[str], rows: list[tuple[str, ...]]) -> str:
    """Generic n-column HTML table, wrapped so it scrolls instead of
    widening the page when it has more columns than a narrow viewport fits
    (e.g. the 6-column recall-by-type table inside a model card)."""
    if not rows:
        return "<p><em>ninguno</em></p>"
    out = ["<div class='table-wrap'><table>", "<tr>" + "".join(
        f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>"]
    for row in rows:
        out.append("<tr>" + "".join(
            f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def _img_data_uri(path: str, log) -> Optional[str]:
    """Base64 data URI for a PNG so the HTML stays portable/offline."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/{mime};base64,{b64}"
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not embed figure %s (%s); skipping.", path, exc)
        return None


#: CSS class (fixed, matches `_HTML_CSS`) -> visible Spanish label. The class
#: name itself must stay in English/lowercase (it is also the badge value
#: `_badge_for` returns and the `tile-badge-*` class it is composed into).
_BADGE_LABELS = {"good": "Bueno", "warning": "Alerta", "serious": "Crítico"}


def _chip_html(badge: Optional[str]) -> str:
    if not badge:
        return ""
    return f"<span class='chip {badge}'>{_BADGE_LABELS.get(badge, badge)}</span>"


def _stat_tile_html(
    label: str, value: str, badge: Optional[str] = None, sub: str = "",
) -> str:
    """One KPI tile: uppercase label, the value, and optional context beneath.

    ``sub`` is where a bare count gets its denominator ("de 6,000 filas"). A
    number without one is the most common way a KPI misleads, so the slot is
    part of the tile rather than something a caller has to fold into the label.
    """
    cls = f" badge-{badge}" if badge else ""
    # A status color never carries meaning alone: the badge always ships as a
    # visible text chip beside the value, not just a color change on the number.
    chip = _chip_html(badge)
    sub_html = f"<div class='sub'>{html.escape(sub)}</div>" if sub else ""
    return (
        f"<div class='tile{cls}'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}{chip}</div>{sub_html}</div>"
    )


def _hero_html(dataset: dict) -> str:
    """The one number the report leads with: how many anomalies were flagged.

    Returns ``''`` when the run reported no anomaly count -- a hero figure with
    nothing to say is worse than none, and there must never be two.
    """
    n_anom = dataset.get("n_anomalies")
    rate = dataset.get("anomaly_rate")
    if n_anom is None:
        return ""
    oot = str(dataset.get("oot_period") or "").split("T")[0]
    bits = []
    if rate is not None:
        bits.append(f"una tasa del {float(rate):.2%} del panel evaluado")
    if oot:
        bits.append(f"período fuera de tiempo {oot}")
    sub = " · ".join(bits).capitalize() if bits else ""
    return (
        "<div class='hero'>"
        f"<div class='figure'>{int(n_anom):,}</div>"
        "<div class='hero-text'>"
        "<div class='hero-label'>anomalías marcadas para revisión</div>"
        + (f"<div class='hero-sub'>{html.escape(sub)}</div>" if sub else "")
        + "</div></div>"
    )


def _dataset_tiles_html(dataset: dict) -> str:
    """Headline KPI tiles for the dataset: volume, imbalance, and the OOT split."""
    if not dataset:
        return ""
    rows_total = dataset.get("rows")

    def _pct_of_rows(n) -> str:
        """``'34% del panel'`` -- the denominator a raw row count needs."""
        if n is None or not rows_total:
            return ""
        return f"{int(n) / float(rows_total):.0%} del panel"

    # -- Group 1: how much data ------------------------------------------- #
    volume = []
    if "rows" in dataset:
        ent = dataset.get("entities")
        per = dataset.get("periods")
        sub = f"{int(ent):,} entidades x {int(per):,} períodos" if ent and per else ""
        volume.append(_stat_tile_html("Filas", f"{int(dataset['rows']):,}", sub=sub))
    if "entities" in dataset:
        volume.append(_stat_tile_html("Entidades", f"{int(dataset['entities']):,}",
                                      sub="individuos únicos del panel"))
    if "periods" in dataset:
        volume.append(_stat_tile_html("Períodos", f"{int(dataset['periods']):,}",
                                      sub="meses observados"))
    if "features" in dataset:
        volume.append(_stat_tile_html("Features", f"{int(dataset['features']):,}",
                                      sub="tras el preprocesamiento"))

    # -- Group 2: how the panel was split --------------------------------- #
    split = []
    if "evaluation_mode" in dataset:
        _mode_es = {"supervised": "Supervisado", "unsupervised": "No supervisado"}
        mode_raw = str(dataset["evaluation_mode"])
        split.append(_stat_tile_html(
            "Modo de evaluación", _mode_es.get(mode_raw, mode_raw.title()),
            sub=("sin etiquetas en el entrenamiento"
                 if mode_raw == "unsupervised" else "con verdad base reservada"),
        ))
    for key, label, sub in (
        ("train_rows", "Entrenamiento", "ajusta preprocesamiento y modelos"),
        ("val_rows", "Validación", "selecciona umbral e hiperparámetros"),
        ("test_rows", "Prueba (test)", "métricas reportadas, nunca se ajusta"),
        ("oot_rows", "Fuera de tiempo (OOT)", "bloque exclusivo del Excel final"),
    ):
        if dataset.get(key) is not None:
            pct = _pct_of_rows(dataset[key])
            split.append(_stat_tile_html(
                label, f"{int(dataset[key]):,}",
                sub=f"{pct} · {sub}" if pct else sub,
            ))
    if dataset.get("oot_period"):
        oot_str = str(dataset["oot_period"]).split("T")[0]
        split.append(_stat_tile_html("Período OOT", oot_str,
                                     sub="el bloque reservado que se reporta"))

    parts = []
    if volume:
        parts.append("<div class='tile-group-label'>Volumen de datos</div>")
        parts.append(f"<div class='tile-row'>{''.join(volume)}</div>")
    if split:
        parts.append("<div class='tile-group-label'>División cronológica</div>")
        parts.append(f"<div class='tile-row'>{''.join(split)}</div>")
    return "".join(parts)


def _oot_callout_html(oot_excel, out_dir: str) -> str:
    items = _oot_excel_items(oot_excel)
    if not items:
        return ""
    accents = [_MODEL_ACCENT.get(name, "var(--series-1)") for name, _ in items]
    rows = []
    for (name, path), accent in zip(items, accents):
        rel = os.path.relpath(path, out_dir).replace(os.sep, "/") \
            if os.path.isabs(path) or os.path.exists(path) else path
        label = _MODEL_TITLES.get(name, (name or "Modelo",))[0]
        rows.append(
            f"<li><span class='dot' style='background:{accent}'></span>"
            f"<a href='{html.escape(rel)}'>{html.escape(label)}</a>"
            f"<span class='fname'>{html.escape(os.path.basename(str(path)))}</span></li>"
        )
    return (
        "<div class='card accent' id='oot'>"
        "<h2 style='margin-top:0;border:none;padding-bottom:0'>Entregable OOT"
        "</h2>"
        f"<p>{html.escape(_OOT_EXPLANATION)}</p>"
        f"<ul class='oot-list'>{''.join(rows)}</ul>"
        "</div>"
    )


def _model_headline_tiles_html(metrics: Optional[dict]) -> str:
    groups = _metric_prefix_split(metrics)
    headline_keys = _HEADLINE_SUPERVISED if groups["oot"] else _HEADLINE_UNSUPERVISED
    source = groups["oot"] if groups["oot"] else groups["unsup"]
    tiles = []
    for key in headline_keys:
        if key not in source:
            continue
        val = source[key]
        badge = _badge_for(key, val)
        tiles.append(_stat_tile_html(_metric_label(key), _fmt_value(val), badge))
    if not tiles:
        return ""
    return f"<div class='tile-row'>{''.join(tiles)}</div>"


def _latent_health_html(health: Optional[dict]) -> str:
    """Posterior-collapse readout for a VAE card; ``''`` for other models.

    Rendered as a status callout rather than another row in the metrics table
    because it is a *gate*, not a measurement to compare: a collapsed latent
    space invalidates the model's whole score column, so it has to be visible
    without expanding anything.
    """
    if not isinstance(health, dict) or not health.get("latent_dim"):
        return ""
    severity = str(health.get("severity", "ok"))
    active = int(health.get("active_units", 0))
    total = int(health.get("latent_dim", 0))
    chip = {"ok": "good", "warning": "warning", "critical": "serious"}.get(severity, "good")
    label = {"ok": "Saludable", "warning": "Revisar",
             "critical": "Colapso"}.get(severity, severity)
    return (
        f"<div class='callout-inline latent-health {chip}'>"
        f"<div class='callout-title'>Espacio latente "
        f"<span class='chip {chip}'>{html.escape(label)}</span></div>"
        f"<p><strong>{active} de {total}</strong> dimensiones latentes activas "
        f"(A<sub>j</sub> &gt; {html.escape(str(health.get('delta')))}; "
        f"KL media {float(health.get('mean_kl', float('nan'))):.4f}). "
        f"{html.escape(str(health.get('reason', '')))}</p>"
        "<p class='subtitle'>Criterio de unidades activas de Burda, Grosse y "
        "Salakhutdinov (IWAE, ICLR 2016). Importa porque el puntaje de anomalía "
        "<em>es</em> el error de reconstrucción: si el decodificador deja de "
        "usar el código latente, el puntaje sigue siendo finito pero deja de "
        "discriminar.</p>"
        "</div>"
    )


def _model_card_html(name: str, spec: dict) -> str:
    title, subtitle = _MODEL_TITLES.get(name, (name, ""))
    accent = _MODEL_ACCENT.get(name, "var(--series-1)")
    metrics = (spec or {}).get("metrics")
    best_params = (spec or {}).get("best_params")

    out = [f"<div class='card model-card' style='--accent:{accent}'>"]
    out.append(f"<h3 class='model-name'>{html.escape(title)}</h3>")
    if subtitle:
        out.append(f"<p class='subtitle'>{html.escape(subtitle)}</p>")
    out.append(_model_headline_tiles_html(metrics))

    scope, by_type_rows = _by_type_rows(metrics)
    if by_type_rows:
        out.append(
            f"<p><strong>Recall por tipo de anomalía</strong> "
            f"<span class='subtitle'>({html.escape(scope)})</span></p>"
        )
        out.append(
            "<p class='subtitle'>Contado contra el top-k global -- el "
            "presupuesto de alertas es una única cola compartida, por lo que "
            "lo que importa es cuántas anomalías de cada tipo sobreviven la "
            "competencia con el resto de las filas.</p>"
        )
        out.append(
            _html_table(
                ["Tipo"] + [label for _, label in _BY_TYPE_COLUMNS], by_type_rows
            )
        )

    out.append(_latent_health_html((spec or {}).get("latent_health")))

    out.append("<div class='two-col'>")
    out.append(
        "<div><p><strong>Mejores parámetros</strong></p>"
        + _html_kv_table(_kv_rows(best_params), headers=("Parámetro", "Valor"))
        + "</div>"
    )
    out.append(
        "<div><details class='metrics-more'><summary>Todas las métricas</summary>"
        + _html_kv_table(_kv_rows(_flat_metrics(metrics)), headers=("Métrica", "Valor"))
        + "</details></div>"
    )
    out.append("</div></div>")
    return "\n".join(out)


#: Figures superseded by an interactive Plotly chart, or by another static
#: figure that shows the same thing. Matched as lowercase substrings against
#: the figure *title*. Dropping these is the "remove charts that do not earn
#: their place" pass:
#:   * score/recon distributions and ROC/PR are now interactive, with the
#:     alert threshold drawn on them -- the PNGs are strictly less informative;
#:   * "vae latent space" was emitted TWICE by two different modules
#:     (`models.vae.plot_latent_space` and
#:     `interpretability.vae_explain.latent_space_plot`), so the duplicate
#:     without the interpretability suffix is dropped;
#:   * PCA embeddings sit beside a UMAP embedding of the same matrix; UMAP is
#:     kept because it preserves local neighbourhood structure, which is what
#:     the chart is being read for.
_SUPERSEDED_FIGURE_TITLES = (
    "score distribution",
    "reconstruction-error distribution",
    "roc/pr",
    "pca embedding",
)
#: Groups demoted below the fold: real diagnostics, but not modelling results.
#: Ten raw-feature histograms answer "is the input sane", already asserted by
#: the reliability section, so they start collapsed instead of leading.
_SECONDARY_FIGURE_GROUPS = (_FIGURE_GROUP_FALLBACK,)


def _is_superseded(title: str) -> bool:
    low = str(title).lower()
    if any(s in low for s in _SUPERSEDED_FIGURE_TITLES):
        return True
    # The duplicated latent-space plot: keep only the interpretability one.
    return "latent space" in low and "interpretability" not in low


def _figures_section_html(figures: list[dict], log) -> str:
    from src.reporting.report_content import figure_note

    existing = [f for f in figures if f.get("path") and os.path.isfile(f["path"])]
    kept = [f for f in existing if not _is_superseded(f.get("title", ""))]
    n_dropped = len(existing) - len(kept)
    if log and n_dropped:
        log.info(
            "Report: %d static figure(s) omitted as superseded by an interactive "
            "chart or duplicated elsewhere.", n_dropped,
        )
    if not kept:
        return ""
    grouped = _group_figures(kept)
    parts = [
        "<h2 id='figures'>Figuras de diagnóstico</h2>",
        "<div class='note'>Detalle de respaldo para los resultados anteriores. "
        "Los gráficos que ahora son interactivos (distribuciones de puntaje, "
        "ROC/PR) se omiten aquí en lugar de duplicarse, al igual que las "
        "proyecciones redundantes de la misma matriz.</div>",
    ]
    first = True
    for group_name, group_figs in grouped.items():
        items = []
        for fig in group_figs:
            ftitle = fig.get("title", os.path.basename(fig["path"]))
            uri = _img_data_uri(fig["path"], log)
            if uri is None:
                continue
            note = figure_note(str(ftitle))
            note_html = (
                f"<p class='fig-note'>{html.escape(note)}</p>" if note else ""
            )
            items.append(
                "<figure class='fig-item'>"
                f"<figcaption>{html.escape(str(ftitle))}</figcaption>"
                f"{note_html}"
                f"<img alt='{html.escape(str(ftitle))}' src='{uri}' loading='lazy'>"
                "</figure>"
            )
        if not items:
            continue
        secondary = group_name in _SECONDARY_FIGURE_GROUPS
        open_attr = " open" if (first and not secondary) else ""
        if not secondary:
            first = False
        parts.append(
            f"<details class='fig-group'{open_attr}><summary>{html.escape(group_name)}"
            f"<span class='count'>{len(items)} figura(s)</span></summary>"
            f"<div class='fig-grid'>{''.join(items)}</div></details>"
        )
    return "\n".join(parts)


def _statistical_checks_html() -> str:
    """The 'why these results are trustworthy' section.

    Each row is a real gate in the pipeline (`src/utils/assumptions.py`,
    `src/evaluation/splits.py`, `src/evaluation/thresholds.py`), not a generic
    checklist -- see `report_content.STATISTICAL_CHECKS` for the source text.
    """
    from src.reporting.report_content import STATISTICAL_CHECKS

    rows = []
    for name, tests, why, failure in STATISTICAL_CHECKS:
        rows.append(
            "<div class='check'>"
            f"<div class='check-name'>{html.escape(name)}</div>"
            f"<div class='check-body'><p><span class='lbl'>Qué verifica</span>"
            f"{html.escape(tests)}</p>"
            f"<p><span class='lbl'>Por qué importa</span>{html.escape(why)}</p>"
            f"<p><span class='lbl'>Si fallara</span>{html.escape(failure)}</p></div>"
            "</div>"
        )
    return (
        "<h2 id='reliability'>Por qué estos resultados son estadísticamente confiables</h2>"
        "<div class='card'><p class='lead'>Es fácil hacer detección de anomalías de una "
        "forma que produce números que lucen impresionantes pero que no pueden "
        "reproducirse con datos en vivo. Las verificaciones a continuación existen "
        "específicamente para evitarlo, y todas se ejecutan en cada corrida -- las "
        "bloqueantes detienen el pipeline por completo en lugar de dejar que una "
        "corrida comprometida llegue a este reporte.</p>"
        + "".join(rows) + "</div>"
    )


def _ml_vs_econometrics_html() -> str:
    """Positioning section: which discipline this pipeline belongs to, and why."""
    from src.reporting.report_content import ML_VS_ECONOMETRICS as MVE

    paras = "".join(f"<p>{html.escape(p)}</p>" for p in MVE["paragraphs"])
    return (
        "<h2 id='approach'>¿Machine learning o econometría?</h2>"
        "<div class='card'>"
        f"<div class='verdict'>{html.escape(MVE['verdict'])}</div>"
        f"{paras}"
        "<div class='callout-inline'><div class='callout-title'>Qué determina "
        "realmente la clasificación</div>"
        f"<p>{MVE['classification_driver']}</p></div>"
        "</div>"
    )


def _param_glossary_html(models: dict) -> str:
    """Per-model parameter table: value used, and what that value implies.

    The *value used* column prefers the run's own tuned/selected parameters
    (`spec['best_params']`) and falls back to the documented default, so the
    table always says what this run actually did rather than only what the
    defaults are.
    """
    from src.reporting.report_content import PARAM_GLOSSARY

    blocks = []
    for name in models:
        glossary = PARAM_GLOSSARY.get(name)
        if not glossary:
            continue
        used = (models.get(name) or {}).get("best_params") or {}
        rows = []
        for param, default, controls, implies in glossary:
            keys = [k.strip() for k in param.split("/")]
            actual = next((used[k] for k in keys if k in used), None)
            if actual is not None:
                val = f"<code>{html.escape(_fmt_value(actual))}</code>" \
                      "<span class='tag tuned'>esta corrida</span>"
            else:
                val = f"<code>{html.escape(default)}</code>" \
                      "<span class='tag default'>por defecto</span>"
            rows.append(
                f"<tr><td class='p-name'><code>{html.escape(param)}</code></td>"
                f"<td class='p-val'>{val}</td>"
                f"<td>{html.escape(controls)}</td>"
                f"<td class='p-why'>{html.escape(implies)}</td></tr>"
            )
        title = _MODEL_TITLES.get(name, (name, ""))[0]
        blocks.append(
            f"<h3>{html.escape(title)}</h3>"
            "<div class='table-wrap'><table class='param-table'>"
            "<thead><tr><th>Parámetro</th><th>Valor usado</th><th>Qué controla</th>"
            "<th>Qué implica este valor</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"
        )
    if not blocks:
        return ""
    return (
        "<h2 id='parameters'>Parámetros del modelo, y qué significa cada valor</h2>"
        "<div class='card'><p class='lead'>Cada parámetro que da forma a un detector "
        "se muestra con el valor que esta corrida realmente usó. "
        "<span class='tag tuned'>esta corrida</span> marca un valor elegido por el "
        "tuner para esta ejecución; <span class='tag default'>por defecto</span> "
        "marca el valor por defecto del proyecto, que está documentado y no es "
        "arbitrario -- la justificación está en la última columna.</p>"
        + "".join(blocks) + "</div>"
    )


def _metric_glossary_html(models: dict) -> str:
    """Plain-language reading guide, restricted to indicators this run produced.

    Filtered rather than exhaustive on purpose: the default strategy is
    unsupervised, and explaining ROC-AUC / precision / recall in a report that
    (correctly) never computed them invites the reader to look for numbers
    that are not there, or worse to assume they were measured.
    """
    from src.reporting.report_content import METRIC_GLOSSARY

    produced: set[str] = set()
    for spec in (models or {}).values():
        for key in _flat_metrics((spec or {}).get("metrics")):
            for prefix in ("oot_", "overall_", "unsup_"):
                if key.startswith(prefix):
                    produced.add(key[len(prefix):])
                    break
            else:
                produced.add(key)
    rows = []
    for key, (what, how) in METRIC_GLOSSARY.items():
        if key not in produced:
            continue
        rows.append(
            f"<tr><td class='m-name'>{html.escape(_metric_label(key))}</td>"
            f"<td>{html.escape(what)}</td>"
            f"<td class='m-read'>{html.escape(how)}</td></tr>"
        )
    if not rows:
        return ""
    return (
        "<h2 id='indicators'>Cómo leer cada indicador</h2>"
        "<div class='card'><p class='lead'>Solo se listan los indicadores que esta "
        "corrida realmente produjo. Las métricas de detección de anomalías no se "
        "leen como las métricas de clasificación habituales, porque la clase "
        "positiva es rara -- la tercera columna es la que importa, ya que dice qué "
        "significa el número en contexto, incluyendo cuando una cifra que luce "
        "débil es en realidad sólida.</p>"
        "<div class='table-wrap'><table class='glossary'>"
        "<thead><tr><th>Indicador</th><th>Qué es</th><th>Cómo leerlo</th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table></div></div>"
    )


def _plotly_section_html(chart_data: Optional[dict], log) -> dict:
    """Build every interactive chart, grouped into placeable sections.

    Returns ``{"results": html, "explain": html, "diagnostics": html}`` rather
    than one blob, because the sections belong in different places in the
    document: results and explainability lead, while the raw-input diagnostics
    go last -- which is also the order the jump-nav advertises. Emitting them
    as one block put "Diagnósticos" physically before "Modelos" while the nav
    listed it last.

    ``{}`` when there is nothing to plot.
    """
    empty: dict = {}
    if not chart_data:
        return empty
    try:
        import json as _json

        from src.reporting.report_content import (
            SERIES_COLORS, THEME_RESTYLE_JS, build_plotly_figures,
        )

        figs = build_plotly_figures(chart_data, log=log)
    except Exception as exc:
        if log:
            log.warning("Interactive charts skipped (%s).", exc)
        return empty
    if not figs:
        return empty

    def _blocks(items) -> str:
        return "".join(
            "<figure class='chart'>"
            f"<figcaption class='chart-note'>{html.escape(f['note'])}</figcaption>"
            f"{f['html']}"
            "</figure>"
            for f in items
        )

    # Grouped so ~20 charts do not stack into one undifferentiated wall.
    # Order is the reading order: results, then per-model explanation, then the
    # raw-input diagnostics (collapsed -- they answer "is the input sane",
    # which the reliability section already asserts).
    by_group: dict[str, list] = {}
    for f in figs:
        by_group.setdefault(f.get("group", "resultados"), []).append(f)
    blocks = [_blocks(by_group.get("resultados", []))]
    series_map = {f["id"]: f["series"] for f in figs}
    js = (
        THEME_RESTYLE_JS
        .replace("__SERIES_COLORS__", _json.dumps(SERIES_COLORS))
        .replace("__FIGURE_SERIES__", _json.dumps(series_map))
    )
    supervised_run = any(
        (m or {}).get("supervised") for m in (chart_data.get("models") or {}).values()
    )
    mode_note = (
        "Esta corrida se evaluó contra una verdad base reservada, por lo que se "
        "incluyen curvas basadas en etiquetas (ROC / precisión-recall)."
        if supervised_run else
        "Esta corrida es <strong>no supervisada</strong> -- los detectores nunca "
        "vieron una etiqueta, por lo que no se reporta ninguna métrica basada en "
        "etiquetas (ROC, precisión, recall). Los gráficos a continuación son los "
        "diagnósticos válidos sin verdad base: dónde cae cada distribución de "
        "puntaje respecto a su umbral de alerta calibrado, y qué tanto coinciden "
        "dos detectores construidos de forma independiente sobre quién es "
        "anómalo."
    )
    sections: dict = {
        # The theme-restyle script rides with the first section so it is
        # defined once, after at least one chart div exists.
        "results": (
            "<h2 id='results'>Resultados</h2>"
            f"<div class='card'><p class='lead'>{mode_note} Pase el cursor sobre "
            "cualquier marca para ver los valores exactos. Cada detector mantiene un "
            "color fijo en todo el reporte (un par validado para deficiencia de "
            "visión cromática) y siempre está identificado por nombre, de modo que "
            "su identidad nunca depende solo del color.</p>"
            + "".join(blocks) + "</div>"
        ),
    }

    modelo = by_group.get("modelo", [])
    if modelo:
        sections["explain"] = (
            "<h2 id='explain'>Explicabilidad por modelo</h2>"
            "<div class='card'><p class='lead'>Qué mira cada detector para "
            "asignar su puntaje: los features que más lo mueven, el mecanismo "
            "interno que produce el puntaje, y la geometría del espacio en el "
            "que separa. Todos son interactivos -- acerque el zoom o pase el "
            "cursor para leer valores exactos.</p>"
            + _blocks(modelo) + "</div>"
        )

    diag = by_group.get("diagnostico", [])
    if diag:
        # Collapsed: these answer "is the input sane", which the reliability
        # section already asserts. They are context, not a modelling result.
        sections["diagnostics"] = (
            "<h2 id='figures'>Diagnósticos de preprocesamiento</h2>"
            f"<details class='fig-group'><summary>Distribuciones de los features "
            f"crudos<span class='count'>{len(diag)} gráfico(s)</span></summary>"
            "<div class='chart-grid'>" + _blocks(diag) + "</div></details>"
        )

    # One script for every chart, appended to whatever section renders last.
    sections["script"] = f"<script>{js}</script>"
    return sections


def _build_html(context: dict, log, out_dir: str = paths.REPORTS_DIR) -> str:
    title = context.get("title", "Reporte de Detección de Anomalías")
    generated_at = context.get("generated_at", "")
    dataset = context.get("dataset") or {}
    models = context.get("models") or {}
    parts: list[str] = []

    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='es'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append(f"<style>{_HTML_CSS}</style>")
    parts.append("</head><body>")

    # -- sticky header / jump nav -------------------------------------------- #
    parts.append("<header class='topbar'><div class='topbar-inner'>")
    parts.append(
        f"<div class='topbar-title'>{html.escape(title)}"
        + (f"<span class='ts'>Generado {html.escape(str(generated_at))}</span>"
           if generated_at else "") + "</div>"
    )
    parts.append(
        "<nav class='jump'>"
        "<a href='#overview'>Resumen</a>"
        "<a href='#oot'>Entregable</a>"
        "<a href='#results'>Resultados</a>"
        "<a href='#explain'>Explicabilidad</a>"
        "<a href='#models'>Modelos</a>"
        "<a href='#indicators'>Indicadores</a>"
        "<a href='#reliability'>Confiabilidad</a>"
        "<a href='#parameters'>Parámetros</a>"
        "<a href='#approach'>Enfoque</a>"
        "</nav>"
    )
    parts.append("<button id='theme-toggle' type='button'>Modo oscuro</button>")
    parts.append("</div></header>")

    parts.append("<div class='wrap'>")
    parts.append("<h1 id='overview'>Resumen</h1>")
    # Hero first, then the supporting KPI groups: one headline number, then
    # the context needed to read it.
    parts.append(_hero_html(dataset))
    parts.append(_dataset_tiles_html(dataset))

    # -- OOT deliverable (headline business output) --------------------------- #
    oot_html = _oot_callout_html(context.get("oot_excel"), out_dir)
    if oot_html:
        parts.append(oot_html)

    # -- results: interactive charts, before the dense metric tables ---------- #
    # Ordering is deliberate: a reader wants "how did it do" (charts), then the
    # per-model explanation, then the numbers, the reading guide, the trust
    # argument, and only last the input diagnostics. The sections are placed
    # individually (rather than as one block) so the document order matches the
    # jump-nav exactly.
    charts = _plotly_section_html(context.get("chart_data"), log)
    parts.append(charts.get("results", ""))
    parts.append(charts.get("explain", ""))

    # -- models ---------------------------------------------------------------- #
    if models:
        parts.append("<h2 id='models'>Detalle por modelo</h2>")
        parts.append("<div class='model-grid'>")
        for name, spec in models.items():
            if not spec:
                continue
            parts.append(_model_card_html(name, spec))
        parts.append("</div>")

    # -- reading guide + trust argument + parameter meanings ------------------- #
    parts.append(_metric_glossary_html(models))
    parts.append(_statistical_checks_html())
    parts.append(_param_glossary_html(models))
    parts.append(_ml_vs_econometrics_html())

    # -- input diagnostics, last: context rather than a modelling result ------ #
    # No static-figure gallery: every chart in this report is Plotly, including
    # the ones that used to be base64 PNGs. The matplotlib PNGs are still
    # written to `artifacts/reports/figures/` as run evidence, but the HTML no
    # longer embeds them (and no PDF exists to rasterise them into either).
    parts.append(charts.get("diagnostics", ""))
    parts.append(charts.get("script", ""))

    # -- methodology (formulas as preformatted text, no network) --------------- #
    parts.append("<h2 id='methodology'>Metodología</h2>")
    parts.append("<div class='card'>")
    parts.append(
        "<p>Se utilizan dos detectores complementarios: un <strong>Isolation "
        "Forest</strong> (las anomalías se aíslan con longitudes de camino "
        "promedio más cortas) y un <strong>Variational Autoencoder</strong> "
        "(las anomalías presentan un error de reconstrucción alto). Ambos "
        "siguen la convención del proyecto <em>mayor puntaje = más "
        "anómalo</em>.</p>"
    )
    parts.append(
        "<p>El VAE maximiza la cota inferior de evidencia (ELBO), lo que "
        "equivale a minimizar el error de reconstrucción más un regularizador "
        "KL. La KL gaussiana está disponible en forma cerrada:</p>"
    )
    parts.append(
        "<pre>"
        + html.escape(_ELBO_TEXT) + "\n\n" + html.escape(_KL_TEXT)
        + "</pre>"
    )
    parts.append("</div>")

    notes = context.get("notes")
    if notes:
        parts.append("<h2>Notas</h2>")
        parts.append(f"<div class='note'>{html.escape(str(notes))}</div>")

    parts.append(
        "<footer class='report-footer'>Generado por el pipeline de detección de "
        "anomalías Modelo-v0.1 &middot; todas las figuras se renderizan en línea "
        "(sin solicitudes externas)</footer>"
    )
    parts.append("</div>")  # .wrap
    parts.append(f"<script>{_THEME_TOGGLE_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Technical documentation (Markdown -- hyperparameters, math, artifact map)   #
# --------------------------------------------------------------------------- #
# Everything the business-facing HTML/MD deliberately omit: exact
# hyperparameters (winner and alternative alike -- the pipeline runs and
# reports every model, it does not pick a single "winner"), the full
# threshold-calibration record, the VAE's math, the preprocessing settings,
# and a catalog of every artifact path this run wrote.
def _artifact_catalog_rows(
    context: dict, report_paths: dict, doc_path: str,
) -> list[tuple[str, str]]:
    from src.utils import paths as _paths

    rows: list[tuple[str, str]] = []
    for label, path in (
        ("Modelo Isolation Forest", _paths.IFOREST_MODEL),
        ("Parámetros ajustados de Isolation Forest", _paths.IFOREST_BEST_PARAMS),
        ("Modelo VAE", _paths.VAE_MODEL),
        ("Parámetros ajustados del VAE", _paths.VAE_BEST_PARAMS),
    ):
        if os.path.exists(path):
            rows.append((label, path))
    for name, path in _oot_excel_items(context.get("oot_excel")):
        label = f"Entregable OOT ({_MODEL_TITLES.get(name, (name or 'modelo', ''))[0]})"
        rows.append((label, str(path)))
    for fmt, label in (("html", "Reporte HTML"), ("md", "Reporte Markdown")):
        p = report_paths.get(fmt)
        if p:
            rows.append((label, p))
    rows.append(("Documentación técnica (este archivo)", doc_path))
    return rows


def _build_model_documentation(
    context: dict, out_dir: str, report_paths: dict, doc_path: str,
) -> str:
    parts: list[str] = []
    parts.append("# Documentación del Modelo\n")
    parts.append(
        "Companion técnico del reporte de anomalías: hiperparámetros exactos, "
        "la formulación matemática del VAE, el pipeline de preprocesamiento y "
        "el mapa completo de artefactos de esta corrida. "
        "`anomaly_report.{html,md}` se mantiene orientado a negocio; este "
        "archivo es el rastro de auditoría para MLOps / auditores.\n"
    )

    dataset = context.get("dataset") or {}
    parts.append("## Metadatos de la corrida\n")
    parts.append(_md_kv_table(
        [("Generado", context.get("generated_at", "n/a")),
         ("Modo de evaluación", dataset.get("evaluation_mode", "n/a"))],
        headers=("Campo", "Valor"),
    ))

    models = context.get("models") or {}
    if models:
        parts.append("## Hiperparámetros de los modelos\n")
        for name, spec in models.items():
            if not spec:
                continue
            title = _MODEL_TITLES.get(name, (name, ""))[0]
            parts.append(f"### {title}\n")
            parts.append(_md_kv_table(_kv_rows((spec or {}).get("best_params")),
                                      headers=("Parámetro", "Valor")))
            cal = (spec or {}).get("threshold")
            if isinstance(cal, dict) and cal:
                parts.append(f"**Calibración del umbral ({title})**\n")
                parts.append(_md_kv_table(_kv_rows(cal), headers=("Campo", "Valor")))

    parts.append("## Fundamentos matemáticos (VAE)\n")
    parts.append(
        "El VAE maximiza la cota inferior de evidencia (ELBO), lo que "
        "equivale a minimizar el error de reconstrucción más un "
        "regularizador KL:\n"
    )
    parts.append(f"$$\n{_ELBO_LATEX}\n$$\n")
    parts.append("con la KL gaussiana disponible en forma cerrada:\n")
    parts.append(f"$$\n{_KL_LATEX}\n$$\n")

    preprocessing = context.get("preprocessing")
    if preprocessing:
        parts.append("## Pipeline de preprocesamiento\n")
        parts.append(_md_kv_table(_kv_rows(preprocessing), headers=("Parámetro", "Valor")))

    notes = context.get("notes")
    if notes:
        parts.append("## Notas de la corrida\n")
        parts.append(f"{notes}\n")

    parts.append("## Catálogo de artefactos\n")
    parts.append(_md_kv_table(_artifact_catalog_rows(context, report_paths, doc_path),
                              headers=("Artefacto", "Ruta")))

    figures = context.get("figures") or []
    existing = [f for f in figures if f.get("path") and os.path.isfile(f["path"])]
    diagnostics = _group_figures(existing).get(_FIGURE_GROUP_FALLBACK) or []
    if diagnostics:
        parts.append("## Diagnósticos de preprocesamiento\n")
        parts.append(
            "Histogramas de distribución de features crudos, generados antes "
            "del modelado para verificar la sanidad del panel de entrada. "
            "Enlazados en lugar de incrustados aquí -- son diagnósticos, no "
            "resultados de modelado.\n"
        )
        for fig in diagnostics:
            ftitle = fig.get("title", os.path.basename(fig["path"]))
            rel = os.path.relpath(fig["path"], out_dir).replace(os.sep, "/")
            parts.append(f"- [{ftitle}]({rel})\n")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def build_report(
    context: dict,
    out_dir: str = paths.REPORTS_DIR,
    basename: str = "anomaly_report",
    formats: Iterable[str] = ("html", "md", "model_doc"),
) -> dict:
    """Assemble a run's artifacts into HTML / Markdown / documentation.

    Args:
        context: Plain dict with keys ``title``, ``generated_at``, ``dataset``,
            ``models`` (each ``{'best_params': {...}, 'metrics': {...},
            'threshold': {...}}``), ``figures`` (list of ``{'title', 'path'}``),
            ``oot_excel``, ``preprocessing`` (flat dict of pipeline settings)
            and ``notes``. ``oot_excel`` accepts either a single path (one
            model) or a ``{model_name: path}`` dict (multiple models, e.g.
            iForest and VAE both export a top-decile OOT Excel) -- all entries
            are rendered. Missing keys are tolerated and omitted from the
            output.
        out_dir: Directory the report files are written to (created lazily).
        basename: Filename stem for the html/md outputs. The technical
            documentation is always written as ``model_documentation.md`` in
            the same directory, independent of ``basename``.
        formats: Which outputs to emit; any of ``'html'``, ``'md'``,
            ``'model_doc'``. No PDF format exists -- see the module docstring.

    Returns:
        ``{'html': path_or_None, 'md': path_or_None, 'model_doc': path_or_None}``.
    """
    log = setup_logging()
    fmts = {str(f).lower() for f in formats}
    result: dict = {"html": None, "md": None, "model_doc": None}

    with log_phase("reporting.build_report", log):
        os.makedirs(out_dir, exist_ok=True)

        if "md" in fmts or "markdown" in fmts:
            md_path = os.path.join(out_dir, f"{basename}.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(_build_markdown(context, out_dir))
            result["md"] = os.path.abspath(md_path)
            log.info("Wrote Markdown report -> %s", md_path)

        if "html" in fmts:
            html_path = os.path.join(out_dir, f"{basename}.html")
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(_build_html(context, log, out_dir))
            result["html"] = os.path.abspath(html_path)
            log.info("Wrote HTML report -> %s", html_path)

        if "model_doc" in fmts or "doc" in fmts or "documentation" in fmts:
            doc_path = os.path.join(out_dir, "model_documentation.md")
            with open(doc_path, "w", encoding="utf-8") as fh:
                fh.write(_build_model_documentation(context, out_dir, result, doc_path))
            result["model_doc"] = os.path.abspath(doc_path)
            log.info("Wrote model documentation -> %s", doc_path)

    return result
