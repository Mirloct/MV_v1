from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

#: Fixed per-candidate colors so a reader tracks "the VAE bar" across every
#: chart in a run instead of a legend order that can reshuffle between
#: figures. Anything outside this set (a custom `if_score_col` name would
#: not appear here since `score_name` is always one of these four) falls
#: back to matplotlib's default cycle.
_SCORE_COLORS = {
    "if_percentile": "#2a78d6",
    "vae_percentile": "#eb6834",
    "ensemble_max": "#4c9a4c",
    "ensemble_mean": "#9467bd",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True),
        encoding="utf-8",
    )


def disagreement_plot(scored: pd.DataFrame, path: Path, label_col: str | None) -> None:
    fig, axis = plt.subplots(figsize=(8, 7))
    has_labels = label_col is not None and label_col in scored
    colors = scored[label_col].map({0: "#7b8da8", 1: "#d1495b"}) if has_labels else "#7b8da8"
    axis.scatter(
        scored["if_percentile"],
        scored["vae_percentile"],
        c=colors,
        alpha=0.72,
        edgecolors="none",
    )
    axis.set(xlabel="Isolation Forest anomaly percentile", ylabel="VAE anomaly percentile")
    title = "Detector disagreement map (red = known positive)" if has_labels else (
        "Detector disagreement map (no ground truth available)"
    )
    axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def metrics_bar_plot(metrics: pd.DataFrame, path: Path) -> bool:
    """Precision@K by score candidate, grouped by alert budget -- the visual
    counterpart to `metrics.csv`, for monitoring each layer's (IF, VAE, each
    ensemble) operational performance at a glance rather than only as a
    table. The disagreement map shows where two detectors *agree*; this
    shows which one actually *performs* at the alert budgets that matter.

    Writes nothing and returns `False` when `metrics` is empty -- the
    label-free run (`config.label_col is None`) has no precision/recall to
    plot, and an empty chart would be worse than no chart.
    """
    if metrics.empty:
        return False
    budgets = sorted(metrics["k"].unique())
    scores = list(dict.fromkeys(metrics["score_name"]))  # first-seen order
    x = np.arange(len(budgets))
    width = 0.8 / max(len(scores), 1)

    fig, axis = plt.subplots(figsize=(8, 5))
    for i, score in enumerate(scores):
        by_k = metrics[metrics["score_name"] == score].set_index("k")
        values = by_k.reindex(budgets)["precision_at_k"].to_numpy()
        axis.bar(
            x + i * width, values, width=width, label=score,
            color=_SCORE_COLORS.get(score),
        )
    axis.set_xticks(x + width * (len(scores) - 1) / 2)
    axis.set_xticklabels([f"k={k}" for k in budgets])
    axis.set_ylabel("precision@k")
    axis.set_title("Operational performance by detector/ensemble")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    display = frame.head(max_rows).copy()
    if display.empty:
        return "_No rows._"
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _warning_lines(warnings: list[dict[str, Any]]) -> str:
    if not warnings:
        return "- No automated warning fired. This is not proof that leakage is absent."
    return "\n".join(
        f"- **{item.get('severity', 'info').upper()} — {item.get('code', 'warning')}:** "
        f"{item.get('message', item.get('reason', ''))}"
        for item in warnings
    )


def write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    drift: pd.DataFrame,
    warnings: list[dict[str, Any]],
    has_metrics_plot: bool = False,
) -> None:
    quadrant_rows = pd.DataFrame(
        [{"quadrant": key, "count": value} for key, value in summary["quadrants"].items()]
    )
    known_positives = summary["known_positives"]
    known_positives_text = (
        "not available (no label column -- label-free run)"
        if known_positives is None else str(known_positives)
    )
    # Both charts are embedded (not just catalogued as sibling files) so a
    # reader monitoring "how is each layer -- IF, VAE, each ensemble --
    # performing" sees it without leaving the report: the disagreement map
    # answers where the two detectors *agree*, the bar chart which one
    # actually *performs* at the alert budgets that matter. The bar chart is
    # omitted (not a broken image link) in label-free runs, where there is
    # no precision/recall to show.
    metrics_plot_section = (
        "\n![Operational performance by detector/ensemble](metrics.png)\n"
        if has_metrics_plot else
        "\n_No operational-performance chart: this run has no label column "
        "(label-free mode), so precision/recall/AP are unavailable._\n"
    )
    content = f"""# IF–VAE diagnostic report

## Decision summary

- Rows scored: **{summary['rows_scored']}**
- Known positives: **{known_positives_text}**
- Primary VAE score: **{summary['vae_primary_score']}**
- Alert percentile used for disagreement quadrants: **{summary['percentile_threshold']:.3f}**
- Synthetic evidence, if used, validates the suite only—not operational model quality.

### Detector quadrants

{_markdown_table(quadrant_rows)}

![Detector disagreement map](disagreement.png)

### Operational metrics

{_markdown_table(metrics.round(4), max_rows=40)}
{metrics_plot_section}

## Risks and failed assumptions

{_warning_lines(warnings)}

## Largest population shifts

{_markdown_table(drift.round(4), max_rows=20)}

## How to read the result

- `IF_ONLY`: isolated in input space but reconstructed/represented as ordinary by the VAE.
- `VAE_ONLY`: not strongly isolated by IF but violates the VAE reconstruction or latent model.
- `BOTH`: high-priority agreement, not automatic proof of a true anomaly.
- `NEITHER`: below this diagnostic threshold; not proof of normality.

Use `metrics.csv`, `coverage.json`, `autopsies.csv`, `drift.csv`, and
`scored_diagnostics.csv` for investigator-level analysis. Thresholds and ensemble
weights must be chosen on validation data under the real alert budget, never on
the final test set or one flagship case.
"""
    path.write_text(content, encoding="utf-8")

