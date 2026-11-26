"""Cumulative unique-person OOT tail summaries, before workbook filtering.

Data sources / inputs: chart_data.models.<detector>.true_oot_entity_scores.
Created: 2026-11-26
Last modified: 2026-11-26
Changelog: 2026-11-26 — report P95/P97/P99 union and detector breakdown.
"""
from __future__ import annotations

from src.evaluation.review_policy import review_ids


def _detector_scores(models: dict, name: str) -> dict:
    return (models.get(name) or {}).get("true_oot_entity_scores") or {}


def percentile_summary(context: dict) -> list[dict]:
    """Count each person once in the union at each cumulative operating point."""
    models = (context.get("chart_data") or {}).get("models") or {}
    if_scores = _detector_scores(models, "iforest")
    vae_scores = _detector_scores(models, "vae")
    population = {str(key) for key in if_scores} | {str(key) for key in vae_scores}
    if not population:
        return []
    rows = []
    for percentile in (95, 97, 99):
        forest = review_ids(if_scores, percentile)
        vae = review_ids(vae_scores, percentile)
        rows.append(dict(percentile=percentile, count=len(forest | vae),
                         if_count=len(forest), vae_count=len(vae),
                         intersection=len(forest & vae), total=len(population)))
    return rows


def percentile_cards_html(context: dict) -> str:
    rows = percentile_summary(context)
    if not rows:
        return ""
    cards = []
    for row in rows:
        cards.append(
            f"<div class='tile' data-percentile='{row['percentile']}'>"
            f"<div class='label'>P{row['percentile']}+ · personas OOT</div>"
            f"<div class='value'>{row['count']:,}</div>"
            f"<div class='sub'>IF: {row['if_count']:,} · IF+VAE: {row['vae_count']:,} · "
            f"Ambos: {row['intersection']:,}<br>"
            f"{row['count'] / row['total']:.1%} de {row['total']:,} personas OOT</div></div>"
        )
    return (
        "<section id='oot-percentile-counts'><h2>Personas por percentil OOT</h2>"
        "<p>Conteos acumulados: unión de ambos detectores, sin duplicar personas. "
        "Cortes calculados sobre el bloque OOT completo; incluyen empates. "
        "El dashboard de analista utiliza únicamente P99.</p>"
        "<div class='tile-row'>" + "".join(cards) + "</div></section>"
    )


def percentile_cards_markdown(context: dict) -> str:
    rows = percentile_summary(context)
    if not rows:
        return ""
    parts = ["## Personas por percentil OOT\n",
             "Conteos acumulados de personas únicas en la unión de detectores; "
             "incluyen empates. El dashboard utiliza únicamente P99.\n",
             "| Corte | Personas (unión) | IF | IF+VAE | Ambos | Población OOT |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        parts.append(f"| P{row['percentile']}+ | {row['count']} | {row['if_count']} | "
                     f"{row['vae_count']} | {row['intersection']} | {row['total']} |")
    return "\n".join(parts)
