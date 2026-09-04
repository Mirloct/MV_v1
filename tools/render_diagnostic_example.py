"""Render the "Diagnóstico cruzado IF-VAE" chapter from SYNTHETIC data.

Produces a standalone HTML page and a Markdown file so the chapter's structure
can be reviewed without running the full pipeline. The inputs come from the
acceptance tests' own synthetic fixture (``tests/test_diagnostic_section.py``):
the numbers are random draws with a fixed seed and describe no real model,
population, or run. Every output file says so, in the page itself.

Usage:
    python tools/render_diagnostic_example.py [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_diagnostic_section import synthetic_frames  # noqa: E402

from src.evaluation.ifvae_diagnostic import diagnose_frames  # noqa: E402
from src.reporting.diagnostic_section import (  # noqa: E402
    DIAGNOSTIC_CSS,
    DIAGNOSTIC_JS,
    render_diagnostic_html,
    render_diagnostic_markdown,
)
from src.reporting.report import _HTML_CSS  # noqa: E402

BANNER = (
    "EJEMPLO CON DATOS SINTÉTICOS — generado por "
    "tools/render_diagnostic_example.py a partir de la fixture de pruebas. "
    "Las cifras son extracciones aleatorias con semilla fija: no describen "
    "ningún modelo, población ni corrida real."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        _REPO_ROOT, "artifacts", "reports", "ifvae_diagnostics_example"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reference, scored, features = synthetic_frames(
        n_reference=400, n_scored=300, n_periods=3, seed=11,
    )
    payload = diagnose_frames(
        reference, scored, features, os.path.join(args.out, "suite_output"),
        run_meta={
            "run_id": "ejemplo-sintetico",
            "generated_at": "SIN FECHA (ejemplo sintético)",
            "architecture_mode": "Apilado",
            "detector_dependency": (
                "El VAE recibe el puntaje del Isolation Forest como feature de "
                "entrada; los detectores no son independientes."
            ),
            "derived_features": [],
            "entity_aggregation_rule": "Máximo puntaje de la entidad en la ventana",
            "detectors": {
                "iforest": {
                    "label": "Isolation Forest",
                    "model_id": "(ejemplo sintético)",
                    "score_origin": "Columna precalculada de la fixture",
                    "score_column": "if_score",
                    "score_direction": "Mayor = más anómalo",
                },
                "vae": {
                    "label": "VAE",
                    "model_id": "(ejemplo sintético)",
                    "score_origin": "Reconstrucción y latentes de la fixture",
                    "score_column": "recon__<feature>, mu__<i>, logvar__<i>",
                    "score_direction": "Mayor residual = más anómalo",
                },
            },
        },
        sensitivity_grid=(0.90, 0.95, 0.99),
        autopsy_rows_per_quadrant=2,
        entity_view=True,
    )
    contract = payload["contract"]

    html_path = os.path.join(args.out, "diagnostic_chapter_example.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(
            "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Ejemplo sintético — Diagnóstico cruzado IF-VAE</title>"
            f"<style>{_HTML_CSS}</style><style>{DIAGNOSTIC_CSS}</style></head>"
            "<body><div class='wrap'>"
            f"<div class='callout-inline serious'><div class='callout-title'>"
            f"Datos sintéticos</div><p>{BANNER}</p></div>"
            f"{render_diagnostic_html(contract)}"
            f"</div><script>{DIAGNOSTIC_JS}</script></body></html>"
        )

    md_path = os.path.join(args.out, "diagnostic_chapter_example.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"> **{BANNER}**\n\n{render_diagnostic_markdown(contract)}")

    print(f"HTML: {html_path}")
    print(f"Markdown: {md_path}")
    print(f"Secciones: {len(contract['sections'])} · "
          f"observaciones evaluadas: {payload['rows_scored']}")


if __name__ == "__main__":
    main()
