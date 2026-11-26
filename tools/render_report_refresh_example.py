"""Render a labelled synthetic long-tail example without training models.

Data sources / inputs: deterministic example scores; writes only to the
requested output directory (default artifacts/reports/report_refresh_example).
Created: 2026-11-26
Last modified: 2026-11-26
Changelog: 2026-11-26 — reproducible report/card/P95 workbook visual fixture.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema
from src.evaluation.oot_report import export_oot_top_anomalies
from src.evaluation.review_policy import review_union
from src.reporting.report import build_report


def main():
    output = Path("artifacts/reports/report_refresh_example")
    schema = PanelSchema(time_col="period", entity_col="entity_id", target_col=None)
    forest = {str(i): float(i) for i in range(1000)}
    vae = {str(i): float(3 * (999 - i)) for i in range(1000)}
    vae["0"] = 150000.0
    populations = {"iforest": forest, "vae": vae}
    workbooks = {}
    for name, scores in populations.items():
        frame = pd.DataFrame({"entity_id": list(scores), "period": ["2026-10"] * 1000,
                              "anomaly_score": list(scores.values())})
        path, _ = export_oot_top_anomalies(
            frame, schema, out_path=str(output / f"oot_p95_{name}.xlsx"), model_name=name,
        )
        workbooks[name] = path
    total = len(review_union(forest, vae))
    context = {
        "title": "Muestra sintética: distribución VAE con extremo de 150k",
        "dataset": {"rows": 1000, "entities": 1000, "n_anomalies": total,
                    "anomaly_rate": total / 1000},
        "chart_data": {"models": {
            name: {"oot_scores": list(scores.values()),
                   "true_oot_entity_scores": scores, "threshold": float(np.percentile(list(scores.values()), 99))}
            for name, scores in populations.items()
        }},
        "oot_excel": workbooks,
        "notes": "Ejemplo sintético exclusivo de validación visual; no representa rendimiento operativo.",
    }
    print(build_report(context, out_dir=str(output), formats=("html", "md")))


if __name__ == "__main__":
    main()
