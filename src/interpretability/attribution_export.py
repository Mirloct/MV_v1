"""Full per-feature attribution as an Excel workbook, one sheet per model.

The report and the figures (`shap_summary_iforest`, `plot_score_distribution`
-style bar charts) deliberately cap what they draw at the top N features -- a
beeswarm or bar chart with 200 rows is unreadable, so cropping there is
correct. But cropping the *deliverable* to the same top N would silently drop
whatever fell outside it, and a reviewer monitoring drift or auditing the
model has no use for a chart -- they need every value. This module is that
uncropped counterpart: the same `{feature: value}` dicts the figures are drawn
from (`shap_summary_iforest`, `reconstruction_error_by_feature`), written out
whole.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

__all__ = ["export_attribution_workbook"]

_DEFAULT_OUT = os.path.join(paths.REPORTS_DIR, "feature_attribution.xlsx")

#: Sheet name, column header for the value, and a one-line description of
#: what the number means -- written as a small header block on each sheet so
#: a reader does not have to guess which methodology produced which numbers
#: (SHAP and reconstruction error are not the same kind of quantity, and
#: nothing about an xlsx says so on its own).
_SHEETS: dict = {
    "iforest": (
        "mean_abs_shap",
        "Importancia media |SHAP| por variable (o, si SHAP no estuvo "
        "disponible, importancia por permutación sobre el puntaje de "
        "anomalía) -- ver iforest_shap_summary.png para el gráfico "
        "recortado a las 20 variables principales.",
    ),
    "vae": (
        "mean_reconstruction_error",
        "Error de reconstrucción cuadrático medio por variable -- las "
        "columnas que el VAE reconstruye peor son las que más presionan "
        "el puntaje de anomalía. Ver vae_recon_by_feature.png para el "
        "gráfico recortado a las 20 variables principales.",
    ),
}


def export_attribution_workbook(
    per_model: dict, out_path: str = _DEFAULT_OUT,
) -> Optional[str]:
    """Write every model's full per-feature attribution to one ``.xlsx``.

    Args:
        per_model: ``{model_name: {feature: value}}``, one entry per model
            that produced an attribution -- typically
            ``{"iforest": shap_summary_iforest(...), "vae":
            reconstruction_error_by_feature(...)}``. A model whose dict is
            empty or missing is skipped rather than writing a blank sheet.
        out_path: Destination ``.xlsx``.

    Returns:
        The written path, or ``None`` if no model had any attribution to
        write (nothing failed -- there was nothing to export).
    """
    log = setup_logging()
    with log_phase("interpretability.export_attribution_workbook", log):
        sheets_written = []
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for model_name, importances in per_model.items():
                if not importances:
                    log.warning(
                        "No attribution values for %r; skipping its sheet.",
                        model_name,
                    )
                    continue
                value_col, description = _SHEETS.get(
                    model_name, (f"{model_name}_value", "")
                )
                # Sorted descending, not just passed through: the dict is
                # already sorted by every producer today, but the sheet must
                # not depend on that staying true.
                rows = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
                df = pd.DataFrame(rows, columns=["variable", value_col])
                # openpyxl sheet names cap at 31 chars and reject a few
                # punctuation characters; model names in this project are
                # short identifiers ("iforest", "vae") so this is a defensive
                # trim, not something expected to trigger.
                sheet_name = str(model_name)[:31]
                # The description goes in row 1, the table starts at row 3
                # (startrow=2, 0-indexed): a reader opening the sheet cold
                # sees what methodology produced the numbers before they see
                # the numbers themselves, and it costs one row.
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                if description:
                    writer.sheets[sheet_name]["A1"] = description
                sheets_written.append((sheet_name, len(df)))

        if not sheets_written:
            log.warning(
                "export_attribution_workbook: no model had attribution values; "
                "no file written."
            )
            if os.path.exists(out_path):
                os.remove(out_path)  # ExcelWriter creates the file eagerly
            return None

        log.info(
            "Wrote feature-attribution workbook -> %s (%s)",
            out_path,
            ", ".join(f"{name}: {n} variables" for name, n in sheets_written),
        )
        return os.path.abspath(out_path)
