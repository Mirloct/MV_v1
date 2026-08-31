"""Reporting module: assemble run artifacts into HTML / Markdown, plus a
separate technical ``model_documentation.md``. No PDF output is generated.

See :func:`src.reporting.report.build_report` for the ``context`` schema and the
per-format rendering details (every chart an interactive Plotly figure in
HTML, ``$$...$$`` LaTeX in Markdown). The HTML/Markdown pair stays
business-facing (results, KPIs, the OOT deliverable); exact hyperparameters,
the VAE math, preprocessing settings and the full artifact catalog live in
``model_documentation.md`` instead.

:func:`src.reporting.flow_visualization.build_flow_visualization` renders a
separate, self-contained HTML flow diagram from the structured run-events log
(``src.utils.observability``) -- an n8n/Databricks-style view of the pipeline's
phases, not part of the anomaly-detection report itself.
:func:`src.reporting.flow_visualization.start_live_view` serves the same view
live, on localhost only, while the run is still in progress.

:func:`src.reporting.analyst_dashboard.build_analyst_dashboard` renders one
self-contained "Cola de Revisión" HTML (not one per model -- both Isolation
Forest and VAE percentiles are shown for every individual, joined in memory
from each detector's own `true_oot_entity_scores`) fed exclusively by this
project's own OOT block: the primary deliverable's OOT Excel table plus the
OOT block's per-month recurrence -- no other data source, no business
categorization layered on top. See `CONTEXT.md` "Downstream analyst
dashboard".
"""

from src.reporting.analyst_dashboard import build_analyst_dashboard
from src.reporting.flow_visualization import build_flow_visualization, start_live_view
from src.reporting.report import build_report

__all__ = [
    "build_report", "build_flow_visualization", "start_live_view",
    "build_analyst_dashboard",
]
