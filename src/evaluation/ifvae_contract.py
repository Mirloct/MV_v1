"""Versioned, serializable contract for the "Diagnóstico cruzado IF-VAE"
report chapter (the factual ficha) and its numeric building blocks.

This module is the ONLY place that decides *what is available*, *why it is
not*, and *where a value came from*, for the FACTUAL chapter. The HTML and
Markdown renderers (`src/reporting/diagnostic_section.py`) consume the
structure this builds and do nothing but lay it out -- no availability
rules, no severity rules, no arithmetic, no narrative.

Design constraints this file exists to enforce (unsupervised diagnostic
record, not a verdict):

* No value is ever invented, defaulted silently, or back-filled. A field that
  cannot be sourced is emitted with an explicit non-``EXECUTED`` status and a
  machine-readable ``reason``.
* Nothing here ranks the detectors, scores their quality, assigns operational
  priority to a quadrant, or turns agreement/drift/latent/stability numbers
  into evidence of performance. Those are interpretations and live in a
  SEPARATE, explicitly-labelled contract
  (``src/evaluation/ifvae_interpretation.py``) built from the same numbers --
  see that module's docstring for why the split exists and stays enforced.
* Every static string is a title, a label, a neutral methodological
  definition, a reading aid, or an availability description.

Sections removed 2026-09-05 (by explicit editorial request, not because they
could not run): the disponibilidad matrix, the standalone "unidad de
análisis" section, the per-quadrant alert autopsy tables, the raw
calidad/desplazamiento (drift) table, the label-conditioned containers, and
the standalone riesgos/procedencia section. The unit-of-analysis statement
(entidad-periodo vs. entidad) now travels as a one-line note on every table
that needs it instead of its own section; drift and autopsy numbers still
feed the interpretation contract as distilled signals -- only the raw,
row-per-observation tables were removed from the factual ficha.

The contract is a plain ``dict`` (JSON-serializable) so it can be snapshotted,
diffed between runs, and asserted against in tests.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

__all__ = [
    "CONTRACT_VERSION",
    "STATUS_EXECUTED",
    "STATUS_NOT_APPLICABLE",
    "STATUS_UNAVAILABLE",
    "STATUS_FAILED",
    "STATUS_NOT_REQUESTED",
    "STATUSES",
    "STATUS_LABELS",
    "NO_VALUE_TEXT",
    "REASON_NO_LABELS",
    "REASON_NO_SEGMENT",
    "REASON_NO_SENSITIVITY_GRID",
    "REASON_NO_ENTITY_VIEW",
    "REASON_NO_EXPERIMENT_TRACKING",
    "REASON_NO_STABILITY_REFITS",
    "FEATURE_FAMILY_PREFIXES",
    "build_diagnostic_contract",
    "field",
    "missing",
    "feature_family",
]

#: Bumped whenever the *shape* of the contract changes, so a stored contract
#: can be told apart from one produced by a different renderer generation.
CONTRACT_VERSION = "2.0.0"

# -- Structured availability states (the only ones any status may take) ----- #
STATUS_EXECUTED = "EXECUTED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_FAILED = "FAILED"
STATUS_NOT_REQUESTED = "NOT_REQUESTED"
STATUSES = (
    STATUS_EXECUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_UNAVAILABLE,
    STATUS_FAILED,
    STATUS_NOT_REQUESTED,
)

#: Neutral, human-readable gloss for each state. Rendered as-is; carries no
#: judgement about the run, only about whether a diagnostic produced output.
STATUS_LABELS = {
    STATUS_EXECUTED: "Ejecutado",
    STATUS_NOT_APPLICABLE: "No aplica",
    STATUS_UNAVAILABLE: "No disponible",
    STATUS_FAILED: "Falló",
    STATUS_NOT_REQUESTED: "No solicitado",
}

#: Text shown in place of a value that has no ``EXECUTED`` status. The reason
#: always travels with it -- a bare "No disponible" would hide *why*.
NO_VALUE_TEXT = "No disponible"

#: Reason strings reused across sections. Kept here (not in the renderers) so
#: the same absence always reads the same way and can be asserted in tests.
REASON_NO_LABELS = (
    "La corrida es no supervisada: no hay columna de verdad base declarada "
    "en la configuración."
)
REASON_NO_SEGMENT = (
    "No hay columna de segmentación disponible en el panel de esta corrida."
)
REASON_NO_SENSITIVITY_GRID = (
    "No se declaró una malla de umbrales en la configuración de la corrida."
)
REASON_NO_ENTITY_VIEW = (
    "No se solicitó una vista agregada por entidad para este apartado."
)
REASON_NO_EXPERIMENT_TRACKING = (
    "Este pipeline no registra corridas de la matriz de experimentos; no hay "
    "fuente de la que leer su estado."
)
REASON_NO_STABILITY_REFITS = (
    "Se configuraron 0 reajustes de estabilidad "
    "(--diagnostic-stability-refits 0); no se midió estabilidad multisemilla."
)


def field(
    value: Any,
    *,
    source: str,
    status: str = STATUS_EXECUTED,
    reason: Optional[str] = None,
    note: Optional[str] = None,
) -> dict:
    """One traceable value: what it is, where it came from, and its state.

    ``source`` is mandatory and non-empty by design -- a value a reader cannot
    trace back to a config key, a metadata field, or an artifact file has no
    place in this record.
    """
    if not source:
        raise ValueError("every field must declare a source")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if status != STATUS_EXECUTED and not reason:
        raise ValueError(f"status {status!r} requires a reason")
    return {
        "value": value,
        "status": status,
        "source": source,
        "reason": reason,
        "note": note,
    }


def missing(
    status: str, reason: str, *, source: str = "n/a", note: Optional[str] = None
) -> dict:
    """A field that could not be sourced. Never falls back to a stand-in value."""
    if status == STATUS_EXECUTED:
        raise ValueError("missing() cannot carry EXECUTED")
    return field(None, source=source, status=status, reason=reason, note=note)


# --------------------------------------------------------------------------- #
# Block constructors (the vocabulary the renderers know how to lay out)       #
# --------------------------------------------------------------------------- #
def _fields_block(rows: Sequence[tuple[str, dict]], *, title: Optional[str] = None) -> dict:
    return {"kind": "fields", "title": title,
            "rows": [{"label": label, "field": f} for label, f in rows]}


def _table_block(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: Optional[str] = None,
    caption: Optional[str] = None,
    empty_text: str = "Sin filas.",
    filters: Optional[dict] = None,
) -> dict:
    return {
        "kind": "table", "title": title, "caption": caption,
        "columns": [str(c) for c in columns],
        "rows": [[("" if c is None else c) for c in row] for row in rows],
        "empty_text": empty_text,
        "filters": filters,
    }


def _note_block(text: str) -> dict:
    """Neutral methodological text / reading aid. Never a finding."""
    return {"kind": "note", "text": text}


def _scatter_block(
    points: dict,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_threshold: Optional[float],
    y_threshold: Optional[float],
    quadrants: Sequence[dict],
    alt_text: str,
    caption: Optional[str] = None,
) -> dict:
    return {
        "kind": "scatter", "title": title, "points": points,
        "x_label": x_label, "y_label": y_label,
        "x_threshold": x_threshold, "y_threshold": y_threshold,
        "quadrants": list(quadrants), "alt_text": alt_text, "caption": caption,
    }


def _section(section_id: str, title: str, blocks: Sequence[dict], *,
             status: str = STATUS_EXECUTED, reason: Optional[str] = None) -> dict:
    return {"id": section_id, "title": title, "status": status,
            "reason": reason, "blocks": [b for b in blocks if b]}


# --------------------------------------------------------------------------- #
# Small, purely descriptive helpers                                           #
# --------------------------------------------------------------------------- #
def _fmt_float(value: Any, digits: int = 4) -> Any:
    """Round for display without inventing precision. ``None``/NaN stay absent."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if as_float != as_float:  # NaN
        return None
    return round(as_float, digits)


def _fmt_pct(numerator: float, denominator: float, digits: int = 1) -> Optional[str]:
    if not denominator:
        return None
    return f"{100.0 * numerator / denominator:.{digits}f}%"


def _artifact_entry(directory: str, filename: str) -> dict:
    """Existence + non-emptiness of one artifact, as a traceable field.

    A linked file that does not exist, or exists at zero bytes, is reported as
    ``UNAVAILABLE`` rather than linked -- the reader must never follow a dead
    or empty link out of this chapter.
    """
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return {"name": filename, "path": None, "size_bytes": 0,
                "status": STATUS_UNAVAILABLE,
                "reason": "El archivo no fue escrito por esta corrida."}
    size = os.path.getsize(path)
    if size <= 0:
        return {"name": filename, "path": None, "size_bytes": 0,
                "status": STATUS_UNAVAILABLE,
                "reason": "El archivo existe pero está vacío."}
    return {"name": filename, "path": path, "size_bytes": int(size),
            "status": STATUS_EXECUTED, "reason": None}


#: Prefix -> feature-family label. The prefixes are this project's own
#: preprocessing output convention (`src/preprocessing/pipeline.py` emits
#: `num__`, `cat__`, `cyc__`, `bool__`, `missing__`), so this is a metadata
#: mapping, not an interpretation of any individual feature.
FEATURE_FAMILY_PREFIXES = (
    ("num__", "Numérico de negocio"),
    ("cat__", "Categórico de negocio"),
    ("cyc__", "Calendario / cíclico"),
    ("bool__", "Booleano de negocio"),
    ("missing__", "Indicador de faltante"),
)
FEATURE_FAMILY_PANEL_MARKERS = ("_lag", "_diff", "_ratio", "_own_z")
FEATURE_FAMILY_PANEL = "Panel / rezago"
FEATURE_FAMILY_DERIVED = "Salida de otro modelo"
FEATURE_FAMILY_UNKNOWN = "Sin familia declarada"


def feature_family(feature: str, derived_features: Sequence[str]) -> str:
    if feature in set(derived_features):
        return FEATURE_FAMILY_DERIVED
    if any(marker in feature for marker in FEATURE_FAMILY_PANEL_MARKERS):
        return FEATURE_FAMILY_PANEL
    for prefix, label in FEATURE_FAMILY_PREFIXES:
        if feature.startswith(prefix):
            return label
    return FEATURE_FAMILY_UNKNOWN


# --------------------------------------------------------------------------- #
# Section builders                                                            #
# --------------------------------------------------------------------------- #
def _section_scope(populations: dict, run_meta: dict, config: dict) -> dict:
    """§1 Alcance: what was evaluated, over which populations, under which
    architecture. Nothing here characterises how well anything performed."""
    ref, sco = populations["reference"], populations["scored"]
    arch = run_meta.get("architecture_mode")
    arch_field = (
        field(arch, source="main.py::PipelineConfig.stack_iforest_into_vae")
        if arch else
        missing(STATUS_UNAVAILABLE,
                "La corrida no declaró un modo arquitectónico.",
                source="run_meta")
    )
    dependency = run_meta.get("detector_dependency")
    dependency_field = (
        field(dependency, source="main.py::PipelineConfig.stack_iforest_into_vae")
        if dependency else
        missing(STATUS_UNAVAILABLE,
                "La corrida no declaró la relación de dependencia entre detectores.",
                source="run_meta")
    )
    entity_rule = run_meta.get("entity_aggregation_rule")
    rows = [
        ("Objetivo del apartado",
         field("Registro estructurado del diagnóstico cruzado entre el Isolation "
               "Forest y el VAE de esta corrida.", source="contrato (texto fijo)")),
        ("Carácter de la evaluación",
         field("No supervisada", source="ifvae_diag.config::label_col")),
        ("Disponibilidad de verdad base",
         field("Sin verdad base declarada",
               source="ifvae_diag.config::label_col")
         if config.get("label_col") is None else
         field(str(config.get("label_col")), source="ifvae_diag.config::label_col")),
        ("Unidad de observación",
         field(populations["observation_unit"], source="src/evaluation/ifvae_diagnostic.py",
               note="Todos los conteos de este capítulo son observaciones "
                    "entidad-periodo salvo que una tabla indique lo contrario.")),
        ("Población de referencia", field(ref["name"], source=ref["source"])),
        ("Población evaluada", field(sco["name"], source=sco["source"])),
        ("Rango temporal de la referencia", _range_field(ref)),
        ("Rango temporal de la población evaluada", _range_field(sco)),
        ("Observaciones (referencia)", field(ref["rows"], source=ref["source"])),
        ("Observaciones (evaluación)", field(sco["rows"], source=sco["source"])),
        ("Entidades únicas (referencia)", field(ref["entities"], source=ref["source"])),
        ("Entidades únicas (evaluación)", field(sco["entities"], source=sco["source"])),
        ("Periodos (referencia)", field(ref["periods"], source=ref["source"])),
        ("Periodos (evaluación)", field(sco["periods"], source=sco["source"])),
        ("Regla de agregación por entidad",
         field(entity_rule, source="main.py::run_meta")
         if entity_rule else
         missing(STATUS_NOT_REQUESTED, REASON_NO_ENTITY_VIEW, source="run_meta")),
        ("Modo arquitectónico de los detectores", arch_field),
        ("Relación entre IF y VAE", dependency_field),
        ("Versión de la suite diagnóstica",
         field(run_meta.get("suite_version"), source="ifvae_diag.__version__")
         if run_meta.get("suite_version") else
         missing(STATUS_UNAVAILABLE, "El paquete no expone versión.",
                 source="ifvae_diag")),
        ("Identificador de la corrida",
         field(run_meta.get("run_id"), source="observability::run_id")
         if run_meta.get("run_id") else
         missing(STATUS_UNAVAILABLE, "La corrida no expuso identificador.",
                 source="run_meta")),
        ("Fecha de la corrida",
         field(run_meta.get("generated_at"), source="main.py::generated_at")
         if run_meta.get("generated_at") else
         missing(STATUS_UNAVAILABLE, "La corrida no expuso fecha.", source="run_meta")),
    ]
    return _section("scope", "1. Alcance del diagnóstico", [_fields_block(rows)])


def _range_field(population: dict) -> dict:
    lo, hi = population.get("period_min"), population.get("period_max")
    if lo is None or hi is None:
        return missing(STATUS_UNAVAILABLE,
                       "La población no expone columna temporal.",
                       source=population["source"])
    return field(f"{lo} … {hi}", source=population["source"])


def _section_configuration(config: dict, run_meta: dict) -> dict:
    """§2 Configuración efectiva: every knob that shaped the numbers below,
    read from the resolved configuration the suite itself wrote."""
    src = "ifvae_diagnostics/resolved_config.json"
    detectors = run_meta.get("detectors") or {}

    def _detector_row(key: str) -> list:
        meta = detectors.get(key) or {}
        return [
            meta.get("label", key),
            meta.get("model_id") or NO_VALUE_TEXT,
            meta.get("score_origin") or NO_VALUE_TEXT,
            meta.get("score_column") or NO_VALUE_TEXT,
            meta.get("score_direction") or NO_VALUE_TEXT,
        ]

    detector_table = _table_block(
        ["Detector", "Modelo / versión", "Origen del puntaje",
         "Columna de puntaje", "Dirección del puntaje"],
        [_detector_row(k) for k in sorted(detectors)],
        title="Detectores",
        caption="Valores tomados de la configuración resuelta y de los "
                "metadatos de la corrida.",
        empty_text="La corrida no expuso metadatos de detectores.",
    )

    budgets = config.get("alert_budgets")
    seeds = config.get("random_seeds")
    rows = [
        ("Puntaje VAE principal seleccionado",
         field(config.get("vae_primary_score"), source=f"{src}::vae_primary_score")),
        ("Residuales usados por el puntaje top-K",
         field(config.get("top_k_residuals"), source=f"{src}::top_k_residuals")),
        ("Método de umbralización",
         field("Percentil sobre la distribución de referencia",
               source=f"{src}::percentile_threshold")),
        ("Valor del umbral (percentil)",
         field(config.get("percentile_threshold"),
               source=f"{src}::percentile_threshold")),
        ("Origen del umbral",
         field("Configuración de la suite diagnóstica", source=src)),
        ("Presupuestos de alertas configurados",
         field(budgets, source=f"{src}::alert_budgets") if budgets else
         missing(STATUS_NOT_REQUESTED,
                 "No se declararon presupuestos de alertas.", source=src)),
        ("Semillas configuradas",
         field(seeds, source=f"{src}::random_seeds") if seeds else
         missing(STATUS_NOT_REQUESTED, "No se declararon semillas.", source=src)),
        ("Regla de desempate del percentil",
         field("Cuenta de valores de referencia menores o iguales al puntaje "
               "(lado derecho)",
               source="ifvae_diag.scoring::anomaly_percentile")),
        ("Regla de agregación por entidad",
         field(run_meta.get("entity_aggregation_rule"), source="main.py::run_meta")
         if run_meta.get("entity_aggregation_rule") else
         missing(STATUS_NOT_REQUESTED, REASON_NO_ENTITY_VIEW, source="run_meta")),
        ("Modo paralelo o apilado",
         field(run_meta.get("architecture_mode"), source="main.py::PipelineConfig")
         if run_meta.get("architecture_mode") else
         missing(STATUS_UNAVAILABLE, "La corrida no declaró el modo.",
                 source="run_meta")),
        ("Features derivados de otros modelos",
         field(list(run_meta.get("derived_features") or []),
               source="main.py: vae_feature_names \\ feature_names")
         if run_meta.get("derived_features") else
         field([], source="main.py: vae_feature_names \\ feature_names",
               note="La matriz del VAE no incorpora salidas de otro modelo.")),
        ("Reajustes de estabilidad configurados",
         field(run_meta.get("stability_refits"),
               source="main.py::PipelineConfig.diagnostic_stability_refits")),
    ]
    return _section("configuration", "2. Configuración efectiva",
                    [detector_table, _fields_block(rows, title="Parámetros efectivos")])


def _section_agreement(agreement: dict, config: dict, populations: dict) -> dict:
    """§3 Concordancia y desacuerdo: counts, set relations, and the scatter.

    The set statistics are arithmetic on the quadrant counts the suite already
    assigned; the rank correlation is a descriptive statistic over the two
    percentile columns it already wrote. Neither is turned into a judgement.
    """
    quadrants = agreement["quadrants"]
    total = agreement["total"]
    src = "ifvae_diagnostics/summary.json::quadrants"
    quadrant_rows = [
        [key, quadrants.get(key, 0), _fmt_pct(quadrants.get(key, 0), total) or NO_VALUE_TEXT,
         definition]
        for key, definition in (
            ("BOTH", "Ambos detectores marcan la observación en o sobre el umbral."),
            ("IF_ONLY", "Solo el Isolation Forest marca la observación."),
            ("VAE_ONLY", "Solo el VAE marca la observación."),
            ("NEITHER", "Ningún detector marca la observación."),
        )
    ]
    set_rows = [
        ("Unidad de análisis",
         field("Observación entidad–periodo (no deduplicada por entidad)",
               source="src/evaluation/ifvae_diagnostic.py")),
        ("Alertas totales del Isolation Forest",
         field(agreement["if_alerts"], source=src)),
        ("Alertas totales del VAE", field(agreement["vae_alerts"], source=src)),
        ("Intersección", field(agreement["intersection"], source=src)),
        ("Unión", field(agreement["union"], source=src)),
        ("Índice de Jaccard",
         field(_fmt_float(agreement["jaccard"]), source=src)
         if agreement["jaccard"] is not None else
         missing(STATUS_UNAVAILABLE,
                 "La unión de alertas es cero; el índice no está definido.",
                 source=src)),
        ("Correlación de rangos (Spearman)",
         field(_fmt_float(agreement["rank_correlation"]),
               source="ifvae_diagnostics/scored_diagnostics.csv")
         if agreement["rank_correlation"] is not None else
         missing(STATUS_UNAVAILABLE, agreement["rank_correlation_reason"],
                 source="ifvae_diagnostics/scored_diagnostics.csv")),
        ("Umbral utilizado",
         field(config.get("percentile_threshold"),
               source="ifvae_diagnostics/resolved_config.json::percentile_threshold")),
        ("Fuente de calibración del umbral",
         field("Distribución de puntajes de la población de referencia",
               source="ifvae_diag.scoring::anomaly_percentile")),
    ]
    blocks = [
        _table_block(["Cuadrante", "Observaciones", "% del universo evaluado",
                      "Definición"],
                     quadrant_rows, title="Cuadrantes",
                     caption="Universo: observaciones entidad-periodo de la "
                             "población evaluada."),
        _fields_block(set_rows, title="Relaciones de conjunto"),
    ]
    scatter = agreement.get("scatter")
    if scatter:
        blocks.append(_scatter_block(
            scatter["points"],
            title="Percentil IF frente a percentil VAE",
            x_label=scatter["x_label"], y_label=scatter["y_label"],
            x_threshold=scatter["threshold"], y_threshold=scatter["threshold"],
            quadrants=[
                {"key": key, "label": key, "count": quadrants.get(key, 0),
                 "position": pos}
                for key, pos in (("NEITHER", "abajo-izquierda"),
                                 ("IF_ONLY", "abajo-derecha"),
                                 ("VAE_ONLY", "arriba-izquierda"),
                                 ("BOTH", "arriba-derecha"))
            ],
            alt_text=scatter["alt_text"],
            caption=scatter.get("caption"),
        ))
    blocks.append(_note_block(
        "Cada punto es una observación entidad-periodo de la población "
        "evaluada, situada por su percentil respecto de la distribución de "
        "referencia de cada detector. Las líneas marcan el umbral "
        "configurado. El cuadrante describe coincidencia de marcado; no "
        "expresa severidad, prioridad ni acierto."
    ))
    return _section("agreement", "3. Concordancia y desacuerdo", blocks)


def _section_vae_candidates(candidates: Sequence[dict], config: dict) -> dict:
    """§4 Comparación de candidatos VAE: same statistics for every candidate,
    with the configured primary marked as *configured*, not as best."""
    primary = config.get("vae_primary_score")
    rows = []
    for cand in candidates:
        rows.append([
            cand["name"] + (" (configurado como principal)"
                            if cand["name"] == primary else ""),
            STATUS_LABELS.get(cand["status"], cand["status"]),
            cand.get("parameters") or "",
            _fmt_pct(cand["alerts"], cand["total"]) if cand["status"] == STATUS_EXECUTED else NO_VALUE_TEXT,
            cand.get("quadrant_text") or NO_VALUE_TEXT,
            cand.get("intersection") if cand["status"] == STATUS_EXECUTED else NO_VALUE_TEXT,
            cand.get("union") if cand["status"] == STATUS_EXECUTED else NO_VALUE_TEXT,
            _fmt_float(cand.get("jaccard")) if cand.get("jaccard") is not None else NO_VALUE_TEXT,
            _fmt_float(cand.get("rank_correlation")) if cand.get("rank_correlation") is not None else NO_VALUE_TEXT,
            cand.get("artifact") or NO_VALUE_TEXT,
        ])
    return _section("vae-candidates", "4. Comparación de puntajes VAE", [
        _table_block(
            ["Candidato", "Disponibilidad", "Parámetros", "Tasa de alertas",
             "Cuadrantes (BOTH / IF_ONLY / VAE_ONLY / NEITHER)", "Intersección",
             "Unión", "Jaccard", "Correlación de rangos", "Artefacto"],
            rows,
            caption="Cada candidato se compara contra el mismo puntaje IF y el "
                    "mismo umbral configurado.",
            empty_text="La corrida no expuso columnas de candidatos VAE.",
        ),
        _note_block(
            "La marca «configurado como principal» indica cuál candidato eligió "
            "la configuración de la corrida para construir el percentil VAE. No "
            "expresa que sea preferible a los demás."
        ),
    ])


def _section_sensitivity(sensitivity: dict) -> dict:
    """§5 Sensibilidad: quadrant counts recomputed over a configured grid."""
    if sensitivity["status"] != STATUS_EXECUTED:
        return _section(
            "sensitivity", "5. Sensibilidad al umbral",
            [_fields_block([("Malla de umbrales",
                             missing(sensitivity["status"], sensitivity["reason"],
                                     source=sensitivity["source"]))])],
            status=sensitivity["status"], reason=sensitivity["reason"],
        )
    rows = [
        [r["threshold"], r["score_candidate"], r["BOTH"], r["IF_ONLY"],
         r["VAE_ONLY"], r["NEITHER"], r["if_rate"], r["vae_rate"], r["union"],
         _fmt_float(r["jaccard"]) if r["jaccard"] is not None else NO_VALUE_TEXT,
         _fmt_float(r["rank_correlation"]) if r["rank_correlation"] is not None else NO_VALUE_TEXT,
         r.get("budget") or ""]
        for r in sensitivity["rows"]
    ]
    return _section("sensitivity", "5. Sensibilidad al umbral", [
        _table_block(
            ["Umbral", "Candidato de puntaje", "BOTH", "IF_ONLY", "VAE_ONLY",
             "NEITHER", "Tasa marginal IF", "Tasa marginal VAE", "Unión",
             "Jaccard", "Correlación de rangos", "Presupuesto asociado"],
            rows,
            caption=f"Malla declarada en configuración: {sensitivity['grid']}.",
            empty_text="La malla configurada no produjo filas.",
        ),
        _note_block(
            "Los conteos se recalculan sobre los percentiles ya escritos por la "
            "suite; ningún modelo se reajusta al variar el umbral."
        ),
    ])


def _section_latent(latent: dict) -> dict:
    """§6 Diagnóstico del espacio latente, sin traducirlo a capacidad."""
    if latent["status"] != STATUS_EXECUTED:
        return _section(
            "latent", "6. Diagnóstico del espacio latente",
            [_fields_block([("Diagnóstico latente",
                             missing(latent["status"], latent["reason"],
                                     source=latent["source"]))])],
            status=latent["status"], reason=latent["reason"],
        )
    src = latent["source"]
    rows = [
        ("Dimensiones latentes totales", field(latent["latent_dimensions"], source=src)),
        ("Dimensiones activas", field(latent["active_units"], source=src)),
        ("Fracción de unidades inactivas o colapsadas",
         field(_fmt_float(latent["collapsed_fraction"]), source=src)),
        ("KL global (promedio de las unidades)",
         field(_fmt_float(latent["mean_kl"]), source=src)
         if latent["mean_kl"] is not None else
         missing(STATUS_UNAVAILABLE, "La corrida no expuso KL por unidad.", source=src)),
        ("Umbral de actividad",
         field(latent["active_threshold"],
               source="ifvae_diagnostics/resolved_config.json::active_variance_threshold")
         if latent["active_threshold"] is not None else
         missing(STATUS_UNAVAILABLE, "La configuración no expuso el umbral.",
                 source="ifvae_diagnostics/resolved_config.json")),
        ("Origen del umbral de actividad",
         field("Configuración de la suite diagnóstica",
               source="ifvae_diagnostics/resolved_config.json")),
        ("Disponibilidad de mu", field(latent["mu_available"], source=src)),
        ("Disponibilidad de logvar", field(latent["logvar_available"], source=src)),
        ("Artefacto de origen", field(latent["artifact"], source=src)),
    ]
    unit_rows = [
        [i, _fmt_float(var), _fmt_float(kl)]
        for i, (var, kl) in enumerate(zip(latent["mu_variance_by_unit"],
                                          latent["mean_kl_by_unit"]))
    ]
    return _section("latent", "6. Diagnóstico del espacio latente", [
        _fields_block(rows),
        _table_block(["Unidad latente", "Varianza de mu", "KL media"], unit_rows,
                     title="Por unidad latente",
                     empty_text="La corrida no expuso métricas por unidad."),
        _note_block(
            "El criterio de unidad activa compara la varianza de mu contra el "
            "umbral configurado, calculado sobre la población de referencia. "
            "Describe el uso del código latente; no cuantifica capacidad de "
            "detección."
        ),
    ])


def _section_stability(stability: dict) -> dict:
    """§7 Estabilidad, un bloque por detector, con reajustes reales."""
    blocks = []
    for key, title in (("iforest", "Isolation Forest"), ("vae", "VAE")):
        block = stability[key]
        if block["status"] != STATUS_EXECUTED:
            blocks.append(_fields_block([
                ("Estado de ejecución",
                 missing(block["status"], block["reason"], source=block["source"])),
            ], title=title))
            continue
        blocks.append(_fields_block([
            ("Estado de ejecución", field(STATUS_LABELS[STATUS_EXECUTED],
                                          source=block["source"])),
            ("Semillas utilizadas", field(block.get("seeds"), source=block["source"])),
            ("Número de reajustes", field(block.get("refits"), source=block["source"])),
            ("Definición de top-K", field(block.get("top_k"), source=block["source"])),
            ("Jaccard medio", field(_fmt_float(block.get("mean_jaccard")),
                                    source=block["source"])),
            ("Jaccard mínimo", field(_fmt_float(block.get("min_jaccard")),
                                     source=block["source"])),
            ("Unidad de remuestreo", field(block.get("resampling_unit"),
                                           source=block["source"])),
            ("Artefacto de origen", field(block.get("artifact"), source=block["source"])),
        ], title=title))
    blocks.append(_note_block(
        "«Reajuste» significa reentrenar el detector con una semilla distinta "
        "sobre el mismo conjunto de entrenamiento, no revalidar contra verdad "
        "base. Un Jaccard bajo indica que el conjunto de alerta cambia al "
        "cambiar la semilla; no indica, por sí solo, que el detector esté mal "
        "calibrado -- ver la interpretación de esta sección para el contraste "
        "con la literatura de estabilidad de espacios latentes."
    ))
    return _section("stability", "7. Estabilidad", blocks)


def _section_temporal(temporal: dict, segmentation: dict) -> dict:
    """§8 Evolución temporal y segmentación, sin leer tendencias."""
    blocks = []
    if temporal["status"] == STATUS_EXECUTED:
        blocks.append(_table_block(
            ["Periodo", "Observaciones", "BOTH", "IF_ONLY", "VAE_ONLY", "NEITHER",
             "Tasa marginal IF", "Tasa marginal VAE", "Advertencia de tamaño"],
            temporal["rows"], title="Por periodo",
            caption=temporal["caption"],
            empty_text="La corrida no expuso periodos.",
        ))
    else:
        blocks.append(_fields_block([
            ("Resultados por periodo",
             missing(temporal["status"], temporal["reason"], source=temporal["source"])),
        ], title="Por periodo"))
    if segmentation["status"] == STATUS_EXECUTED:
        blocks.append(_table_block(
            ["Segmento", "Observaciones", "BOTH", "IF_ONLY", "VAE_ONLY", "NEITHER",
             "Tasa marginal IF", "Tasa marginal VAE", "Advertencia de tamaño"],
            segmentation["rows"], title="Por segmento",
            empty_text="La corrida no expuso segmentos.",
        ))
    else:
        blocks.append(_fields_block([
            ("Resultados por segmento",
             missing(segmentation["status"], segmentation["reason"],
                     source=segmentation["source"])),
        ], title="Por segmento"))
    blocks.append(_note_block(
        "Los grupos por debajo del tamaño mínimo declarado se marcan en la "
        "columna de advertencia. Las diferencias entre grupos no se "
        "interpretan en este apartado."
    ))
    return _section("temporal", "8. Evolución temporal y segmentación", blocks)


def _section_experiments(experiments: Sequence[dict]) -> dict:
    """§9 Matriz de experimentos: las familias que son genuinamente baratas o
    ya se calculan en otra parte se ejecutan de verdad para esta corrida
    (Ensembles, variantes de reconstrucción siempre; contaminación IF por
    defecto; capacidad/beta VAE si se configura una malla); las que
    requieren tocar código de modelo/preprocesamiento o múltiples ventanas
    temporales quedan `NOT_REQUESTED` con el motivo específico de cada una,
    no una limitación genérica."""
    rows = [
        [e["experiment"], STATUS_LABELS.get(e["status"], e["status"]) + f" ({e['status']})",
         e.get("detail") or NO_VALUE_TEXT, e.get("configuration") or NO_VALUE_TEXT,
         e.get("artifact") or NO_VALUE_TEXT, e.get("reason") or ""]
        for e in experiments
    ]
    return _section("experiments", "9. Experimentos diagnósticos", [
        _table_block(
            ["Experimento", "Estado", "Resultado", "Configuración",
             "Artefacto", "Motivo de no ejecución"],
            rows,
            caption="Familias tomadas de la matriz de experimentos de la suite "
                    "diagnóstica (evidence/EXPERIMENT_MATRIX.md). Se ejecutan "
                    "las que son baratas o ya se calculan en otra sección de "
                    "esta misma corrida (Ensembles, variantes de "
                    "reconstrucción, contaminación IF); las que exigen tocar "
                    "código de modelo o preprocesamiento, o comparar entre "
                    "múltiples ventanas OOT que esta corrida no conserva, "
                    "quedan NOT_REQUESTED con el motivo puntual de cada una.",
            empty_text="No se declararon experimentos.",
        ),
    ])


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def build_diagnostic_contract(
    *,
    populations: dict,
    config: dict,
    run_meta: dict,
    agreement: dict,
    candidates: Sequence[dict],
    sensitivity: dict,
    latent: dict,
    stability: dict,
    temporal: dict,
    segmentation: dict,
    experiments: Sequence[dict],
) -> dict:
    """Assemble the versioned contract from already-computed inputs.

    Every argument is data that ``src/evaluation/ifvae_diagnostic.py`` read
    from the suite's own artifacts or derived arithmetically from them. This
    function adds structure, labels, and neutral definitions -- never a new
    measurement and never a conclusion.
    """
    return {
        "contract_version": CONTRACT_VERSION,
        "id": "diagnostic-suite",
        "title": "Diagnóstico cruzado IF-VAE",
        "subtitle": (
            "Ficha estructurada de diagnóstico no supervisado: qué se evaluó, "
            "con qué configuración, sobre qué población y qué diagnósticos "
            "están disponibles. Sin interpretación -- ver el capítulo "
            "«Interpretación y recomendaciones» para la lectura heurística "
            "de estos mismos números."
        ),
        "sections": [
            _section_scope(populations, run_meta, config),
            _section_configuration(config, run_meta),
            _section_agreement(agreement, config, populations),
            _section_vae_candidates(candidates, config),
            _section_sensitivity(sensitivity),
            _section_latent(latent),
            _section_stability(stability),
            _section_temporal(temporal, segmentation),
            _section_experiments(experiments),
        ],
    }
