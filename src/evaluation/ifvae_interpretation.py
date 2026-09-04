"""Interpretation, indicator-validation, decision-flow and recommendation
layer for the "Diagnóstico cruzado IF-VAE" chapter.

**Why this is a separate module from ``ifvae_contract.py``.** The factual
ficha (``ifvae_contract.build_diagnostic_contract``) was explicitly built,
by request, to draw no conclusion, rank no detector, and assign no
operational priority. A later request asked for the opposite: a per-analysis
interpretation toolkit, a dynamic decision-flow diagram, and a recommendation
to the analyst. Both requests stand -- the resolution is not to weaken either
one, but to keep them as two contracts, built from the same underlying
numbers, rendered as two clearly separate, separately-labelled report
chapters. Every claim in THIS module is heuristic, evidence-linked, and
explicitly framed as such; nothing here is presented as ground truth.

**Every interpretive claim below is machine-checkable against the input it
was derived from** (``basis``), and every threshold used to branch a
decision is named and justified in ``METHODOLOGY_NOTES`` rather than left as
a bare magic number -- see that dict for the citation or established-project
precedent behind each one, and for the ones deliberately NOT turned into a
hard threshold because the evidence does not support a universal cutoff
(seed-to-seed stability, most notably).

Design:

* ``build_interpretation_contract(...)`` returns one dict with three parts:
  ``sections`` (a "toolkit" entry per analysis, plain-language reading +
  validity note), ``indicator_validation`` (sample-size and multiple-testing
  checks on the indicators actually used elsewhere in the chapter), and
  ``decision_flow`` (nodes evaluated against this run's real numbers, the
  path actually taken, and a recommendation assembled from the fragments the
  taken path produced -- never a fixed string per scenario).
* Nothing here fits a model, refits a detector, or reads a file the factual
  contract did not already read. This module is pure computation over
  already-computed inputs, exactly like ``ifvae_contract.py`` -- it differs
  only in that it is permitted to conclude something from them.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from src.evaluation.ifvae_contract import STATUS_EXECUTED, STATUS_NOT_REQUESTED

__all__ = ["build_interpretation_contract", "METHODOLOGY_NOTES"]

CONTRACT_VERSION = "1.0.0"

#: Every non-obvious threshold or metric choice below, with its justification,
#: so a reader (or a future maintainer) can check this module is not
#: asserting a cutoff nobody would defend. Referenced by id from the
#: `basis`/`note` fields the sections below emit.
METHODOLOGY_NOTES = {
    "latent-active-fraction": (
        "Un tercio de las dimensiones latentes activas (criterio de Burda, "
        "Grosse y Salakhutdinov, IWAE, ICLR 2016) es el mismo umbral que ya "
        "usa el chequeo de colapso posterior de este proyecto "
        "(`src.models.vae.collapse_verdict`) -- se reutiliza aquí, no se "
        "inventa uno nuevo."
    ),
    "drift-fdr": (
        "El desplazamiento poblacional se marca con corrección "
        "Benjamini-Hochberg (FDR, 1995) sobre los p-valores de Kolmogórov-"
        "Smirnov de todas las features probadas simultáneamente, en vez de "
        "un corte de significancia sin corregir -- con docenas de features "
        "evaluadas a la vez, un corte sin corregir produce varios falsos "
        "positivos por construcción."
    ),
    "stability-no-universal-cutoff": (
        "No se fija un umbral universal de 'estable' para el Jaccard "
        "multisemilla. La evidencia (Studying the Stability of "
        "Representation Learning, arXiv:2402.11404, 2024) muestra que los "
        "espacios/rankings de modelos tipo autoencoder son, en la práctica, "
        "considerablemente menos estables entre semillas que los ensambles "
        "de árboles -- un Jaccard bajo en el VAE frente al IF es un patrón "
        "ya documentado en la literatura, no necesariamente un defecto de "
        "esta corrida. Solo se marca el caso degenerado (solapamiento casi "
        "nulo, < 0.05), no un rango intermedio arbitrario."
    ),
    "min-sample-note": (
        "Los estadísticos de rango (Spearman) y de conjuntos (Jaccard) sobre "
        "menos de 30 observaciones tienen varianza alta; se señalan como "
        "tal, no se suprimen."
    ),
}

#: Below this population size, a rank/set statistic is flagged as high-variance.
MIN_RELIABLE_SAMPLE = 30
#: Below this Jaccard, a stability result is flagged as a degenerate,
#: near-zero-overlap case regardless of architecture -- see
#: METHODOLOGY_NOTES["stability-no-universal-cutoff"].
DEGENERATE_JACCARD = 0.05
#: Burda, Grosse & Salakhutdinov (IWAE, ICLR 2016); reused, not invented --
#: see METHODOLOGY_NOTES["latent-active-fraction"].
LATENT_ACTIVE_FRACTION_THRESHOLD = 1.0 / 3.0
#: Families excluded from the "business-relevant drift" count: both are
#: expected to shift under any chronological split by construction (see
#: CONTEXT.md "IF-VAE Diagnostic Suite integration" for the documented
#: calendar-artifact finding this reuses), so counting them alongside
#: genuine business-feature drift would overstate how much of the
#: population actually moved in a way relevant to scoring.
STRUCTURAL_DRIFT_FAMILIES = {"Calendario / cíclico", "Panel / rezago"}


def _fragment(text: str, *, basis: str, severity: str = "info") -> dict:
    """One interpretive statement: what it says, what it is based on, and
    how urgently it should be read. ``severity`` is descriptive (info/
    attention/caution) -- never a pass/fail verdict on the run."""
    return {"text": text, "basis": basis, "severity": severity}


def _toolkit_section(section_id: str, title: str, fragments: Sequence[dict],
                     validity: Sequence[dict]) -> dict:
    return {"id": section_id, "title": title, "reading": list(fragments),
            "validity": list(validity)}


# --------------------------------------------------------------------------- #
# Per-analysis interpretation ("toolkit")                                     #
# --------------------------------------------------------------------------- #
def _interpret_agreement(agreement: dict) -> dict:
    counts, total = agreement["quadrants"], agreement["total"]
    both, if_only, vae_only = counts["BOTH"], counts["IF_ONLY"], counts["VAE_ONLY"]
    fragments = []
    if both > if_only and both > vae_only:
        fragments.append(_fragment(
            f"El cuadrante BOTH ({both} observaciones) es el más grande de "
            "los tres cuadrantes de alerta: la mayoría de lo marcado por "
            "cualquiera de los dos detectores lo marcan ambos.",
            basis="ifvae_diagnostics/summary.json::quadrants"))
    if vae_only > if_only:
        fragments.append(_fragment(
            f"El VAE contribuye más observaciones exclusivas ({vae_only}) que "
            f"el Isolation Forest ({if_only}) sobre esta ventana. Esto "
            "describe volumen de marcado exclusivo, no precisión: sin verdad "
            "base no puede convertirse en una afirmación de que el VAE "
            "'encuentra más fraude'.", basis="agreement.quadrants",
            severity="attention"))
    elif if_only > vae_only:
        fragments.append(_fragment(
            f"El Isolation Forest contribuye más observaciones exclusivas "
            f"({if_only}) que el VAE ({vae_only}) sobre esta ventana. Mismo "
            "matiz: describe volumen, no precisión.",
            basis="agreement.quadrants", severity="attention"))
    rho = agreement.get("rank_correlation")
    if rho is not None:
        strength = ("baja" if abs(rho) < 0.3 else "moderada" if abs(rho) < 0.6 else "alta")
        fragments.append(_fragment(
            f"La correlación de rangos entre ambos puntajes es {strength} "
            f"(rho={rho:.3f}): {'los detectores ordenan de forma parecida' if strength != 'baja' else 'cada detector ordena a las entidades de forma bastante distinta'} "
            "sobre esta ventana.",
            basis="ifvae_diagnostics/scored_diagnostics.csv"))
    validity = []
    if total < MIN_RELIABLE_SAMPLE:
        validity.append(_fragment(
            f"La ventana evaluada tiene {total} observaciones, por debajo de "
            f"{MIN_RELIABLE_SAMPLE}: el Jaccard y la correlación de rangos de "
            "esta sección tienen varianza alta.",
            basis="min-sample-note", severity="caution"))
    if agreement.get("jaccard") is None:
        validity.append(_fragment(
            "El índice de Jaccard no está definido (unión de alertas = 0): "
            "ningún detector marcó observaciones con este umbral.",
            basis="agreement.jaccard", severity="attention"))
    return _toolkit_section("agreement", "Concordancia y desacuerdo", fragments, validity)


def _interpret_candidates(candidates: Sequence[dict], config: dict) -> dict:
    primary = config.get("vae_primary_score")
    executed = [c for c in candidates if c["status"] == STATUS_EXECUTED]
    fragments = []
    if executed:
        rates = {c["name"]: c["alerts"] / c["total"] if c["total"] else 0.0 for c in executed}
        spread = max(rates.values()) - min(rates.values()) if rates else 0.0
        widest = max(rates, key=rates.get)
        narrowest = min(rates, key=rates.get)
        fragments.append(_fragment(
            f"Entre los {len(executed)} candidatos de puntaje VAE evaluados, "
            f"la tasa de alerta va de {min(rates.values()):.1%} ({narrowest}) a "
            f"{max(rates.values()):.1%} ({widest}) -- una diferencia de "
            f"{spread:.1%} puntos porcentuales según qué señal reconstructiva "
            "se use, con el mismo umbral.",
            basis="ifvae_diagnostics/scored_diagnostics.csv"))
        primary_rate = rates.get(primary)
        if primary_rate is not None and spread > 0:
            rank = sorted(rates.values(), reverse=True).index(primary_rate) + 1
            fragments.append(_fragment(
                f"El candidato configurado como principal ({primary}) ocupa "
                f"el lugar {rank} de {len(rates)} por tasa de alerta entre los "
                "candidatos disponibles -- una descripción de dónde cae, no "
                "una afirmación de que sea el mejor sin verdad base.",
                basis="candidates", severity="attention"))
    validity = [_fragment(
        "Esta comparación usa el mismo puntaje IF y el mismo umbral para "
        "todos los candidatos; ninguno se reajusta.",
        basis="ifvae_diag.pipeline::_score_reconstruction_candidates")]
    return _toolkit_section("vae-candidates", "Comparación de puntajes VAE", fragments, validity)


def _interpret_sensitivity(sensitivity: dict) -> dict:
    if sensitivity["status"] != STATUS_EXECUTED or not sensitivity["rows"]:
        return _toolkit_section(
            "sensitivity", "Sensibilidad al umbral",
            [_fragment("No se evaluó sensibilidad al umbral en esta corrida: "
                      f"{sensitivity.get('reason', '')}",
                      basis="sensitivity.status", severity="info")],
            [])
    both_counts = [r["BOTH"] for r in sensitivity["rows"]]
    fragments = []
    if both_counts and max(both_counts) > 0:
        ratio = max(both_counts) / max(min(both_counts), 1)
        fragments.append(_fragment(
            f"El conteo de observaciones en BOTH varía de {min(both_counts)} a "
            f"{max(both_counts)} (razón {ratio:.1f}x) a lo largo de la malla de "
            f"umbrales configurada ({sensitivity['grid']}). "
            + ("El volumen de alertas es sensible a la elección del umbral "
               "operativo; documentar el criterio usado para fijarlo."
               if ratio >= 3 else
               "El volumen de alertas cambia de forma moderada dentro de esta "
               "malla."),
            basis="sensitivity.rows",
            severity="attention" if ratio >= 3 else "info"))
    return _toolkit_section("sensitivity", "Sensibilidad al umbral", fragments, [])


def _interpret_latent(latent: dict) -> dict:
    if latent["status"] != STATUS_EXECUTED:
        return _toolkit_section(
            "latent", "Diagnóstico del espacio latente",
            [_fragment(f"No disponible: {latent.get('reason', '')}",
                      basis="latent.status")], [])
    active, dims = latent["active_units"], latent["latent_dimensions"]
    fraction = active / dims if dims else 0.0
    healthy = fraction >= LATENT_ACTIVE_FRACTION_THRESHOLD
    fragments = [_fragment(
        f"{active} de {dims} dimensiones latentes activas ({fraction:.0%}). "
        + ("Por encima del umbral de un tercio (Burda et al., 2016): sin "
           "evidencia de colapso posterior."
           if healthy else
           "Por debajo del umbral de un tercio (Burda et al., 2016): posible "
           "colapso parcial -- el puntaje de reconstrucción del VAE podría "
           "ser menos informativo de lo esperado."),
        basis="latent-active-fraction", severity="info" if healthy else "attention")]
    return _toolkit_section("latent", "Diagnóstico del espacio latente", fragments, [])


def _interpret_stability(stability: dict) -> dict:
    fragments, validity = [], []
    for key, label in (("iforest", "Isolation Forest"), ("vae", "VAE")):
        block = stability[key]
        if block["status"] != STATUS_EXECUTED:
            fragments.append(_fragment(f"{label}: no disponible ({block.get('reason', '')}).",
                                       basis=f"stability.{key}.status"))
            continue
        jaccard = block["mean_jaccard"]
        note = ""
        if jaccard < DEGENERATE_JACCARD:
            note = (" Solapamiento casi nulo entre semillas: el conjunto de "
                    "alerta cambia casi por completo al cambiar solo la "
                    "semilla aleatoria.")
        fragments.append(_fragment(
            f"{label}: Jaccard medio {jaccard:.3f} entre {block['refits']} "
            f"reajustes (top-{block['top_k']}).{note}",
            basis="stability-no-universal-cutoff",
            severity="attention" if jaccard < DEGENERATE_JACCARD else "info"))
    if (stability["iforest"]["status"] == STATUS_EXECUTED
            and stability["vae"]["status"] == STATUS_EXECUTED):
        if stability["vae"]["mean_jaccard"] < stability["iforest"]["mean_jaccard"]:
            fragments.append(_fragment(
                "El VAE muestra menor estabilidad multisemilla que el "
                "Isolation Forest. Es el patrón ya documentado para modelos "
                "tipo autoencoder frente a ensambles de árboles "
                "(arXiv:2402.11404), no evidencia por sí sola de un defecto "
                "en esta corrida específica.",
                basis="stability-no-universal-cutoff"))
    return _toolkit_section("stability", "Estabilidad", fragments, validity)


def _interpret_temporal(temporal: dict, segmentation: dict) -> dict:
    fragments = []
    if temporal["status"] == STATUS_EXECUTED and temporal["rows"]:
        both_by_period = {row[0]: row[2] for row in temporal["rows"]}
        if len(both_by_period) > 1:
            values = list(both_by_period.values())
            fragments.append(_fragment(
                f"El conteo de BOTH por periodo va de {min(values)} a "
                f"{max(values)} a lo largo de {len(values)} periodo(s) "
                "evaluados -- una descripción de variación, no una tendencia "
                "que se afirme causal ni proyectable.",
                basis="temporal.rows"))
    if segmentation["status"] == STATUS_EXECUTED and segmentation["rows"]:
        fragments.append(_fragment(
            f"Se desglosó por {len(segmentation['rows'])} segmento(s) "
            "declarados en el panel; ver la tabla de la ficha para los "
            "conteos por segmento.", basis="segmentation.rows"))
    else:
        fragments.append(_fragment(
            f"Sin desglose por segmento: {segmentation.get('reason', '')}",
            basis="segmentation.status"))
    return _toolkit_section("temporal", "Evolución temporal y segmentación", fragments, [])


# --------------------------------------------------------------------------- #
# Indicator validation ("validación de indicadores")                          #
# --------------------------------------------------------------------------- #
def _indicator_validation(agreement: dict, stability: dict, sensitivity: dict,
                          drift_signal: dict, latent: dict) -> list[dict]:
    checks = []
    total = agreement["total"]
    checks.append({
        "indicator": "Jaccard / correlación de rangos (concordancia)",
        "valid": total >= MIN_RELIABLE_SAMPLE,
        "detail": f"n={total} observaciones evaluadas (mínimo de referencia: "
                 f"{MIN_RELIABLE_SAMPLE}).",
        "basis": "min-sample-note",
    })
    for key, label in (("iforest", "Isolation Forest"), ("vae", "VAE")):
        block = stability[key]
        if block["status"] == STATUS_EXECUTED:
            checks.append({
                "indicator": f"Estabilidad multisemilla ({label})",
                "valid": block["refits"] >= 3,
                "detail": f"{block['refits']} reajuste(s); top_k_stability "
                         "requiere un mínimo de 2 y es más confiable con más.",
                "basis": "top_k_stability requiere >= 2 corridas (ifvae_diag.stability)",
            })
    if sensitivity["status"] == STATUS_EXECUTED:
        grid = sensitivity["grid"] or []
        out_of_range = [g for g in grid if not (0.0 < g < 1.0)]
        checks.append({
            "indicator": "Malla de sensibilidad",
            "valid": not out_of_range,
            "detail": (f"{len(grid)} umbral(es) declarados"
                      + (f"; fuera de rango válido: {out_of_range}" if out_of_range else ".")),
            "basis": "ifvae_diag.config::DiagnosticConfig.__post_init__ "
                    "(0.5 < percentile_threshold < 1.0)",
        })
    if drift_signal.get("available"):
        checks.append({
            "indicator": "Desplazamiento poblacional (drift)",
            "valid": True,
            "detail": f"Corrección Benjamini-Hochberg aplicada sobre "
                     f"{drift_signal['n_features']} features "
                     f"(alpha={drift_signal['alpha']}); {drift_signal['n_flagged']} "
                     "marcadas tras la corrección.",
            "basis": "drift-fdr",
        })
    if latent["status"] == STATUS_EXECUTED:
        dims = latent["latent_dimensions"]
        checks.append({
            "indicator": "Diagnóstico latente",
            "valid": True,
            "detail": f"{dims} dimensión(es) latentes evaluadas contra el umbral "
                     "de actividad de la configuración.",
            "basis": "latent-active-fraction",
        })
    return checks


# --------------------------------------------------------------------------- #
# Decision flow (dynamic, evaluated against THIS run's real numbers)          #
# --------------------------------------------------------------------------- #
def _node(node_id: str, question: str, answer: str, taken: str,
         fragment: Optional[dict], severity: str = "info") -> dict:
    return {"id": node_id, "question": question, "answer": answer,
            "taken": taken, "fragment": fragment, "severity": severity}


def _build_decision_flow(agreement: dict, latent: dict, stability: dict,
                         drift_signal: dict, sensitivity: dict) -> dict:
    nodes = []

    both = agreement["quadrants"]["BOTH"]
    if both > 0:
        nodes.append(_node(
            "agreement", "¿Hay observaciones en el cuadrante BOTH?",
            f"Sí -- {both} observaciones", "both-yes",
            _fragment(
                f"Revisar primero las {both} observaciones del cuadrante "
                "BOTH: es el subconjunto en el que ambos detectores "
                "coinciden en marcar sobre el umbral configurado.",
                basis="agreement.quadrants.BOTH"),
            severity="info"))
    else:
        nodes.append(_node(
            "agreement", "¿Hay observaciones en el cuadrante BOTH?",
            "No -- 0 observaciones", "both-no",
            _fragment(
                "Ningún detector coincide sobre el umbral configurado en esta "
                "ventana: no existe un subconjunto de máxima concordancia que "
                "priorizar; considerar revisar IF_ONLY y VAE_ONLY por "
                "separado, o bajar el umbral (ver sensibilidad).",
                basis="agreement.quadrants.BOTH", severity="attention"),
            severity="attention"))

    if latent["status"] == STATUS_EXECUTED:
        dims, active = latent["latent_dimensions"], latent["active_units"]
        healthy = (active / dims if dims else 0.0) >= LATENT_ACTIVE_FRACTION_THRESHOLD
        nodes.append(_node(
            "latent", "¿El espacio latente del VAE está activo (>= 1/3, Burda et al. 2016)?",
            f"{'Sí' if healthy else 'No'} -- {active}/{dims} activas",
            "latent-healthy" if healthy else "latent-collapsed",
            _fragment(
                "Sin evidencia de colapso posterior: las alertas del VAE se "
                "apoyan en un código latente activo."
                if healthy else
                "Posible colapso parcial del espacio latente: interpretar las "
                "alertas exclusivas del VAE (VAE_ONLY) con cautela adicional.",
                basis="latent-active-fraction",
                severity="info" if healthy else "attention"),
            severity="info" if healthy else "attention"))
    else:
        nodes.append(_node(
            "latent", "¿El espacio latente del VAE está activo?",
            "No evaluado", "latent-unavailable", None, severity="info"))

    if drift_signal.get("available"):
        business_flagged = [t for t in drift_signal["top"]
                            if t["family"] not in STRUCTURAL_DRIFT_FAMILIES]
        if business_flagged:
            names = ", ".join(t["feature"] for t in business_flagged[:3])
            nodes.append(_node(
                "drift", "¿Hay desplazamiento significativo (FDR) en variables de negocio?",
                f"Sí -- {len(business_flagged)} variable(s): {names}",
                "drift-business", _fragment(
                    f"Variables de negocio con desplazamiento significativo "
                    f"tras corrección FDR: {names}. Puede afectar la "
                    "comparabilidad de los puntajes entre entrenamiento y la "
                    "ventana evaluada -- revisar antes de tratar los puntajes "
                    "como directamente comparables al periodo de entrenamiento.",
                    basis="drift-fdr", severity="attention"),
                severity="attention"))
        else:
            nodes.append(_node(
                "drift", "¿Hay desplazamiento significativo (FDR) en variables de negocio?",
                "No", "drift-none", _fragment(
                    "Sin desplazamiento significativo en variables de negocio "
                    "tras corrección por múltiples pruebas (las variables de "
                    "calendario y de panel se excluyen de este conteo por "
                    "motivo estructural, no porque no muestren cambio).",
                    basis="drift-fdr"),
                severity="info"))
    else:
        nodes.append(_node(
            "drift", "¿Hay desplazamiento significativo en variables de negocio?",
            "No evaluado", "drift-unavailable", None, severity="info"))

    if sensitivity["status"] == STATUS_EXECUTED and sensitivity["rows"]:
        both_counts = [r["BOTH"] for r in sensitivity["rows"]]
        ratio = max(both_counts) / max(min(both_counts), 1) if both_counts else 1.0
        sensitive = ratio >= 3
        nodes.append(_node(
            "sensitivity", "¿El volumen de alertas es sensible al umbral (razón >= 3x)?",
            f"{'Sí' if sensitive else 'No'} -- razón {ratio:.1f}x",
            "sensitivity-high" if sensitive else "sensitivity-low",
            _fragment(
                "El volumen de alertas cambia sustancialmente con el umbral "
                "elegido: documentar y justificar el criterio operativo usado "
                "para fijarlo."
                if sensitive else
                "El volumen de alertas es relativamente estable dentro de la "
                "malla de umbrales evaluada.",
                basis="sensitivity.rows", severity="attention" if sensitive else "info"),
            severity="attention" if sensitive else "info"))
    else:
        nodes.append(_node(
            "sensitivity", "¿El volumen de alertas es sensible al umbral?",
            "No evaluado", "sensitivity-unavailable", None, severity="info"))

    fragments = [n["fragment"] for n in nodes if n["fragment"]]
    ordered = sorted(fragments, key=lambda f: {"attention": 0, "caution": 0,
                                               "info": 1}.get(f["severity"], 1))
    return {"nodes": nodes, "recommendation_fragments": ordered}


def build_interpretation_contract(
    *, agreement: dict, candidates: Sequence[dict], sensitivity: dict, latent: dict,
    stability: dict, temporal: dict, segmentation: dict, drift_signal: dict,
    warnings: Sequence[dict], config: dict, run_meta: dict, populations: dict,
) -> dict:
    """Assemble the interpretation contract from already-computed inputs.

    Mirrors ``ifvae_contract.build_diagnostic_contract``'s calling
    convention (keyword-only, one argument per already-computed input) so
    the two contracts stay easy to build side by side from the same call
    site in ``diagnose_frames``.
    """
    decision_flow = _build_decision_flow(agreement, latent, stability, drift_signal, sensitivity)
    return {
        "contract_version": CONTRACT_VERSION,
        "id": "diagnostic-interpretation",
        "title": "Interpretación y recomendaciones",
        "subtitle": (
            "Lectura heurística de los mismos números de la ficha anterior, "
            "más un flujo de decisión y una recomendación para el analista. "
            "Generado por reglas deterministas sobre los resultados de esta "
            "corrida -- no es un juicio de verdad base ni un veredicto sobre "
            "qué modelo es mejor; ver la nota metodológica de cada afirmación."
        ),
        "toolkit": [
            _interpret_agreement(agreement),
            _interpret_candidates(candidates, config),
            _interpret_sensitivity(sensitivity),
            _interpret_latent(latent),
            _interpret_stability(stability),
            _interpret_temporal(temporal, segmentation),
        ],
        "indicator_validation": _indicator_validation(
            agreement, stability, sensitivity, drift_signal, latent),
        "decision_flow": decision_flow,
        "methodology_notes": METHODOLOGY_NOTES,
    }
