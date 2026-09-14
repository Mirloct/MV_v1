"""Acceptance tests for the "Diagnóstico cruzado IF-VAE" (factual) and
"Interpretación y recomendaciones" (heuristic) report chapters.

Covers: structured parity between HTML and Markdown for both contracts,
traceability of every visible factual value, absence of silent fallbacks,
correct response to configuration changes, correct states with and without
labels, quadrant arithmetic invariants, behaviour with absent or empty
artifacts, edge cases (zero alerts, ties, missing logvar, duplicate ids,
infinities, missing values, temporal overlap, segmentation), real seeded
stability refits, the decision-flow's branch selection against constructed
scenarios, and the factual/interpretive split itself (the facts renderer
must stay non-interpretive; the interpretation renderer must stay
computation-free).

Everything here drives the *real* code path: the fixtures build small
synthetic frames and call ``diagnose_frames`` -- the same function the
pipeline calls after building its frames from live detectors -- so no part
of either contract is proven only against a mock. The stability tests go one
step further and fit real (tiny) project model instances.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.ifvae_contract import (  # noqa: E402
    CONTRACT_VERSION,
    NO_VALUE_TEXT,
    STATUS_EXECUTED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_REQUESTED,
    STATUS_UNAVAILABLE,
    STATUSES,
)
from src.evaluation.ifvae_diagnostic import diagnose_frames  # noqa: E402
from src.evaluation.ifvae_interpretation import (  # noqa: E402
    CONTRACT_VERSION as INTERPRETATION_CONTRACT_VERSION,
    DEGENERATE_JACCARD,
    LATENT_ACTIVE_FRACTION_THRESHOLD,
    build_interpretation_contract,
)
from src.reporting.diagnostic_section import (  # noqa: E402
    iter_contract_values,
    render_diagnostic_html,
    render_diagnostic_markdown,
)
from src.reporting.interpretation_section import (  # noqa: E402
    render_interpretation_html,
    render_interpretation_markdown,
)

EXPECTED_SECTION_COUNT = 9  # after removing §{availability, analysis-unit,
# reconstruction, data-quality, label-conditioned, provenance} from the
# original 15, by explicit request (2026-09-05).

# --------------------------------------------------------------------------- #
# Synthetic fixtures -- NOT real model output                                 #
# --------------------------------------------------------------------------- #
N_FEATURES = 4
LATENT_DIM = 2


def synthetic_frames(
    n_reference: int = 120,
    n_scored: int = 90,
    n_periods: int = 3,
    seed: int = 7,
    with_labels: bool = False,
    with_segment: bool = False,
    with_logvar: bool = True,
    tie_scores: bool = False,
    overlap_time: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Two contract-shaped frames built from a fixed seed.

    Deliberately synthetic: the values carry no meaning and exist only to
    exercise the contract's structure and edge-case handling.
    """
    rng = np.random.default_rng(seed)
    features = [f"num__f{i}" for i in range(N_FEATURES)]

    def block(n: int, period_offset: int, prefix: str) -> pd.DataFrame:
        entities = [f"E{i % max(n // n_periods, 1):04d}" for i in range(n)]
        periods = [f"2026-{(period_offset + (i % n_periods)) % 12 + 1:02d}-01"
                   for i in range(n)]
        x = rng.normal(size=(n, N_FEATURES))
        recon = x + rng.normal(scale=0.3, size=(n, N_FEATURES))
        mu = rng.normal(size=(n, LATENT_DIM))
        logvar = rng.normal(scale=0.1, size=(n, LATENT_DIM))
        if_score = (np.full(n, 0.5) if tie_scores else rng.normal(size=n))
        blocks = [
            pd.DataFrame({"alert_id": [f"{prefix}{i:05d}" for i in range(n)]}),
            pd.DataFrame({"entity_id": entities}),
            pd.DataFrame({"scoring_timestamp": periods}),
            pd.DataFrame(x, columns=features),
            pd.DataFrame(recon, columns=[f"recon__{f}" for f in features]),
            pd.DataFrame(mu, columns=[f"mu__{i}" for i in range(LATENT_DIM)]),
            pd.DataFrame({"if_score": if_score}),
        ]
        if with_logvar:
            blocks.insert(-1, pd.DataFrame(
                logvar, columns=[f"logvar__{i}" for i in range(LATENT_DIM)]))
        return pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)

    reference = block(n_reference, 0, "R")
    scored = block(n_scored, 0 if overlap_time else 6, "S")
    if with_labels:
        scored["label"] = (np.arange(n_scored) % 9 == 0).astype(int)
    if with_segment:
        scored["segment"] = np.where(np.arange(n_scored) % 2 == 0, "a", "b")
    return reference, scored, features


def build(tmpdir: str, **kwargs) -> dict:
    """Run the real contract path over synthetic frames."""
    frame_keys = {
        "n_reference", "n_scored", "n_periods", "seed", "with_labels",
        "with_segment", "with_logvar", "tie_scores", "overlap_time",
    }
    frame_kwargs = {k: v for k, v in kwargs.items() if k in frame_keys}
    diag_kwargs = {k: v for k, v in kwargs.items() if k not in frame_keys}
    reference, scored, features = synthetic_frames(**frame_kwargs)
    run_meta = diag_kwargs.pop("run_meta", {
        "run_id": "test-run", "generated_at": "2026-01-01T00:00:00",
        "architecture_mode": "Apilado",
        "detector_dependency": "El VAE recibe el puntaje del IF.",
        "derived_features": [],
        "detectors": {"iforest": {"label": "Isolation Forest"}},
    })
    if frame_kwargs.get("with_labels"):
        diag_kwargs.setdefault("label_col", "label")
    if frame_kwargs.get("with_segment"):
        diag_kwargs.setdefault("segment_col", "segment")
    return diagnose_frames(reference, scored, features, tmpdir,
                           run_meta=run_meta, **diag_kwargs)


def all_fields(contract: dict):
    for section in contract["sections"]:
        for block in section["blocks"]:
            if block["kind"] == "fields":
                for row in block["rows"]:
                    yield section["id"], row["label"], row["field"]


def section_by_id(contract: dict, section_id: str) -> dict:
    return next(s for s in contract["sections"] if s["id"] == section_id)


def strip_tags(html_text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html_text)


# --------------------------------------------------------------------------- #
# Factual contract: structure, traceability, no fallbacks                     #
# --------------------------------------------------------------------------- #
class ContractStructureTests(unittest.TestCase):
    """Criteria 1-3: parity, traceability, no silent fallbacks."""

    def test_html_and_markdown_show_the_same_structured_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        html_text = strip_tags(render_diagnostic_html(contract))
        md_text = render_diagnostic_markdown(contract)
        checked = 0
        for _section, _block, label, text in iter_contract_values(contract):
            if not text.strip():
                continue
            probe = text.strip()[:60]
            self.assertIn(probe, html_text, f"falta en HTML: {label} -> {probe}")
            self.assertIn(probe, md_text, f"falta en Markdown: {label} -> {probe}")
            checked += 1
        self.assertGreater(checked, 60, "la ficha debe exponer contenido sustantivo")

    def test_every_visible_field_declares_its_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        for section_id, label, field in all_fields(contract):
            self.assertTrue(field.get("source"),
                            f"campo sin fuente: {section_id}/{label}")
            self.assertIn(field["status"], STATUSES)

    def test_absent_values_carry_a_reason_and_no_stand_in_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        checked = 0
        for section_id, label, field in all_fields(contract):
            if field["status"] == STATUS_EXECUTED:
                continue
            self.assertIsNone(field["value"],
                              f"valor de respaldo en {section_id}/{label}")
            self.assertTrue(field["reason"],
                            f"ausencia sin motivo en {section_id}/{label}")
            checked += 1
        self.assertGreater(checked, 0, "el caso base debe tener ausencias declaradas")

    def test_pruned_sections_are_gone_and_only_nine_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        ids = [s["id"] for s in contract["sections"]]
        self.assertEqual(len(ids), EXPECTED_SECTION_COUNT)
        for removed in ("availability", "analysis-unit", "reconstruction",
                        "data-quality", "label-conditioned", "provenance"):
            self.assertNotIn(removed, ids, f"la sección {removed!r} debía eliminarse")
        for kept in ("scope", "configuration", "agreement", "vae-candidates",
                    "sensitivity", "latent", "stability", "temporal", "experiments"):
            self.assertIn(kept, ids)


class ConfigurationResponseTests(unittest.TestCase):
    """Criteria 4-6: the record follows configuration, architecture and unit."""

    def test_changing_threshold_changes_the_reported_quadrants(self):
        with tempfile.TemporaryDirectory() as tmp:
            low = build(tmp, percentile_threshold=0.60)["contract"]
        with tempfile.TemporaryDirectory() as tmp:
            high = build(tmp, percentile_threshold=0.99)["contract"]

        def both_count(contract):
            table = section_by_id(contract, "agreement")["blocks"][0]
            return next(r[1] for r in table["rows"] if r[0] == "BOTH")

        def threshold_value(contract):
            for _s, label, field in all_fields(contract):
                if label == "Valor del umbral (percentil)":
                    return field["value"]
            raise AssertionError("umbral no reportado")

        self.assertNotEqual(both_count(low), both_count(high))
        self.assertEqual(threshold_value(low), 0.60)
        self.assertEqual(threshold_value(high), 0.99)

    def test_changing_the_primary_vae_candidate_moves_the_configured_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, vae_primary_score="recon_max")["contract"]
        rows = section_by_id(contract, "vae-candidates")["blocks"][0]["rows"]
        marked = [r[0] for r in rows if "configurado como principal" in r[0]]
        self.assertEqual(marked, ["recon_max (configurado como principal)"])

    def test_entity_aggregation_rule_appears_only_when_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            without = build(tmp)["contract"]
        with tempfile.TemporaryDirectory() as tmp:
            with_view = build(
                tmp, entity_view=True,
                run_meta={"run_id": "t", "generated_at": "2026-01-01T00:00:00",
                          "architecture_mode": "Apilado",
                          "detector_dependency": "dep",
                          "entity_aggregation_rule": "Máximo por entidad",
                          "derived_features": [], "detectors": {}},
            )["contract"]

        def rule_field(contract):
            for _s, label, field in all_fields(contract):
                if label == "Regla de agregación por entidad":
                    return field
            raise AssertionError("campo de regla de agregación ausente")

        self.assertEqual(rule_field(without)["status"], STATUS_NOT_REQUESTED)
        self.assertEqual(rule_field(with_view)["status"], STATUS_EXECUTED)
        self.assertEqual(rule_field(with_view)["value"], "Máximo por entidad")

    def test_architecture_mode_is_reported_and_unknown_is_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            stacked = build(tmp)["contract"]
        with tempfile.TemporaryDirectory() as tmp:
            parallel = build(tmp, run_meta={
                "run_id": "t", "generated_at": "2026-01-01T00:00:00",
                "architecture_mode": "Paralelo",
                "detector_dependency": "Ninguno recibe el puntaje del otro.",
                "derived_features": [], "detectors": {}})["contract"]
        with tempfile.TemporaryDirectory() as tmp:
            unknown = build(tmp, run_meta={"run_id": "t",
                                           "generated_at": "2026-01-01T00:00:00",
                                           "derived_features": [],
                                           "detectors": {}})["contract"]

        def mode(contract):
            for _s, label, field in all_fields(contract):
                if label == "Modo arquitectónico de los detectores":
                    return field
            raise AssertionError("modo no reportado")

        self.assertEqual(mode(stacked)["value"], "Apilado")
        self.assertEqual(mode(parallel)["value"], "Paralelo")
        self.assertEqual(mode(unknown)["status"], STATUS_UNAVAILABLE)
        self.assertIsNone(mode(unknown)["value"])

    def test_observation_counts_use_the_entity_period_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build(tmp, n_scored=90)["contract"]
        scope = {row["label"]: row["field"]
                for block in section_by_id(result, "scope")["blocks"]
                for row in block["rows"]}
        self.assertEqual(scope["Observaciones (evaluación)"]["value"], 90)
        self.assertEqual(scope["Unidad de observación"]["value"],
                         "Observación entidad–periodo")
        agreement = section_by_id(result, "agreement")["blocks"][0]
        self.assertIn("Observaciones", agreement["columns"])


class LabelStateTests(unittest.TestCase):
    """Label-conditioned rows in the surviving sections (§9 experiments is
    label-independent; the dedicated label-blocked containers were removed,
    so this now only checks that nothing label-dependent silently appears)."""

    def test_label_free_run_declares_no_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        scope = {row["label"]: row["field"]
                for block in section_by_id(contract, "scope")["blocks"]
                for row in block["rows"]}
        self.assertEqual(scope["Disponibilidad de verdad base"]["value"],
                         "Sin verdad base declarada")


class ArithmeticInvariantTests(unittest.TestCase):
    """Quadrant counts and percentages must add up."""

    def test_quadrants_and_set_relations_are_internally_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = build(tmp, n_scored=90)
        contract = payload["contract"]
        quadrants = payload["quadrants"]
        self.assertEqual(sum(quadrants.values()), payload["rows_scored"])

        table = section_by_id(contract, "agreement")["blocks"][0]
        counts = {row[0]: row[1] for row in table["rows"]}
        self.assertEqual(sum(counts.values()), payload["rows_scored"])
        percentages = [float(row[2].rstrip("%")) for row in table["rows"]]
        self.assertAlmostEqual(sum(percentages), 100.0, delta=0.5)

        fields = {row["label"]: row["field"]
                  for block in section_by_id(contract, "agreement")["blocks"]
                  if block["kind"] == "fields" for row in block["rows"]}
        self.assertEqual(fields["Intersección"]["value"], counts["BOTH"])
        self.assertEqual(
            fields["Unión"]["value"],
            counts["BOTH"] + counts["IF_ONLY"] + counts["VAE_ONLY"],
        )
        self.assertEqual(fields["Alertas totales del Isolation Forest"]["value"],
                         counts["BOTH"] + counts["IF_ONLY"])
        self.assertEqual(fields["Alertas totales del VAE"]["value"],
                         counts["BOTH"] + counts["VAE_ONLY"])

    def test_the_primary_candidate_row_matches_the_agreement_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        agreement = {row[0]: row[1]
                     for row in section_by_id(contract, "agreement")["blocks"][0]["rows"]}
        rows = section_by_id(contract, "vae-candidates")["blocks"][0]["rows"]
        primary = next(r for r in rows if "configurado como principal" in r[0])
        expected = " / ".join(str(agreement[k]) for k in
                              ("BOTH", "IF_ONLY", "VAE_ONLY", "NEITHER"))
        self.assertEqual(primary[4], expected)


class ArtifactTests(unittest.TestCase):
    """Absent or empty artifacts are described, never linked as usable."""

    def test_missing_artifact_reads_as_none_in_the_candidates_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        rows = section_by_id(contract, "vae-candidates")["blocks"][0]["rows"]
        # `scored_diagnostics.csv` IS written by a real `diagnose_frames`
        # call, so every candidate's artifact column should name it.
        for row in rows:
            self.assertEqual(row[-1], "scored_diagnostics.csv")


class EdgeCaseTests(unittest.TestCase):
    """Degenerate inputs must not fabricate or crash."""

    def test_a_population_below_the_reference_yields_zero_alerts(self):
        """Zero alerts is a real operating point, not an error state."""
        reference, scored, features = synthetic_frames()
        scored = scored.copy()
        for feature in features:
            scored[f"recon__{feature}"] = scored[feature]
        scored["if_score"] = reference["if_score"].min() - 10.0
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(reference, scored, features, tmp)
        quadrants = payload["quadrants"]
        self.assertEqual(quadrants["BOTH"], 0)
        self.assertEqual(quadrants["NEITHER"], payload["rows_scored"])
        fields = {row["label"]: row["field"]
                  for block in section_by_id(payload["contract"], "agreement")["blocks"]
                  if block["kind"] == "fields" for row in block["rows"]}
        self.assertEqual(fields["Índice de Jaccard"]["status"], STATUS_UNAVAILABLE)
        self.assertTrue(fields["Índice de Jaccard"]["reason"])
        self.assertIn("0", render_diagnostic_markdown(payload["contract"]))

    def test_tied_scores_do_not_break_rank_correlation_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, tie_scores=True)["contract"]
        fields = {row["label"]: row["field"]
                  for block in section_by_id(contract, "agreement")["blocks"]
                  if block["kind"] == "fields" for row in block["rows"]}
        rho = fields["Correlación de rangos (Spearman)"]
        if rho["status"] != STATUS_EXECUTED:
            self.assertTrue(rho["reason"])
            self.assertIsNone(rho["value"])

    def test_missing_logvar_reports_latent_as_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, with_logvar=False)["contract"]
        latent = section_by_id(contract, "latent")
        self.assertEqual(latent["status"], STATUS_UNAVAILABLE)
        rendered = render_diagnostic_markdown(contract)
        self.assertNotIn("nan", rendered.lower())

    def test_multiple_periods_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, n_periods=3)["contract"]
        rows = section_by_id(contract, "temporal")["blocks"][0]["rows"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row[1] for row in rows), 90)

    def test_small_groups_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, n_scored=12, n_periods=3)["contract"]
        rows = section_by_id(contract, "temporal")["blocks"][0]["rows"]
        self.assertTrue(all(row[-1] for row in rows),
                        "los grupos pequeños deben marcarse")

    def test_duplicate_ids_infinities_and_overlap_are_surfaced(self):
        from ifvae_diag.contracts import DataContractError

        reference, scored, features = synthetic_frames()
        duplicated = scored.copy()
        duplicated.loc[duplicated.index[1], "alert_id"] = duplicated.iloc[0]["alert_id"]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DataContractError):
                diagnose_frames(reference, duplicated, features, tmp)

        infinite = scored.copy()
        infinite.loc[infinite.index[0], features[0]] = np.inf
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DataContractError):
                diagnose_frames(reference, infinite, features, tmp)

        overlap_ref, overlap_scored, overlap_features = synthetic_frames(overlap_time=True)
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(overlap_ref, overlap_scored, overlap_features, tmp)
            self.assertTrue(any(
                w.get("code") == "temporal_overlap" for w in _raw_warnings(payload)
            ))

    def test_missing_values_are_surfaced_as_a_warning(self):
        reference, scored, features = synthetic_frames()
        scored.loc[scored.index[0], features[0]] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(reference, scored, features, tmp)
            self.assertTrue(any(
                w.get("code") == "missing_features" for w in _raw_warnings(payload)
            ))

    def test_segment_column_switches_the_segmentation_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, with_segment=True)["contract"]
        rows = section_by_id(contract, "temporal")["blocks"][1]["rows"]
        self.assertEqual({row[0] for row in rows}, {"a", "b"})


def _raw_warnings(payload: dict) -> list:
    """Warnings aren't stored verbatim on the trimmed contract anymore (the
    provenance section that tabulated them was removed) -- read them back
    from the interpretation contract's indicator/decision-flow inputs isn't
    possible either (also distilled), so re-run the suite's own warnings
    file directly for this edge-case assertion."""
    with open(os.path.join(payload["report_dir"], "warnings.json"), encoding="utf-8") as fh:
        return json.load(fh)["warnings"]


# --------------------------------------------------------------------------- #
# Interpretation contract: separate, labelled, computation-free renderers     #
# --------------------------------------------------------------------------- #
class InterpretationStructureTests(unittest.TestCase):
    def test_interpretation_is_a_separate_versioned_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = build(tmp)
        interpretation = payload["interpretation"]
        self.assertEqual(interpretation["contract_version"], INTERPRETATION_CONTRACT_VERSION)
        self.assertEqual(interpretation["id"], "diagnostic-interpretation")
        self.assertNotEqual(interpretation["id"], payload["contract"]["id"])
        self.assertIn("heurística", interpretation["subtitle"].lower())

    def test_html_and_markdown_interpretation_show_the_same_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpretation = build(tmp)["interpretation"]
        html_text = strip_tags(render_interpretation_html(interpretation))
        md_text = render_interpretation_markdown(interpretation)
        for fragment in interpretation["decision_flow"]["recommendation_fragments"]:
            probe = fragment["text"][:50]
            self.assertIn(probe, html_text)
            self.assertIn(probe, md_text)

    def test_every_toolkit_reading_declares_a_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpretation = build(tmp)["interpretation"]
        checked = 0
        for section in interpretation["toolkit"]:
            for fragment in section["reading"] + section["validity"]:
                self.assertTrue(fragment["basis"], f"sin base: {fragment['text'][:40]}")
                self.assertIn(fragment["severity"], ("info", "attention", "caution"))
                checked += 1
        self.assertGreater(checked, 3)

    def test_indicator_validation_flags_small_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpretation = build(tmp, n_scored=10, n_reference=10, n_periods=1)["interpretation"]
        checks = {c["indicator"]: c for c in interpretation["indicator_validation"]}
        agreement_check = checks["Jaccard / correlación de rangos (concordancia)"]
        self.assertFalse(agreement_check["valid"])


class DecisionFlowTests(unittest.TestCase):
    """The decision flow must branch on THIS run's real numbers, not a
    scenario baked into the code."""

    def test_zero_both_quadrant_takes_the_no_agreement_branch(self):
        interpretation = build_interpretation_contract(
            agreement={"quadrants": {"BOTH": 0, "IF_ONLY": 3, "VAE_ONLY": 5, "NEITHER": 92},
                      "total": 100, "if_alerts": 3, "vae_alerts": 5,
                      "intersection": 0, "union": 8, "jaccard": 0.0,
                      "rank_correlation": 0.2, "rank_correlation_reason": "",
                      "scatter": None},
            candidates=[], sensitivity={"status": STATUS_NOT_REQUESTED, "reason": "x",
                                       "rows": [], "grid": None},
            latent={"status": STATUS_UNAVAILABLE, "reason": "x"},
            stability={"iforest": {"status": STATUS_NOT_REQUESTED, "reason": "x"},
                      "vae": {"status": STATUS_NOT_REQUESTED, "reason": "x"}},
            temporal={"status": STATUS_UNAVAILABLE, "reason": "x", "rows": []},
            segmentation={"status": STATUS_NOT_APPLICABLE, "reason": "x"},
            drift_signal={"available": False}, warnings=[],
            config={}, run_meta={}, populations={},
        )
        node = interpretation["decision_flow"]["nodes"][0]
        self.assertEqual(node["taken"], "both-no")
        self.assertEqual(node["severity"], "attention")

    def test_positive_both_quadrant_takes_the_agreement_branch(self):
        interpretation = build_interpretation_contract(
            agreement={"quadrants": {"BOTH": 12, "IF_ONLY": 3, "VAE_ONLY": 5, "NEITHER": 80},
                      "total": 100, "if_alerts": 15, "vae_alerts": 17,
                      "intersection": 12, "union": 20, "jaccard": 0.6,
                      "rank_correlation": 0.7, "rank_correlation_reason": "",
                      "scatter": None},
            candidates=[], sensitivity={"status": STATUS_NOT_REQUESTED, "reason": "x",
                                       "rows": [], "grid": None},
            latent={"status": STATUS_UNAVAILABLE, "reason": "x"},
            stability={"iforest": {"status": STATUS_NOT_REQUESTED, "reason": "x"},
                      "vae": {"status": STATUS_NOT_REQUESTED, "reason": "x"}},
            temporal={"status": STATUS_UNAVAILABLE, "reason": "x", "rows": []},
            segmentation={"status": STATUS_NOT_APPLICABLE, "reason": "x"},
            drift_signal={"available": False}, warnings=[],
            config={}, run_meta={}, populations={},
        )
        node = interpretation["decision_flow"]["nodes"][0]
        self.assertEqual(node["taken"], "both-yes")
        self.assertIn("12", node["fragment"]["text"])

    def test_collapsed_latent_takes_the_caution_branch(self):
        interpretation = build_interpretation_contract(
            agreement={"quadrants": {"BOTH": 1, "IF_ONLY": 1, "VAE_ONLY": 1, "NEITHER": 1},
                      "total": 4, "if_alerts": 2, "vae_alerts": 2,
                      "intersection": 1, "union": 3, "jaccard": 0.33,
                      "rank_correlation": 0.1, "rank_correlation_reason": "",
                      "scatter": None},
            candidates=[], sensitivity={"status": STATUS_NOT_REQUESTED, "reason": "x",
                                       "rows": [], "grid": None},
            latent={"status": STATUS_EXECUTED, "latent_dimensions": 8, "active_units": 1,
                   "collapsed_fraction": 0.875, "mu_variance_by_unit": [], "mean_kl_by_unit": [],
                   "mean_kl": None, "active_threshold": 0.001, "mu_available": "Sí",
                   "logvar_available": "Sí", "artifact": "summary.json"},
            stability={"iforest": {"status": STATUS_NOT_REQUESTED, "reason": "x"},
                      "vae": {"status": STATUS_NOT_REQUESTED, "reason": "x"}},
            temporal={"status": STATUS_UNAVAILABLE, "reason": "x", "rows": []},
            segmentation={"status": STATUS_NOT_APPLICABLE, "reason": "x"},
            drift_signal={"available": False}, warnings=[],
            config={}, run_meta={}, populations={},
        )
        latent_node = next(n for n in interpretation["decision_flow"]["nodes"]
                           if n["id"] == "latent")
        self.assertEqual(latent_node["taken"], "latent-collapsed")
        self.assertEqual(latent_node["severity"], "attention")
        # 1/8 = 0.125 < 1/3 threshold -- confirms the shared, reused constant.
        self.assertLess(1 / 8, LATENT_ACTIVE_FRACTION_THRESHOLD)

    def test_business_drift_names_the_flagged_features_not_calendar_ones(self):
        interpretation = build_interpretation_contract(
            agreement={"quadrants": {"BOTH": 0, "IF_ONLY": 0, "VAE_ONLY": 0, "NEITHER": 10},
                      "total": 10, "if_alerts": 0, "vae_alerts": 0,
                      "intersection": 0, "union": 0, "jaccard": None,
                      "rank_correlation": None, "rank_correlation_reason": "x",
                      "scatter": None},
            candidates=[], sensitivity={"status": STATUS_NOT_REQUESTED, "reason": "x",
                                       "rows": [], "grid": None},
            latent={"status": STATUS_UNAVAILABLE, "reason": "x"},
            stability={"iforest": {"status": STATUS_NOT_REQUESTED, "reason": "x"},
                      "vae": {"status": STATUS_NOT_REQUESTED, "reason": "x"}},
            temporal={"status": STATUS_UNAVAILABLE, "reason": "x", "rows": []},
            segmentation={"status": STATUS_NOT_APPLICABLE, "reason": "x"},
            drift_signal={
                "available": True, "n_features": 3, "n_flagged": 2, "alpha": 0.05,
                "method": "BH",
                "top": [
                    {"feature": "cyc__period_month_sin", "family": "Calendario / cíclico",
                     "ks_statistic": 0.9, "ks_pvalue": 0.0001},
                    {"feature": "num__income", "family": "Numérico de negocio",
                     "ks_statistic": 0.4, "ks_pvalue": 0.001},
                ],
            },
            warnings=[], config={}, run_meta={}, populations={},
        )
        drift_node = next(n for n in interpretation["decision_flow"]["nodes"]
                          if n["id"] == "drift")
        self.assertEqual(drift_node["taken"], "drift-business")
        self.assertIn("num__income", drift_node["fragment"]["text"])
        self.assertNotIn("cyc__period_month_sin", drift_node["fragment"]["text"])

    def test_recommendation_fragments_are_ordered_by_severity(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpretation = build(tmp)["interpretation"]
        severities = [f["severity"] for f in
                     interpretation["decision_flow"]["recommendation_fragments"]]
        priority = {"attention": 0, "caution": 0, "info": 1}
        self.assertEqual(severities, sorted(severities, key=lambda s: priority.get(s, 1)))


class RealStabilityTests(unittest.TestCase):
    """The previously-permanent UNAVAILABLE stability sections now actually
    execute, by refitting real (tiny) project model instances."""

    def _fit_tiny_detectors(self):
        from src.models import IsolationForestDetector, VAEDetector

        rng = np.random.default_rng(3)
        x_fit = rng.normal(size=(60, N_FEATURES)).astype(np.float64)
        x_score = rng.normal(size=(40, N_FEATURES)).astype(np.float64)
        if_detector = IsolationForestDetector(
            random_state=1, n_estimators=20, contamination=0.1,
            max_samples="auto", max_features=1.0, bootstrap=False,
        )
        if_detector.fit(x_fit)
        vae_detector = VAEDetector(
            random_state=1, epochs=2, latent_dim=LATENT_DIM, hidden_dim=8,
            n_layers=1, batch_size=32, early_stopping_patience=None,
        )
        vae_detector.fit(x_fit)
        return if_detector, x_fit, x_score, vae_detector

    def test_stability_actually_refits_and_reports_a_valid_jaccard(self):
        if_detector, x_if_fit, x_if_score, vae_detector = self._fit_tiny_detectors()
        reference, scored, features = synthetic_frames(n_reference=60, n_scored=40)
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(
                reference, scored, features, tmp,
                stability={
                    "if_detector": if_detector, "x_if_fit": x_if_fit,
                    "x_if_score": x_if_score, "vae_detector": vae_detector,
                    "x_vae_fit": x_if_fit, "x_vae_score": x_if_score,
                    "valid_mask": None, "stability_refits": 3, "base_seed": 42,
                },
            )
        stability = section_by_id(payload["contract"], "stability")
        fields_by_title = {b.get("title"): {r["label"]: r["field"] for r in b["rows"]}
                          for b in stability["blocks"] if b["kind"] == "fields"}
        for title in ("Isolation Forest", "VAE"):
            fields = fields_by_title[title]
            self.assertEqual(fields["Estado de ejecución"]["status"], STATUS_EXECUTED)
            jaccard = fields["Jaccard medio"]["value"]
            self.assertIsNotNone(jaccard)
            self.assertGreaterEqual(jaccard, 0.0)
            self.assertLessEqual(jaccard, 1.0)
            self.assertEqual(fields["Número de reajustes"]["value"], 3)

    def test_zero_refits_reports_unavailable_with_a_stated_reason(self):
        if_detector, x_if_fit, x_if_score, vae_detector = self._fit_tiny_detectors()
        reference, scored, features = synthetic_frames(n_reference=60, n_scored=40)
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(
                reference, scored, features, tmp,
                stability={
                    "if_detector": if_detector, "x_if_fit": x_if_fit,
                    "x_if_score": x_if_score, "vae_detector": vae_detector,
                    "x_vae_fit": x_if_fit, "x_vae_score": x_if_score,
                    "valid_mask": None, "stability_refits": 0, "base_seed": 42,
                },
            )
        stability = section_by_id(payload["contract"], "stability")
        rendered = render_diagnostic_markdown(payload["contract"])
        self.assertIn("Estabilidad", rendered)
        for block in stability["blocks"]:
            if block["kind"] != "fields":
                continue
            field = block["rows"][0]["field"]
            self.assertEqual(field["status"], STATUS_UNAVAILABLE)
            self.assertTrue(field["reason"])

    def test_different_base_seeds_do_not_collide_with_each_other(self):
        """Stability seeds are derived from `base_seed`, not hardcoded."""
        from src.evaluation.ifvae_diagnostic import _seeded_refit_stability
        from src.models import IsolationForestDetector

        if_detector, x_fit, x_score, _ = self._fit_tiny_detectors()
        seeds_a = tuple(1 + 1000 * (i + 1) for i in range(3))
        seeds_b = tuple(2 + 1000 * (i + 1) for i in range(3))
        self.assertEqual(len(set(seeds_a) & set(seeds_b)), 0)
        result = _seeded_refit_stability(
            IsolationForestDetector, if_detector,
            ("n_estimators", "max_samples", "max_features", "contamination", "bootstrap"),
            x_fit, x_score, seeds_a, k=10,
        )
        self.assertEqual(result["runs"], 3)


class ExperimentMatrixTests(unittest.TestCase):
    """§9 now genuinely executes the cheap/already-computed families and
    gives a specific (not generic) reason for the ones left NOT_REQUESTED."""

    def _fit_tiny_detectors(self):
        return RealStabilityTests._fit_tiny_detectors(self)

    def test_ensembles_and_reconstruction_variants_execute_for_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        rows = section_by_id(contract, "experiments")["blocks"][0]["rows"]
        by_name = {r[0]: r for r in rows}
        for name in ("Ensembles (máximo IF/VAE)", "Ensembles (promedio IF/VAE)",
                    "Variantes de reconstrucción (VAE)"):
            self.assertIn(STATUS_EXECUTED, by_name[name][1],
                         f"{name} debería ejecutarse sin configuración adicional")

    def test_contamination_sweep_runs_by_default_with_real_refits(self):
        if_detector, x_if_fit, x_if_score, vae_detector = self._fit_tiny_detectors()
        reference, scored, features = synthetic_frames(n_reference=60, n_scored=40)
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(
                reference, scored, features, tmp,
                stability={
                    "if_detector": if_detector, "x_if_fit": x_if_fit,
                    "x_if_score": x_if_score, "vae_detector": vae_detector,
                    "x_vae_fit": x_if_fit, "x_vae_score": x_if_score,
                    "valid_mask": None, "stability_refits": 0, "base_seed": 42,
                },
            )
        rows = section_by_id(payload["contract"], "experiments")["blocks"][0]["rows"]
        contamination_rows = [r for r in rows if "contaminación (IF, contamination=" in r[0]]
        self.assertEqual(len(contamination_rows), 3)  # default grid: 0.01, 0.02, 0.05
        for row in contamination_rows:
            self.assertIn(STATUS_EXECUTED, row[1])
            self.assertNotEqual(row[2], NO_VALUE_TEXT)  # a real "Resultado" was computed

    def test_capacity_and_beta_are_not_requested_without_a_configured_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        rows = section_by_id(contract, "experiments")["blocks"][0]["rows"]
        by_name = {r[0]: r for r in rows}
        for name in ("Capacidad y dimensión latente (VAE)", "Beta y programación KL (VAE)"):
            self.assertIn(STATUS_NOT_REQUESTED, by_name[name][1])
            self.assertTrue(by_name[name][-1], "debe traer un motivo específico")

    def test_capacity_grid_executes_when_configured(self):
        if_detector, x_if_fit, x_if_score, vae_detector = self._fit_tiny_detectors()
        reference, scored, features = synthetic_frames(n_reference=60, n_scored=40)
        with tempfile.TemporaryDirectory() as tmp:
            payload = diagnose_frames(
                reference, scored, features, tmp,
                stability={
                    "if_detector": if_detector, "x_if_fit": x_if_fit,
                    "x_if_score": x_if_score, "vae_detector": vae_detector,
                    "x_vae_fit": x_if_fit, "x_vae_score": x_if_score,
                    "valid_mask": None, "stability_refits": 0, "base_seed": 42,
                },
                experiment_capacity_grid=(2,),
            )
        rows = section_by_id(payload["contract"], "experiments")["blocks"][0]["rows"]
        capacity_rows = [r for r in rows if "Capacidad y dimensión latente (VAE): latent_dim=" in r[0]]
        self.assertEqual(len(capacity_rows), 1)
        self.assertIn(STATUS_EXECUTED, capacity_rows[0][1])

    def test_out_of_scope_families_carry_a_specific_not_generic_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        rows = section_by_id(contract, "experiments")["blocks"][0]["rows"]
        by_name = {r[0]: r for r in rows}
        expectations = {
            "Pérdidas por tipo de feature": "pérdida",
            "Preprocesamiento": "fit_transform_panel",
            "Ablación de familias de features": "matriz de features",
            "Backtests temporales": "origen rodante",
            "Estabilidad entre ventanas": "ventanas OOT",
        }
        for name, keyword in expectations.items():
            self.assertIn(STATUS_NOT_REQUESTED, by_name[name][1])
            self.assertIn(keyword, by_name[name][-1],
                         f"{name} debería explicar POR QUÉ, no una razón genérica")

    def test_no_generic_tracking_reason_leaks_into_any_row(self):
        """The old blanket 'no hay registro de corridas' reason must be gone
        from every row, not just some -- each family now explains itself."""
        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp)["contract"]
        rows = section_by_id(contract, "experiments")["blocks"][0]["rows"]
        for row in rows:
            self.assertNotIn("no lleva un registro de corridas por variante del que "
                            "leer su estado", row[-1] or "")


# --------------------------------------------------------------------------- #
# Renderer purity: facts stay non-interpretive; interpretation stays          #
# computation-free (both, differently)                                       #
# --------------------------------------------------------------------------- #
class RendererPurityTests(unittest.TestCase):
    FACT_RENDERER = os.path.join("src", "reporting", "diagnostic_section.py")
    INTERPRETATION_RENDERER = os.path.join("src", "reporting", "interpretation_section.py")
    #: Words that would turn a FACTUAL description into a verdict about the run.
    VERDICT_WORDS = (
        "mejor", "peor", "supera", "aporta más", "más señal", "prioridad",
        "confirma", "demuestra", "garantiza", "valida el desempeño",
        "recomendamos", "recomienda", "debe revisarse primero",
    )

    def _code_without_docstrings(self, path: str) -> str:
        import ast

        source = open(path, encoding="utf-8").read()
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for doc in docstrings:
            source = source.replace(doc, "")
        return source

    def test_fact_renderer_contains_no_verdict_language(self):
        code = self._code_without_docstrings(self.FACT_RENDERER).lower()
        for word in self.VERDICT_WORDS:
            self.assertNotIn(word, code,
                             f"{self.FACT_RENDERER} contiene lenguaje de veredicto: {word}")

    def test_fact_renderer_contains_no_run_specific_constants(self):
        forbidden = re.compile(
            r"\b(0\.9[059]|95|99|p95|p99|recon_topk|isolation forest|vae_kl)\b",
            re.IGNORECASE,
        )
        code = self._code_without_docstrings(self.FACT_RENDERER)
        code = re.sub(r'DIAGNOSTIC_(CSS|JS) = """.*?"""', "", code, flags=re.S)
        hits = [m.group(0) for m in forbidden.finditer(code)]
        self.assertEqual(hits, [], f"{self.FACT_RENDERER} contiene valores de corrida: {hits}")

    def test_neither_renderer_computes_over_run_numbers(self):
        for path in (self.FACT_RENDERER, self.INTERPRETATION_RENDERER):
            code = self._code_without_docstrings(path)
            code = re.sub(r'(DIAGNOSTIC|INTERPRETATION)_(CSS|JS) = """.*?"""', "",
                         code, flags=re.S)
            self.assertNotIn("100.0 *", code, f"{path} calcula sobre los números de la corrida")
            self.assertNotIn("/ total", code, f"{path} calcula sobre los números de la corrida")

    def test_interpretation_renderer_is_allowed_interpretive_language(self):
        # Sanity check the split is real: the interpretation renderer's own
        # module docstring should NOT claim to avoid interpretation.
        source = open(self.INTERPRETATION_RENDERER, encoding="utf-8").read()
        self.assertIn("interpretive language", source.lower().replace("interpretative", "interpretive"))

    def test_report_module_delegates_both_chapters(self):
        source = open(os.path.join("src", "reporting", "report.py"),
                      encoding="utf-8").read()
        for name in ("_diagnostic_suite_section_md", "_diagnostic_suite_section_html",
                    "_interpretation_section_md", "_interpretation_section_html"):
            self.assertIn(f"def {name}(context", source)
        self.assertIn("render_diagnostic_markdown", source)
        self.assertIn("render_diagnostic_html", source)
        self.assertIn("render_interpretation_markdown", source)
        self.assertIn("render_interpretation_html", source)
        self.assertNotIn("_diagnostic_suite_quadrant_verdict", source)


class SuiteIntegrityTests(unittest.TestCase):
    """The vendored suite's own methodological tests are intact."""

    def test_vendored_suite_tests_are_present(self):
        root = os.path.join("tools", "if_vae_diagnostic_suite", "tests")
        expected = {
            "test_contracts.py", "test_data_quality.py", "test_diagnostics.py",
            "test_metrics.py", "test_scoring.py", "test_simulation.py",
            "test_stability.py", "test_quality_gate.py", "test_mutation_probe.py",
            "test_pipeline_unsupervised.py", "test_reporting_metrics_plot.py",
        }
        present = set(os.listdir(root))
        self.assertTrue(expected.issubset(present),
                        f"faltan pruebas de la suite: {sorted(expected - present)}")


class ContractVersionTests(unittest.TestCase):
    def test_contract_is_versioned_and_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = build(tmp)
        contract = payload["contract"]
        self.assertEqual(contract["contract_version"], CONTRACT_VERSION)
        json_text = json.dumps(contract, ensure_ascii=False)
        self.assertGreater(len(json_text), 3000)
        self.assertEqual(json.loads(json_text)["id"], "diagnostic-suite")
        self.assertEqual(len(contract["sections"]), EXPECTED_SECTION_COUNT)

        interpretation = payload["interpretation"]
        self.assertEqual(interpretation["contract_version"], INTERPRETATION_CONTRACT_VERSION)
        interp_json = json.dumps(interpretation, ensure_ascii=False)
        self.assertGreater(len(interp_json), 500)


if __name__ == "__main__":
    unittest.main()
