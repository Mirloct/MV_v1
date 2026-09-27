"""The single configuration file: precedence, strict keys, early segment validation, report name.

Data sources / inputs: temporary ``pipeline.yaml`` fixtures and ``main.PipelineConfig``.
Created: 2026-09-25
Last modified: 2026-09-25
Changelog:
- 2026-09-25: Added adversarial range, duplicate-key, unknown-section, and
  cross-field validation coverage so invalid YAML cannot silently disable §9.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import main  # noqa: E402
from src.utils.config_file import (  # noqa: E402
    ConfigFileError, apply_config_file, configured_value, find_config_file,
)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name

    def write(self, text: str, name: str = "pipeline.yaml") -> str:
        path = os.path.join(self.d, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class TestPrecedence(Base):
    def test_file_sets_values_that_are_still_at_their_default(self):
        cfg = main.PipelineConfig()
        path = self.write("diagnostic:\n  segment_column: region\n  stability_refits: 1\n"
                          "dashboard:\n  identity_column: cargo\n")
        report = apply_config_file(cfg, path)
        self.assertEqual((cfg.diagnostic_segment_column, cfg.diagnostic_stability_refits,
                          cfg.analyst_identity_column), ("region", 1, "cargo"))
        self.assertEqual(report["sources"]["diagnostic_segment_column"], "file")
        self.assertEqual(report["sources"]["diagnostic_entity_view"], "default")

    def test_cli_and_code_values_beat_the_file(self):
        path = self.write("diagnostic:\n  segment_column: region\n  stability_refits: 1\n")
        cfg = main.PipelineConfig(diagnostic_stability_refits=7)               # set in code
        report = apply_config_file(cfg, path, protected=frozenset({"diagnostic_segment_column"}))
        self.assertEqual(cfg.diagnostic_segment_column, "segment")             # CLI-protected: untouched
        self.assertEqual(cfg.diagnostic_stability_refits, 7)                   # code differs from default
        self.assertEqual(report["sources"]["diagnostic_segment_column"], "cli")
        self.assertEqual(report["sources"]["diagnostic_stability_refits"], "code")

    def test_empty_string_in_the_file_disables_the_breakdown(self):
        cfg = main.PipelineConfig()
        apply_config_file(cfg, self.write("diagnostic:\n  segment_column: ''\n"))
        self.assertEqual(cfg.diagnostic_segment_column, "")

    def test_no_file_changes_nothing(self):
        cfg = main.PipelineConfig()
        report = apply_config_file(cfg, None)
        self.assertEqual(cfg.diagnostic_segment_column, "segment")
        self.assertTrue(all(s == "default" for s in report["sources"].values()))

    def test_unknown_key_and_bad_values_are_errors_that_name_the_valid_keys(self):
        with self.assertRaises(ConfigFileError) as ctx:
            apply_config_file(main.PipelineConfig(), self.write("diagnostic:\n  segmnt_column: x\n"))
        self.assertIn("diagnostic.segment_column", str(ctx.exception))
        with self.assertRaises(ConfigFileError):
            apply_config_file(main.PipelineConfig(), self.write("diagnostic:\n  stability_refits: abc\n"))
        with self.assertRaises(ConfigFileError):
            apply_config_file(main.PipelineConfig(), self.write("- just\n- a list\n"))
        with self.assertRaises(ConfigFileError):
            find_config_file(os.path.join(self.d, "missing.yaml"))

    def test_standalone_tools_read_the_same_value(self):
        path = self.write("diagnostic:\n  segment_column: canal\n")
        self.assertEqual(configured_value("diagnostic.segment_column", "segment", path), "canal")
        self.assertEqual(configured_value("dashboard.identity_column", "puesto", path), "puesto")

    def test_shipped_file_does_not_turn_the_default_segment_into_an_explicit_request(self):
        cfg = main.PipelineConfig()
        report = apply_config_file(cfg, find_config_file(None))
        self.assertEqual(report["sources"]["diagnostic_segment_column"], "default")
        cfg.config_sources = report["sources"]
        df = pd.DataFrame({"entity_id": ["a"], "region": ["x"]})          # a panel without 'segment'
        with self.assertLogs(logging.getLogger("modelo.t"), level="WARNING"):
            main._validate_segment_column(cfg, df, logging.getLogger("modelo.t"))   # warns, does not raise

    def test_value_ranges_encoding_and_abbreviated_flags(self):
        for bad in ("experiments:\n  epoch_cap: 0\n", "experiments:\n  max_fit_rows: 0\n",
                    "diagnostic:\n  sensitivity_grid: [1.5, -1]\n", "experiments:\n  families: [capacidad]\n",
                    "diagnostic:\n  stability_refits: -2\n",
                    "experiments:\n  capacity_grid: [0, -3]\n",
                    "experiments:\n  beta_grid: [-1, .inf]\n",
                    "experiments:\n  kl_anneal_grid: [-5]\n",
                    "experiments:\n  backtest_origins: 1\n  backtest_vae_origins: 2\n"):
            with self.assertRaises(ConfigFileError, msg=bad):
                apply_config_file(main.PipelineConfig(), self.write(bad))
        cp1252 = os.path.join(self.d, "ansi.yaml")
        with open(cp1252, "wb") as fh:
            fh.write("# configuración\ndiagnostic:\n  segment_column: región\n".encode("cp1252"))
        with self.assertRaises(ConfigFileError) as ctx:
            apply_config_file(main.PipelineConfig(), cp1252)
        self.assertIn("UTF-8", str(ctx.exception))
        with self.assertRaises(SystemExit):                              # no silent prefix matching
            main.build_arg_parser().parse_args(["--diagnostic-segment", "x"])

    def test_unknown_empty_sections_and_duplicate_keys_are_errors(self):
        for bad in ("diagnostc:\n", "diagnostic:\n  segment_column: a\n  segment_column: b\n"):
            with self.assertRaises(ConfigFileError, msg=bad):
                apply_config_file(main.PipelineConfig(), self.write(bad))

    def test_the_shipped_file_is_valid_and_only_uses_known_keys(self):
        path = find_config_file(None)
        self.assertIsNotNone(path, "configs/pipeline.yaml must ship with the project")
        apply_config_file(main.PipelineConfig(), path)      # an unknown or badly typed key raises ConfigFileError


class TestCliWiring(Base):
    def _cfg(self, *argv):
        parser = main.build_arg_parser()
        args = parser.parse_args(list(argv))
        args._explicit_dests = main._explicit_dests(parser, list(argv))
        return main.config_from_args(args)

    def test_only_flags_really_given_are_explicit(self):
        cfg = self._cfg()
        self.assertEqual(cfg.cli_explicit, ())
        cfg = self._cfg("--diagnostic-segment-column", "region", "--analyst-identity-column=cargo")
        self.assertEqual(set(cfg.cli_explicit), {"diagnostic_segment_column", "analyst_identity_column"})
        self.assertEqual(self._cfg("--config", "x.yaml").config_file, "x.yaml")

    def test_passing_the_default_value_on_the_command_line_still_wins_over_the_file(self):
        cfg = self._cfg("--diagnostic-entity-view")          # equals the argparse default (True)
        self.assertIn("diagnostic_entity_view", cfg.cli_explicit)
        apply_config_file(cfg, self.write("diagnostic:\n  entity_view: false\n"),
                          frozenset(cfg.cli_explicit))
        self.assertTrue(cfg.diagnostic_entity_view)

    def test_identity_column_default_is_no_longer_a_dead_default(self):
        cfg = self._cfg()
        apply_config_file(cfg, self.write("dashboard:\n  identity_column: cargo\n"),
                          frozenset(cfg.cli_explicit))
        self.assertEqual(cfg.analyst_identity_column, "cargo")

    def test_identification_columns_flag_wins_over_the_file(self):
        cfg = self._cfg("--identification-columns", "puesto", "area")
        self.assertEqual(cfg.identification_columns, ("puesto", "area"))
        self.assertIn("identification_columns", cfg.cli_explicit)
        apply_config_file(cfg, self.write("data:\n  identification_columns: [other]\n"),
                          frozenset(cfg.cli_explicit))
        self.assertEqual(cfg.identification_columns, ("puesto", "area"))   # the file never wins


class TestIdentificationColumnsFile(Base):
    def test_file_sets_a_list_of_column_names(self):
        cfg = main.PipelineConfig()
        apply_config_file(cfg, self.write("data:\n  identification_columns: [puesto, area]\n"))
        self.assertEqual(cfg.identification_columns, ("puesto", "area"))

    def test_duplicates_and_blanks_are_dropped_order_preserved(self):
        cfg = main.PipelineConfig()
        apply_config_file(cfg, self.write("data:\n  identification_columns: [puesto, '', area, puesto]\n"))
        self.assertEqual(cfg.identification_columns, ("puesto", "area"))

    def test_a_single_string_is_accepted_not_only_a_list(self):
        cfg = main.PipelineConfig()
        apply_config_file(cfg, self.write("data:\n  identification_columns: puesto\n"))
        self.assertEqual(cfg.identification_columns, ("puesto",))


class TestResolveIdentificationColumns(unittest.TestCase):
    """`main._resolve_identification_columns`: the single place `data.identification_columns` and
    `dashboard.identity_column` are folded into the one set every modelling phase reads from
    `PanelSchema.identification_columns` via `src.data.loader.key_columns`."""

    df = pd.DataFrame({"entity_id": ["a"], "puesto": ["x"], "area": ["y"]})
    log = logging.getLogger("modelo.test_identification_columns")

    def test_identity_column_is_folded_in_without_repeating_it(self):
        cfg = main.PipelineConfig(identification_columns=("area",))    # analyst_identity_column stays 'puesto'
        self.assertEqual(main._resolve_identification_columns(cfg, self.df, self.log),
                         ("area", "puesto"))

    def test_already_listed_identity_column_is_not_duplicated(self):
        cfg = main.PipelineConfig(identification_columns=("puesto", "area"))
        self.assertEqual(main._resolve_identification_columns(cfg, self.df, self.log),
                         ("puesto", "area"))

    def test_a_missing_column_only_warns_and_is_dropped_from_the_result(self):
        cfg = main.PipelineConfig(identification_columns=("area", "no_existe"))
        with self.assertLogs(self.log, level="WARNING") as captured:
            resolved = main._resolve_identification_columns(cfg, self.df, self.log)
        self.assertEqual(resolved, ("area", "puesto"))                 # missing one silently dropped
        self.assertIn("no_existe", " ".join(captured.output))

    def test_disabling_identity_column_leaves_only_the_explicit_list(self):
        cfg = main.PipelineConfig(identification_columns=("area",), analyst_identity_column="")
        self.assertEqual(main._resolve_identification_columns(cfg, self.df, self.log), ("area",))


class TestSegmentValidation(unittest.TestCase):
    df = pd.DataFrame({"entity_id": ["a"], "Region": ["x"], "product_type": ["p"]})
    log = logging.getLogger("modelo.test_segment")

    def test_explicit_missing_column_stops_early_with_suggestion_and_columns(self):
        cfg = main.PipelineConfig(diagnostic_segment_column="region")
        cfg.config_sources = {"diagnostic_segment_column": "file"}
        with self.assertRaises(ValueError) as ctx:
            main._validate_segment_column(cfg, self.df, self.log)
        msg = str(ctx.exception)
        self.assertIn("'Region'", msg)                     # closest match ignoring case
        self.assertIn("product_type", msg)                 # every available column
        self.assertIn("configs/pipeline.yaml", msg)

    def test_default_missing_column_only_warns(self):
        cfg = main.PipelineConfig()
        cfg.config_sources = {"diagnostic_segment_column": "default"}
        with self.assertLogs(self.log, level="WARNING"):
            main._validate_segment_column(cfg, self.df, self.log)

    def test_present_or_disabled_segment_is_silent(self):
        for name in ("Region", "", None):
            cfg = main.PipelineConfig(diagnostic_segment_column=name)
            cfg.config_sources = {"diagnostic_segment_column": "file"}
            main._validate_segment_column(cfg, self.df, self.log)


class TestReportShowsTheRealColumnName(unittest.TestCase):
    def test_section_8_names_the_configured_column(self):
        from test_diagnostic_section import build, section_by_id

        with tempfile.TemporaryDirectory() as tmp:
            contract = build(tmp, with_segment=True, segment_name="region")["contract"]
        text = repr(section_by_id(contract, "temporal"))
        self.assertIn("Por segmento (region)", text)
        with tempfile.TemporaryDirectory() as tmp:
            missing = build(tmp, segment_name="region")["contract"]
        self.assertIn("Columna solicitada: 'region'", repr(section_by_id(missing, "temporal")))


if __name__ == "__main__":
    unittest.main()
