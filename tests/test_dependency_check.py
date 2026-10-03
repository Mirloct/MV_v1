"""Startup dependency check: parses requirements.txt, compares against what is
actually installed, and -- by default -- installs whatever is missing or
outdated directly, rather than stopping the run over it (explicit user
request, 2026-10-02: never raise and cut execution, fix it in code instead).

Data sources / inputs: temporary requirements.txt fixtures, a stub logger, and
mocked `importlib.metadata`/`subprocess` calls (no real package is ever
installed or queried from the live environment).
Created: 2026-10-01
Last modified: 2026-10-02
Changelog:
- 2026-10-02: `auto_install` default flipped to `True`. Every test that only
  wants to exercise the detection logic now passes `auto_install=False`
  explicitly, instead of relying on the old default -- otherwise, a test
  with an unmocked `subprocess` would try to actually run `pip install` for
  a fake package during the suite.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from importlib import metadata
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.dependency_check import (  # noqa: E402
    DependencyIssue, _parse_requirements, _pip_install_command, _version_tuple,
    check_dependencies,
)


class _ListLogger:
    """Minimal stand-in that records calls instead of touching real logging
    handlers -- these tests only care about what was said, not how it is
    formatted/handled."""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def _record(self, level, msg, *args):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args):
        self._record("INFO", msg, *args)

    def warning(self, msg, *args):
        self._record("WARNING", msg, *args)

    def error(self, msg, *args):
        self._record("ERROR", msg, *args)

    def text(self) -> str:
        return "\n".join(m for _, m in self.records)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name
        self.log = _ListLogger()

    def write_requirements(self, text: str) -> str:
        path = os.path.join(self.d, "requirements.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class ParsingTests(unittest.TestCase):
    def test_comments_and_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "requirements.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("numpy>=1.26\n\n# a comment\npandas>=2.2  # inline note\n")
            self.assertEqual(
                _parse_requirements(path), [("numpy", "1.26"), ("pandas", "2.2")],
            )

    def test_an_unsupported_operator_is_skipped_not_guessed_at(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "requirements.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("numpy==1.26\nscipy>=1.12\n")
            self.assertEqual(_parse_requirements(path), [("scipy", "1.12")])

    def test_version_tuple_ignores_prerelease_suffix(self):
        self.assertEqual(_version_tuple("2.2.0rc1"), (2, 2, 0))
        self.assertEqual(_version_tuple("1.26"), (1, 26))
        self.assertLess(_version_tuple("1.9"), _version_tuple("1.10"))

    def test_install_command_pins_every_issue_to_its_floor(self):
        issues = [DependencyIssue("numpy", "1.26", "1.20"),
                  DependencyIssue("torch", "2.2", None)]
        cmd = _pip_install_command(issues)
        self.assertIn('"numpy>=1.26"', cmd)
        self.assertIn('"torch>=2.2"', cmd)
        self.assertIn("pip install --upgrade", cmd)


class CheckDependenciesTests(Base):
    def test_missing_requirements_file_warns_and_passes(self):
        missing = os.path.join(self.d, "nope.txt")
        self.assertTrue(check_dependencies(missing, logger=self.log))
        self.assertIn("WARNING", [lvl for lvl, _ in self.log.records])

    def test_everything_satisfied_passes_without_an_install_command(self):
        path = self.write_requirements("numpy>=1.20\n")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="1.26.4",
        ):
            self.assertTrue(check_dependencies(path, logger=self.log))
        self.assertNotIn("pip install", self.log.text())

    def test_a_missing_package_with_auto_install_off_fails_and_names_the_fix(self):
        path = self.write_requirements("torch>=2.2\n")

        with mock.patch(
            "src.utils.dependency_check.metadata.version",
            side_effect=metadata.PackageNotFoundError("torch"),
        ):
            self.assertFalse(check_dependencies(path, logger=self.log, auto_install=False))
        self.assertIn("no instalado", self.log.text())
        self.assertIn("pip install --upgrade", self.log.text())
        self.assertIn('"torch>=2.2"', self.log.text())

    def test_an_outdated_package_with_auto_install_off_states_the_installed_version(self):
        path = self.write_requirements("numpy>=1.26\n")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="1.20.0",
        ):
            self.assertFalse(check_dependencies(path, logger=self.log, auto_install=False))
        self.assertIn("instalado 1.20.0", self.log.text())

    def test_auto_install_defaults_to_true_without_passing_it_explicitly(self):
        # Explicit user request: a missing/outdated dependency is fixed in
        # code by default, never left for a human to run a command -- so
        # OMITTING `auto_install` must behave exactly like passing `True`.
        # version() is called twice: once for the initial check (outdated),
        # once for the post-install re-check (now satisfied).
        path = self.write_requirements("numpy>=1.26\n")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "src.utils.dependency_check.metadata.version",
            side_effect=["1.20.0", "1.26.0"],
        ), mock.patch("subprocess.run", return_value=completed) as run:
            self.assertTrue(check_dependencies(path, logger=self.log))
            run.assert_called_once()

    def test_pip_exit_zero_but_version_still_short_is_caught(self):
        # pip returning 0 does not guarantee the floor is actually met (a
        # conflicting pin elsewhere in the environment could win) -- the
        # post-install re-check must catch that instead of trusting the exit
        # code alone.
        path = self.write_requirements("numpy>=1.26\n")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="1.20.0",
        ), mock.patch("subprocess.run", return_value=completed):
            self.assertFalse(check_dependencies(path, logger=self.log, auto_install=True))
        self.assertIn("pip terminó sin error pero", self.log.text())
        self.assertIn("1.20.0", self.log.text())

    def test_a_newer_than_required_version_passes(self):
        path = self.write_requirements("numpy>=1.26\n")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="2.0.0",
        ):
            self.assertTrue(check_dependencies(path, logger=self.log))

    def test_auto_install_false_never_touches_subprocess(self):
        path = self.write_requirements("numpy>=1.26\n")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="1.20.0",
        ), mock.patch("subprocess.run") as run:
            check_dependencies(path, logger=self.log, auto_install=False)
            run.assert_not_called()

    def test_auto_install_true_runs_pip_and_passes_on_success(self):
        path = self.write_requirements("numpy>=1.26\n")
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "src.utils.dependency_check.metadata.version",
            side_effect=["1.20.0", "1.26.0"],
        ), mock.patch("subprocess.run", return_value=completed) as run:
            self.assertTrue(
                check_dependencies(path, logger=self.log, auto_install=True)
            )
            run.assert_called_once()
            cmd = run.call_args[0][0]
            self.assertIn("pip", cmd)
            self.assertIn("install", cmd)
            self.assertTrue(any("numpy>=1.26" in part for part in cmd))

    def test_auto_install_true_fails_cleanly_on_a_nonzero_pip_exit(self):
        path = self.write_requirements("numpy>=1.26\n")
        completed = mock.Mock(returncode=1, stdout="", stderr="no matching distribution")
        with mock.patch(
            "src.utils.dependency_check.metadata.version", return_value="1.20.0",
        ), mock.patch("subprocess.run", return_value=completed):
            self.assertFalse(
                check_dependencies(path, logger=self.log, auto_install=True)
            )
        self.assertIn("pip devolvió código 1", self.log.text())


if __name__ == "__main__":
    unittest.main()
