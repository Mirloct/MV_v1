import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.quality_gate import (
    baseline_violations,
    complexity_scores,
    complexity_violations,
    load_baseline,
    reference_baseline,
    tautology_violations,
)


GATE = Path(__file__).resolve().parents[1] / "scripts" / "quality_gate.py"


def branching_source(decisions, name="example"):
    branches = "".join(f"    if value == {n}:\n        return {n}\n" for n in range(decisions))
    return f"def {name}(value):\n{branches}    return -1\n"


class QualityGateTests(unittest.TestCase):
    def run_gate(self, source, baseline=None, filename="example.py"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/complexity_baseline.json").write_text(
                json.dumps(baseline or {}), encoding="utf-8"
            )
            (root / filename).write_text(source, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(GATE), "--root", str(root)],
                capture_output=True, text=True, check=False,
            )

    def test_ceiling_ten_passes_and_eleven_rejects(self):
        accepted = self.run_gate(branching_source(9))
        rejected = self.run_gate(branching_source(10))
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        self.assertEqual(rejected.returncode, 1, rejected.stdout)
        self.assertIn("complexity 11 exceeds 10", rejected.stdout)

    def test_existing_debt_passes_as_noncompliant(self):
        result = self.run_gate(branching_source(10), {"example.py::example": 11})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("NOT compliant", result.stdout)
        self.assertIn("DEBT example.py::example=11", result.stdout)

    def test_increased_debt_rejects(self):
        result = self.run_gate(branching_source(11), {"example.py::example": 11})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("complexity 12 exceeds 11", result.stdout)

    def test_improvement_requires_baseline_ratchet(self):
        result = self.run_gate(branching_source(9), {"example.py::example": 11})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("ratchet baseline to 10", result.stdout)
        self.assertEqual(complexity_violations({"f": 10}, {}), [])

    def test_removed_debt_requires_baseline_removal(self):
        self.assertIn("ratchet baseline to None", complexity_violations({}, {"old": 11})[0])

    def test_baseline_cannot_add_or_inflate_debt(self):
        violations = baseline_violations({"new": 11, "old": 13}, {"old": 12})
        self.assertEqual(len(violations), 2)
        self.assertEqual(baseline_violations({"old": 11}, {"old": 12}), [])

    def test_baseline_rejects_invalid_json_schema(self):
        fixtures = ['[]', '{"f": 11}', '{"f::g": true}', '{"f::g": 10}',
                    '{"f::g": 11.5}', '{"f::g": 11, "f::g": 12}', '{']
        for source in fixtures:
            with self.subTest(source=source), self.assertRaises(ValueError):
                load_baseline(source)

    def test_assertion_rejecting_fixtures(self):
        fixtures = ["assert True", "assert 1", "assert value == value",
                    "assert value is value", "self.assertTrue(True)",
                    "self.assertTrue(value == value)", "self.assertEqual(value, value)",
                    "self.assertAlmostEqual(value, value)", "self.assertListEqual(value, value)",
                    "self.assertTrue(expr=True)", "self.assertEqual(first=value, second=value)",
                    "self.assertEqual(value, second=value)"]
        for statement in fixtures:
            with self.subTest(statement=statement):
                result = self.run_gate(f"def test_bad():\n    {statement}\n", filename="test_bad.py")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("REJECT test_bad.py:", result.stdout)

    def test_empty_rejecting_fixtures(self):
        for statement in ("pass", '"only a docstring"', "...", "return", "return None"):
            with self.subTest(statement=statement):
                result = self.run_gate(f"async def test_bad():\n    {statement}\n", filename="test_bad.py")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("empty test", result.stdout)

    def test_meaningful_assertions_and_helpers_pass(self):
        source = ('def helper():\n    pass\n'
                  'def test_real():\n    actual = helper()\n    assert actual == 3\n'
                  '    self.assertEqual(actual, 3)\n'
                  '    fixture = "assert True"\n')
        result = self.run_gate(source, filename="test_real.py")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_syntax_error_fails_closed(self):
        result = self.run_gate("def broken(:")
        self.assertEqual(result.returncode, 1)
        self.assertIn("QUALITY ERROR", result.stdout)

    def test_malformed_baseline_fails_closed_in_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/complexity_baseline.json").write_text('{"f::g": true}', encoding="utf-8")
            result = subprocess.run([sys.executable, str(GATE), "--root", str(root)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("invalid baseline entry", result.stdout)

    def test_nested_test_files_are_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/complexity_baseline.json").write_text('{}', encoding="utf-8")
            (root / "tests/nested").mkdir(parents=True)
            (root / "tests/nested/test_bad.py").write_text('def test_bad():\n    pass\n', encoding="utf-8")
            result = subprocess.run([sys.executable, str(GATE), "--root", str(root)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("tests/nested/test_bad.py: line 1: empty test", result.stdout)

    def test_nested_methods_and_closures_have_independent_scores(self):
        source = ('class Outer:\n    async def method(self):\n'
                  '        def inner(x):\n            return 1 if x else 0\n'
                  '        return inner(1)\n')
        scores = complexity_scores(ast.parse(source), "module.py")
        self.assertEqual(scores, {"module.py::Outer.method": 1, "module.py::Outer.method.inner": 2})

    def test_radon_counts_boolean_comprehension_and_match(self):
        source = ('def f(xs, flag):\n    values = [x for x in xs if x and flag]\n'
                  '    match flag:\n        case 1: return values\n        case _: return []\n')
        self.assertEqual(complexity_scores(ast.parse(source), "m.py"), {"m.py::f": 5})

    def test_ast_check_ignores_comments_and_strings(self):
        self.assertEqual(tautology_violations(ast.parse('text = "assert True"\n# assert True\n')), [])

    def test_baseline_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(GATE), "--root", directory],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("QUALITY ERROR", result.stdout)

    def test_repeated_legacy_names_keep_distinct_identities(self):
        source = 'def f(x):\n    if x:\n        def callback(): return 1\n    else:\n        def callback(): return 2\n'
        scores = complexity_scores(ast.parse(source), "m.py")
        self.assertEqual(scores, {"m.py::f": 2, "m.py::f.callback": 1, "m.py::f.callback#2": 1})

    def test_git_reference_bootstrap_and_protected_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.git(root, "init")
            self.git(root, "config", "user.email", "fixture@example.invalid")
            self.git(root, "config", "user.name", "Fixture")
            (root / "old.py").write_text(branching_source(10), encoding="utf-8")
            self.git(root, "add", ".")
            self.git(root, "commit", "-m", "fixture")
            self.assertEqual(reference_baseline(root, "HEAD"), {"old.py::example": 11})
            (root / "scripts").mkdir()
            (root / "scripts/complexity_baseline.json").write_text('{"old.py::example": 11}', encoding="utf-8")
            self.git(root, "add", ".")
            self.git(root, "commit", "-m", "baseline fixture")
            self.assertEqual(reference_baseline(root, "HEAD"), {"old.py::example": 11})
            (root / "old.py").write_text(branching_source(11), encoding="utf-8")
            (root / "scripts/complexity_baseline.json").write_text('{"old.py::example": 12}', encoding="utf-8")
            result = subprocess.run([sys.executable, str(GATE), "--root", str(root), "--base-ref", "HEAD"],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("baseline debt 12 exceeds reference 11", result.stdout)

    def git(self, root, *args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
