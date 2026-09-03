import ast
import unittest

from scripts.quality_gate import cyclomatic_complexity, tautology_violations


class QualityGateTests(unittest.TestCase):
    def test_complexity_counts_decision_paths(self):
        tree = ast.parse("def f(x):\n    if x:\n        return 1\n    return 0\n")
        fn = tree.body[0]
        self.assertEqual(cyclomatic_complexity(fn), 2)

    def test_tautology_scanner_rejects_assert_true(self):
        violations = tautology_violations("def test_bad(self):\n    self.assertTrue(True)\n")
        self.assertTrue(any("assertTrue(True)" in item for item in violations))


if __name__ == "__main__":
    unittest.main()

