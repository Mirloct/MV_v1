"""Regression test for a genuine, root-caused cross-platform bug found
integrating this suite into Modelo v0.1 (Windows): `mutation_probe.py`
joined the mutated-source `PYTHONPATH` entry with a hardcoded POSIX `:`
separator. On Windows, `os.pathsep` is `;`, and a drive-letter path like
`C:\\Users\\...\\src` already contains a colon, so the hardcoded join
produced one unparseable PYTHONPATH string. Python silently fell back to
importing the real, pip-installed (unmutated) package for every mutant,
so `_mutant_is_killed` always returned `False` -- a false "the gate is
broken" reading that had nothing to do with test quality.

This is a real behavior assertion, not a tautology: it actually applies one
mutant, runs its target test module against the mutated copy in a fresh
subprocess, and asserts the (correct, still-passing) tests catch the
mutation -- exactly what `_mutant_is_killed` is supposed to measure.
"""

import unittest

from scripts.mutation_probe import MUTANTS, _mutant_is_killed


class MutationProbeCrossPlatformTests(unittest.TestCase):
    def test_lift_formula_mutant_is_actually_killed(self):
        mutant = next(m for m in MUTANTS if m.name == "lift_formula")
        self.assertTrue(
            _mutant_is_killed(mutant),
            "mutation_probe must run tests against the mutated copy, not the "
            "installed package -- see PYTHONPATH construction in "
            "_mutant_is_killed",
        )


if __name__ == "__main__":
    unittest.main()
