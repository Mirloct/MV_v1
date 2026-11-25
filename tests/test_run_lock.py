"""Real descriptor contention and exception recovery checks.

Data sources / inputs: temporary OS lock files.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — verify overlap rejection and automatic release.
"""
import tempfile
import unittest
from pathlib import Path

from src.utils.run_lock import artifact_lock


class ArtifactLockTests(unittest.TestCase):
    def test_overlap_rejected_and_released_on_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.lock"
            with self.assertRaisesRegex(ValueError, "work failed"):
                with artifact_lock(path):
                    with self.assertRaisesRegex(RuntimeError, "Another pipeline run owns"):
                        with artifact_lock(path):
                            self.fail("overlap entered")
                    raise ValueError("work failed")
            with artifact_lock(path):
                self.assertTrue(path.exists())
