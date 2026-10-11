"""`_warn_on_extreme_magnitudes` (src/preprocessing/pipeline.py): the scaler-blow-up
guard that names which columns exceed `_EXTREME_MAGNITUDE` after transformation.
Pins its sparse-input behavior across both an implicit-zero column (no stored
entries at all, e.g. an all-zero one-hot level in this batch) and an
explicit-value column, since the per-column max must treat a sparse matrix's
unstored entries as zero, not skip the column -- a case a naive "max of stored
data only" loop could get wrong.
"""
from __future__ import annotations

import logging
import unittest

import numpy as np
import scipy.sparse as sp

from src.preprocessing.pipeline import _warn_on_extreme_magnitudes


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())


class ExtremeMagnitudeWarningTests(unittest.TestCase):
    def _run(self, X, names):
        log = logging.getLogger("test_pipeline_diagnostics")
        log.setLevel(logging.WARNING)
        handler = _ListHandler()
        log.addHandler(handler)
        try:
            _warn_on_extreme_magnitudes(X, names, log)
        finally:
            log.removeHandler(handler)
        return handler.records

    def test_sparse_column_with_only_implicit_zeros_is_never_flagged(self):
        # Column 1 has no stored entries at all (an all-zero one-hot level in
        # this batch) -- its true max is the implicit 0.0, well under the
        # threshold, and must not be misread as "no data -> skip/flag".
        X = sp.csr_matrix(np.array([[1.0, 0.0, 2e6], [3.0, 0.0, 0.0]]))
        records = self._run(X, ["num__a", "cat__rare_level", "num__blown_up"])
        self.assertEqual(len(records), 1)
        self.assertIn("num__blown_up", records[0])
        self.assertNotIn("cat__rare_level", records[0])

    def test_sparse_and_dense_agree_on_which_columns_exceed_the_threshold(self):
        rng = np.random.default_rng(0)
        dense = rng.normal(size=(20, 6)).astype(float)
        dense[3, 4] = 5e6   # one deliberate blow-up
        names = [f"c{i}" for i in range(6)]
        dense_records = self._run(dense, names)
        sparse_records = self._run(sp.csr_matrix(dense), names)
        self.assertEqual(len(dense_records), 1)
        self.assertEqual(len(sparse_records), 1)
        self.assertIn("c4", dense_records[0])
        self.assertIn("c4", sparse_records[0])

    def test_no_columns_exceed_threshold_logs_nothing(self):
        X = sp.csr_matrix(np.ones((5, 3)))
        self.assertEqual(self._run(X, ["a", "b", "c"]), [])


if __name__ == "__main__":
    unittest.main()
