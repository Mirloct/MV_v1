"""Validation suite for `src.evaluation` (OOT split, labels, metrics, scoring, OOT Excel).

Reuses the conftest session sandbox. Small panel, conventional ground-truth
naming so the loader finds the labels. Excel/figures land under tmp_path.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import IsolationForestDetector
from src.evaluation import (
    THRESHOLD_METHODS,
    apply_threshold,
    build_scored_frame,
    calibrate_threshold,
    chronological_split,
    export_oot_top_anomalies,
    export_oot_top_decile,
    load_ground_truth_labels,
    load_ground_truth_types,
    metrics_by_anomaly_type,
    oot_period,
    oot_split,
    plot_embedding,
    plot_roc_pr,
    supervised_metrics,
    unsupervised_metrics,
)

N_ENTITIES = 400
N_PERIODS = 6
SEED = 20260724


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    dest = tmp_path_factory.mktemp("eval_panel")
    df, schema = load_or_generate_panel(
        data_path=str(dest / "data.csv"),
        ground_truth_path=str(dest / "ground_truth.parquet"),
        n_individuals=N_ENTITIES, n_periods=N_PERIODS, seed=SEED,
    )
    X, keys, names = fit_transform_panel(
        df, schema, numeric_transform="yeo-johnson", categorical_encoding="frequency")
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    y = load_ground_truth_labels(schema, keys)
    in_mask, oot_mask = oot_split(keys, time_col=schema.time_col)
    det = IsolationForestDetector(n_estimators=150, random_state=0).fit(X[in_mask])
    scores = det.score_samples(X)
    return dict(df=df, schema=schema, X=X, keys=keys, names=names, y=y,
                in_mask=in_mask, oot_mask=oot_mask, scores=scores)


# --------------------------------------------------------------------------- #
class TestOOTSplit:
    def test_last_period_is_oot_and_masks_partition_rows(self, prepared):
        keys, schema = prepared["keys"], prepared["schema"]
        in_mask, oot_mask = prepared["in_mask"], prepared["oot_mask"]
        assert in_mask.shape[0] == oot_mask.shape[0] == len(keys)
        # partition: every row is exactly one of in-time / OOT
        assert np.all(in_mask ^ oot_mask)
        # OOT is the max period
        periods = pd.to_datetime(keys[schema.time_col])
        oot_p = pd.to_datetime(oot_period(keys, schema.time_col))
        assert periods[oot_mask.astype(bool)].max() == periods.max()
        assert periods[oot_mask.astype(bool)].min() == periods.max()  # single OOT period
        assert periods[in_mask.astype(bool)].max() < periods.max()


class TestLabels:
    def test_labels_aligned_and_have_positives(self, prepared):
        y, keys = prepared["y"], prepared["keys"]
        assert y.shape[0] == len(keys)
        assert set(np.unique(y).tolist()) <= {0, 1}
        assert y.sum() > 0, "ground-truth join produced no anomalies"


class TestSupervisedMetrics:
    def test_returns_full_suite_and_sane_ranges(self, prepared):
        y, scores, oot = prepared["y"], prepared["scores"], prepared["oot_mask"]
        m = supervised_metrics(y[oot], scores[oot])
        for k in ("roc_auc", "pr_auc", "best_f1", "best_f2", "mcc",
                  "precision_at_10pct", "recall_at_10pct", "lift_at_10pct",
                  "expected_loss", "base_rate", "n", "n_positive"):
            assert k in m, f"missing metric {k}"
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert -1.0 <= m["mcc"] <= 1.0
        assert m["lift_at_10pct"] >= 0.0

    def test_no_positive_case_is_graceful(self, prepared):
        scores = prepared["scores"]
        y0 = np.zeros_like(prepared["y"])
        m = supervised_metrics(y0, scores)  # must not raise / divide by zero
        assert m["n_positive"] == 0
        assert np.isfinite(m["expected_loss"]) or math.isnan(m["expected_loss"])


class TestUnsupervisedMetrics:
    def test_returns_expected_keys(self, prepared):
        m = unsupervised_metrics(prepared["X"], prepared["scores"])
        for k in ("silhouette", "calinski_harabasz", "rank_stability",
                  "contamination", "n_flagged"):
            assert k in m


class TestChronologicalSplit:
    """Fase 2: strictly chronological train/val/test, never random."""

    @pytest.fixture(scope="class")
    @classmethod
    def split(cls, prepared):
        return chronological_split(
            prepared["keys"], prepared["schema"].time_col,
            n_val_periods=2, n_test_periods=2,
        )

    def test_masks_partition_every_row_exactly_once(self, prepared, split):
        n = len(prepared["keys"])
        stacked = split.train_mask.astype(int) + split.val_mask.astype(int) + split.test_mask.astype(int)
        assert stacked.tolist() == [1] * n

    def test_blocks_are_strictly_ordered_in_time(self, prepared, split):
        t = pd.to_datetime(prepared["keys"][prepared["schema"].time_col])
        assert t[split.train_mask].max() < t[split.val_mask].min()
        assert t[split.val_mask].max() < t[split.test_mask].min()

    def test_no_period_appears_in_two_blocks(self, split):
        tr, va, te = set(map(str, split.train_periods)), set(map(str, split.val_periods)), set(map(str, split.test_periods))
        assert tr.isdisjoint(va) and va.isdisjoint(te) and tr.isdisjoint(te)

    def test_short_panel_shrinks_instead_of_failing(self, prepared):
        """A 6-period panel cannot give 2 val + 5 test; it must still return a
        usable split rather than raise."""
        keys = prepared["keys"]
        sp6 = chronological_split(keys, prepared["schema"].time_col,
                                  n_val_periods=2, n_test_periods=5)
        assert sp6.train_mask.sum() > 0
        assert sp6.val_mask.sum() > 0 and sp6.test_mask.sum() > 0

    def test_too_few_periods_raises(self, prepared):
        keys = prepared["keys"].copy()
        keys = keys[keys[prepared["schema"].time_col].isin(
            sorted(keys[prepared["schema"].time_col].unique())[:2])]
        with pytest.raises(ValueError, match="at least 3 distinct periods"):
            chronological_split(keys, prepared["schema"].time_col)


class TestThresholdCalibration:
    """Fase 6: the cut-off comes from validation, never from test."""

    @pytest.fixture(scope="class")
    @classmethod
    def val_scores(cls):
        rng = np.random.default_rng(7)
        return np.concatenate([rng.normal(0.0, 1.0, 3000), rng.normal(7.0, 1.0, 30)])

    @pytest.mark.parametrize("method", list(THRESHOLD_METHODS))
    def test_returns_a_finite_threshold_and_documented_fields(self, val_scores, method):
        out = calibrate_threshold(val_scores, method=method)
        for key in ("threshold", "method", "requested_method", "n", "n_flagged", "flagged_rate"):
            assert key in out
        assert np.isfinite(out["threshold"])
        assert out["requested_method"] == method

    def test_percentile_flags_the_expected_share(self, val_scores):
        out = calibrate_threshold(val_scores, method="percentile", percentile=99.0)
        assert out["flagged_rate"] == pytest.approx(0.01, abs=0.004)

    def test_pot_is_stricter_than_a_p99_for_a_small_target_far(self, val_scores):
        pot = calibrate_threshold(val_scores, method="pot", target_far=1e-3)
        pct = calibrate_threshold(val_scores, method="percentile", percentile=99.0)
        assert pot["threshold"] > pct["threshold"], (
            "a 1-in-1000 false-alarm budget must sit above the 1-in-100 percentile"
        )

    def test_pot_falls_back_and_says_so_when_the_tail_is_too_thin(self):
        out = calibrate_threshold(np.arange(40.0), method="pot", tail_percentile=95.0)
        assert out["method"] == "percentile"
        assert "fallback_reason" in out

    def test_apply_threshold_is_the_inequality_it_claims(self, val_scores):
        out = calibrate_threshold(val_scores, method="percentile", percentile=90.0)
        flags = apply_threshold(val_scores, out["threshold"])
        assert set(np.unique(flags)) <= {0, 1}
        assert flags.sum() == int((val_scores >= out["threshold"]).sum())

    def test_unknown_method_raises(self, val_scores):
        with pytest.raises(ValueError, match="Unknown threshold method"):
            calibrate_threshold(val_scores, method="magic")


class TestTopNExport:
    """The headline deliverable: a fixed-size, risk-ranked queue of individuals."""

    def _scored(self, prepared):
        return build_scored_frame(prepared["df"], prepared["keys"],
                                  prepared["scores"], prepared["schema"])

    def test_exports_exactly_top_n_individuals(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        path, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "top.xlsx"),
            top_n=50, model_name="iforest",
        )
        assert os.path.isfile(path)
        assert len(table) == 50
        ent = prepared["schema"].entity_col
        assert table[ent].nunique() == 50, "the export must hold 50 distinct people"

    def test_is_sorted_by_descending_risk(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        _, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "s.xlsx"), top_n=25)
        scores = table["anomaly_score"].to_numpy()
        assert np.all(np.diff(scores) <= 0)

    def test_top_n_is_parameterisable(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        for n in (10, 50, 200):
            _, table = export_oot_top_anomalies(
                sdf, prepared["schema"], out_path=str(tmp_path / f"n{n}.xlsx"), top_n=n)
            assert len(table) == n

    def test_top_n_is_capped_by_the_population(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        n_oot = int(prepared["oot_mask"].sum())
        _, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "big.xlsx"), top_n=10**6)
        assert len(table) == n_oot, "asking for more people than exist must not fail"

    def test_one_row_per_individual_across_multiple_test_months(self, prepared, tmp_path):
        """With several OOT months an entity has several rows; the queue must
        still be N people, each represented by their worst month."""
        sdf = self._scored(prepared)
        ent = prepared["schema"].entity_col
        _, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "m.xlsx"),
            top_n=40, n_oot_periods=3)
        assert len(table) == 40
        assert table[ent].nunique() == 40

    def test_threshold_adds_an_alert_column(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        cal = calibrate_threshold(prepared["scores"], method="percentile", percentile=95.0)
        _, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "a.xlsx"),
            top_n=30, threshold=cal["threshold"])
        assert "alert" in table.columns
        assert set(np.unique(table["alert"])) <= {0, 1}

    def test_default_filename_carries_the_queue_size(self, prepared):
        sdf = self._scored(prepared)
        path, _ = export_oot_top_anomalies(sdf, prepared["schema"], top_n=50,
                                           model_name="iforest")
        assert os.path.basename(path) == "oot_top50_iforest.xlsx"

    def test_top_fraction_still_works_when_top_n_is_none(self, prepared, tmp_path):
        sdf = self._scored(prepared)
        _, table = export_oot_top_anomalies(
            sdf, prepared["schema"], out_path=str(tmp_path / "f.xlsx"),
            top_n=None, top_fraction=0.10)
        n_oot = int(prepared["oot_mask"].sum())
        assert len(table) == math.ceil(0.10 * n_oot)


class TestGroundTruthTypes:
    def test_types_are_aligned_and_cover_the_vocabulary(self, prepared):
        types = load_ground_truth_types(prepared["schema"], prepared["keys"])
        y = prepared["y"]
        assert types.shape[0] == len(prepared["keys"])
        assert set(np.unique(types)) <= {"none", "global", "local",
                                         "contextual", "collective"}
        # `anomaly_type` and `is_anomaly` must agree row-for-row: both come from
        # the same join, so a mismatch means the two views drifted apart.
        assert np.array_equal((types != "none").astype(int), y)

    def test_all_four_geometries_are_present(self, prepared):
        types = load_ground_truth_types(prepared["schema"], prepared["keys"])
        for name in ("global", "local", "contextual", "collective"):
            assert (types == name).sum() > 0, f"no {name} anomalies in the panel"


class TestMetricsByAnomalyType:
    """Which geometry is the detector blind to? An aggregate PR-AUC cannot say."""

    @pytest.fixture(scope="class")
    @classmethod
    def by_type(cls, prepared):
        types = load_ground_truth_types(prepared["schema"], prepared["keys"])
        return metrics_by_anomaly_type(prepared["y"], types, prepared["scores"])

    def test_covers_every_observed_type_plus_overall(self, prepared, by_type):
        types = load_ground_truth_types(prepared["schema"], prepared["keys"])
        observed = {t for t in np.unique(types) if t != "none"}
        assert observed <= set(by_type)
        assert "__overall__" in by_type

    def test_positive_counts_sum_to_the_total(self, prepared, by_type):
        per_type = sum(
            block["n_positive"] for name, block in by_type.items() if name != "__overall__"
        )
        assert per_type == float(prepared["y"].sum())
        assert by_type["__overall__"]["n_positive"] == float(prepared["y"].sum())

    def test_each_block_has_the_documented_fields(self, by_type):
        for name, block in by_type.items():
            for key in ("n", "n_positive", "mean_score_percentile",
                        "recall_at_1pct", "recall_at_5pct", "recall_at_10pct"):
                assert key in block, f"{name} missing {key}"
            for frac in ("1pct", "5pct", "10pct"):
                assert 0.0 <= block[f"recall_at_{frac}"] <= 1.0

    def test_recall_is_monotone_in_the_alert_budget(self, by_type):
        for name, block in by_type.items():
            assert (
                block["recall_at_1pct"] <= block["recall_at_5pct"]
                <= block["recall_at_10pct"]
            ), f"{name}: recall must not decrease as the budget grows"

    def test_recall_is_counted_against_the_global_top_k(self, prepared, by_type):
        """A within-type ranking would report near-perfect recall for a type the
        detector never surfaces; assert the global-queue semantics instead."""
        y, scores = prepared["y"], prepared["scores"]
        types = load_ground_truth_types(prepared["schema"], prepared["keys"])
        n = len(scores)
        k = max(1, math.ceil(0.10 * n))
        top = np.argsort(-scores, kind="stable")[:k]
        in_top = np.zeros(n, dtype=bool)
        in_top[top] = True
        for name, block in by_type.items():
            if name == "__overall__":
                continue
            pos = (types == name) & (y == 1)
            expected = float((pos & in_top).sum()) / float(pos.sum())
            assert block["recall_at_10pct"] == pytest.approx(expected)

    def test_length_mismatch_raises(self, prepared):
        with pytest.raises(ValueError, match="length mismatch"):
            metrics_by_anomaly_type(prepared["y"], np.array(["none"]), prepared["scores"])

    def test_empty_input_returns_empty_dict(self):
        assert metrics_by_anomaly_type([], [], []) == {}


class TestScoredFrame:
    def test_has_keys_score_and_raw_features(self, prepared):
        sdf = build_scored_frame(prepared["df"], prepared["keys"],
                                 prepared["scores"], prepared["schema"])
        assert len(sdf) == len(prepared["keys"])
        assert prepared["schema"].entity_col in sdf.columns
        assert prepared["schema"].time_col in sdf.columns
        assert "anomaly_score" in sdf.columns
        # raw human-readable features are present (e.g. region is categorical text)
        assert "region" in sdf.columns
        assert sdf["region"].dtype == object or str(sdf["region"].dtype).startswith("categor")


class TestOOTExcelDeliverable:
    """The headline deliverable: top-10% OOT, format ID - SCORE - VARIABLES."""

    def test_excel_layout_topdecile_sorted(self, prepared, tmp_path):
        sdf = build_scored_frame(prepared["df"], prepared["keys"],
                                 prepared["scores"], prepared["schema"])
        out = tmp_path / "oot_top10_iforest.xlsx"
        path, table = export_oot_top_decile(
            sdf, prepared["schema"], out_path=str(out),
            top_fraction=0.10, model_name="iforest")
        assert os.path.exists(path)
        # column 0 = ID (entity), column 1 = SCORE, remaining = VARIABLES
        assert table.columns[0] == prepared["schema"].entity_col
        assert table.columns[1] == "anomaly_score"
        assert table.shape[1] >= 3, "no VARIABLES columns after ID/SCORE"
        # size == ceil(0.10 * n_oot_rows)
        n_oot = int(prepared["oot_mask"].sum())
        assert len(table) == math.ceil(0.10 * n_oot)
        # sorted descending by score
        sc = table["anomaly_score"].to_numpy()
        assert np.all(sc[:-1] >= sc[1:]), "OOT table not sorted by score descending"
        # only OOT-period individuals selected
        oot_ids = set(prepared["keys"][prepared["oot_mask"].astype(bool)][
            prepared["schema"].entity_col].astype(str))
        assert set(table[prepared["schema"].entity_col].astype(str)) <= oot_ids
        # readable back as a valid xlsx
        back = pd.read_excel(path)
        assert len(back) == len(table)


class TestFigures:
    def test_embedding_and_roc_pr_written(self, prepared, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "reports" / "figures" / "evaluation"
        paths = []
        try:
            p1 = plot_embedding(prepared["X"], prepared["scores"], method="pca",
                                out_dir=str(out), y=prepared["y"])
            p2 = plot_roc_pr(prepared["y"][prepared["oot_mask"]],
                             prepared["scores"][prepared["oot_mask"]], out_dir=str(out))
            paths = [p for p in (p1, p2) if p]
            assert p1 and os.path.exists(p1)
            assert p2 and os.path.exists(p2)  # OOT has positives, so curves are defined
            for p in paths:
                assert "reports" in Path(p).parts and "figures" in Path(p).parts
        finally:
            for p in paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
