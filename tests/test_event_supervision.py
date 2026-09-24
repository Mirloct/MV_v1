"""Acceptance tests for reviewed event labels, gate 4.5 and the challengers.

The main panel has no target column; a separate CSV gives ``(entity_id, codmes)``
rows a reviewed target. These tests write real CSV files to disk and drive the
real loader, gate, evaluation and challengers -- no mocks -- against a synthetic
panel whose positives are built to be learnable, so a correct implementation has
a checkable answer.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import PanelSchema  # noqa: E402
from src.evaluation.event_evaluation import calibration_summary, evaluate_ranking  # noqa: E402
from src.evaluation.event_labels import (  # noqa: E402
    EventLabelsContractError,
    LabelSpec,
    find_labels_file,
    load_event_labels,
)
from src.evaluation.event_supervision import EventSupervisionConfig, run_event_supervision  # noqa: E402
from src.evaluation.label_gate import GateThresholds, decide_gate, sufficiency_metrics  # noqa: E402
from src.models.event_challenger import ChallengerConfig  # noqa: E402

N_MONTHS = 24
SCHEMA = PanelSchema(time_col="period", entity_col="entity_id", target_col=None)


def month_start(i: int) -> pd.Timestamp:
    return pd.Timestamp(2022, 1, 1) + pd.DateOffset(months=i)


def codmes(i: int) -> str:
    m = month_start(i)
    return f"{m.year}{m.month:02d}"


def make_keys(n_entities: int = 150) -> pd.DataFrame:
    ents = [f"{i:05d}" for i in range(n_entities)]   # leading zeros on purpose
    return pd.DataFrame({
        "entity_id": np.repeat(ents, N_MONTHS),
        "period": np.tile([month_start(i) for i in range(N_MONTHS)], n_entities),
    })


def make_masks(keys: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = ((keys["period"].dt.year - 2022) * 12 + keys["period"].dt.month - 1).to_numpy()
    return {"train": idx <= 13, "val": (idx >= 14) & (idx <= 15),
            "test": (idx >= 16) & (idx <= 19), "oot": idx >= 20}


def episode_plan(n_episodes: int, n_entities: int = 150, seed: int = 0) -> list[tuple[str, int, int]]:
    """``(entity, start_month, length)`` for ``n_episodes`` distinct entities."""
    rng = np.random.default_rng(seed)
    ents = rng.choice(n_entities, size=n_episodes, replace=False)
    return [(f"{e:05d}", int(rng.integers(0, 21)), int(rng.integers(1, 4))) for e in ents]


def positive_rows(plan) -> set[tuple[str, int]]:
    return {(e, s + j) for e, s, ln in plan for j in range(ln) if s + j < N_MONTHS}


def write_labels(path: str, keys: pd.DataFrame, plan, *, status: bool = True, listed=None) -> None:
    pos = positive_rows(plan)
    rows = []
    for e, p in zip(keys["entity_id"], keys["period"]):
        i = (p.year - 2022) * 12 + p.month - 1
        if listed is not None and (e, i) not in listed:
            continue
        rows.append({"entity_id": e, "codmes": codmes(i), "target": int((e, i) in pos),
                     **({"label_status": "confirmed"} if status else {})})
    pd.DataFrame(rows).to_csv(path, index=False)


def learnable_scores(keys: pd.DataFrame, plan, seed: int = 1):
    """IF/VAE-like scores and a feature matrix that carry real signal."""
    pos = positive_rows(plan)
    rng = np.random.default_rng(seed)
    idx = ((keys["period"].dt.year - 2022) * 12 + keys["period"].dt.month - 1).to_numpy()
    y = np.array([(e, i) in pos for e, i in zip(keys["entity_id"], idx)], dtype=float)
    if_score = 1.5 * y + rng.normal(size=len(y))
    vae_score = 0.8 * y + rng.normal(size=len(y))
    X = np.column_stack([y + rng.normal(scale=0.8, size=len(y))] + [rng.normal(size=len(y)) for _ in range(5)])
    return {"if_score": if_score, "vae_score": vae_score}, X, [f"f{i}" for i in range(6)]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "reviewed_labels.csv")
        self.keys = make_keys(20)

    def _write(self, rows):
        pd.DataFrame(rows).to_csv(self.path, index=False)

    def test_consecutive_positive_months_collapse_into_one_episode_and_ids_keep_zeros(self):
        self._write([{"entity_id": "00003", "codmes": codmes(m), "target": 1} for m in (2, 3, 4)]
                    + [{"entity_id": "00003", "codmes": codmes(m), "target": 0} for m in (0, 1, 5, 6)])
        labels = load_event_labels(self.path, SCHEMA, self.keys)
        self.assertEqual(len(labels.episodes), 1)
        ep = labels.episodes.iloc[0]
        self.assertEqual((ep["entity"], int(ep["end_m"] - ep["start_m"]), int(ep["n_rows"])), ("00003", 2, 3))
        self.assertEqual(labels.audit["episode_source"], "derived")

    def test_washout_lets_a_short_gap_stay_inside_one_episode(self):
        rows = [{"entity_id": "00001", "codmes": codmes(m), "target": 1} for m in (2, 4)]
        self._write(rows)
        one = load_event_labels(self.path, SCHEMA, self.keys, LabelSpec(washout_months=1))
        two = load_event_labels(self.path, SCHEMA, self.keys, LabelSpec(washout_months=2))
        self.assertEqual((len(one.episodes), len(two.episodes)), (2, 1))

    def test_rows_absent_from_the_file_are_unknown_never_negative(self):
        self._write([{"entity_id": "00001", "codmes": codmes(3), "target": 0},
                     {"entity_id": "00002", "codmes": codmes(5), "target": 1}])
        labels = load_event_labels(self.path, SCHEMA, self.keys)
        self.assertEqual(int(labels.known.sum()), 2)
        self.assertEqual(int(np.isnan(labels.target).sum()), len(self.keys) - 2)
        self.assertEqual(labels.audit["unlisted_policy"], "unknown")

    def test_unlisted_as_negative_is_opt_in_and_still_skips_listed_but_excluded_rows(self):
        self._write([{"entity_id": "00001", "codmes": codmes(3), "target": 1, "label_status": "confirmed"},
                     {"entity_id": "00002", "codmes": codmes(4), "target": 1, "label_status": "pending"}])
        labels = load_event_labels(self.path, SCHEMA, self.keys, LabelSpec(unlisted_as_negative=True))
        pending_row = ((self.keys["entity_id"] == "00002")
                       & (self.keys["period"] == month_start(4))).to_numpy()
        self.assertFalse(labels.known[pending_row].any())          # pending stays unknown
        self.assertEqual(int(labels.known.sum()), len(self.keys) - 1)
        self.assertEqual(labels.audit["unlisted_policy"], "negative")

    def test_status_filter_conflicts_and_invalid_targets_are_counted_not_used(self):
        self._write([
            {"entity_id": "00001", "codmes": codmes(2), "target": 1, "label_status": "confirmed"},
            {"entity_id": "00001", "codmes": codmes(3), "target": 1, "label_status": "uncertain"},
            {"entity_id": "00001", "codmes": codmes(4), "target": 1, "label_status": "withdrawn"},
            {"entity_id": "00002", "codmes": codmes(2), "target": 1, "label_status": "confirmed"},
            {"entity_id": "00002", "codmes": codmes(2), "target": 0, "label_status": "confirmed"},
            {"entity_id": "00003", "codmes": codmes(2), "target": 7, "label_status": "confirmed"},
            {"entity_id": "00004", "codmes": codmes(2), "target": None, "label_status": "confirmed"},
        ])
        labels = load_event_labels(self.path, SCHEMA, self.keys)
        self.assertEqual(int(labels.known.sum()), 1)                 # only the first row
        self.assertEqual(labels.audit["excluded"].get("status:uncertain"), 1)
        self.assertEqual(labels.audit["excluded"].get("status:withdrawn"), 1)
        self.assertEqual(labels.audit["conflicting_duplicate_rows"], 2)
        self.assertEqual(labels.audit["invalid_targets"], 1)
        self.assertEqual(labels.audit["null_targets"], 1)

    def test_recent_episodes_and_negatives_are_immature_and_excluded_from_evaluation(self):
        last = N_MONTHS - 1
        self._write([{"entity_id": "00001", "codmes": codmes(last), "target": 1},
                     {"entity_id": "00002", "codmes": codmes(last), "target": 0},
                     {"entity_id": "00003", "codmes": codmes(last - 5), "target": 1},
                     {"entity_id": "00004", "codmes": codmes(last - 5), "target": 0}])
        labels = load_event_labels(self.path, SCHEMA, self.keys, LabelSpec(horizon_months=1))
        row = lambda e, m: int(np.flatnonzero((self.keys["entity_id"] == e).to_numpy()  # noqa: E731
                                              & (self.keys["period"] == month_start(m)).to_numpy())[0])
        self.assertFalse(labels.usable[row("00001", last)])   # positive whose horizon has not elapsed
        self.assertFalse(labels.usable[row("00002", last)])   # negative too recent to confirm
        self.assertTrue(labels.usable[row("00003", last - 5)])
        self.assertTrue(labels.usable[row("00004", last - 5)])
        self.assertEqual(labels.audit["episodes_mature"], 1)
        self.assertEqual(labels.audit["episodes"], 2)

    def test_provided_episode_ids_are_used_and_incomplete_ones_are_flagged(self):
        self._write([{"entity_id": "00001", "codmes": codmes(2), "target": 1, "episode_id": "A"},
                     {"entity_id": "00001", "codmes": codmes(8), "target": 1, "episode_id": "A"}])
        labels = load_event_labels(self.path, SCHEMA, self.keys)
        self.assertEqual((len(labels.episodes), labels.audit["episode_source"]), (1, "file"))
        self._write([{"entity_id": "00001", "codmes": codmes(2), "target": 1, "episode_id": "A"},
                     {"entity_id": "00001", "codmes": codmes(8), "target": 1, "episode_id": None}])
        flagged = load_event_labels(self.path, SCHEMA, self.keys)
        self.assertEqual(flagged.audit["positive_rows_without_episode_id"], 1)
        self.assertEqual(flagged.audit["episode_source"], "derived")

    def test_contract_violations_raise_a_clear_error(self):
        self._write([{"entity_id": "00001", "codmes": codmes(2), "outcome": 1}])
        with self.assertRaisesRegex(EventLabelsContractError, "target"):
            load_event_labels(self.path, SCHEMA, self.keys)
        self._write([{"entity_id": "NOPE", "codmes": codmes(2), "target": 1}])
        with self.assertRaisesRegex(EventLabelsContractError, "matches any panel"):
            load_event_labels(self.path, SCHEMA, self.keys)
        self._write([{"entity_id": "00001", "codmes": "not-a-month", "target": 1}])
        with self.assertRaises(EventLabelsContractError):
            load_event_labels(self.path, SCHEMA, self.keys)

    def test_column_names_can_be_mapped_and_as_of_sets_the_cutoff(self):
        self._write([{"cliente": "00001", "mes": codmes(2), "y": 1}])
        spec = LabelSpec(entity_col="cliente", period_col="mes", target_col="y", as_of="202206")
        labels = load_event_labels(self.path, SCHEMA, self.keys, spec)
        self.assertEqual(labels.audit["cutoff_month"], "2022-06")
        self.assertEqual(labels.audit["episodes_mature"], 1)  # ended in March, 3 months before as_of

    def test_find_labels_file_prefers_current_and_never_replaces_a_missing_explicit_path(self):
        os.makedirs(os.path.join(self.tmp.name, "current"))
        target = os.path.join(self.tmp.name, "current", "reviewed_labels.csv")
        pd.DataFrame({"entity_id": ["1"], "codmes": ["202201"], "target": [0]}).to_csv(target, index=False)
        pd.DataFrame({"entity_id": ["1"]}).to_csv(os.path.join(self.tmp.name, "other.csv"), index=False)
        self.assertEqual(find_labels_file(None, self.tmp.name), os.path.abspath(target))
        self.assertEqual(find_labels_file(target, None), os.path.abspath(target))
        self.assertIsNone(find_labels_file(os.path.join(self.tmp.name, "missing.csv"), self.tmp.name))
        self.assertIsNone(find_labels_file(None, os.path.join(self.tmp.name, "empty_dir_that_is_absent")))


class HorizonTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "labels.csv")
        self.keys = make_keys(5)
        rows = [{"entity_id": "00001", "codmes": codmes(m), "target": int(m in (10, 11))}
                for m in range(N_MONTHS)]
        pd.DataFrame(rows).to_csv(self.path, index=False)
        self.labels = load_event_labels(self.path, SCHEMA, self.keys)

    def _row(self, m):
        return int(np.flatnonzero((self.keys["entity_id"] == "00001").to_numpy()
                                  & (self.keys["period"] == month_start(m)).to_numpy())[0])

    def test_rows_before_an_onset_are_positive_and_rows_inside_the_episode_are_not_at_risk(self):
        y, eligible, episode = self.labels.horizon_target(3)
        for m in (7, 8, 9):
            self.assertEqual((y[self._row(m)], eligible[self._row(m)]), (1.0, True), m)
            self.assertTrue(episode[self._row(m)].startswith("00001#"))
        self.assertEqual((y[self._row(6)], eligible[self._row(6)]), (0.0, True))  # onset is 4 months away
        for m in (10, 11):
            self.assertFalse(eligible[self._row(m)])                # already inside the episode
        self.assertEqual((y[self._row(15)], eligible[self._row(15)]), (0.0, True))

    def test_right_censored_rows_are_dropped_not_called_negative(self):
        _y, eligible, _e = self.labels.horizon_target(3)
        self.assertTrue(eligible[self._row(N_MONTHS - 4)])          # t+3 = last observed month
        self.assertFalse(eligible[self._row(N_MONTHS - 3)])
        self.assertFalse(eligible[self._row(N_MONTHS - 1)])


class GateTests(unittest.TestCase):
    def _metrics(self, **over):
        base = dict(eligible_rows=1000, n_positive_rows=40, n_confirmed_negative_rows=900,
                    n_mature_positive_episodes=30, n_positive_entities=12, n_temporal_origins=3,
                    max_single_month_share=0.2, oos_positive_episodes=10, n_audited_nonalerts=None)
        return {**base, **over}

    def test_fewer_than_30_mature_episodes_keeps_if_vae_only(self):
        gate = decide_gate(self._metrics(n_mature_positive_episodes=29), {})
        self.assertEqual((gate["level"], gate["supervised_challengers_allowed"]), ("rojo", False))
        self.assertEqual(gate["authorized_families"], ["iforest", "vae"])
        self.assertTrue(any("30" in u for u in gate["unmet_for_next_level"]))

    def test_30_to_99_authorises_penalised_logistic_and_hazard_only(self):
        gate = decide_gate(self._metrics(), {})
        self.assertEqual(gate["level"], "ambar_1")
        self.assertEqual(gate["implemented_families"],
                         ["iforest", "vae", "logit_scores_head", "ridge_logistic", "discrete_hazard"])
        self.assertTrue(gate["supervised_challengers_allowed"])

    def test_count_alone_is_not_enough_evidence_requirements_must_hold_too(self):
        for over in ({"n_positive_entities": 9}, {"n_temporal_origins": 1},
                     {"max_single_month_share": 0.8}):
            gate = decide_gate(self._metrics(**over), {})
            self.assertEqual(gate["level"], "rojo", over)
            self.assertTrue(gate["unmet_for_next_level"], over)

    def test_higher_levels_and_the_formal_calculation_requirement(self):
        pilot = self._metrics(n_mature_positive_episodes=120, n_positive_entities=25, oos_positive_episodes=60)
        self.assertEqual(decide_gate(pilot, {})["level"], "ambar_2")
        big = self._metrics(n_mature_positive_episodes=250, n_positive_entities=40, oos_positive_episodes=80)
        self.assertEqual(decide_gate(big, {})["level"], "ambar_2")   # no formal calc yet
        green = decide_gate(big, {}, formal_calculation_ok=True)
        self.assertEqual(green["level"], "verde_condicionado")
        self.assertIn("gradient_boosting_restricted", green["not_implemented_families"])

    def test_any_veto_overrides_the_count(self):
        big = self._metrics(n_mature_positive_episodes=250, n_positive_entities=40, oos_positive_episodes=80)
        vetoed = decide_gate(big, {"invalid_targets": 3}, formal_calculation_ok=True)
        self.assertEqual((vetoed["level"], vetoed["level_by_count"]), ("rojo", "verde_condicionado"))
        self.assertEqual([v["code"] for v in vetoed["vetoes"]], ["invalid_target_values"])
        for audit, code in (({"conflicting_duplicate_rows": 1}, "label_conflicts"),
                            ({"unlisted_policy": "negative"}, "unreviewed_converted_to_negative"),
                            ({"positive_rows_without_episode_id": 2}, "episodes_not_collapsed")):
            self.assertIn(code, [v["code"] for v in decide_gate(big, audit)["vetoes"]])
        attested = decide_gate(big, {"unlisted_policy": "negative"}, unlisted_attested=True)
        self.assertFalse(attested["vetoed"])
        none = decide_gate(self._metrics(eligible_rows=0), {})
        self.assertIn("no_usable_labels", [v["code"] for v in none["vetoes"]])
        no_neg = decide_gate(self._metrics(n_confirmed_negative_rows=0), {})
        self.assertIn("no_confirmed_negatives", [v["code"] for v in no_neg["vetoes"]])

    def test_sufficiency_metrics_come_from_the_real_loader_output(self):
        keys = make_keys(150)
        plan = episode_plan(45)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "l.csv")
            write_labels(path, keys, plan)
            labels = load_event_labels(path, SCHEMA, keys)
        m = sufficiency_metrics(labels, make_masks(keys)["test"] | make_masks(keys)["oot"])
        mature = [p for p in plan if p[1] + p[2] - 1 + 1 <= N_MONTHS - 1]  # end + lag(1) <= last month
        self.assertEqual(m["n_positive_episodes"], len(plan))
        self.assertEqual(m["n_mature_positive_episodes"], len(mature))
        self.assertEqual(m["n_positive_entities"], len({p[0] for p in mature}))
        self.assertEqual(m["n_unreviewed_rows"], 0)
        self.assertIsNone(m["n_audited_nonalerts"])          # unknown, not invented
        self.assertGreaterEqual(m["n_temporal_origins"], 2)
        self.assertAlmostEqual(m["event_rate_row"], m["n_positive_rows"] / m["eligible_rows"])


class EvaluationTests(unittest.TestCase):
    def _frame(self, n=1000, n_pos=50, seed=0):
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        y[rng.choice(n, n_pos, replace=False)] = 1
        ent = np.array([f"e{i % 200}" for i in range(n)], dtype=object)
        episode = np.where(y == 1, [f"ep{i}" for i in range(n)], "").astype(object)
        return y, ent, episode, np.ones(n, dtype=bool)

    def test_a_perfect_ranking_scores_one_and_a_random_one_scores_near_the_base_rate(self):
        y, ent, ep, elig = self._frame()
        perfect = evaluate_ranking(y + 0.0, y, elig, ent, ep, k=50, n_boot=0)
        self.assertEqual((perfect["ap"], perfect["precision_at_k"], perfect["recall_at_k"]), (1.0, 1.0, 1.0))
        self.assertEqual((perfect["false_positives_at_k"], perfect["episode_recall_at_k"]), (0, 1.0))
        random = evaluate_ranking(np.random.default_rng(3).normal(size=len(y)), y, elig, ent, ep, k=50, n_boot=0)
        self.assertLess(random["ap"], 0.15)
        self.assertLess(abs(random["lift_at_k"] - 1.0), 1.0)

    def test_unknown_and_ineligible_rows_never_count_as_negatives(self):
        y, ent, ep, elig = self._frame()
        y_nan = y.copy()
        y_nan[500:] = np.nan
        res = evaluate_ranking(y + 0.0, y_nan, elig, ent, ep, k=10, n_boot=0)
        self.assertEqual(res["eligible_rows"], 500)
        half = elig.copy()
        half[:500] = False
        self.assertEqual(evaluate_ranking(y + 0.0, y, half, ent, ep, n_boot=0)["eligible_rows"], 500)

    def test_fewer_than_20_positive_episodes_is_descriptive_only(self):
        y, ent, ep, elig = self._frame(n_pos=15)
        few = evaluate_ranking(y + 0.0, y, elig, ent, ep, n_boot=0)
        self.assertFalse(few["conclusive"])
        self.assertIn("descriptivo", few["note"])
        y2, ent2, ep2, elig2 = self._frame(n_pos=25)
        self.assertTrue(evaluate_ranking(y2 + 0.0, y2, elig2, ent2, ep2, n_boot=0)["conclusive"])

    def test_windows_without_positives_or_rows_report_a_status_instead_of_a_metric(self):
        y, ent, ep, elig = self._frame()
        self.assertEqual(evaluate_ranking(y, y * 0, elig, ent, ep)["status"], "no_positives")
        self.assertEqual(evaluate_ranking(y, y, elig & False, ent, ep)["status"], "no_eligible_rows")

    def test_cluster_bootstrap_gives_an_interval_that_contains_the_estimate_for_a_noisy_ranking(self):
        y, ent, ep, elig = self._frame(n=1500, n_pos=90)
        s = y * 1.2 + np.random.default_rng(5).normal(size=len(y))
        res = evaluate_ranking(s, y, elig, ent, ep, n_boot=60, seed=1)
        low, high = res["ci"]["ap"]
        self.assertLess(low, res["ap"] + 1e-9)
        self.assertGreater(high, res["ap"] - 0.05)
        self.assertLess(low, high)

    def test_calibration_slope_needs_enough_events(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.6, size=500)
        y = (rng.uniform(size=500) < p).astype(float)
        cal = calibration_summary(p, y)
        self.assertIsNotNone(cal["calibration_slope"])
        self.assertLess(cal["brier"], 0.25)
        few = calibration_summary(p[:40], np.r_[np.ones(3), np.zeros(37)])
        self.assertIsNone(few["calibration_slope"])


class SupervisionPhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keys = make_keys(150)
        self.masks = make_masks(self.keys)

    def _run(self, plan, *, mode="auto", labels_path="auto", n_boot=20, **kw):
        path = os.path.join(self.tmp.name, "reviewed_labels.csv")
        if labels_path == "auto":
            write_labels(path, self.keys, plan)
            labels_path = path
        scores, X, names = learnable_scores(self.keys, plan)
        cfg = EventSupervisionConfig(
            labels_path=labels_path, labels_dir=None, out_dir=os.path.join(self.tmp.name, "out"),
            challenger=ChallengerConfig(n_boot=n_boot), challenger_mode=mode, **kw)
        return run_event_supervision(cfg=cfg, schema=SCHEMA, keys=self.keys, masks=self.masks,
                                     scores=scores, X=X, feature_names=names)

    def test_no_labels_file_falls_back_to_the_unsupervised_run(self):
        result = self._run(episode_plan(45), labels_path=os.path.join(self.tmp.name, "absent.csv"))
        self.assertEqual(result["status"], "no_labels_file")
        self.assertIn("IF/VAE", result["reason"])

    def test_a_file_that_breaks_the_contract_falls_back_with_the_reason(self):
        bad = os.path.join(self.tmp.name, "bad.csv")
        pd.DataFrame({"entity_id": ["00001"], "codmes": ["202201"]}).to_csv(bad, index=False)
        result = self._run(episode_plan(45), labels_path=bad)
        self.assertEqual(result["status"], "contract_error")
        self.assertIn("target", result["reason"])

    def _run_dir(self, labels_dir):
        plan = episode_plan(45)
        scores, X, names = learnable_scores(self.keys, plan)
        cfg = EventSupervisionConfig(labels_path=None, labels_dir=labels_dir,
                                     out_dir=os.path.join(self.tmp.name, "out"))
        return run_event_supervision(cfg=cfg, schema=SCHEMA, keys=self.keys, masks=self.masks,
                                     scores=scores, X=X, feature_names=names)

    def test_an_empty_or_missing_labels_folder_is_ignored_without_error(self):
        empty = os.path.join(self.tmp.name, "reviewed_labels")
        os.makedirs(empty)
        self.assertEqual(self._run_dir(empty)["status"], "no_labels_file")
        self.assertEqual(self._run_dir(os.path.join(self.tmp.name, "never_created"))["status"],
                         "no_labels_file")
        open(os.path.join(empty, "README.md"), "w").write("not a table")
        os.makedirs(os.path.join(empty, "quarantine"))
        pd.DataFrame({"entity_id": ["1"]}).to_csv(os.path.join(empty, "quarantine", "x.csv"), index=False)
        self.assertEqual(self._run_dir(empty)["status"], "no_labels_file")   # only README + quarantine

    def test_a_file_with_nothing_in_it_is_ignored_without_error(self):
        cases = {
            "zero_bytes": b"",
            "header_only": b"entity_id,codmes,target\n",
            "blank_lines": b"\n\n   \n",
            "all_null_target": b"entity_id,codmes,target\n00001,202201,\n00002,202202,\n",
        }
        for name, content in cases.items():
            path = os.path.join(self.tmp.name, f"{name}.csv")
            with open(path, "wb") as fh:
                fh.write(content)
            result = self._run(episode_plan(45), labels_path=path)
            self.assertEqual(result["status"], "empty_labels_file", name)
            self.assertIn("IF/VAE", result["reason"], name)
            self.assertNotIn("artifacts", result, name)              # nothing was written

    def test_a_corrupt_binary_file_is_ignored_without_error(self):
        path = os.path.join(self.tmp.name, "garbage.csv")
        with open(path, "wb") as fh:
            fh.write(os.urandom(2048))
        result = self._run(episode_plan(45), labels_path=path)
        self.assertIn(result["status"], ("contract_error", "empty_labels_file"))
        self.assertIn("IF/VAE", result["reason"])

    def test_a_file_where_every_row_is_pending_runs_but_uses_nothing(self):
        path = os.path.join(self.tmp.name, "pending.csv")
        pd.DataFrame({"entity_id": ["00001", "00002"], "codmes": [codmes(3), codmes(4)],
                      "target": [1, 0], "label_status": ["pending", "pending"]}).to_csv(path, index=False)
        result = self._run(episode_plan(45), labels_path=path)
        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["gate"]["level"], "rojo")
        self.assertIn("no_usable_labels", [v["code"] for v in result["gate"]["vetoes"]])
        self.assertEqual(result["challengers"]["status"], "skipped")

    def test_below_30_episodes_evaluates_if_vae_but_runs_no_challenger(self):
        result = self._run(episode_plan(12))
        self.assertEqual(result["gate"]["level"], "rojo")
        self.assertEqual(result["challengers"]["status"], "skipped")
        self.assertIn("IF/VAE", result["challengers"]["reason"])
        cur = result["evaluation"]["current_month"]["if_score"]["oot"]
        self.assertIn(cur["status"], ("ok", "no_positives"))
        for path in result["artifacts"].values():
            self.assertTrue(os.path.isfile(path), path)
        acta = json.load(open(result["artifacts"]["label_gate"], encoding="utf-8"))
        self.assertEqual(acta["gate"]["level"], "rojo")

    def test_30_to_99_episodes_runs_the_exploratory_challengers_and_compares_them_to_if_vae(self):
        result = self._run(episode_plan(60))
        gate = result["gate"]
        self.assertEqual(gate["level"], "ambar_1", gate["unmet_for_next_level"])
        chal = result["challengers"]
        self.assertEqual(chal["status"], "executed")
        for name in ("logit_scores_head", "ridge_logistic", "discrete_hazard"):
            fam = chal["families"][name]
            self.assertEqual(fam["status"], "executed", (name, fam.get("reason")))
            self.assertGreater(fam["fit_positive_episodes"], 4)
            warned = any("10 episodios por par" in w for w in fam["warnings"])
            # events/parameter = fit episodes / (features + 1): the 2-score head is
            # the only family cheap enough to clear 10 here; the 6-feature ridge and
            # the hazard (fewer onset episodes) must say they are exploratory.
            self.assertEqual(warned, fam["events_per_parameter"] < 10, name)
            if name != "logit_scores_head":
                self.assertTrue(warned, name)
            self.assertEqual(set(fam["windows"]), {"test", "oot"})
            self.assertEqual(set(fam["baselines"]), {"if_score", "vae_score"})
        ridge = chal["families"]["ridge_logistic"]
        self.assertEqual(ridge["top_coefficients"][0]["feature"], "f0")     # the only informative column
        oot = ridge["windows"]["oot"]
        self.assertEqual(oot["status"], "ok")
        self.assertIn("brier", oot["calibration"])
        self.assertGreater(oot["ap"], oot["ap_baseline"])                    # learnable signal is found
        csv = pd.read_csv(result["artifacts"]["event_evaluation"])
        self.assertTrue({"detector", "challenger"} <= set(csv["kind"]))
        self.assertIn("onset_within_3m", set(csv["target"]))

    def test_fit_rows_never_reach_into_the_evaluation_months(self):
        result = self._run(episode_plan(60))
        fam = result["challengers"]["families"]["ridge_logistic"]
        # train+val end at month 15; evaluation starts at month 16; the purge (lag 1)
        # removes month 15, so no fit row can see the evaluation window's labels.
        self.assertEqual(fam["purge_months"], 1)
        hazard = result["challengers"]["families"]["discrete_hazard"]
        self.assertEqual(hazard["purge_months"], 3)
        self.assertLess(hazard["fit_rows"], fam["fit_rows"])

    def test_challenger_mode_off_and_force(self):
        off = self._run(episode_plan(60), mode="off")
        self.assertEqual(off["challengers"]["status"], "skipped")
        forced = self._run(episode_plan(12), mode="force")
        self.assertTrue(forced["challengers"]["forced"])
        statuses = {n: f["status"] for n, f in forced["challengers"]["families"].items()}
        self.assertTrue(set(statuses.values()) <= {"executed", "skipped", "failed"})
        skipped = [f for f in forced["challengers"]["families"].values() if f["status"] == "skipped"]
        for fam in skipped:
            self.assertTrue(fam["reason"])            # every fallback says why

    def test_too_few_training_episodes_skips_a_family_with_its_reason(self):
        plan = [(e, 18, 2) for e, _s, _l in episode_plan(40)]   # every episode in the evaluation window
        result = self._run(plan, mode="force")
        for fam in result["challengers"]["families"].values():
            self.assertEqual(fam["status"], "skipped")
            self.assertTrue("clase" in fam["reason"] or "episodio" in fam["reason"])

    def test_a_partial_file_leaves_unlisted_rows_out_of_every_denominator(self):
        plan = episode_plan(60)
        listed = {(e, i) for e, i in positive_rows(plan)}
        path = os.path.join(self.tmp.name, "partial.csv")
        write_labels(path, self.keys, plan, listed=listed)          # positives only, no negatives
        result = self._run(plan, labels_path=path)
        self.assertEqual(result["gate"]["level"], "rojo")
        self.assertIn("no_confirmed_negatives", [v["code"] for v in result["gate"]["vetoes"]])
        self.assertGreater(result["gate"]["metrics"]["n_unreviewed_rows"], 3000)


if __name__ == "__main__":
    unittest.main()
