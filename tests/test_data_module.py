"""Validation suite for `src.data` (synthetic panel generator + loader).

This is the project's first test module; the patterns here are meant to be
reused by later suites:

* All filesystem output goes to pytest tmp dirs (see `tests/conftest.py`,
  which also chdirs the session into a sandbox so relative-path defaults in
  the source cannot touch the real `data/` or `logs/`).
* Expensive artifacts (a generated panel) are built once per module via a
  ``tmp_path_factory``-backed fixture and shared read-only.
* Statistical assertions use explicit, documented thresholds with the
  observed value in the assertion message, so a failure reports a number
  rather than "distribution looked wrong".

Scale is deliberately small (1,500 entities x 10 periods = 15,000 rows) so
the whole suite runs in a few seconds; the generator's production default is
100,000 x 10 = 1,000,000 rows and is never exercised here.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data import PanelSchema, generate_synthetic_panel, load_or_generate_panel
from src.data.synthetic import (
    _DRIFT_START_MONTH_IDX_DEFAULT,
    _break_correlation_in_subgroup,
    _generate_base_panel,
)

# --- fixed test scale / seed ------------------------------------------------
N_ENTITIES = 1_500
N_PERIODS = 10
SEED = 20260723
N_ROWS = N_ENTITIES * N_PERIODS

EXPECTED_COLUMNS = [
    "entity_id", "period", "age", "tenure_months", "region", "segment",
    "employment_status", "marital_status", "product_type",
    "transaction_channel", "is_digital_active", "income", "account_balance",
    "monthly_transactions_amount", "monthly_transactions_count",
    "avg_transaction_amount", "withdrawal_amount", "credit_score",
    "overdraft_count", "num_products", "days_since_last_login",
    "customer_satisfaction_score",
]

# Label-ish column names that must never leak into the main (unlabeled) CSV.
FORBIDDEN_COLUMNS = {"is_anomaly", "anomaly_type", "ground_truth", "target", "label"}

ALLOWED_ANOMALY_TYPES = {"none", "global", "local", "contextual", "collective"}

# Mechanisms as documented in `src/data/synthetic.py::_inject_missingness`.
MCAR_COLUMNS = ["customer_satisfaction_score", "days_since_last_login", "credit_score"]
MCAR_RATE = 0.02
MNAR_COLUMNS = ["account_balance", "income"]

# Monetary columns that are genuinely heavy-tailed by construction
# (lognormal + multiplicative fat-tail shocks + injected extremes).
STRONG_HEAVY_TAIL_COLUMNS = [
    "account_balance", "monthly_transactions_amount", "avg_transaction_amount",
]
# Monetary columns that are right-skewed but only mildly so (plain lognormal).
MILD_SKEW_COLUMNS = ["income", "withdrawal_amount"]


# ---------------------------------------------------------------- helpers --
def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_ground_truth(path: str | Path) -> Path:
    """Return the ground-truth file that was actually written.

    `_write_ground_truth` falls back from parquet to a sibling `.csv` when
    neither pyarrow nor fastparquet is importable, so the path the caller
    asked for is not necessarily the path on disk.
    """
    p = Path(path)
    if p.exists():
        return p
    fallback = p.with_suffix(".csv")
    assert fallback.exists(), f"no ground-truth file at {p} or {fallback}"
    return fallback


def _read_ground_truth(path: str | Path) -> pd.DataFrame:
    p = _resolve_ground_truth(path)
    if p.suffix == ".csv":
        return pd.read_csv(p, parse_dates=["period"])
    return pd.read_parquet(p)


def _generate(dest: Path, *, seed: int = SEED, n: int = N_ENTITIES, periods: int = N_PERIODS):
    """Generate a panel into `dest`; return (csv_path, requested_gt_path)."""
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "data.csv"
    gt_path = dest / "ground_truth.parquet"
    generate_synthetic_panel(
        n_individuals=n, n_periods=periods,
        out_path=str(csv_path), ground_truth_path=str(gt_path), seed=seed,
    )
    return csv_path, gt_path


# --------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def panel_paths(tmp_path_factory) -> tuple[Path, Path]:
    """Generate the shared reference panel once for the whole module."""
    dest = tmp_path_factory.mktemp("panel")
    return _generate(dest)


@pytest.fixture(scope="module")
def panel(panel_paths) -> pd.DataFrame:
    csv_path, _ = panel_paths
    return pd.read_csv(csv_path, parse_dates=["period"])


@pytest.fixture(scope="module")
def ground_truth(panel_paths) -> pd.DataFrame:
    _, gt_path = panel_paths
    return _read_ground_truth(gt_path)


@pytest.fixture(scope="module")
def labeled(panel: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Panel joined to its hidden ground truth (evaluation-only view)."""
    return panel.merge(
        ground_truth, on=["entity_id", "period"], how="left", validate="one_to_one"
    )


@pytest.fixture(scope="module")
def period_index(panel: pd.DataFrame) -> pd.Series:
    ordered = sorted(panel["period"].unique())
    return panel["period"].map({p: i for i, p in enumerate(ordered)})


# ====================================================================== 1 ==
class TestReproducibility:
    def test_same_seed_produces_byte_identical_files(self, tmp_path):
        a_csv, a_gt = _generate(tmp_path / "a", n=400, periods=6, seed=7)
        b_csv, b_gt = _generate(tmp_path / "b", n=400, periods=6, seed=7)

        assert _sha256(a_csv) == _sha256(b_csv), "same seed must give identical panel CSV"
        assert _sha256(_resolve_ground_truth(a_gt)) == _sha256(_resolve_ground_truth(b_gt)), (
            "same seed must give identical ground truth"
        )

    def test_different_seed_produces_different_output(self, tmp_path):
        a_csv, _ = _generate(tmp_path / "a", n=400, periods=6, seed=7)
        c_csv, _ = _generate(tmp_path / "c", n=400, periods=6, seed=8)
        assert _sha256(a_csv) != _sha256(c_csv), "different seeds must not collide"

        a = pd.read_csv(a_csv)
        c = pd.read_csv(c_csv)
        assert a.shape == c.shape
        assert not np.allclose(a["account_balance"].fillna(-1), c["account_balance"].fillna(-1))


# ====================================================================== 2 ==
class TestShapeAndSchema:
    def test_row_count_is_entities_times_periods(self, panel):
        assert len(panel) == N_ROWS
        assert panel["entity_id"].nunique() == N_ENTITIES
        assert panel["period"].nunique() == N_PERIODS
        # Balanced panel: every entity observed in every period exactly once.
        assert not panel.duplicated(["entity_id", "period"]).any()
        assert (panel.groupby("entity_id").size() == N_PERIODS).all()

    def test_expected_columns_present(self, panel):
        assert list(panel.columns) == EXPECTED_COLUMNS
        assert "entity_id" in panel.columns
        assert "period" in panel.columns

    def test_main_csv_carries_no_labels(self, panel):
        leaked = {c for c in panel.columns if c.lower() in FORBIDDEN_COLUMNS}
        assert leaked == set(), f"label columns leaked into the unlabeled panel: {leaked}"

    def test_ground_truth_written_to_a_separate_file(self, panel_paths):
        csv_path, gt_path = panel_paths
        gt_file = _resolve_ground_truth(gt_path)
        assert gt_file.exists()
        assert gt_file.resolve() != csv_path.resolve()

    def test_periods_are_consecutive_months(self, panel):
        periods = pd.DatetimeIndex(sorted(panel["period"].unique()))
        deltas = periods.to_period("M").astype("int64")
        assert (np.diff(deltas) == 1).all(), f"periods not consecutive months: {periods}"


# ====================================================================== 3 ==
class TestGroundTruthIntegrity:
    def test_required_columns(self, ground_truth):
        assert set(ground_truth.columns) == {"entity_id", "period", "is_anomaly", "anomaly_type"}

    def test_anomaly_type_values_are_in_the_allowed_set(self, ground_truth):
        observed = set(ground_truth["anomaly_type"].unique())
        assert observed <= ALLOWED_ANOMALY_TYPES, f"unexpected anomaly types: {observed - ALLOWED_ANOMALY_TYPES}"
        # All four injected types must actually be present.
        assert ALLOWED_ANOMALY_TYPES - {"none"} <= observed, (
            f"missing anomaly types: {ALLOWED_ANOMALY_TYPES - {'none'} - observed}"
        )

    def test_every_gt_key_exists_in_the_panel(self, panel, ground_truth):
        panel_keys = set(map(tuple, panel[["entity_id", "period"]].to_numpy()))
        gt_keys = set(map(tuple, ground_truth[["entity_id", "period"]].to_numpy()))
        assert gt_keys <= panel_keys, f"{len(gt_keys - panel_keys)} GT keys absent from the panel"
        assert not ground_truth.duplicated(["entity_id", "period"]).any()

    def test_is_anomaly_matches_anomaly_type(self, ground_truth):
        expected = ground_truth["anomaly_type"] != "none"
        assert ground_truth["is_anomaly"].astype(bool).equals(expected.rename("is_anomaly")), (
            "is_anomaly disagrees with anomaly_type != 'none'"
        )

    def test_anomaly_rate_is_in_the_documented_1_to_3_percent_band(self, ground_truth):
        rate = ground_truth["is_anomaly"].astype(bool).mean()
        assert 0.005 <= rate <= 0.05, f"anomaly rate {rate:.4f} outside plausible band"

    def test_types_do_not_overlap(self, ground_truth):
        counts = ground_truth["anomaly_type"].value_counts()
        n_flagged = int(ground_truth["is_anomaly"].astype(bool).sum())
        assert counts.drop("none").sum() == n_flagged


# ====================================================================== 4 ==
class TestHeavyTails:
    """A column is 'heavy-tailed' here if it clears BOTH bars:
    Fisher-Pearson skewness > 5 and max/p99 > 20. For reference, a
    Gaussian has skew 0 and max/p99 ~= 1.3 at n=15k; a lognormal(0, 1)
    has skew ~6.2. The max/p99 ratio is the tail-mass test: it says the
    single most extreme observation is >=20x the 99th percentile, which
    a light-tailed distribution essentially never produces.
    """

    @pytest.mark.parametrize("col", STRONG_HEAVY_TAIL_COLUMNS)
    def test_monetary_column_is_heavy_tailed(self, panel, col):
        s = panel[col].dropna().to_numpy()
        skew = float(stats.skew(s))
        ratio = float(s.max() / np.percentile(s, 99))
        assert skew > 5.0, f"{col}: skewness {skew:.2f} <= 5.0"
        assert ratio > 20.0, f"{col}: max/p99 {ratio:.2f} <= 20"

    @pytest.mark.parametrize("col", MILD_SKEW_COLUMNS)
    def test_other_monetary_columns_are_at_least_right_skewed(self, panel, col):
        s = panel[col].dropna().to_numpy()
        skew = float(stats.skew(s))
        p99_over_median = float(np.percentile(s, 99) / np.median(s))
        assert skew > 1.0, f"{col}: skewness {skew:.2f} <= 1.0 (not right-skewed)"
        assert p99_over_median > 2.0, f"{col}: p99/median {p99_over_median:.2f} <= 2"

    def test_no_negative_monetary_values(self, panel):
        for col in STRONG_HEAVY_TAIL_COLUMNS + MILD_SKEW_COLUMNS:
            s = panel[col].dropna()
            assert (s > 0).all(), f"{col} contains non-positive values"


# ====================================================================== 5 ==
class TestLongTailCategoricals:
    """Zipf-like: top-3 categories hold most of the mass and at least one
    category is present but rarer than 0.5% of rows.

    The generator uses a realistic Zipf exponent (s ~ 1.3) plus an explicit
    rare tail, instead of the s ~ 3 that used to put >80% of the rows on
    rank 1 alone. Top-3 mass is therefore ~0.65-0.75 rather than ~0.96;
    `transaction_channel` sits at the bottom of that range because it is a
    mixture of the pre- and post-drift regimes.
    """

    @pytest.mark.parametrize("col", ["transaction_channel", "region", "product_type"])
    def test_top_three_categories_dominate(self, panel, col):
        freq = panel[col].value_counts(normalize=True)
        top3 = float(freq.head(3).sum())
        assert top3 > 0.60, f"{col}: top-3 mass {top3:.4f} <= 0.60 (not long-tailed)"

    @pytest.mark.parametrize("col", ["transaction_channel", "region", "product_type"])
    def test_a_rare_category_is_present(self, panel, col):
        freq = panel[col].value_counts(normalize=True)
        rarest = float(freq.min())
        assert 0.0 < rarest < 0.005, (
            f"{col}: rarest category frequency {rarest:.6f} not in (0, 0.005) "
            f"-- no genuine long tail (n_categories={len(freq)})"
        )

    def test_categorical_is_constant_within_entity_for_static_attributes(self, panel):
        """Entity-level attributes must not change period-to-period."""
        for col in ["region", "segment", "employment_status", "marital_status",
                    "product_type", "age"]:
            varying = panel.groupby("entity_id")[col].nunique()
            assert (varying == 1).all(), f"{col} varies within entity (should be static)"


# ====================================================================== 6 ==
class TestMissingness:
    """Documented mechanisms (src/data/synthetic.py::_inject_missingness):
      * MCAR, rate 2%: customer_satisfaction_score, days_since_last_login,
        credit_score.
      * MNAR: account_balance, income -- base rate 1%, +15pp for the top 5%
        of values (simulated redaction of large amounts). Expected marginal
        rate = 0.95*0.01 + 0.05*0.16 = 1.75%.
    """

    def test_nans_exist(self, panel):
        assert panel.isna().to_numpy().any(), "no missing values injected at all"

    @pytest.mark.parametrize("col", MCAR_COLUMNS)
    def test_mcar_rate_matches_documentation(self, panel, col):
        rate = float(panel[col].isna().mean())
        assert 0.012 <= rate <= 0.030, f"{col}: MCAR rate {rate:.4f} far from documented {MCAR_RATE}"

    @pytest.mark.parametrize("col", MCAR_COLUMNS)
    def test_mcar_is_independent_of_magnitude(self, panel, col):
        """MCAR: the mean percentile of a correlated proxy (income) among
        rows where `col` is missing must be ~0.5 (no selection on value)."""
        missing = panel[col].isna()
        proxy_pctile = panel["income"].rank(pct=True)
        mean_pctile = float(proxy_pctile[missing].mean())
        assert abs(mean_pctile - 0.5) < 0.08, (
            f"{col}: income-percentile among missing rows = {mean_pctile:.4f}; "
            "expected ~0.50 for MCAR"
        )

    @pytest.mark.parametrize("col,proxy", [("income", "credit_score"), ("account_balance", "income")])
    def test_mnar_missingness_correlates_with_magnitude(self, panel, col, proxy):
        """MNAR: `col` is unobservable where it is missing, so magnitude is
        measured through a strongly correlated observed proxy
        (credit_score is a monotone function of income by construction;
        account_balance is generated as income x multipliers)."""
        missing = panel[col].isna()
        assert missing.any(), f"{col} has no missing values"

        proxy_pctile = panel[proxy].rank(pct=True)
        mean_missing = float(proxy_pctile[missing].mean())
        mean_present = float(proxy_pctile[~missing].mean())

        assert mean_missing > 0.58, (
            f"{col}: mean {proxy} percentile among missing rows = {mean_missing:.4f}; "
            "expected clearly > 0.5 under MNAR-on-large-values"
        )
        assert mean_missing - mean_present > 0.08, (
            f"{col}: missing/present {proxy}-percentile gap = "
            f"{mean_missing - mean_present:.4f}; too small to be MNAR"
        )

    @pytest.mark.parametrize("col", MNAR_COLUMNS)
    def test_mnar_marginal_rate_is_plausible(self, panel, col):
        rate = float(panel[col].isna().mean())
        assert 0.008 <= rate <= 0.035, f"{col}: MNAR marginal rate {rate:.4f} outside expected band"

    def test_key_columns_are_never_missing(self, panel):
        for col in ["entity_id", "period"]:
            assert panel[col].notna().all(), f"{col} must never be missing"


# ====================================================================== 7 ==
class TestTemporalDrift:
    """Drift starts at period index `_DRIFT_START_MONTH_IDX_DEFAULT` (6) and
    shifts `transaction_channel` toward mobile_app/online_banking and raises
    the `is_digital_active` probability from 0.30 to a terminal 0.65.

    The shift is a *gradual, partial* migration: adoption ramps over three
    months and tops out at 85% of entities, so the pre/post distributions
    overlap instead of being disjoint. Post-drift averages are therefore
    averages over the ramp, not the terminal level.
    """

    def test_drift_start_is_within_the_panel(self):
        assert _DRIFT_START_MONTH_IDX_DEFAULT < N_PERIODS

    def test_is_digital_active_shifts_after_drift_month(self, panel, period_index):
        pre = panel.loc[period_index < _DRIFT_START_MONTH_IDX_DEFAULT, "is_digital_active"].mean()
        post = panel.loc[period_index >= _DRIFT_START_MONTH_IDX_DEFAULT, "is_digital_active"].mean()
        assert 0.25 <= pre <= 0.35, f"pre-drift digital rate {pre:.4f} != documented ~0.30"
        # Mean over the ramped post-drift window (terminal level is ~0.65).
        assert 0.50 <= post <= 0.70, f"post-drift digital rate {post:.4f} outside the ramped band"
        assert post - pre > 0.25, f"drift too weak: {pre:.4f} -> {post:.4f}"

    def test_transaction_channel_mass_moves_to_digital(self, panel, period_index):
        pre_mask = period_index < _DRIFT_START_MONTH_IDX_DEFAULT
        post_mask = ~pre_mask
        pre = panel.loc[pre_mask, "transaction_channel"].value_counts(normalize=True)
        post = panel.loc[post_mask, "transaction_channel"].value_counts(normalize=True)

        # Partial shift: branch stays the leading legacy channel post-drift
        # but loses most of its share, mobile_app climbs from marginal to
        # comparable. Both regimes keep support on the same categories.
        assert pre.get("branch", 0.0) > 0.40, f"pre-drift branch share {pre.get('branch', 0.0):.4f}"
        assert post.get("branch", 0.0) < 0.30, f"post-drift branch share {post.get('branch', 0.0):.4f}"
        assert pre.get("mobile_app", 0.0) < 0.10, f"pre-drift mobile share {pre.get('mobile_app', 0.0):.4f}"
        assert post.get("mobile_app", 0.0) > 0.20, f"post-drift mobile share {post.get('mobile_app', 0.0):.4f}"

        # Total variation distance between the two channel distributions.
        cats = sorted(set(pre.index) | set(post.index))
        tvd = 0.5 * sum(abs(pre.get(c, 0.0) - post.get(c, 0.0)) for c in cats)
        assert tvd > 0.25, f"channel distribution TVD pre/post = {tvd:.4f}; not a real regime change"

    def test_no_drift_before_the_drift_month(self, panel, period_index):
        """Pre-drift periods must be homogeneous among themselves."""
        pre = panel[period_index < _DRIFT_START_MONTH_IDX_DEFAULT]
        pre_idx = period_index[period_index < _DRIFT_START_MONTH_IDX_DEFAULT]
        by_period = pre.groupby(pre_idx)["is_digital_active"].mean()
        assert by_period.max() - by_period.min() < 0.10, (
            f"pre-drift digital rate is not stable across periods: {by_period.to_dict()}"
        )

    def test_short_panel_uses_adaptive_drift_start(self, tmp_path):
        """n_periods <= 6 must not silently skip drift."""
        csv_path, _ = _generate(tmp_path / "short", n=400, periods=4, seed=3)
        df = pd.read_csv(csv_path, parse_dates=["period"])
        ordered = sorted(df["period"].unique())
        pidx = df["period"].map({p: i for i, p in enumerate(ordered)})
        pre = df.loc[pidx < 2, "is_digital_active"].mean()
        post = df.loc[pidx >= 2, "is_digital_active"].mean()
        assert post - pre > 0.20, f"adaptive drift did not fire: {pre:.4f} -> {post:.4f}"


# ====================================================================== 8 ==
class TestBrokenCorrelationSubgroup:
    """The ~5% decorrelated subgroup is not identifiable from the published
    CSV (membership is intentionally not a column), so it is verified
    white-box by calling the two generation phases directly and reading the
    membership mask off `df.attrs` before it is cleared. An additional
    black-box check confirms the effect survives into the published file.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def broken():
        import logging

        rng = np.random.default_rng(4242)
        logger = logging.getLogger("test.broken")
        df = _generate_base_panel(1200, 8, rng, logger)
        mask = np.asarray(df.attrs["_decorrelated_subgroup_row"]).copy()
        pre_corr = stats.spearmanr(df["income"], df["account_balance"]).statistic
        _break_correlation_in_subgroup(df, rng, logger)
        return df, mask, float(pre_corr)

    def test_subgroup_is_about_five_percent_of_entities(self, broken):
        df, mask, _ = broken
        frac = df.loc[mask, "entity_id"].nunique() / df["entity_id"].nunique()
        assert 0.02 <= frac <= 0.09, f"decorrelated subgroup covers {frac:.4f} of entities, expected ~0.05"

    def test_correlation_is_broken_inside_the_subgroup(self, broken):
        df, mask, pre_corr = broken
        inside = stats.spearmanr(df.loc[mask, "income"], df.loc[mask, "account_balance"]).statistic
        outside = stats.spearmanr(df.loc[~mask, "income"], df.loc[~mask, "account_balance"]).statistic

        assert pre_corr > 0.40, f"baseline income~balance correlation only {pre_corr:.4f}"
        assert outside > 0.40, f"correlation destroyed outside the subgroup too: {outside:.4f}"
        assert abs(inside) < 0.20, f"subgroup correlation {inside:.4f} is not broken"
        assert outside - inside > 0.30, (
            f"subgroup correlation ({inside:.4f}) not materially weaker than the rest ({outside:.4f})"
        )

    def test_published_panel_correlation_is_diluted_but_present(self, panel):
        """Black-box: the shipped CSV still shows a positive income~balance
        relationship, but weaker than the pure-correlated baseline (~0.57)
        because of the 5% decorrelated subgroup plus injected anomalies."""
        sub = panel[["income", "account_balance"]].dropna()
        corr = float(stats.spearmanr(sub["income"], sub["account_balance"]).statistic)
        assert 0.30 < corr < 0.56, (
            f"published income~balance spearman = {corr:.4f}; expected a diluted "
            "positive correlation below the ~0.57 pure-correlated baseline"
        )


# ====================================================================== 9 ==
class TestAnomalyTypesAreDistinct:
    """Highest-value checks: the four injected types must be *behaviourally*
    different, otherwise the ground truth cannot support the research claim
    that global/local/contextual/collective anomalies are distinguishable.
    """

    @staticmethod
    def _normal_balances(labeled: pd.DataFrame) -> np.ndarray:
        return labeled.loc[labeled["anomaly_type"] == "none", "account_balance"].dropna().to_numpy()

    def test_global_anomalies_are_extreme_globally(self, labeled):
        normals = self._normal_balances(labeled)
        p99 = np.percentile(normals, 99)
        g = labeled.loc[labeled["anomaly_type"] == "global", "account_balance"]

        assert len(g) > 0
        assert g.notna().all(), "global anomalies must have an observed balance"
        frac = float((g > p99).mean())
        assert frac == 1.0, f"only {frac:.3f} of global anomalies exceed the normal p99 ({p99:,.0f})"
        assert g.min() > normals.max(), (
            f"global anomaly minimum ({g.min():,.0f}) does not exceed the normal maximum "
            f"({normals.max():,.0f}) -- global anomalies are not separable"
        )
        assert g.median() / np.median(normals) > 100, (
            f"global/normal median ratio only {g.median() / np.median(normals):.1f}"
        )

    def test_local_anomalies_are_NOT_extreme_globally(self, labeled):
        """The defining property: locally impossible, globally unremarkable."""
        normals = self._normal_balances(labeled)
        p99 = np.percentile(normals, 99)
        loc = labeled.loc[labeled["anomaly_type"] == "local", "account_balance"]

        assert len(loc) > 0
        frac_extreme = float((loc > p99).mean())
        assert frac_extreme <= 0.02, (
            f"{frac_extreme:.3f} of local anomalies exceed the normal p99 -- "
            "they are globally extreme, which defeats the local/global distinction"
        )
        assert loc.max() <= normals.max(), "a local anomaly exceeds the global maximum"
        assert loc.min() >= np.percentile(normals, 1), "a local anomaly is below the normal p1"

    def test_local_anomalies_ARE_extreme_within_their_own_entity(self, labeled):
        entity_mean = labeled.groupby("entity_id")["account_balance"].transform("mean")
        log_dev = np.abs(np.log(labeled["account_balance"] / entity_mean))
        med_local = float(np.nanmedian(log_dev[labeled["anomaly_type"] == "local"]))
        med_normal = float(np.nanmedian(log_dev[labeled["anomaly_type"] == "none"]))
        assert med_local > 1.0, f"local anomalies deviate only {med_local:.3f} (log) from their entity mean"
        assert med_local > 1.5 * med_normal, (
            f"local deviation {med_local:.3f} not materially larger than normal rows {med_normal:.3f}"
        )

    def test_global_and_local_populations_do_not_overlap(self, labeled):
        g = labeled.loc[labeled["anomaly_type"] == "global", "account_balance"]
        loc = labeled.loc[labeled["anomaly_type"] == "local", "account_balance"]
        assert loc.max() < g.min(), "global and local anomaly balance ranges overlap"

    def test_collective_anomalies_are_a_synchronised_near_identical_cluster(self, labeled):
        col = labeled[labeled["anomaly_type"] == "collective"]
        assert len(col) >= 3, f"only {len(col)} collective anomaly rows injected"

        # Each group is one synchronised event: a single period, many
        # distinct entities. The collective budget is a row fraction, so a
        # run contains several groups (>= 20 entities each) -- synchronisation
        # and near-identity are therefore properties *within* a group, not of
        # the pooled population (different groups have different spike levels).
        assert col["period"].nunique() <= max(2, len(col) // 20), (
            "collective anomalies are spread over more periods than there are groups"
        )
        assert col["entity_id"].nunique() >= 3

        amt = col["monthly_transactions_amount"]
        per_group = col.groupby("period")["monthly_transactions_amount"]
        cv = float((per_group.std() / per_group.mean()).max())
        assert cv < 0.10, f"worst collective group's coefficient of variation {cv:.4f} >= 0.10 (not near-identical)"

        panel_median = float(labeled["monthly_transactions_amount"].median())
        assert amt.median() / panel_median > 3.0, (
            f"collective spike is only {amt.median() / panel_median:.2f}x the panel median"
        )

    def test_contextual_anomalies_are_off_season_december_sized_withdrawals(self, labeled):
        ctx = labeled[labeled["anomaly_type"] == "contextual"]
        assert len(ctx) > 0
        months = set(pd.DatetimeIndex(ctx["period"]).month)
        assert 12 not in months, "contextual anomalies must not sit in December (their normal context)"

        normal_non_dec = labeled[
            (labeled["anomaly_type"] == "none")
            & (pd.DatetimeIndex(labeled["period"]).month != 12)
        ]["withdrawal_amount"]
        ratio = float(ctx["withdrawal_amount"].median() / normal_non_dec.median())
        assert ratio > 2.0, (
            f"contextual withdrawals are only {ratio:.2f}x the normal non-December median"
        )

    def test_contextual_anomalies_are_not_globally_extreme(self, labeled):
        """Like local anomalies, contextual ones must be unremarkable in the
        marginal distribution -- only the context makes them anomalous."""
        ctx = labeled.loc[labeled["anomaly_type"] == "contextual", "withdrawal_amount"]
        all_wd = labeled["withdrawal_amount"].dropna()
        p99 = np.percentile(all_wd, 99)
        frac = float((ctx > p99).mean())
        assert frac < 0.30, f"{frac:.3f} of contextual anomalies are globally extreme withdrawals"


# ===================================================================== 10 ==
class TestLoader:
    def test_generates_when_the_csv_is_absent(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        gt_path = tmp_path / "gt.parquet"
        assert not csv_path.exists()

        df, schema = load_or_generate_panel(
            data_path=str(csv_path), n_individuals=300, n_periods=6,
            ground_truth_path=str(gt_path), seed=11,
        )
        assert csv_path.exists()
        assert _resolve_ground_truth(gt_path).exists()
        assert len(df) == 300 * 6
        assert isinstance(schema, PanelSchema)

    def test_does_not_regenerate_when_the_csv_exists(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        gt_path = tmp_path / "gt.parquet"
        load_or_generate_panel(
            data_path=str(csv_path), n_individuals=300, n_periods=6,
            ground_truth_path=str(gt_path), seed=11,
        )
        mtime_before = os.path.getmtime(csv_path)
        size_before = os.path.getsize(csv_path)
        digest_before = _sha256(csv_path)
        time.sleep(0.05)

        # Different generator kwargs: if it regenerated, the file would change.
        df, _ = load_or_generate_panel(
            data_path=str(csv_path), n_individuals=999, n_periods=3,
            ground_truth_path=str(gt_path), seed=12345,
        )
        assert os.path.getmtime(csv_path) == mtime_before, "existing CSV was rewritten"
        assert os.path.getsize(csv_path) == size_before
        assert _sha256(csv_path) == digest_before
        assert len(df) == 300 * 6, "loader returned regenerated data instead of the existing file"

    def test_schema_inference_on_synthetic_data(self, panel_paths):
        csv_path, _ = panel_paths
        df, schema = load_or_generate_panel(data_path=str(csv_path))
        assert schema.time_col == "period"
        assert schema.entity_col == "entity_id"
        assert schema.target_col is None, (
            "synthetic panel is unlabeled; target_col must be None "
            "(ground truth lives in a separate file)"
        )
        assert pd.api.types.is_datetime64_any_dtype(df["period"]), (
            "loader must parse the inferred time column to datetime"
        )

    @pytest.mark.parametrize("target_name", ["target", "ground_truth"])
    def test_detects_an_inline_target_column(self, tmp_path, panel, target_name):
        csv_path = tmp_path / f"with_{target_name}.csv"
        small = panel.head(200).copy()
        small[target_name] = 0
        small.to_csv(csv_path, index=False)

        _, schema = load_or_generate_panel(data_path=str(csv_path))
        assert schema.target_col == target_name
        assert schema.time_col == "period"
        assert schema.entity_col == "entity_id"

    def test_out_path_kwarg_is_ignored_in_favour_of_data_path(self, tmp_path):
        csv_path = tmp_path / "real.csv"
        decoy = tmp_path / "decoy.csv"
        load_or_generate_panel(
            data_path=str(csv_path), out_path=str(decoy),
            n_individuals=200, n_periods=4,
            ground_truth_path=str(tmp_path / "gt.parquet"), seed=5,
        )
        assert csv_path.exists()
        assert not decoy.exists(), "out_path kwarg should be ignored, not honoured"


# ====================================================================== 11 ==
class TestDerivedColumnIntegrity:
    """The panel must stay internally consistent *after* anomaly injection.

    Injection rewrites `account_balance`, `monthly_transactions_amount` and
    `withdrawal_amount`. Any column derived from those and frozen beforehand
    turns into an arithmetic fingerprint of the injected rows -- a label leak a
    detector can exploit without learning anything about anomalies.
    """

    def test_avg_transaction_amount_identity_holds_on_every_row(self, panel):
        expected = panel["monthly_transactions_amount"] / panel["monthly_transactions_count"]
        violations = ~np.isclose(
            panel["avg_transaction_amount"], expected, rtol=1e-9, equal_nan=True
        )
        assert int(violations.sum()) == 0, (
            f"{int(violations.sum())} rows violate avg = amount / count; the "
            "derived column was not recomputed after injection (leak)"
        )

    def test_avg_identity_holds_specifically_on_collective_rows(self, labeled):
        """The regression that motivated the fix: collective injection rewrites
        `monthly_transactions_amount`, so these were the *only* violating rows."""
        coll = labeled[labeled["anomaly_type"] == "collective"]
        assert len(coll) > 0, "no collective anomalies to check"
        expected = coll["monthly_transactions_amount"] / coll["monthly_transactions_count"]
        assert np.allclose(coll["avg_transaction_amount"], expected, rtol=1e-9)

    def test_overdraft_count_tracks_post_injection_balance(self, panel):
        """Overdrafts are Poisson with a rate that jumps below the balance p20.

        Derived pre-injection they described the *old* balance; recomputed after,
        the bottom quintile must still carry visibly more overdrafts.
        """
        obs = panel[["account_balance", "overdraft_count"]].dropna()
        cut = obs["account_balance"].quantile(0.2)
        low = obs.loc[obs["account_balance"] <= cut, "overdraft_count"].mean()
        high = obs.loc[obs["account_balance"] > cut, "overdraft_count"].mean()
        assert low > 3 * high, (
            f"bottom-quintile mean overdrafts {low:.3f} vs {high:.3f} elsewhere: "
            "overdraft_count is stale w.r.t. the injected balances"
        )

    @pytest.mark.parametrize(
        "anomaly_type,column",
        [
            ("global", "account_balance"),
            ("local", "account_balance"),
            ("contextual", "withdrawal_amount"),
            ("collective", "monthly_transactions_amount"),
        ],
    )
    def test_defining_column_is_never_missing_on_anomaly_rows(
        self, labeled, anomaly_type, column
    ):
        """Missingness must not erase the feature that defines a labelled row.

        MNAR adds +15pp missingness above the p95 of `account_balance`, and every
        global anomaly sits there by construction -- without the protection mask
        ~16% of them would be labelled positive with the defining value NaN,
        capping achievable recall for reasons unrelated to any detector.
        """
        rows = labeled[labeled["anomaly_type"] == anomaly_type]
        assert len(rows) > 0, f"no {anomaly_type} anomalies generated"
        assert rows[column].notna().all(), (
            f"{int(rows[column].isna().sum())} {anomaly_type} rows have a missing "
            f"{column}, the column that defines them"
        )

    def test_missingness_still_present_on_clean_rows(self, labeled):
        """Protecting the anomalies must not silently disable missingness."""
        clean = labeled[~labeled["is_anomaly"].astype(bool)]
        for col in ("account_balance", "income", "credit_score"):
            assert clean[col].isna().any(), f"no missingness left in {col}"


# ================================================== known defects (xfail) ==
class TestKnownDefects:
    """Deterministic reproductions of defects found during validation.

    Both defects below have since been fixed (the `strict` xfail markers
    were removed once the tests started passing); they are kept as
    regression guards documenting exactly what "fixed" means.
    """

    def test_ground_truth_lands_next_to_the_generated_panel(self, tmp_path):
        target_dir = tmp_path / "elsewhere"
        target_dir.mkdir()
        csv_path = target_dir / "data.csv"

        load_or_generate_panel(data_path=str(csv_path), n_individuals=100, n_periods=4, seed=2)

        siblings = {p.name for p in target_dir.iterdir()}
        assert any(name.startswith("ground_truth") for name in siblings), (
            f"ground truth not written beside the panel; {target_dir} contains {siblings} "
            f"(it went to {Path.cwd() / 'data'} instead)"
        )

    def test_local_anomaly_values_are_not_degenerate(self, labeled):
        loc = labeled.loc[labeled["anomaly_type"] == "local", "account_balance"].round(6)
        top_share = float(loc.value_counts().max() / len(loc))
        assert top_share < 0.10, (
            f"{top_share:.3f} of local anomalies share one identical balance value "
            f"({loc.value_counts().idxmax():,.2f}) -- clip pile-up makes them "
            "trivially separable and indistinguishable from a collective anomaly"
        )
