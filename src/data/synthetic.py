"""Synthetic banking panel data generator.

Generates a realistic, unlabeled monthly banking panel (``data.csv``) plus a
separate hidden ground-truth file (``ground_truth.parquet``/``.csv``) used
only for offline evaluation. The main data file intentionally carries no
anomaly labels: downstream detection modules (Isolation Forest / VAE) must
work unsupervised by default, per ``CONTEXT.md``.

Public entry point: :func:`generate_synthetic_panel`, which returns a
:class:`PanelGenerationResult` describing the files that were *actually*
written (the ground-truth writer falls back from parquet to CSV when no
parquet engine is installed, so the requested path is not necessarily the
path on disk).

Anomaly semantics
-----------------
The four injected types are defined by *where* the anomaly is visible, not by
magnitude. Detectors are expected to be differentially good at each, so the
definitions are load-bearing for evaluation:

``global``
    Extreme in the overall (marginal) distribution -- an ``account_balance``
    far beyond anything the population produces. Detectable from a single row
    with no context.
``local``
    Perfectly ordinary *globally* but irreconcilable with the entity's own
    history (>= ``_LOCAL_MIN_RATIO``x away from its own mean balance).
    Invariant, enforced by the generator and asserted by the test suite:
    local anomalies must NOT be globally extreme -- their values are drawn
    from inside the population's [p2, p95] band. A local anomaly that is also
    a global outlier collapses the two categories and is a bug.
``contextual``
    Anomalous only given a context variable -- here a December-sized
    ``withdrawal_amount`` occurring in a non-December month. Unremarkable
    both marginally and for the entity; only the month makes it wrong.
``collective``
    Individually unremarkable rows that are anomalous *as a set*: a group of
    20-50 distinct entities posting a near-identical
    ``monthly_transactions_amount`` spike in the same period (coordinated
    fraud). The signal is the synchronisation and near-identity, not the
    per-row value.

Reproducibility
---------------
Output is a pure function of ``(n_individuals, n_periods, seed)`` and the
anomaly-fraction arguments. Periods are anchored on the fixed
``_REFERENCE_END_MONTH`` rather than wall-clock "today", so re-running the
generator months later still yields byte-identical files.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import paths
from src.utils.logging_config import log_phase, setup_logging

# Fixed anchor date (NOT derived from wall-clock "today") so that repeated
# runs with the same seed always produce identical `period` values,
# regardless of the day the generator is executed on.
_REFERENCE_END_MONTH = pd.Timestamp("2026-06-01")

# Month index (0-based) at which the temporal-drift regime change kicks in
# (e.g. a "digital-first" policy push). Documented default: month 6.
_DRIFT_START_MONTH_IDX_DEFAULT = 6

# --- Categorical shape --------------------------------------------------------
# Zipf exponent for the "wide" vocabularies. A realistic long tail sits around
# s = 1.1-1.5; anything near s = 3 collapses ~80% of the rows onto rank 1,
# which makes the column effectively constant (and one-hot/target encodings
# rank-deficient downstream).
_ZIPF_S_WIDE = 1.3
# Mass of the genuinely rare tail categories, expressed as a ratio of the
# smallest *head* probability. A pure Zipf law with s ~ 1.3 never reaches the
# "<0.05% of rows" regime the project wants, so the tail is attached
# explicitly instead of being bought with an unrealistic exponent.
_RARE_TAIL_RATIOS = (0.10, 0.02, 0.004)
_N_RARE_TAIL = len(_RARE_TAIL_RATIOS)

# --- Within-entity persistence (panel dynamics) -------------------------------
# Probability that an entity uses its own "home" channel in a given month;
# the complement is a one-off switch drawn from the population distribution.
_CHANNEL_STICKINESS = 0.75
# Loading of the entity-level latent in the digital-adoption latent variable
# (within-entity correlation of `is_digital_active` is the square of this).
_DIGITAL_ENTITY_LOADING = 0.8

# Split of the total log-scale dispersion of each dynamic monetary feature
# into an entity-level (persistent) part and a month-level (transient) part.
# The pairs are chosen so that sqrt(entity^2 + month^2) reproduces the
# marginal sigma the column had before persistence was introduced.
_BALANCE_SIGMA_ENTITY, _BALANCE_SIGMA_MONTH = 0.72, 0.35          # total ~0.80
_TXN_AMOUNT_SIGMA_ENTITY, _TXN_AMOUNT_SIGMA_MONTH = 0.75, 0.50    # total ~0.90
_WITHDRAWAL_SIGMA_ENTITY, _WITHDRAWAL_SIGMA_MONTH = 0.45, 0.40    # total ~0.60
_TXN_COUNT_SIGMA_ENTITY = 0.35
_DECORR_SIGMA_ENTITY, _DECORR_SIGMA_MONTH = 0.95, 0.55            # total ~1.10

# --- Temporal drift -----------------------------------------------------------
_DIGITAL_BASE_P = 0.30
_DIGITAL_DRIFTED_P = 0.65
# Fraction of the population that has migrated to the new regime once the
# ramp is complete. Deliberately < 1 so the pre/post distributions overlap
# instead of being disjoint (a total regime replacement makes drift
# detection trivial and unrealistic).
_DRIFT_MAX_ADOPTION = 0.85
# Months over which adoption ramps from 0 to `_DRIFT_MAX_ADOPTION`.
_DRIFT_RAMP_MONTHS = 3

# --- Anomaly budgets ----------------------------------------------------------
_FRAC_GLOBAL_DEFAULT = 0.005
_FRAC_LOCAL_DEFAULT = 0.005
_FRAC_CONTEXTUAL_DEFAULT = 0.005
_FRAC_COLLECTIVE_DEFAULT = 0.005
# "Global" anomaly magnitude. The injected value is
# `max(p99.9 * _GLOBAL_P999_MULT, observed_max * _GLOBAL_MAX_HEADROOM)` scaled
# by U(1, _GLOBAL_SPREAD). The first term reproduces the historical 20x-100x
# p99.9 range; the second guarantees separation from the deliberately unlabeled
# organic fat-tail shocks (derivation in `_inject_anomalies`).
_GLOBAL_P999_MULT = 20.0
_GLOBAL_MAX_HEADROOM = 1.5
_GLOBAL_SPREAD = 5.0
# Realistic coordinated-fraud ring sizes (entities per synchronised group).
_COLLECTIVE_GROUP_MIN = 20
_COLLECTIVE_GROUP_MAX = 50
# Local anomalies: minimum multiplicative deviation from the entity's own
# historical mean, and the global quantile band the replacement value is
# drawn from (so the value stays unremarkable in the marginal distribution).
_LOCAL_MIN_RATIO = 6.0
_LOCAL_Q_LO, _LOCAL_Q_HI = 0.02, 0.95
# Narrowest quantile window still considered usable for a given direction.
_LOCAL_MIN_Q_WINDOW = 0.01

_NORMAL = NormalDist()

# --- Categorical vocabularies -------------------------------------------------
# The last `_N_RARE_TAIL` entries of each "wide" vocabulary below are the
# genuinely rare categories (see `_long_tail_probs`); they are ordered so the
# semantically exotic products/regions/channels land in that tail.
_REGIONS = [
    "North", "South", "East", "West", "Central", "Metro",
    "Coastal", "Highlands", "Border",
    "Islands", "Overseas_Territory", "Remote_Outpost",
]
_SEGMENTS = ["retail", "sme", "corporate", "private_banking"]
_EMPLOYMENT = ["employed", "self_employed", "unemployed", "retired", "student"]
_MARITAL = ["single", "married", "divorced", "widowed"]
_PRODUCT_TYPES = [
    "checking", "savings", "credit_card", "personal_loan", "mortgage",
    "auto_loan", "investment_account", "insurance_product", "business_account",
    "student_account", "foreign_currency_account", "precious_metals_account",
]
# Channels: ordered so index 0 is highest pre-drift probability mass
# (branch-first world). Drift reshuffles the probability vector (see
# `_inject_temporal_drift`) so mobile/online channels gain the mass that
# branch/phone previously held, without changing the category set itself.
_CHANNELS = [
    "branch", "phone_banking", "atm", "mail_order", "agent_network",
    "cash_deposit_kiosk", "mobile_app", "online_banking",
    "international_wire", "legacy_terminal", "crypto_gateway",
]
# Legacy -> digital probability-mass swaps applied by the drift phase.
_DRIFT_SWAP_PAIRS = [("branch", "mobile_app"), ("phone_banking", "online_banking")]


@dataclass
class PanelGenerationResult:
    """What :func:`generate_synthetic_panel` actually produced.

    Attributes:
        out_path: Path of the main (unlabeled) panel CSV that was written.
        ground_truth_path: Path of the ground-truth file that was written.
            This is the *real* path: if no parquet engine is installed the
            writer falls back to a sibling ``.csv`` and this attribute
            reports that CSV, so downstream consumers never have to guess.
        n_rows: Rows in the generated panel; always
            ``n_entities * n_periods`` (the panel is balanced).
        n_entities: Distinct entities in the panel.
        n_periods: Consecutive monthly periods per entity.
        anomaly_counts: Rows injected per anomaly type, keyed by
            ``"global"``, ``"local"``, ``"contextual"``, ``"collective"``.
            Actual counts can fall slightly below the requested budgets
            (collective groups skip rows already claimed by another type,
            contextual injection skips December rows).
    """

    out_path: str
    ground_truth_path: str
    n_rows: int
    n_entities: int
    n_periods: int
    anomaly_counts: dict[str, int] = field(default_factory=dict)


def _zipf_like_probs(n: int, s: float = 2.5) -> np.ndarray:
    """Return an n-length probability vector with pure Zipf-like decay."""
    ranks = np.arange(1, n + 1)
    weights = 1.0 / np.power(ranks, s)
    return weights / weights.sum()


def _long_tail_probs(n: int, s: float = _ZIPF_S_WIDE, n_rare: int = _N_RARE_TAIL) -> np.ndarray:
    """Realistic long tail: Zipf head + an explicitly rare tail.

    The head (ranks 1..n-n_rare) follows a Zipf law with a *realistic*
    exponent, so no single category swallows the column. The last `n_rare`
    ranks are pushed far below the head (down to ~0.01% of the mass) so the
    dataset still contains genuinely rare categories -- a property a Zipf
    law with s ~ 1.3 cannot deliver on its own.
    """
    n_rare = int(np.clip(n_rare, 0, max(0, n - 1)))
    head_n = n - n_rare
    head = 1.0 / np.power(np.arange(1, head_n + 1), s)
    head = head / head.sum()
    if n_rare:
        tail = head[-1] * np.asarray(_RARE_TAIL_RATIOS[:n_rare], dtype=float)
        probs = np.concatenate([head, tail])
    else:
        probs = head
    return probs / probs.sum()


def _entity_ids(n_individuals: int) -> np.ndarray:
    width = max(6, len(str(n_individuals)))
    return np.array([f"CUST_{i + 1:0{width}d}" for i in range(n_individuals)])


def _prob_to_threshold(p: float) -> float:
    """Latent-normal cutoff whose exceedance probability is `p`."""
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return _NORMAL.inv_cdf(1.0 - p)


def _bernoulli_from_latent(latent: np.ndarray, p_row: np.ndarray) -> np.ndarray:
    """Threshold a standard-normal latent so P(True) equals `p_row` per row.

    `p_row` takes only a handful of distinct values (one per period), so the
    inverse CDF is evaluated once per distinct value rather than per row.
    """
    uniq, inverse = np.unique(p_row, return_inverse=True)
    thresholds = np.array([_prob_to_threshold(p) for p in uniq])
    return latent > thresholds[inverse]


# --------------------------------------------------------------------------- #
# Derived columns -- single source of truth                                     #
# --------------------------------------------------------------------------- #
# Every column below is a deterministic or stochastic *function* of another
# column of the panel. Each formula is defined exactly once, here, and the whole
# chain is replayed by `_recompute_derived_columns` after any phase that
# rewrites one of the inputs.
#
# TEORÍA: a derived column computed *before* a later phase overwrites its input
# stops being a function of that input and becomes a fingerprint of the phase
# itself. Concretely: collective anomalies rewrite `monthly_transactions_amount`,
# so an `avg_transaction_amount` frozen at generation time violates the identity
# `avg = amount / count` on *exactly* the injected rows. A detector (or any lag/
# diff feature built on the pair) can then recover the label from an arithmetic
# inconsistency rather than from the anomaly's statistical signature -- textbook
# label leakage, and it silently inflates every downstream metric.


def _derive_avg_transaction_amount(amount: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Mean ticket size: ``amount / count``.

    `monthly_transactions_count` is Poisson + 1, so the denominator is >= 1 and
    the division needs no zero guard.
    """
    return np.asarray(amount, dtype=float) / np.asarray(count, dtype=float)


def _derive_credit_score(income: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Credit score: the income z-score mapped onto a 300-850 FICO-like range.

    NaN-safe moments so the formula stays valid if it is ever replayed after the
    missingness phase.
    """
    income = np.asarray(income, dtype=float)
    income_z = (income - np.nanmean(income)) / (np.nanstd(income) + 1e-9)
    noisy = 650 + 40 * income_z + rng.normal(0, 50, size=income.size)
    return np.clip(np.nan_to_num(noisy, nan=650.0), 300, 850).round().astype(int)


def _derive_overdraft_count(account_balance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overdrafts: Poisson whose rate jumps for the bottom balance quintile."""
    pctile = pd.Series(np.asarray(account_balance, dtype=float)).rank(pct=True).to_numpy()
    lam = np.where(pctile < 0.2, 1.8, 0.15)
    return rng.poisson(np.nan_to_num(lam, nan=0.15))


def _derive_customer_satisfaction(
    overdraft_count: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Satisfaction on a 1-10 scale, decreasing in the number of overdrafts."""
    mean = 7.0 - 0.3 * np.asarray(overdraft_count, dtype=float)
    return np.clip(rng.normal(mean, 1.5).round(), 1, 10).astype(int)


def _recompute_derived_columns(
    df: pd.DataFrame,
    rng: np.random.Generator,
    logger: logging.Logger,
    context: str = "",
) -> None:
    """Replay the derived-column chain in dependency order, in place.

    Order matters -- ``overdraft_count`` reads the (possibly rewritten)
    ``account_balance`` and ``customer_satisfaction_score`` in turn reads the
    freshly drawn ``overdraft_count``::

        avg_transaction_amount <- monthly_transactions_amount / monthly_transactions_count
        credit_score           <- income
        overdraft_count        <- account_balance (percentile)
        customer_satisfaction  <- overdraft_count

    Call this after **every** phase that rewrites a monetary input:
    `_break_correlation_in_subgroup` (rewrites `account_balance`) and
    `_inject_anomalies` (rewrites `account_balance`,
    `monthly_transactions_amount`, `withdrawal_amount`). Calling it once at the
    end of injection covers both, since injection is the last such phase.

    Must run *before* the missingness phase: replaying the chain on NaN inputs
    would resurrect values the MNAR/MCAR mechanism deliberately removed.
    """
    df["avg_transaction_amount"] = _derive_avg_transaction_amount(
        df["monthly_transactions_amount"].to_numpy(),
        df["monthly_transactions_count"].to_numpy(),
    )
    df["credit_score"] = _derive_credit_score(df["income"].to_numpy(), rng)
    df["overdraft_count"] = _derive_overdraft_count(df["account_balance"].to_numpy(), rng)
    df["customer_satisfaction_score"] = _derive_customer_satisfaction(
        df["overdraft_count"].to_numpy(), rng
    )
    logger.info(
        "Recomputed derived columns%s: avg_transaction_amount, credit_score, "
        "overdraft_count, customer_satisfaction_score",
        f" ({context})" if context else "",
    )


def _generate_base_panel(
    n_individuals: int, n_periods: int, rng: np.random.Generator, logger: logging.Logger
) -> pd.DataFrame:
    """Phase 1: vectorized generation of the base (pre-drift, pre-anomaly) panel.

    Row order is entity-major: row position = entity_idx * n_periods + period_idx.
    This fixed layout is relied on later (anomaly injection, drift injection)
    to address specific (entity, period) rows without a lookup table.

    Dynamic features are *persistent within an entity*: every entity carries
    a latent level (monetary features), a latent digital propensity, and a
    "home" transaction channel, so its monthly observations cluster around
    its own baseline instead of being redrawn i.i.d. every month. Without
    this the panel has no within-entity autocorrelation and the local /
    contextual anomaly definitions lose their meaning.
    """
    n_rows = n_individuals * n_periods

    # --- entity-level (static) attributes ---
    entity_id = _entity_ids(n_individuals)
    age_entity = rng.integers(18, 85, size=n_individuals)
    region_entity = rng.choice(_REGIONS, size=n_individuals, p=_long_tail_probs(len(_REGIONS)))
    segment_entity = rng.choice(_SEGMENTS, size=n_individuals, p=_zipf_like_probs(len(_SEGMENTS), s=1.4))
    employment_entity = rng.choice(_EMPLOYMENT, size=n_individuals, p=_zipf_like_probs(len(_EMPLOYMENT), s=1.2))
    marital_entity = rng.choice(_MARITAL, size=n_individuals, p=_zipf_like_probs(len(_MARITAL), s=1.0))
    product_entity = rng.choice(_PRODUCT_TYPES, size=n_individuals, p=_long_tail_probs(len(_PRODUCT_TYPES)))
    account_open_month_idx_entity = rng.integers(-240, 0, size=n_individuals)  # opened up to 20y before panel start
    income_base_entity = rng.lognormal(mean=8.5, sigma=0.6, size=n_individuals)  # heavy-tailed monthly income
    decorrelated_subgroup_entity = rng.random(n_individuals) < 0.05  # 5% "broken correlation" subgroup

    # --- entity-level latent levels for the *dynamic* features ---
    balance_level_entity = (
        rng.uniform(2.0, 6.0, size=n_individuals)
        * rng.lognormal(0.0, _BALANCE_SIGMA_ENTITY, size=n_individuals)
    )
    txn_amount_level_entity = rng.lognormal(mean=7.0, sigma=_TXN_AMOUNT_SIGMA_ENTITY, size=n_individuals)
    # Mean-preserving multiplicative level for the transaction *count*.
    txn_count_level_entity = rng.lognormal(
        mean=-0.5 * _TXN_COUNT_SIGMA_ENTITY ** 2, sigma=_TXN_COUNT_SIGMA_ENTITY, size=n_individuals
    )
    withdrawal_level_entity = rng.lognormal(mean=6.3, sigma=_WITHDRAWAL_SIGMA_ENTITY, size=n_individuals)
    digital_latent_entity = rng.normal(size=n_individuals)
    channel_home_entity = rng.choice(_CHANNELS, size=n_individuals, p=_long_tail_probs(len(_CHANNELS)))

    periods = pd.date_range(end=_REFERENCE_END_MONTH, periods=n_periods, freq="MS")

    # --- broadcast entity-level attrs to row-level (entity-major order) ---
    entity_id_row = np.repeat(entity_id, n_periods)
    period_idx_row = np.tile(np.arange(n_periods), n_individuals)
    period_row = periods.values[period_idx_row]
    age_row = np.repeat(age_entity, n_periods)
    region_row = np.repeat(region_entity, n_periods)
    segment_row = np.repeat(segment_entity, n_periods)
    employment_row = np.repeat(employment_entity, n_periods)
    marital_row = np.repeat(marital_entity, n_periods)
    product_row = np.repeat(product_entity, n_periods)
    income_base_row = np.repeat(income_base_entity, n_periods)
    decorrelated_row = np.repeat(decorrelated_subgroup_entity, n_periods)
    tenure_months_row = period_idx_row - np.repeat(account_open_month_idx_entity, n_periods)

    # --- row-level dynamic draws (entity level x month-level shock) ---
    income = income_base_row * rng.lognormal(mean=0.0, sigma=0.12, size=n_rows)

    # Account balance: correlated with income for the general population, and
    # anchored on the entity's own balance level across months.
    # (The 5% decorrelated subgroup is overwritten in a later, distinct phase.)
    account_balance = (
        income
        * np.repeat(balance_level_entity, n_periods)
        * rng.lognormal(0.0, _BALANCE_SIGMA_MONTH, size=n_rows)
    )

    # Monthly transaction amount: heavy-tailed, independent of income.
    monthly_transactions_amount = (
        np.repeat(txn_amount_level_entity, n_periods)
        * rng.lognormal(mean=0.0, sigma=_TXN_AMOUNT_SIGMA_MONTH, size=n_rows)
    )

    # Natural fat-tail shocks (0.1% of rows) on both monetary variables -
    # unlabeled "extreme but organic" outliers, distinct from injected
    # ground-truth anomalies.
    for col in (account_balance, monthly_transactions_amount):
        shock_mask = rng.random(n_rows) < 0.001
        col[shock_mask] *= rng.uniform(50.0, 500.0, size=shock_mask.sum())

    monthly_transactions_count = rng.poisson(lam=12.0 * np.repeat(txn_count_level_entity, n_periods)) + 1

    month_of_year = pd.DatetimeIndex(period_row).month.to_numpy()
    december_mult = np.where(month_of_year == 12, rng.uniform(2.0, 3.0, size=n_rows), 1.0)
    withdrawal_amount = (
        np.repeat(withdrawal_level_entity, n_periods)
        * rng.lognormal(mean=0.0, sigma=_WITHDRAWAL_SIGMA_MONTH, size=n_rows)
        * december_mult
    )

    num_products = np.clip(rng.poisson(2, size=n_rows) + 1, 1, 6)

    # Digital adoption: a persistent entity propensity plus a monthly shock,
    # thresholded so the marginal probability is the documented baseline
    # (the drift phase re-thresholds the same latent, see
    # `_inject_temporal_drift`).
    digital_latent_row = (
        _DIGITAL_ENTITY_LOADING * np.repeat(digital_latent_entity, n_periods)
        + np.sqrt(1.0 - _DIGITAL_ENTITY_LOADING ** 2) * rng.normal(size=n_rows)
    )
    is_digital_active = digital_latent_row > _prob_to_threshold(_DIGITAL_BASE_P)
    days_since_last_login = rng.exponential(scale=np.where(is_digital_active, 3.0, 20.0)).round().astype(int)

    # Transaction channel: sticky within entity -- most months reuse the
    # entity's "home" channel, the rest are one-off switches drawn from the
    # population distribution (so the marginal is unchanged). Pre-drift
    # probabilities for all rows; the drift phase overwrites post-drift rows.
    channel_row = _draw_sticky_channels(
        np.repeat(channel_home_entity, n_periods), _long_tail_probs(len(_CHANNELS)), rng
    )

    df = pd.DataFrame({
        "entity_id": entity_id_row,
        "period": period_row,
        "age": age_row,
        "tenure_months": tenure_months_row,
        "region": region_row,
        "segment": segment_row,
        "employment_status": employment_row,
        "marital_status": marital_row,
        "product_type": product_row,
        "transaction_channel": channel_row,
        "is_digital_active": is_digital_active,
        "income": income,
        "account_balance": account_balance,
        "monthly_transactions_amount": monthly_transactions_amount,
        "monthly_transactions_count": monthly_transactions_count,
        # Derived columns: placeholders holding the schema position. They are
        # filled by `_recompute_derived_columns` below, which is the one and
        # only definition of these four formulas.
        "avg_transaction_amount": 0.0,
        "withdrawal_amount": withdrawal_amount,
        "credit_score": 0,
        "overdraft_count": 0,
        "num_products": num_products,
        "days_since_last_login": days_since_last_login,
        "customer_satisfaction_score": 0,
    })

    _recompute_derived_columns(df, rng, logger, context="base panel")

    # Stash a couple of intermediate arrays as private attrs (not written to
    # disk) so later phases don't need to recompute them.
    df.attrs["_decorrelated_subgroup_row"] = decorrelated_row
    df.attrs["_decorrelated_subgroup_entity"] = decorrelated_subgroup_entity
    df.attrs["_period_idx_row"] = period_idx_row
    df.attrs["_digital_latent_row"] = digital_latent_row
    df.attrs["_channel_home_entity"] = channel_home_entity
    df.attrs["_n_periods"] = n_periods

    logger.info(
        "Base panel generated: %d rows x %d columns (%d entities x %d periods); "
        "dynamic features carry entity-level persistence (channel stickiness %.2f, "
        "digital latent loading %.2f)",
        len(df), df.shape[1], n_individuals, n_periods,
        _CHANNEL_STICKINESS, _DIGITAL_ENTITY_LOADING,
    )
    return df


def _draw_sticky_channels(
    home_row: np.ndarray, probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return channels that reuse `home_row` most months, switch otherwise.

    Because the one-off switches are drawn from the same `probs` the home
    channels came from, the marginal channel distribution is exactly `probs`
    -- only the within-entity autocorrelation changes.
    """
    n = len(home_row)
    fresh = rng.choice(_CHANNELS, size=n, p=probs)
    switch = rng.random(n) >= _CHANNEL_STICKINESS
    return np.where(switch, fresh, home_row)


def _break_correlation_in_subgroup(df: pd.DataFrame, rng: np.random.Generator, logger: logging.Logger) -> None:
    """Phase 2: for the ~5% decorrelated subgroup, redraw account_balance
    independently of income (simulating a fraud ring / data-quality segment
    where the usual income<->balance relationship does not hold).

    The redraw keeps the same entity-level persistence as the rest of the
    panel -- only the *link to income* is broken, not the panel structure.
    """
    mask = np.asarray(df.attrs["_decorrelated_subgroup_row"])
    n_affected = int(mask.sum())
    if n_affected:
        entity_mask = df.attrs.get("_decorrelated_subgroup_entity")
        n_periods = df.attrs.get("_n_periods")
        if entity_mask is not None and n_periods:
            n_entities_affected = int(np.asarray(entity_mask).sum())
            level = rng.lognormal(mean=8.0, sigma=_DECORR_SIGMA_ENTITY, size=n_entities_affected)
            level_row = np.repeat(level, n_periods)
            values = level_row * rng.lognormal(0.0, _DECORR_SIGMA_MONTH, size=n_affected)
        else:  # pragma: no cover - defensive: attrs stripped by a caller
            values = rng.lognormal(mean=8.0, sigma=1.1, size=n_affected)
        df.loc[mask, "account_balance"] = values
    n_entities = df.loc[mask, "entity_id"].nunique()
    logger.info(
        "Correlation-breaking: redrew account_balance independently of income for %d rows "
        "across %d entities (%.1f%% of entities)",
        n_affected, n_entities, 100.0 * n_entities / df["entity_id"].nunique(),
    )


def _inject_temporal_drift(
    df: pd.DataFrame, n_periods: int, rng: np.random.Generator, logger: logging.Logger
) -> None:
    """Phase 3: from `drift_start_idx` onward, *gradually* shift the
    transaction_channel and digital-adoption distributions (simulates a
    digital-first policy change).

    The shift is a partial, ramped migration rather than a hard regime
    replacement: adoption climbs over `_DRIFT_RAMP_MONTHS` months and tops
    out at `_DRIFT_MAX_ADOPTION`, so the pre- and post-drift distributions
    keep overlapping. Once an entity has migrated it stays migrated (the
    adoption threshold is entity-level), which is what makes the shift look
    like a real behavioural change rather than per-row resampling.
    """
    drift_start_idx = _DRIFT_START_MONTH_IDX_DEFAULT
    if drift_start_idx >= n_periods:
        drift_start_idx = max(1, n_periods // 2)
        logger.info(
            "Panel too short for default drift start (month %d); using adaptive month %d instead",
            _DRIFT_START_MONTH_IDX_DEFAULT, drift_start_idx,
        )

    period_idx_row = df.attrs["_period_idx_row"]
    post_mask = period_idx_row >= drift_start_idx
    n_affected = int(post_mask.sum())
    if n_affected == 0:
        logger.info("No rows fall in the post-drift regime; skipping drift injection")
        return

    n_post_periods = n_periods - drift_start_idx
    n_individuals = len(df) // n_periods
    ramp_len = min(_DRIFT_RAMP_MONTHS, max(2, n_post_periods))
    # ramp in [0, 1]: 0 before the drift month, then linear over `ramp_len`
    # months, then flat.
    ramp_row = np.clip((period_idx_row - drift_start_idx + 1) / ramp_len, 0.0, 1.0)

    # --- digital adoption: re-threshold the *same* latent so an entity's
    # propensity is preserved while the population rate ramps up. ---
    latent = df.attrs["_digital_latent_row"]
    p_row = _DIGITAL_BASE_P + (_DIGITAL_DRIFTED_P - _DIGITAL_BASE_P) * ramp_row
    digital = _bernoulli_from_latent(latent, p_row)
    df["is_digital_active"] = digital
    # Keep the login-recency feature consistent with the new adoption state.
    df.loc[post_mask, "days_since_last_login"] = (
        rng.exponential(scale=np.where(digital[post_mask], 3.0, 20.0)).round().astype(int)
    )

    # --- transaction channel: entity-level migration to the digital regime ---
    pre_probs = _long_tail_probs(len(_CHANNELS))
    post_probs = pre_probs.copy()
    # Swap the probability mass of the two leading legacy channels with the
    # two digital channels -- same shape, different assignment.
    for legacy, digital_channel in _DRIFT_SWAP_PAIRS:
        i, j = _CHANNELS.index(legacy), _CHANNELS.index(digital_channel)
        post_probs[i], post_probs[j] = post_probs[j], post_probs[i]

    adoption_row = _DRIFT_MAX_ADOPTION * ramp_row
    adopt_threshold_entity = rng.random(n_individuals)
    adopted_row = np.repeat(adopt_threshold_entity, n_periods) < adoption_row

    home_pre_row = np.repeat(df.attrs["_channel_home_entity"], n_periods)
    home_post_row = np.repeat(rng.choice(_CHANNELS, size=n_individuals, p=post_probs), n_periods)

    idx = np.where(post_mask)[0]
    adopted = adopted_row[idx]
    home = np.where(adopted, home_post_row[idx], home_pre_row[idx])
    fresh = np.where(
        adopted,
        rng.choice(_CHANNELS, size=len(idx), p=post_probs),
        rng.choice(_CHANNELS, size=len(idx), p=pre_probs),
    )
    switch = rng.random(len(idx)) >= _CHANNEL_STICKINESS
    df.loc[df.index[idx], "transaction_channel"] = np.where(switch, fresh, home)

    drift_period = pd.date_range(end=_REFERENCE_END_MONTH, periods=n_periods, freq="MS")[drift_start_idx]
    logger.info(
        "Temporal drift injected from period index %d (%s) onward: %d rows affected; "
        "adoption ramps over %d month(s) to %.0f%% of entities (partial shift, distributions "
        "still overlap); features shifted: transaction_channel (toward mobile_app/online_banking), "
        "is_digital_active (%.2f -> %.2f terminal probability, %.3f mean over the post-drift window)",
        drift_start_idx, drift_period.date(), n_affected, ramp_len, 100 * _DRIFT_MAX_ADOPTION,
        _DIGITAL_BASE_P, _DIGITAL_DRIFTED_P, float(p_row[post_mask].mean()),
    )


def _inject_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    logger: logging.Logger,
    protect_mask: Optional[np.ndarray] = None,
) -> None:
    """Phase 5 (last): MCAR on 3 columns, MNAR on 2 columns.

    - MCAR (missing completely at random, rate 2%): customer_satisfaction_score,
      days_since_last_login, credit_score. Missingness independent of value.
      The columns are converted to pandas' nullable integer dtype rather than
      to `object`, so they stay numeric for in-memory consumers.
    - MNAR (missing not at random): account_balance, income. Very high values
      are disproportionately more likely to be missing, simulating manual
      review / redaction of large balances before the extract is released.

    Args:
        protect_mask: Boolean row mask (typically the injected anomalies) that
            is never nulled by either mechanism.

            TEORÍA: the MNAR rate is ``base + 0.15`` above the 95th percentile
            of the column. Global anomalies set `account_balance` to 20-100x
            p99.9, so *every one of them* falls in that band and ~16% would have
            the very feature that defines them replaced by NaN -- while still
            carrying a positive label in the ground truth. That makes a slice of
            the oracle unlearnable by construction, capping achievable recall
            for reasons unrelated to the detector. Missingness is kept a
            property of the *background* process, not of the labelled rows.
    """
    n_rows = len(df)
    keep = None if protect_mask is None else np.asarray(protect_mask, dtype=bool)
    if keep is not None and keep.shape[0] != n_rows:
        raise ValueError(
            f"protect_mask has {keep.shape[0]} rows, expected {n_rows} to match the panel"
        )

    def _apply_protection(mask: np.ndarray) -> np.ndarray:
        return mask if keep is None else (mask & ~keep)

    mcar_cols = ["customer_satisfaction_score", "days_since_last_login", "credit_score"]
    mcar_rate = 0.02
    for col in mcar_cols:
        mask = _apply_protection(rng.random(n_rows) < mcar_rate)
        # Nullable integer: keeps integer semantics *and* supports NA, unlike
        # `object` (wasteful, breaks arithmetic) or plain float (silently
        # turns counts/scores into floats).
        df[col] = df[col].astype("Int64")
        df.loc[mask, col] = pd.NA
        logger.info("MCAR: injected %d missing values (%.2f%%) into '%s'", int(mask.sum()), 100 * mcar_rate, col)

    mnar_cols = ["account_balance", "income"]
    base_rate, top_extra_rate, top_quantile = 0.01, 0.15, 0.95
    for col in mnar_cols:
        pctile = df[col].rank(pct=True).to_numpy()
        prob = np.where(pctile >= top_quantile, base_rate + top_extra_rate, base_rate)
        mask = _apply_protection(rng.random(n_rows) < prob)
        df.loc[mask, col] = np.nan
        logger.info(
            "MNAR: injected %d missing values into '%s' (base rate %.1f%%, top %.0f%% of values get "
            "+%.0f%% extra missing probability -- simulates manual redaction of large amounts)",
            int(mask.sum()), col, 100 * base_rate, 100 * (1 - top_quantile), 100 * top_extra_rate,
        )

    if keep is not None:
        logger.info(
            "Missingness: %d injected-anomaly rows were protected from both mechanisms "
            "so the ground-truth oracle stays fully observed",
            int(keep.sum()),
        )


def _empirical_quantile(sorted_values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linearly-interpolated empirical quantile function of `sorted_values`."""
    if len(sorted_values) == 1:
        return np.full(len(q), sorted_values[0], dtype=float)
    pos = np.clip(q, 0.0, 1.0) * (len(sorted_values) - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _empirical_cdf(sorted_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical CDF of `sorted_values` evaluated at `x`."""
    return np.searchsorted(sorted_values, x, side="left") / len(sorted_values)


def _inject_anomalies(
    df: pd.DataFrame,
    n_individuals: int,
    n_periods: int,
    rng: np.random.Generator,
    logger: logging.Logger,
    frac_global: float = _FRAC_GLOBAL_DEFAULT,
    frac_local: float = _FRAC_LOCAL_DEFAULT,
    frac_contextual: float = _FRAC_CONTEXTUAL_DEFAULT,
    frac_collective: float = _FRAC_COLLECTIVE_DEFAULT,
) -> pd.DataFrame:
    """Phase 4: inject global / local / contextual / collective anomalies
    and return a separate ground-truth DataFrame.

    See the module docstring for the semantic definition of each type; the
    columns they act on are `account_balance` (global, local),
    `withdrawal_amount` (contextual) and `monthly_transactions_amount`
    (collective).

    All four budgets are explicit fractions of the total row count, so every
    anomaly type stays measurable at any panel size. Rows are drawn from a
    shrinking availability pool, so the types are mutually exclusive: a row
    carries at most one label. The main `df` is mutated in place (values
    overwritten for the selected rows); the anomaly labels themselves are
    never added as columns to `df`.

    Returns:
        A DataFrame with one row per panel row and columns `entity_id`,
        `period`, `is_anomaly`, `anomaly_type` (`"none"` for clean rows).
        Its `.attrs["_anomaly_counts"]` holds the per-type row counts.
    """
    n_rows = len(df)
    is_anomaly = np.zeros(n_rows, dtype=bool)
    anomaly_type = np.full(n_rows, "none", dtype=object)
    available = np.ones(n_rows, dtype=bool)

    # Snapshot used by the "local" anomaly logic so it reflects the
    # pre-anomaly distribution, regardless of injection order.
    balance_snapshot = df["account_balance"].to_numpy(copy=True)
    entity_mean_balance = pd.Series(balance_snapshot).groupby(df["entity_id"].to_numpy()).transform("mean").to_numpy()
    sorted_balance = np.sort(balance_snapshot[~np.isnan(balance_snapshot)])
    global_p999 = np.nanpercentile(balance_snapshot, 99.9)
    global_median = np.nanmedian(balance_snapshot)
    global_max = float(np.nanmax(balance_snapshot))

    # Floor for the "global" injection. Anchoring on p99.9 alone is not enough:
    # the base panel carries deliberately *unlabeled* organic fat-tail shocks
    # (0.1% of rows multiplied by U(50, 500)), which can land above 20x p99.9.
    #
    # TEORÍA: "global" means "outside the support of the overall distribution".
    # If an unlabeled organic outlier can sit *above* a labelled global anomaly
    # on the very axis that defines it, the label stops being a function of the
    # data: two rows with the same balance carry different labels, so no
    # detector can separate them and the ground truth contradicts itself. Taking
    # the observed maximum into account makes the type well-defined by
    # construction instead of by luck. (This mattered less when missingness ran
    # first, because MNAR censored ~16% of the top balances before injection --
    # a coincidence, not a design.)
    global_floor = max(global_p999 * _GLOBAL_P999_MULT, global_max * _GLOBAL_MAX_HEADROOM)

    counts = {"global": 0, "local": 0, "contextual": 0, "collective": 0}

    # --- Collective anomalies: small clusters of entities, same period,
    # near-identical unusual spike (coordinated-fraud simulation). The number
    # of groups is derived from `frac_collective` so the type does not become
    # statistically invisible at production scale; group sizes stay in the
    # realistic 20-50 entity range. ---
    if n_periods >= 2 and frac_collective > 0:
        group_max = int(min(_COLLECTIVE_GROUP_MAX, max(3, n_individuals // 2)))
        group_min = int(min(_COLLECTIVE_GROUP_MIN, group_max))
        mean_group = 0.5 * (group_min + group_max)
        n_groups = int(max(1, round(frac_collective * n_rows / mean_group)))

        candidate_periods = np.arange(1, n_periods)
        # Spread groups over distinct periods where possible, so each
        # synchronised event stays identifiable.
        reps = int(np.ceil(n_groups / len(candidate_periods)))
        group_periods = np.tile(candidate_periods, reps)[:n_groups].copy()
        rng.shuffle(group_periods)

        for period_idx in tqdm(group_periods, desc="collective anomaly groups"):
            group_size = int(rng.integers(group_min, group_max + 1))
            entities = rng.choice(n_individuals, size=min(group_size, n_individuals), replace=False)
            rows = entities * n_periods + int(period_idx)
            rows = rows[available[rows]]
            if len(rows) == 0:
                continue
            spike_base = global_median * rng.uniform(3.0, 5.0)
            df.loc[df.index[rows], "monthly_transactions_amount"] = spike_base * (1 + rng.normal(0, 0.02, size=len(rows)))
            is_anomaly[rows] = True
            anomaly_type[rows] = "collective"
            available[rows] = False
            counts["collective"] += len(rows)
    else:
        logger.info("n_periods < 2 (or zero budget): skipping collective anomaly injection")

    # --- Remaining budget for global / local / contextual, drawn from the
    # still-available row pool so the four types never overlap. ---
    n_global = int(round(frac_global * n_rows))
    n_local = int(round(frac_local * n_rows))
    n_contextual = int(round(frac_contextual * n_rows))

    pool = np.where(available)[0]
    rng.shuffle(pool)

    # Global: values far outside the overall distribution.
    global_rows = pool[:n_global]
    if len(global_rows):
        df.loc[df.index[global_rows], "account_balance"] = global_floor * rng.uniform(
            1.0, _GLOBAL_SPREAD, size=len(global_rows)
        )
        is_anomaly[global_rows] = True
        anomaly_type[global_rows] = "global"
        available[global_rows] = False
        counts["global"] = len(global_rows)
    pool = pool[n_global:]

    # Local: normal-looking globally, wildly inconsistent with the entity's
    # own historical average.
    #
    # The replacement value is drawn from the *empirical* balance distribution
    # (via its quantile function) inside the [p02, p95] band, restricted to
    # the sub-band that is at least `_LOCAL_MIN_RATIO`x away from the entity's
    # own mean. Sampling a target quantile -- rather than clipping an
    # own-mean multiple to the band edges -- keeps the values spread over the
    # normal range instead of piling them onto the two clip bounds, which
    # would make local anomalies trivially separable and structurally
    # indistinguishable from a collective anomaly.
    local_rows = pool[:n_local]
    if len(local_rows) and len(sorted_balance):
        own_mean = entity_mean_balance[local_rows]
        own_mean = np.where(np.isfinite(own_mean) & (own_mean > 0), own_mean, global_median)

        # Feasible quantile windows for each direction.
        up_start = np.clip(_empirical_cdf(sorted_balance, own_mean * _LOCAL_MIN_RATIO), _LOCAL_Q_LO, _LOCAL_Q_HI)
        down_end = np.clip(_empirical_cdf(sorted_balance, own_mean / _LOCAL_MIN_RATIO), _LOCAL_Q_LO, _LOCAL_Q_HI)
        width_up = _LOCAL_Q_HI - up_start
        width_down = down_end - _LOCAL_Q_LO

        up_ok = width_up >= _LOCAL_MIN_Q_WINDOW
        down_ok = width_down >= _LOCAL_MIN_Q_WINDOW
        coin = rng.random(len(local_rows)) < 0.5
        # Pick a direction that actually has room; if both do, pick at random;
        # if neither does (entity mean sits mid-distribution), take the wider.
        go_up = np.where(up_ok & down_ok, coin, np.where(up_ok, True, np.where(down_ok, False, width_up >= width_down)))

        start = np.where(go_up, up_start, _LOCAL_Q_LO)
        end = np.where(go_up, _LOCAL_Q_HI, down_end)
        end = np.maximum(end, start)
        q = start + (end - start) * rng.random(len(local_rows))
        new_val = _empirical_quantile(sorted_balance, q)

        df.loc[df.index[local_rows], "account_balance"] = new_val
        is_anomaly[local_rows] = True
        anomaly_type[local_rows] = "local"
        available[local_rows] = False
        counts["local"] = len(local_rows)

        n_relaxed = int((~up_ok & ~down_ok).sum())
        achieved = np.abs(np.log(new_val / own_mean))
        logger.info(
            "Local anomalies: %d rows drawn from the global [p%.0f, p%.0f] band; median deviation "
            "from the entity's own mean = %.2fx (%.1f%% of rows met the >=%.0fx target; %d rows had "
            "no feasible window and used the widest available offset)",
            len(local_rows), 100 * _LOCAL_Q_LO, 100 * _LOCAL_Q_HI,
            float(np.exp(np.median(achieved))), 100.0 * float((achieved >= np.log(_LOCAL_MIN_RATIO)).mean()),
            _LOCAL_MIN_RATIO, n_relaxed,
        )
    pool = pool[n_local:]

    # Contextual: a large withdrawal that would be normal in December but is
    # anomalous outside of it (context = calendar month).
    month_of_year = pd.DatetimeIndex(df["period"]).month.to_numpy()
    non_december_pool = pool[month_of_year[pool] != 12]
    contextual_rows = non_december_pool[:n_contextual]
    if len(contextual_rows):
        december_like = rng.lognormal(mean=6.3, sigma=0.6, size=len(contextual_rows)) * rng.uniform(2.0, 3.0, size=len(contextual_rows))
        df.loc[df.index[contextual_rows], "withdrawal_amount"] = december_like
        is_anomaly[contextual_rows] = True
        anomaly_type[contextual_rows] = "contextual"
        available[contextual_rows] = False
        counts["contextual"] = len(contextual_rows)

    total_anomalies = int(is_anomaly.sum())
    logger.info(
        "Anomaly injection summary: %d / %d rows flagged (%.2f%%) -- global=%d, local=%d, "
        "contextual=%d, collective=%d",
        total_anomalies, n_rows, 100 * total_anomalies / n_rows,
        counts["global"], counts["local"], counts["contextual"], counts["collective"],
    )

    ground_truth = pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "period": df["period"].to_numpy(),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    })
    ground_truth.attrs["_anomaly_counts"] = counts
    return ground_truth


def _write_ground_truth(gt_df: pd.DataFrame, ground_truth_path: str, logger: logging.Logger) -> str:
    """Write the ground truth and return the path actually written.

    Falls back from parquet to a sibling `.csv` when no parquet engine is
    installed. The *returned* path is the real one, so callers never have to
    guess which of the two files exists.
    """
    parent = os.path.dirname(ground_truth_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        gt_df.to_parquet(ground_truth_path, index=False)
        logger.info("Wrote ground truth to %s (parquet, %d rows)", ground_truth_path, len(gt_df))
        return ground_truth_path
    except ImportError as exc:
        fallback_path = os.path.splitext(ground_truth_path)[0] + ".csv"
        logger.warning(
            "Neither pyarrow nor fastparquet is importable (%s); falling back to CSV at %s. "
            "Install pyarrow to get parquet output.",
            exc, fallback_path,
        )
        gt_df.to_csv(fallback_path, index=False)
        return fallback_path


def _write_synthetic_marker(out_path: str, logger: logging.Logger, **facts) -> Optional[str]:
    """Stamp the written panel as generated, next to the CSV itself.

    Without this file, only the run that *generates* the panel knows it is
    synthetic: every later run finds an ordinary CSV on disk and cannot tell
    it from real data. That matters because tuned hyperparameters are
    persisted as project artifacts -- a second `python main.py` on the same
    generated panel would otherwise write `best_params_*.yaml` fitted to
    invented numbers and present them as the official ones.

    Best-effort: a marker that cannot be written is logged and skipped rather
    than failing the generation. The consequence of losing it is that a later
    run treats the panel as real, so the failure is reported loudly.
    """
    marker = paths.synthetic_marker_for(out_path)
    payload = {
        "synthetic": True,
        "generator": "src.data.synthetic.generate_synthetic_panel",
        "panel": os.path.basename(out_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **facts,
    }
    try:
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Marked panel as synthetic: %s", marker)
        return marker
    except OSError as exc:
        logger.warning(
            "Could not write the synthetic-provenance marker %s (%s). A later "
            "run will treat this generated panel as real data and may persist "
            "tuned parameters from it.", marker, exc,
        )
        return None


def generate_synthetic_panel(
    n_individuals: int = 100_000,
    n_periods: int = 10,
    out_path: str = paths.DATA_PATH,
    ground_truth_path: str = paths.GROUND_TRUTH_PATH,
    seed: int = 42,
    frac_global: float = _FRAC_GLOBAL_DEFAULT,
    frac_local: float = _FRAC_LOCAL_DEFAULT,
    frac_contextual: float = _FRAC_CONTEXTUAL_DEFAULT,
    frac_collective: float = _FRAC_COLLECTIVE_DEFAULT,
) -> PanelGenerationResult:
    """Generate a synthetic monthly banking panel and write it to disk.

    Writes two files and returns a :class:`PanelGenerationResult` describing
    them; callers should reload the data via
    ``src.data.loader.load_or_generate_panel``.

    The result is a *balanced* panel keyed by ``(entity_id, period)``:
    `n_individuals` x `n_periods` rows, every entity observed exactly once in
    every period. `period` values are the `n_periods` consecutive month
    starts ending at a fixed anchor (`_REFERENCE_END_MONTH`, 2026-06-01) --
    deliberately not derived from wall-clock time, so output depends only on
    the arguments.

    Generation applies, in order: base panel, correlation-breaking on a ~5%
    subgroup, temporal drift, anomaly injection, derived-column recomputation,
    missingness injection.

    That order is load-bearing. Anomaly injection rewrites `account_balance`,
    `monthly_transactions_amount` and `withdrawal_amount`, so the columns
    derived from them are replayed immediately afterwards (see
    `_recompute_derived_columns`) -- otherwise the panel carries an arithmetic
    fingerprint of the injected rows that a detector can exploit as a label
    leak. Missingness comes last so that it cannot be undone by a later
    injection, and it skips the injected rows entirely so the ground-truth
    oracle stays fully observed.

    Dynamic features are persistent within an entity rather than redrawn
    i.i.d. each month: `account_balance`, `monthly_transactions_amount`,
    `monthly_transactions_count` and `withdrawal_amount` are anchored on a
    per-entity latent level, `is_digital_active` on a per-entity latent
    propensity, and `transaction_channel` on a per-entity "home" channel
    reused ~75% of months. Without that persistence the local and contextual
    anomaly definitions would be meaningless.

    Missingness is injected with two different mechanisms (relevant to any
    imputation step downstream):

    * MCAR at a 2% rate, independent of the value, on
      `customer_satisfaction_score`, `days_since_last_login` and
      `credit_score` (kept as nullable ``Int64``, not floats).
    * MNAR on `account_balance` and `income`: the probability of being
      missing depends on the value's own magnitude -- 1% baseline, +15
      percentage points for rows in the top 5% of the column -- simulating
      manual review/redaction of large amounts before release.

    Both mechanisms skip the injected-anomaly rows, so realised missingness
    rates apply to the ~98% of rows labelled `"none"`.

    Side effects: writes `out_path` and the ground-truth file (creating
    parent directories), and logs every phase through
    `src.utils.logging_config`. Anomaly labels are written ONLY to the
    separate ground-truth file and never as a column of `out_path`, so
    downstream detection stays unsupervised by default.

    Args:
        n_individuals: Number of distinct entities (customers). Default
            100_000. Combined with the default `n_periods=10` this yields
            1,000,000 rows. Lower this (and/or `n_periods`) for fast smoke
            tests, e.g. `n_individuals=200, n_periods=6`.
        n_periods: Number of consecutive monthly periods per entity. Default
            10. Must be >= 2 for temporal drift, local, and collective
            anomaly injection to have any effect (with n_periods == 1 those
            phases are skipped/no-ops and only logged as such).
        out_path: Where to write the main (unlabeled) panel as CSV. Parent
            directories are created if missing.
        ground_truth_path: Where to write the hidden ground-truth labels
            (parquet if pyarrow/fastparquet is available, else a sibling
            `.csv` file with a logged warning). This is a *request*: the path
            actually written is reported as
            `PanelGenerationResult.ground_truth_path` and may carry a `.csv`
            extension instead of `.parquet`. Callers must use the returned
            path and never assume the extension.
        seed: Seed for both `numpy` and the stdlib `random` module. A given
            (n_individuals, n_periods, seed) combination reproduces
            byte-identical files on any machine and any date.
        frac_global, frac_local, frac_contextual, frac_collective: Fraction
            of rows to flag with each anomaly type (default 0.5% each, ~2%
            in total); see the module docstring for what each type means.
            The types are mutually exclusive, so realised counts can come in
            marginally under budget. Collective anomalies are emitted as
            synchronised groups of 20-50 entities; the number of groups is
            derived from `frac_collective` so the type stays measurable at
            any panel size.

    Returns:
        A `PanelGenerationResult` with the real `out_path` and
        `ground_truth_path`, the panel shape, and the per-type anomaly counts.

    Main dataset columns (22, unlabeled):
        entity_id, period, age, tenure_months, region, segment,
        employment_status, marital_status, product_type,
        transaction_channel, is_digital_active, income, account_balance,
        monthly_transactions_amount, monthly_transactions_count,
        avg_transaction_amount, withdrawal_amount, credit_score,
        overdraft_count, num_products, days_since_last_login,
        customer_satisfaction_score.

    Ground-truth columns: entity_id, period, is_anomaly, anomaly_type
        (one of "global", "local", "contextual", "collective", "none").
    """
    logger = setup_logging()
    with log_phase("generate_synthetic_panel"):
        rng = np.random.default_rng(seed)
        random.seed(seed)

        with log_phase("base generation", logger):
            df = _generate_base_panel(n_individuals, n_periods, rng, logger)

        with log_phase("correlation-breaking", logger):
            _break_correlation_in_subgroup(df, rng, logger)

        with log_phase("drift injection", logger):
            _inject_temporal_drift(df, n_periods, rng, logger)

        with log_phase("anomaly injection", logger):
            ground_truth = _inject_anomalies(
                df, n_individuals, n_periods, rng, logger,
                frac_global=frac_global, frac_local=frac_local,
                frac_contextual=frac_contextual, frac_collective=frac_collective,
            )
        anomaly_counts = dict(ground_truth.attrs.get("_anomaly_counts", {}))

        # Injection rewrote account_balance / monthly_transactions_amount /
        # withdrawal_amount, so every column derived from them is now stale.
        # Replaying the chain here (and not earlier) is what keeps the panel
        # internally consistent and free of the arithmetic label signature --
        # see `_recompute_derived_columns`.
        with log_phase("derived-column recomputation", logger):
            _recompute_derived_columns(df, rng, logger, context="post-injection")

        # Missingness runs last, and never touches the injected rows: it must
        # not resurrect values (if it ran first, injection would refill NaNs)
        # nor erase the features that define a labelled anomaly.
        with log_phase("missingness injection", logger):
            _inject_missingness(
                df, rng, logger,
                protect_mask=ground_truth["is_anomaly"].to_numpy(dtype=bool),
            )

        # Drop generation-only bookkeeping before writing to disk.
        df.attrs.clear()

        with log_phase("writing to disk", logger):
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            df.to_csv(out_path, index=False)
            logger.info("Wrote main panel to %s (%d rows x %d columns)", out_path, len(df), df.shape[1])
            real_gt_path = _write_ground_truth(ground_truth, ground_truth_path, logger)
            _write_synthetic_marker(
                out_path, logger,
                n_rows=len(df), n_entities=n_individuals, n_periods=n_periods,
                seed=seed,
            )

        return PanelGenerationResult(
            out_path=out_path,
            ground_truth_path=real_gt_path,
            n_rows=len(df),
            n_entities=n_individuals,
            n_periods=n_periods,
            anomaly_counts=anomaly_counts,
        )
