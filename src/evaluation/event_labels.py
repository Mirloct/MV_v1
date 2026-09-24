"""External event labels: give ``(entity, codmes)`` rows a target from a separate
CSV, without ever pretending an unreviewed row is negative.

The main panel (``data.csv``) carries **no target column**. Reviewed labels live
in another file -- by default under ``data/reviewed_labels/`` -- with at least
``entity_id``, ``codmes`` and ``target`` (see ``reports/Supervisión temporal por
eventos.md`` for the full contract). This module is the *only* place that reads
that file and turns it into per-row arrays aligned with the panel's ``keys``:

* ``target`` is ``NaN`` wherever there is no usable decision. ``pending``,
  ``uncertain``, ``conflict``, ``superseded``, ``withdrawn`` statuses, invalid
  or null targets, conflicting duplicates and rows absent from the file are all
  *unknown*, never 0. (``unlisted_as_negative`` is an explicit opt-in for a file
  that is an exhaustive base, and the sufficiency gate vetoes it unless the
  audit is attested.)
* Positive months are collapsed into **episodes** (an ``episode_id`` column if the
  file has one, else consecutive positive months of one entity, closed after
  ``washout_months`` without a positive), because six consecutive positive
  months are one case, not six independent events.
* An episode is **mature** once its end plus the label horizon, the confirmation
  delay and a buffer have all elapsed by the cut-off; a row is a usable
  negative only if its own horizon has also elapsed. Immature rows are excluded
  from evaluation, never counted as negatives.

Nothing here touches training: labels are consumed after the detectors are
fitted (evaluation, sufficiency gate, optional challengers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.loader import PanelSchema, parse_period_column
from src.utils.logging_config import setup_logging

__all__ = [
    "EventLabels", "EventLabelsContractError", "EventLabelsEmptyError", "EventLabelsError",
    "LabelSpec", "find_labels_file", "load_event_labels",
]

#: Column aliases tried (case-insensitively) when the spec names none.
_ALIASES = {
    "entity": ("entity_id", "id_entidad", "entidad", "cliente_id", "id"),
    "period": ("codmes", "period", "periodo", "mes"),
    "target": ("target", "label", "is_anomaly", "y"),
    "status": ("label_status", "status", "estado"),
    "episode": ("episode_id", "id_episodio", "episodio"),
    "maturity": ("maturity_date", "fecha_madurez"),
    "available": ("label_available_at", "available_at", "fecha_disponibilidad"),
}
_TARGET_TOKENS = {"1": 1.0, "0": 0.0, "true": 1.0, "false": 0.0, "si": 1.0, "sí": 1.0,
                  "no": 0.0, "yes": 1.0}
_LABEL_FILE_NAMES = ("reviewed_labels.csv", "labels.csv")


class EventLabelsError(ValueError):
    """Base of every reason the labels file cannot be used; callers ignore the
    file and keep running unsupervised."""


class EventLabelsContractError(EventLabelsError):
    """The labels file exists but is unreadable or does not satisfy the minimum contract."""


class EventLabelsEmptyError(EventLabelsError):
    """The labels file has no rows or no target values at all: nothing to use."""


@dataclass(frozen=True)
class LabelSpec:
    """How to read and interpret the external labels file.

    Column names left ``None`` are auto-detected from :data:`_ALIASES`.
    Month-valued knobs are whole months; ``as_of`` (``YYYY-MM-DD``/``YYYYMM``)
    is the point-in-time cut-off for maturity and availability, default the last
    period of the panel.
    """

    entity_col: Optional[str] = None
    period_col: Optional[str] = None
    target_col: Optional[str] = None
    status_col: Optional[str] = None
    episode_col: Optional[str] = None
    maturity_col: Optional[str] = None
    available_col: Optional[str] = None
    usable_statuses: tuple[str, ...] = ("confirmed", "adjudicated")
    unlisted_as_negative: bool = False
    horizon_months: int = 1
    confirm_delay_months: int = 0
    maturity_buffer_months: int = 0
    washout_months: int = 1
    as_of: Optional[str] = None

    @property
    def maturity_lag_months(self) -> int:
        return int(self.horizon_months + self.confirm_delay_months + self.maturity_buffer_months)


@dataclass
class EventLabels:
    """Per-row labels aligned with the panel ``keys`` (see module docstring)."""

    target: np.ndarray               # float64, NaN = no usable decision
    known: np.ndarray                # bool, a usable 0/1 decision exists
    entity: np.ndarray               # object, str
    month: np.ndarray                # int64, year*12 + month-1
    episode_id: np.ndarray           # object, "" where the row is not positive
    episode_mature: np.ndarray       # bool, row belongs to a mature positive episode
    mature_known: np.ndarray         # bool, known AND its maturity condition holds
    episodes: pd.DataFrame           # one row per positive episode
    cutoff_month: int                # last month usable as "elapsed"
    spec: LabelSpec
    audit: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def usable(self) -> np.ndarray:
        """Rows that may enter an evaluation: known and mature."""
        return self.mature_known

    def horizon_target(self, horizon_months: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(y, eligible, episode)`` for "a NEW episode starts within the next H months".

        A row is at risk only when it is a usable negative (no episode is running
        on it). ``y`` is 1 when a mature episode of the same entity starts in
        ``(t, t+H]``. A row is eligible only when its whole horizon has elapsed
        by the cut-off (right-censored rows are dropped, not called negative) and
        no *immature* episode starts inside its window (its outcome is unknown).
        ``episode`` names the episode that starts inside the window of each ``y == 1``
        row (``""`` elsewhere).
        """
        n = len(self.month)
        y = np.zeros(n, dtype=float)
        episode = np.full(n, "", dtype=object)
        eligible = self.mature_known & (self.target == 0)
        eligible &= self.month + horizon_months <= self.cutoff_month
        if not len(self.episodes) or not eligible.any():
            return y, eligible, episode
        rows = pd.DataFrame({"ent": self.entity, "t": self.month, "i": np.arange(n)})
        rows = rows.loc[eligible].sort_values("t")
        for mature, flag in ((True, "mature"), (False, "immature")):
            starts = self.episodes.loc[self.episodes["mature"] == mature, ["entity", "start_m", "episode_id"]]
            if starts.empty:
                continue
            starts = starts.rename(columns={"entity": "ent"}).sort_values("start_m")
            nxt = pd.merge_asof(
                rows, starts, left_on="t", right_on="start_m", by="ent",
                direction="forward", allow_exact_matches=False,
            )
            within = (nxt["start_m"] - nxt["t"]).le(horizon_months).to_numpy()
            idx = nxt["i"].to_numpy()[within]
            if flag == "mature":
                y[idx] = 1.0
                episode[idx] = nxt["episode_id"].to_numpy()[within]
            else:
                eligible[idx] = False
        return y, eligible, episode


# --------------------------------------------------------------------------- #
# Locating and reading the file                                               #
# --------------------------------------------------------------------------- #
def find_labels_file(path: Optional[str], default_dir: Optional[str] = None) -> Optional[str]:
    """Resolve the labels file: an explicit file, else the best CSV/parquet in a
    directory (``current/reviewed_labels.*`` first, then the newest). ``None``
    when nothing is there -- the caller falls back to the unsupervised run."""
    for candidate in (path, default_dir):
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if os.path.isdir(candidate):
            found = _best_in_directory(candidate)
            if found:
                return found
        elif path and candidate == path:
            return None  # an explicit path that does not exist is not silently replaced
    return None


def _best_in_directory(directory: str) -> Optional[str]:
    for sub in ("current", ""):
        for name in _LABEL_FILE_NAMES:
            for ext in (".csv", ".parquet"):
                p = os.path.join(directory, sub, os.path.splitext(name)[0] + ext)
                if os.path.isfile(p):
                    return os.path.abspath(p)
    files = [
        os.path.join(root, f) for root, _d, names in os.walk(directory) for f in names
        if f.lower().endswith((".csv", ".parquet"))
        and not any(part in ("quarantine", "fixtures", "schema") for part in root.split(os.sep))
    ]
    return os.path.abspath(max(files, key=os.path.getmtime)) if files else None


def _read_table(path: str) -> pd.DataFrame:
    """Read the file as text, turning every way it can be empty, corrupt or
    unreadable into an :class:`EventLabelsError` (never a raw pandas/OS error)."""
    try:
        if os.path.getsize(path) == 0:
            raise EventLabelsEmptyError(f"El archivo {os.path.basename(path)!r} está vacío (0 bytes).")
        if path.lower().endswith((".parquet", ".pq")):
            frame = pd.read_parquet(path).astype(object)
        else:
            # Everything as text first: keeps leading zeros in ids and lets the target
            # and period be validated explicitly instead of silently coerced by pandas.
            frame = pd.read_csv(path, dtype=str, keep_default_na=True, encoding="utf-8-sig")
    except EventLabelsError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise EventLabelsEmptyError(f"El archivo {os.path.basename(path)!r} no tiene datos.") from exc
    except Exception as exc:  # noqa: BLE001 - corrupt/locked/binary file: ignore it, do not crash
        raise EventLabelsContractError(
            f"No se pudo leer {os.path.basename(path)!r} ({type(exc).__name__}: {exc})."
        ) from exc
    frame = frame.dropna(how="all")
    if frame.empty:
        raise EventLabelsEmptyError(
            f"El archivo {os.path.basename(path)!r} no tiene filas (solo encabezado o filas en blanco).")
    return frame


def _resolve(df: pd.DataFrame, given: Optional[str], key: str, required: bool) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    if given:
        if given.strip().lower() in lowered:
            return lowered[given.strip().lower()]
        raise EventLabelsContractError(
            f"Column {given!r} (for {key}) is not in the labels file; it has {list(df.columns)}."
        )
    for alias in _ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    if required:
        raise EventLabelsContractError(
            f"The labels file needs a {key} column (any of {_ALIASES[key]}); "
            f"it has {list(df.columns)}."
        )
    return None


def _month_index(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, errors="coerce")
    return (ts.dt.year * 12 + ts.dt.month - 1).to_numpy(dtype="float64")


def _parse_periods(series: pd.Series, name: str, log: logging.Logger) -> pd.Series:
    parsed = parse_period_column(series, name, log)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise EventLabelsContractError(
            f"Column {name!r} could not be parsed as a month (expected e.g. 202401 or 2024-01-01)."
        )
    return parsed


def _coerce_target(raw: pd.Series) -> tuple[pd.Series, int, int]:
    """``(target, n_null, n_invalid)``: 0/1 floats, NaN for null and invalid."""
    text = raw.astype("string").str.strip().str.lower()
    null = text.isna() | text.isin(["", "nan", "none", "null", "<na>"])
    mapped = text.map(_TARGET_TOKENS)
    numeric = pd.to_numeric(text, errors="coerce")
    value = mapped.astype("float64").fillna(numeric.astype("float64"))
    invalid = ~null & ~value.isin([0.0, 1.0])
    value = value.where(~invalid & ~null)
    return value, int(null.sum()), int(invalid.sum())


# --------------------------------------------------------------------------- #
# Episodes and maturity                                                       #
# --------------------------------------------------------------------------- #
def _derive_episodes(entity: np.ndarray, month: np.ndarray, washout: int) -> np.ndarray:
    """Episode ids for positive rows: an entity's positives belong to one episode
    while fewer than ``washout`` non-positive months separate them."""
    ids = np.full(len(entity), "", dtype=object)
    frame = pd.DataFrame({"e": entity, "m": month, "i": np.arange(len(entity))}).sort_values(["e", "m"])
    same = frame["e"].eq(frame["e"].shift())
    gap = frame["m"].sub(frame["m"].shift()) - 1
    new_episode = ~same | gap.ge(max(1, washout))
    frame["seq"] = new_episode.cumsum()
    first = frame.groupby("seq")["m"].transform("min").astype(int)
    ids[frame["i"].to_numpy()] = (
        frame["e"].astype(str) + "#" + first.astype(str)
    ).to_numpy()
    return ids


def _episode_table(entity, month, episode_id, positive, maturity_month, spec, cutoff_month):
    rows = pd.DataFrame({"episode_id": episode_id[positive], "entity": entity[positive],
                         "m": month[positive]})
    if rows.empty:
        return pd.DataFrame(columns=["episode_id", "entity", "start_m", "end_m", "n_rows", "mature"])
    table = rows.groupby("episode_id", sort=False).agg(
        entity=("entity", "first"), start_m=("m", "min"), end_m=("m", "max"), n_rows=("m", "size"),
    ).reset_index()
    if maturity_month is not None:
        ready = pd.Series(maturity_month[positive]).groupby(rows["episode_id"].to_numpy()).max()
        table["mature"] = table["episode_id"].map(ready).le(cutoff_month).fillna(False).to_numpy()
    else:
        table["mature"] = (table["end_m"] + spec.maturity_lag_months) <= cutoff_month
    return table


# --------------------------------------------------------------------------- #
# Public loader                                                               #
# --------------------------------------------------------------------------- #
def load_event_labels(
    path: str, schema: PanelSchema, keys: pd.DataFrame, spec: LabelSpec = LabelSpec(),
) -> EventLabels:
    """Read the labels file and align it to ``keys`` (see module docstring).

    Raises :class:`EventLabelsContractError` when the file cannot satisfy the
    minimum contract (missing key/target columns, unparseable months, no
    overlap with the panel); the caller decides the fallback.
    """
    log = setup_logging()
    entity_col = schema.entity_col or "entity_id"
    time_col = schema.time_col or "period"
    raw = _read_table(path)
    c_ent = _resolve(raw, spec.entity_col, "entity", True)
    c_per = _resolve(raw, spec.period_col, "period", True)
    c_tgt = _resolve(raw, spec.target_col, "target", True)
    c_sta = _resolve(raw, spec.status_col, "status", False)
    c_epi = _resolve(raw, spec.episode_col, "episode", False)
    c_mat = _resolve(raw, spec.maturity_col, "maturity", False)
    c_ava = _resolve(raw, spec.available_col, "available", False)

    panel_month = _month_index(pd.to_datetime(keys[time_col], errors="coerce"))
    if np.isnan(panel_month).all():
        raise EventLabelsContractError("The panel period column could not be read as months.")
    cutoff = _parse_as_of(spec.as_of)
    cutoff_month = (int(cutoff.year * 12 + cutoff.month - 1) if cutoff is not None
                    else int(np.nanmax(panel_month)))

    labels = pd.DataFrame({
        "ent": raw[c_ent].astype(str).str.strip(),
        "m": _month_index(_parse_periods(raw[c_per], str(c_per), log)),
    })
    labels["target"], n_null, n_invalid = _coerce_target(raw[c_tgt])
    if int(labels["target"].notna().sum()) == 0 and n_invalid == 0:
        raise EventLabelsEmptyError(f"La columna {c_tgt!r} no tiene ningún valor de target.")
    audit: dict[str, Any] = {
        "rows_in_file": int(len(raw)), "null_targets": n_null, "invalid_targets": n_invalid,
        "columns": {"entity": c_ent, "period": c_per, "target": c_tgt, "status": c_sta,
                    "episode": c_epi, "maturity": c_mat, "available": c_ava},
        "unparseable_months": int(labels["m"].isna().sum()),
    }
    labels["excluded"] = ""
    labels.loc[labels["m"].isna(), "excluded"] = "unparseable_month"
    if c_sta is not None:
        status = raw[c_sta].astype("string").str.strip().str.lower().fillna("")
        bad = ~status.isin(spec.usable_statuses) & labels["excluded"].eq("")
        labels.loc[bad, "excluded"] = "status:" + status[bad]
    if c_ava is not None:
        avail = _month_index(pd.to_datetime(raw[c_ava], errors="coerce"))
        late = (avail > cutoff_month) & labels["excluded"].eq("")
        labels.loc[late, "excluded"] = "not_yet_available"
    labels.loc[labels["target"].isna() & labels["excluded"].eq(""), "excluded"] = "no_decision"
    audit["excluded"] = labels.loc[labels["excluded"] != "", "excluded"].value_counts().to_dict()
    audit["status_column_present"] = c_sta is not None

    keep = labels["excluded"].eq("")
    good = labels.loc[keep].copy()
    if c_epi is not None:
        good["episode_src"] = raw.loc[keep, c_epi].astype("string").str.strip().to_numpy()
    if c_mat is not None:
        good["maturity_m"] = _month_index(pd.to_datetime(raw.loc[keep, c_mat], errors="coerce"))
    # One row per (entity, month): identical duplicates collapse, disagreeing ones
    # are a conflict and become unknown (never resolved by picking one).
    disagree = good.groupby(["ent", "m"])["target"].transform("nunique").gt(1)
    audit["conflicting_duplicate_rows"] = int(disagree.sum())
    good = good.loc[~disagree].drop_duplicates(["ent", "m"], keep="first")

    n = len(keys)
    ent = keys[entity_col].astype(str).str.strip().to_numpy(dtype=object)
    frame = pd.DataFrame({"ent": ent, "m": panel_month, "i": np.arange(n)})
    joined = frame.merge(good, on=["ent", "m"], how="left").sort_values("i")
    matched = joined["target"].notna().to_numpy()
    audit["panel_rows"] = int(n)
    audit["panel_rows_matched"] = int(matched.sum())
    audit["file_rows_usable"] = int(len(good))
    audit["file_rows_not_in_panel"] = int(len(good) - matched.sum())
    if audit["file_rows_usable"] and not matched.any() and not spec.unlisted_as_negative:
        raise EventLabelsContractError(
            "No usable label row matches any panel (entity, month); check the entity "
            "normalisation and the codmes format."
        )
    target = joined["target"].to_numpy(dtype=float)
    known = matched.copy()
    if spec.unlisted_as_negative:
        unlisted = ~known & ~np.isnan(panel_month)
        # Only rows the file says nothing about: a listed-but-excluded row
        # (pending, uncertain, ...) stays unknown even under this policy.
        listed = frame.merge(
            labels.loc[labels["excluded"] != "", ["ent", "m"]].drop_duplicates(), on=["ent", "m"],
            how="left", indicator=True).sort_values("i")["_merge"].eq("both").to_numpy()
        target[unlisted & ~listed] = 0.0
        known = ~np.isnan(target)
    audit["unlisted_policy"] = "negative" if spec.unlisted_as_negative else "unknown"

    month = np.where(np.isnan(panel_month), -1, panel_month).astype("int64")
    positive = known & (target == 1.0)
    episode_id = np.full(n, "", dtype=object)
    provided = None
    if c_epi is not None and positive.any():
        provided = joined["episode_src"].astype("string").fillna("").to_numpy(dtype=object)
        missing = positive & (provided == "")
        audit["positive_rows_without_episode_id"] = int(missing.sum())
        if missing.any():
            provided = None  # incomplete column: derive for every row, and the gate vetoes
    if provided is not None:
        episode_id[positive] = np.char.add(ent[positive].astype(str), "#" + provided[positive].astype(str))
        audit["episode_source"] = "file"
    else:
        episode_id[positive] = _derive_episodes(ent[positive], month[positive], spec.washout_months)
        audit["episode_source"] = "derived"
    maturity_m = joined["maturity_m"].to_numpy() if c_mat is not None else None
    episodes = _episode_table(ent, month, episode_id, positive, maturity_m, spec, cutoff_month)
    mature_ids = set(episodes.loc[episodes["mature"], "episode_id"])
    episode_mature = positive & np.isin(episode_id, list(mature_ids))
    mature_neg = known & (target == 0.0) & (month + spec.maturity_lag_months <= cutoff_month)
    mature_known = episode_mature | mature_neg

    audit.update(
        cutoff_month=_month_label(cutoff_month), rows_known=int(known.sum()),
        rows_positive=int(positive.sum()), rows_negative=int((known & (target == 0.0)).sum()),
        rows_unknown=int((~known).sum()), rows_immature=int((known & ~mature_known).sum()),
        episodes=int(len(episodes)), episodes_mature=int(episodes["mature"].sum()),
    )
    log.info(
        "Event labels from %s: %d/%d panel rows labeled (%d positive in %d episode(s), %d mature); "
        "%d unknown rows are NOT treated as negative.", path, audit["rows_known"], n,
        audit["rows_positive"], audit["episodes"], audit["episodes_mature"],
        audit["rows_unknown"] if not spec.unlisted_as_negative else 0,
    )
    return EventLabels(
        target=target, known=known, entity=ent, month=month, episode_id=episode_id,
        episode_mature=episode_mature, mature_known=mature_known, episodes=episodes,
        cutoff_month=cutoff_month, spec=spec, audit=audit, source_path=os.path.abspath(path),
    )


def _parse_as_of(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 6 and text.isdigit():  # codmes style: 202403
        text = f"{text[:4]}-{text[4:]}-01"
    try:
        return pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise EventLabelsContractError(f"as_of {value!r} is not a date (use YYYYMM or YYYY-MM-DD).") from exc


def _month_label(month_index: int) -> str:
    return f"{month_index // 12:04d}-{month_index % 12 + 1:02d}"
