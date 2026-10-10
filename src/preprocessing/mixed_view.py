"""The VAE's mixed-type view of the panel: continuous, binary and *indexed* categoricals.

The Isolation Forest keeps the matrix it always had (one-hot columns are withheld from it by
:func:`src.preprocessing.split_matrix_for_model`). The VAE used to receive every one-hot
column as an independent feature, so a variable with many levels contributed many
reconstruction terms and dominated ``recon_topk``. This module builds the alternative
input: **one integer index per categorical variable** (with explicit MISSING / UNKNOWN
tokens), next to the continuous and binary columns the pipeline already produced.

The result is still a single dense 2-D ``float32`` matrix (categorical indices are stored as
whole floats, exact up to 2**24), so masking, slicing, stacking and every other consumer that
treats the VAE input as "a matrix with row order" keep working. What each column *means* is
carried by a :class:`MixedLayout`, which travels with the detector.

Column roles
------------
``num``   continuous, reconstructed (Huber/MSE) and scored.
``bool``  binary, reconstructed with a BCE-with-logits head and scored.
``cat``   one integer index per original categorical variable; embedded by the encoder,
          reconstructed with one softmax head over its vocabulary, scored by its NLL.

Token policy (``explicit_token``)
---------------------------------
Index ``0`` = MISSING (the value was null), index ``1`` = UNKNOWN (a level the *fit rows* did
not have, or one rarer than ``min_frequency``), indices ``2..`` = the vocabulary **sorted
alphabetically** (so the mapping does not depend on the order in which levels appear in the
data). Vocabularies are learned from the fit rows only, so a level that first appears in
validation/OOT is UNKNOWN there, never silently a normal category.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "MISSING_TOKEN", "UNKNOWN_TOKEN", "MISSING_INDEX", "UNKNOWN_INDEX", "N_TOKENS",
    "MIXED_VIEW_VERSION", "assert_no_onehot", "categorical_sources", "CategoricalSpec", "MixedLayout", "MixedViewConfig", "MixedViewBuilder",
]

MISSING_TOKEN, UNKNOWN_TOKEN = "<MISSING>", "<UNKNOWN>"
MISSING_INDEX, UNKNOWN_INDEX, N_TOKENS = 0, 1, 2
MIXED_VIEW_VERSION = "mixed_view_v1"

ROLES = ("num", "bool", "cat")
_SUPPORTED_POLICIES = ("explicit_token",)


@dataclass(frozen=True)
class CategoricalSpec:
    """One original categorical variable: its X column, source column and vocabulary."""

    name: str                      # original (source) column, e.g. "segment"
    column: str                    # column in the mixed matrix, e.g. "cat__segment"
    vocabulary: tuple[str, ...]    # observed levels, sorted; indices start at N_TOKENS

    @property
    def cardinality(self) -> int:
        return N_TOKENS + len(self.vocabulary)

    def labels(self) -> list[str]:
        return [MISSING_TOKEN, UNKNOWN_TOKEN, *self.vocabulary]


@dataclass(frozen=True)
class MixedViewConfig:
    """How the VAE view is built (the ``vae:`` block of ``configs/pipeline.yaml``)."""

    min_frequency: float = 0.001          # levels rarer than this (of the fit rows) become UNKNOWN
    unknown_category_policy: str = "explicit_token"
    missing_category_policy: str = "explicit_token"

    def __post_init__(self) -> None:
        for name in ("unknown_category_policy", "missing_category_policy"):
            if getattr(self, name) not in _SUPPORTED_POLICIES:
                raise ValueError(f"{name}={getattr(self, name)!r} is not supported; "
                                 f"use one of {_SUPPORTED_POLICIES}.")
        if not 0.0 <= self.min_frequency < 1.0:
            raise ValueError("min_frequency must be in [0, 1).")

    def to_dict(self) -> dict:
        return {"min_frequency": float(self.min_frequency),
                "unknown_category_policy": self.unknown_category_policy,
                "missing_category_policy": self.missing_category_policy}


@dataclass
class MixedLayout:
    """What every column of the mixed matrix is. Serialisable; part of every fingerprint."""

    columns: list[str]
    roles: list[str]
    categoricals: dict[str, CategoricalSpec] = field(default_factory=dict)   # keyed by X column
    policy: dict = field(default_factory=dict)
    version: str = MIXED_VIEW_VERSION

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.roles):
            raise ValueError("columns and roles must have the same length.")
        bad = sorted(set(self.roles) - set(ROLES))
        if bad:
            raise ValueError(f"unknown column role(s) {bad}; valid roles: {list(ROLES)}.")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("layout columns must be unique.")
        for col, role in zip(self.columns, self.roles):
            if (role == "cat") != (col in self.categoricals):
                raise ValueError(f"column {col!r} has role {role!r} but "
                                 f"{'no ' if role == 'cat' else 'an unexpected '}categorical spec.")

    # -- structure -------------------------------------------------------------- #
    def positions(self, role: str) -> np.ndarray:
        return np.array([i for i, r in enumerate(self.roles) if r == role], dtype=np.int64)

    def names(self, role: str) -> list[str]:
        return [c for c, r in zip(self.columns, self.roles) if r == role]

    @property
    def variables(self) -> list[str]:
        """Original scored variables, in contribution order: numeric, then binary, then categorical."""
        return self.names("num") + self.names("bool") + self.names("cat")

    @property
    def variable_kinds(self) -> list[str]:
        return (["num"] * len(self.names("num")) + ["bool"] * len(self.names("bool"))
                + ["cat"] * len(self.names("cat")))

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def cat_specs(self) -> list[CategoricalSpec]:
        return [self.categoricals[c] for c in self.names("cat")]

    def with_extra_numeric(self, name: str) -> "MixedLayout":
        """A copy with one more continuous column at the end (the stacked IF score)."""
        return MixedLayout(self.columns + [name], self.roles + ["num"], dict(self.categoricals),
                           dict(self.policy), self.version)

    def subset(self, drop: Sequence[str]) -> tuple["MixedLayout", np.ndarray]:
        """``(layout without the ``drop`` columns, positions to keep in the matrix)``."""
        dropset = set(drop)
        keep = [i for i, c in enumerate(self.columns) if c not in dropset]
        cols = [self.columns[i] for i in keep]
        return (MixedLayout(cols, [self.roles[i] for i in keep],
                            {c: s for c, s in self.categoricals.items() if c in cols},
                            dict(self.policy), self.version),
                np.asarray(keep, dtype=np.int64))

    # -- identity --------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "version": self.version, "columns": list(self.columns), "roles": list(self.roles),
            "policy": dict(self.policy),
            "categoricals": {c: {"name": s.name, "column": s.column, "vocabulary": list(s.vocabulary)}
                             for c, s in self.categoricals.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MixedLayout":
        cats = {c: CategoricalSpec(v["name"], v["column"], tuple(v["vocabulary"]))
                for c, v in d["categoricals"].items()}
        return cls(list(d["columns"]), list(d["roles"]), cats, dict(d.get("policy", {})),
                   d.get("version", MIXED_VIEW_VERSION))

    def fingerprint(self) -> str:
        """Hash of the column order, roles, categorical names, vocabularies and token policy."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def categorical_sources(df: pd.DataFrame, schema: Any) -> list[str]:
    """Original categorical columns of the panel: EXACTLY the columns the pipeline's categorical branch
    one-hot encodes (``make_column_selector(dtype_include=["object", "category"])``, keys excluded), so the
    embedding representation covers the same variables the one-hot representation did -- no more (e.g. a
    ``string`` or ``timedelta`` column the one-hot branch drops) and no fewer. ``key_columns`` (structural
    keys, target, ``schema.identification_columns``) is excluded, same as the one-hot branch."""
    from sklearn.compose import make_column_selector

    from src.data.loader import key_columns

    keys = key_columns(schema)
    candidates = [c for c in df.columns if c not in keys]
    return list(make_column_selector(dtype_include=["object", "category"])(df[candidates]))


def assert_no_onehot(layout: MixedLayout) -> None:
    """The VAE input must hold no one-hot dummy: every ``cat__*`` column is a single index column."""
    bad = [c for c, r in zip(layout.columns, layout.roles) if c.startswith("cat__") and r != "cat"]
    if bad:
        raise ValueError(f"one-hot columns reached the mixed VAE input: {bad[:5]}")


def _classify(name: str) -> Optional[str]:
    """Role of a column of the preprocessed one-hot matrix; ``None`` = replaced (one-hot)."""
    if name.startswith("cat__"):
        return None
    if name.startswith("bool__"):
        return "bool"
    return "num"


class MixedViewBuilder:
    """Fit the VAE view on the fit rows, then transform any rows (train, validation, OOT, perturbed).

    Numeric / binary columns are *taken from the matrix the pipeline already built*
    (same causal transforms, same scalers fitted on the train block); only the categorical
    variables are re-derived, from the **raw** panel columns, so nulls are visible as MISSING
    instead of having been imputed away.
    """

    def __init__(self, config: Optional[MixedViewConfig] = None):
        self.config = config or MixedViewConfig()
        self.layout: Optional[MixedLayout] = None
        self._source_of: dict[str, str] = {}
        self._positions: np.ndarray = np.array([], dtype=np.int64)
        self._lookup: dict[str, dict[str, int]] = {}

    # -- fit -------------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, X: Any, feature_names: Sequence[str], fit_mask: np.ndarray,
            cat_sources: Sequence[str]) -> "MixedViewBuilder":
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape[0] != len(df) or not fit_mask.any():
            raise ValueError("fit_mask must select a non-empty subset of the rows of df.")
        if X.shape[0] != len(df) or X.shape[1] != len(feature_names):
            raise ValueError("X, feature_names and df are not aligned.")
        cols: list[str] = []
        roles: list[str] = []
        pos: list[int] = []
        for j, name in enumerate(feature_names):
            role = _classify(str(name))
            if role is not None:
                cols.append(str(name))
                roles.append(role)
                pos.append(j)
        n_fit = int(fit_mask.sum())
        floor = self.config.min_frequency * n_fit
        cats: dict[str, CategoricalSpec] = {}
        for src in cat_sources:
            if src not in df.columns:
                raise ValueError(f"categorical source column {src!r} is not in the panel.")
            counts = df.loc[fit_mask, src].dropna().astype(str).value_counts()
            vocab = tuple(sorted(str(k) for k, v in counts.items() if v >= max(floor, 1)))
            spec = CategoricalSpec(name=str(src), column=f"cat__{src}", vocabulary=vocab)
            cats[spec.column] = spec
            cols.append(spec.column)
            roles.append("cat")
            self._source_of[spec.column] = str(src)
            self._lookup[spec.column] = {lvl: N_TOKENS + i for i, lvl in enumerate(vocab)}
        self.layout = MixedLayout(cols, roles, cats, self.config.to_dict())
        self._positions = np.asarray(pos, dtype=np.int64)
        self._n_pre_cols = len(feature_names)
        return self

    # -- transform ---------------------------------------------------------------- #
    def encode_categorical(self, values: pd.Series, column: str) -> np.ndarray:
        """Integer indices for one categorical variable (MISSING / UNKNOWN / vocabulary)."""
        lookup = self._lookup[column]
        isna = pd.isna(values).to_numpy()
        codes = values.astype(str).map(lookup).fillna(UNKNOWN_INDEX).to_numpy(dtype=np.int64)
        codes[isna] = MISSING_INDEX
        return codes

    def transform(self, df: pd.DataFrame, X: Any) -> np.ndarray:
        """The mixed matrix for ``df``'s rows (``X`` is the pipeline matrix of the same rows)."""
        if self.layout is None:
            raise RuntimeError("MixedViewBuilder is not fitted.")
        if X.shape[0] != len(df):
            raise ValueError("X and df must have the same number of rows.")
        if X.shape[1] != self._n_pre_cols:
            raise ValueError(f"X has {X.shape[1]} columns; the builder was fitted on {self._n_pre_cols}.")
        base = X.tocsc()[:, self._positions].toarray() if sp.issparse(X) else np.asarray(X)[:, self._positions]
        n_cat = len(self.layout.names("cat"))
        out = np.empty((len(df), self.layout.n_columns), dtype=np.float32)
        out[:, : base.shape[1]] = base
        # A binary column has no MISSING token and no indicator: a null there (the pipeline only casts
        # booleans to float, so it arrives as NaN) is scored as its neutral value False rather than
        # aborting the whole scoring call. Numeric NaN still raises in the detector: those are imputed upstream.
        bpos = self.layout.positions("bool")
        if len(bpos):
            block = out[:, bpos]
            out[:, bpos] = np.where(np.isnan(block), 0.0, block)
        for k, col in enumerate(self.layout.names("cat")):
            out[:, base.shape[1] + k] = self.encode_categorical(df[self._source_of[col]], col)
        assert base.shape[1] + n_cat == self.layout.n_columns
        return out

    def fit_transform(self, df, X, feature_names, fit_mask, cat_sources) -> np.ndarray:
        return self.fit(df, X, feature_names, fit_mask, cat_sources).transform(df, X)
