"""Variational autoencoder for mixed data: continuous + binary + embedded categoricals.

Why this exists
---------------
With one-hot inputs every level of a categorical variable is an independent reconstruction
feature: a variable with 40 levels contributes 40 squared-error terms, a mostly-zero column has
a tiny robust scale (MAD), and the diagnostic's ``recon_topk`` ends up choosing among *dummy
columns* instead of business variables (measured: 98 % of the top-5 slots were one-hot columns
that make up 66 % of the columns). Here each categorical variable is **one** input (an embedding
looked up from its integer index) and **one** output (a softmax over its vocabulary), so it
produces **exactly one** reconstruction contribution -- the negative log-likelihood of the
observed category -- whatever its cardinality.

Design
------
* Encoder input = ``[numeric | binary | missing-flags | embedding(cat_1) | ... | embedding(cat_k)]``
  -> MLP trunk -> ``mu``, ``logvar`` (same trunk shape rules as :class:`~src.models.vae.VAEModel`).
* Decoder = MLP trunk from ``z`` with three kinds of heads: a linear head for the numeric variables,
  a logit head for the binary ones and one logit vector *per original categorical variable*.
  Embedding vectors are never reconstructed; the categorical head is evaluated against the true index.
* Loss (per row, summed over original variables, then averaged over the batch)::

      recon = w_num  * sum_j huber(x_j, x^_j)            (or MSE)
            + w_bool * sum_j BCE_with_logits(x_j, l_j)
            + w_cat  * sum_j CE(cat_j, logits_j)
      loss  = recon + beta * KL

  The default weights are all 1.0, so they depend on neither the number of columns nor the number of
  categories -- a variable is one term regardless of how it is encoded.
* Anomaly contribution of variable ``v`` = its own term (Huber/MSE, BCE or NLL); the row score is the
  weighted mean over variables. NLL is floored at probability ``1e-6`` (``NLL_CAP`` = 13.8155) so a
  token the model never saw in training cannot produce an unbounded value; MISSING and UNKNOWN keep
  their own identifiable contribution instead of being folded into a normal category.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.mixed_view import MixedLayout

__all__ = [
    "MIXED_ARCHITECTURE", "MIXED_ARCHITECTURE_VERSION", "ONEHOT_ARCHITECTURE", "NLL_CAP",
    "MixedVAEConfig", "MixedVAEModel", "IncompatibleCheckpointError", "ContributionReference",
    "CONTRIBUTION_FLOOR_FRACTION", "CONTRIBUTION_ABS_FLOOR",
]

ONEHOT_ARCHITECTURE = "onehot_v1"          # the original MLP VAE over the one-hot matrix
MIXED_ARCHITECTURE = "mixed_v1"
#: Bump on ANY change of the mixed architecture or loss definition: old checkpoints are then rejected.
MIXED_ARCHITECTURE_VERSION = "1"
#: -log(1e-6): the largest per-variable NLL. Documented numeric floor of the categorical contribution.
NLL_CAP = float(-math.log(1e-6))

#: Normalisation of the per-variable contributions (the "documented numeric floor"): each contribution is
#: centred at its variable's reference median, then divided by a robust scale (MAD of the centred excess,
#: falling back to its mean) that is floored at ``CONTRIBUTION_FLOOR_FRACTION`` x the MEAN OF THE CENTRED
#: EXCESS of the reference and never below ``CONTRIBUTION_ABS_FLOOR``. Same rule as the diagnostic suite's
#: ``residual_contributions(..., scale_floor_fraction=0.1, center=True)``.
CONTRIBUTION_FLOOR_FRACTION = 0.1
CONTRIBUTION_ABS_FLOOR = 1e-6

_NUMERIC_LOSSES = ("huber", "mse")
_FAMILIES = ("numeric", "boolean", "categorical")


class IncompatibleCheckpointError(ValueError):
    """A checkpoint/payload of another architecture (e.g. one-hot) was offered; never loaded partially."""


@dataclass(frozen=True)
class MixedVAEConfig:
    """Loss, embedding and token settings (the ``vae:`` block of ``configs/pipeline.yaml``)."""

    embedding_dimension_strategy: str = "auto"     # "auto" | "fixed"
    embedding_dimension: Optional[int] = None      # used when strategy == "fixed"
    embedding_min_dimension: int = 2
    embedding_max_dimension: int = 32
    numeric_loss: str = "huber"
    huber_delta: float = 1.0
    boolean_loss: str = "binary_cross_entropy"
    categorical_loss: str = "cross_entropy"
    aggregate_by_original_feature: bool = True
    weight_numeric: float = 1.0
    weight_boolean: float = 1.0
    weight_categorical: float = 1.0
    unknown_category_policy: str = "explicit_token"
    missing_category_policy: str = "explicit_token"

    def __post_init__(self) -> None:
        if self.embedding_dimension_strategy not in ("auto", "fixed"):
            raise ValueError("embedding dimension_strategy must be 'auto' or 'fixed'.")
        if self.embedding_dimension_strategy == "fixed" and not (self.embedding_dimension or 0) >= 1:
            raise ValueError("strategy 'fixed' needs embedding_dimension >= 1.")
        if not 1 <= self.embedding_min_dimension <= self.embedding_max_dimension:
            raise ValueError("need 1 <= min_dimension <= max_dimension.")
        if self.numeric_loss not in _NUMERIC_LOSSES:
            raise ValueError(f"numeric loss must be one of {_NUMERIC_LOSSES}.")
        if self.boolean_loss != "binary_cross_entropy":
            raise ValueError("boolean loss must be 'binary_cross_entropy'.")
        if self.categorical_loss != "cross_entropy":
            raise ValueError("categorical loss must be 'cross_entropy'.")
        if not self.aggregate_by_original_feature:
            raise ValueError("aggregate_by_original_feature=false is not supported: one contribution "
                             "per original variable is the point of the mixed VAE.")
        for w in (self.weight_numeric, self.weight_boolean, self.weight_categorical):
            if not w > 0:
                raise ValueError("family loss weights must be > 0.")
        for p in (self.unknown_category_policy, self.missing_category_policy):
            if p != "explicit_token":
                raise ValueError("token policies other than 'explicit_token' are not supported.")

    # -- embedding sizes ------------------------------------------------------- #
    def embedding_dim_for(self, cardinality: int) -> int:
        """Embedding width of a variable with ``cardinality`` classes (tokens included).

        ``auto``: the widely used ``round(1.6 * card ** 0.56)`` rule (fast.ai), clipped to
        ``[min_dimension, max_dimension]`` -- a low-cardinality flag gets 2-3 dimensions and a
        50-level variable about 14, without any dependence on the other variables.
        """
        if self.embedding_dimension_strategy == "fixed":
            dim = int(self.embedding_dimension)
        else:
            dim = int(round(1.6 * float(cardinality) ** 0.56))
        return int(min(max(dim, self.embedding_min_dimension), self.embedding_max_dimension))

    def embedding_dims(self, layout: MixedLayout) -> dict[str, int]:
        return {s.column: self.embedding_dim_for(s.cardinality) for s in layout.cat_specs()}

    def family_weights(self) -> dict[str, float]:
        return {"numeric": self.weight_numeric, "boolean": self.weight_boolean,
                "categorical": self.weight_categorical}

    # -- identity -------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MixedVAEConfig":
        unknown = sorted(set(d) - set(cls.__dataclass_fields__))
        if unknown:   # never a partial load: an unknown key means another version of the loss/embedding config
            raise ValueError(f"unknown MixedVAEConfig key(s) {unknown}")
        return cls(**d)

    def fingerprint(self, layout: MixedLayout) -> str:
        """Everything that changes what a trained mixed VAE *is*: layout (variable names, order,
        vocabularies, tokens), embedding sizes, loss types and weights, token policy, code version."""
        payload = {"arch": MIXED_ARCHITECTURE, "version": MIXED_ARCHITECTURE_VERSION,
                   "layout": layout.fingerprint(), "config": self.to_dict(),
                   "embedding_dims": self.embedding_dims(layout), "nll_cap": NLL_CAP,
                   "normalisation": [CONTRIBUTION_FLOOR_FRACTION, CONTRIBUTION_ABS_FLOOR]}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


class ContributionReference:
    """Per-variable centre and scale of the contributions on the FIT (train) rows.

    ``normalize`` turns a raw contribution into "how far above its variable's usual level, in units of that
    variable's robust scale" -- the quantity every ranking of variables must use, because a raw NLL
    (log 40 for a uniform 40-level variable) is not comparable with a Huber term.
    """

    def __init__(self, median: np.ndarray, scale: np.ndarray):
        self.median = np.asarray(median, dtype=float)
        self.scale = np.asarray(scale, dtype=float)

    @classmethod
    def from_contributions(cls, C: np.ndarray, floor_fraction: float = CONTRIBUTION_FLOOR_FRACTION,
                           abs_floor: float = CONTRIBUTION_ABS_FLOOR) -> "ContributionReference":
        C = np.abs(np.asarray(C, dtype=float))
        median = np.nanmedian(C, axis=0)
        excess = np.maximum(C - median, 0.0)
        med_e = np.nanmedian(excess, axis=0)
        mad = np.nanmedian(np.abs(excess - med_e), axis=0) * 1.4826
        mean_e = np.nanmean(excess, axis=0)
        fallback = np.maximum(med_e, mean_e)
        scale = np.where(mad > 1e-12, mad, np.maximum(fallback, 1e-12))
        scale = np.maximum(np.maximum(scale, floor_fraction * mean_e), abs_floor)
        return cls(median, scale)

    def normalize(self, C: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(np.asarray(C, dtype=float)) - self.median, 0.0) / self.scale

    def to_dict(self) -> dict:
        return {"median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ContributionReference":
        return cls(d["median"], d["scale"])


def _mlp(dims: list[int], act: type, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MixedVAEModel(nn.Module):
    """Encoder with per-variable embeddings, decoder with numeric / binary / categorical heads.

    ``forward(x)`` takes the *mixed matrix* (see :mod:`src.preprocessing.mixed_view`) and returns
    ``(out, mu, logvar)`` where ``out`` is a dict ``{"num", "bool", "cat": [logits per variable]}``.
    """

    def __init__(self, layout: MixedLayout, config: MixedVAEConfig, latent_dim: int, hidden_dims: list[int],
                 dropout: float, activation: type):
        super().__init__()
        self.layout, self.config = layout, config
        self.latent_dim = int(latent_dim)
        self.hidden_dims = [int(h) for h in hidden_dims]
        for role in ("num", "bool", "flag", "cat"):
            self.register_buffer(f"idx_{role}", torch.as_tensor(layout.positions(role), dtype=torch.long))
        specs = layout.cat_specs()
        self.cardinalities = [s.cardinality for s in specs]
        emb_dims = config.embedding_dims(layout)
        self.embedding_dims = [emb_dims[s.column] for s in specs]
        self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(self.cardinalities, self.embedding_dims)])
        self.n_num, self.n_bool = len(layout.names("num")), len(layout.names("bool"))
        enc_in = self.n_num + self.n_bool + len(layout.names("flag")) + sum(self.embedding_dims)
        if enc_in < 1:
            raise ValueError("the mixed layout has no input columns.")
        self.encoder = _mlp([enc_in, *self.hidden_dims], activation, dropout)
        last = self.hidden_dims[-1]
        self.fc_mu, self.fc_logvar = nn.Linear(last, self.latent_dim), nn.Linear(last, self.latent_dim)
        self.decoder = _mlp([self.latent_dim, *reversed(self.hidden_dims)], activation, dropout)
        self.head_num = nn.Linear(last, self.n_num) if self.n_num else None
        self.head_bool = nn.Linear(last, self.n_bool) if self.n_bool else None
        self.heads_cat = nn.ModuleList([nn.Linear(last, c) for c in self.cardinalities])
        w = ([config.weight_numeric] * self.n_num + [config.weight_boolean] * self.n_bool
             + [config.weight_categorical] * len(specs))
        self.register_buffer("weights", torch.as_tensor(w, dtype=torch.float32))

    # -- helpers ---------------------------------------------------------------- #
    def _cat_indices(self, x: torch.Tensor) -> torch.Tensor:
        idx = x[:, self.idx_cat].round().long()
        hi = torch.as_tensor(self.cardinalities, device=x.device, dtype=torch.long) - 1
        return torch.minimum(torch.clamp(idx, min=0), hi.unsqueeze(0))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [x[:, self.idx_num], x[:, self.idx_bool], x[:, self.idx_flag]]
        cat = self._cat_indices(x)
        parts += [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        h = self.encoder(torch.cat(parts, dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> dict:
        h = self.decoder(z)
        return {
            "num": self.head_num(h) if self.head_num is not None else z.new_zeros((z.size(0), 0)),
            "bool": self.head_bool(h) if self.head_bool is not None else z.new_zeros((z.size(0), 0)),
            "cat": [head(h) for head in self.heads_cat],
        }

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    # -- per-variable contributions ------------------------------------------------ #
    def contributions(self, x: torch.Tensor, out: dict) -> torch.Tensor:
        """``(n, n_variables)`` reconstruction term of every ORIGINAL variable, ordered like
        ``layout.variables`` (numeric, binary, categorical)."""
        cols = []
        if self.n_num:
            xn = x[:, self.idx_num]
            if self.config.numeric_loss == "huber":
                cols.append(F.huber_loss(out["num"], xn, reduction="none", delta=self.config.huber_delta))
            else:
                cols.append((out["num"] - xn) ** 2)
        if self.n_bool:
            cols.append(F.binary_cross_entropy_with_logits(
                out["bool"], x[:, self.idx_bool].clamp(0.0, 1.0), reduction="none"))
        if self.heads_cat:
            cat = self._cat_indices(x)
            nll = torch.stack([F.cross_entropy(lg, cat[:, j], reduction="none")
                               for j, lg in enumerate(out["cat"])], dim=1)
            cols.append(nll.clamp(max=NLL_CAP))
        return torch.cat(cols, dim=1)

    def loss_parts(self, x: torch.Tensor, out: dict, mu: torch.Tensor, logvar: torch.Tensor,
                   beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """``(total, recon, kl, parts)``: batch means; ``parts`` holds the numeric / boolean /
        categorical (and per-categorical-variable) reconstruction means, weights included."""
        c = self.contributions(x, out) * self.weights
        recon_row = c.sum(dim=1)
        kl_row = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        n_n, n_b = self.n_num, self.n_bool
        parts = {
            "numeric": c[:, :n_n].sum(1).mean().detach() if n_n else c.new_zeros(()),
            "boolean": c[:, n_n:n_n + n_b].sum(1).mean().detach() if n_b else c.new_zeros(()),
            "categorical": c[:, n_n + n_b:].sum(1).mean().detach() if self.heads_cat else c.new_zeros(()),
            "by_categorical_variable": c[:, n_n + n_b:].mean(0).detach() if self.heads_cat else c.new_zeros((0,)),
        }
        recon, kl = recon_row.mean(), kl_row.mean()
        return recon + float(beta) * kl, recon, kl, parts

    def row_scores(self, x: torch.Tensor, out: dict) -> torch.Tensor:
        """Weighted mean of the per-variable contributions (higher = more anomalous)."""
        return (self.contributions(x, out) * self.weights).sum(1) / self.weights.sum()

    def architecture_config(self) -> dict:
        return {"latent_dim": self.latent_dim, "hidden_dims": self.hidden_dims,
                "embedding_dims": self.embedding_dims, "cardinalities": self.cardinalities}
