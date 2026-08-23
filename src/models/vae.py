"""Variational Autoencoder (VAE) anomaly detector with Optuna tuning and crash recovery.

This module is the sibling of :mod:`src.models.iforest`: it wraps a small
PyTorch VAE in a project-consistent, sklearn-ish detector and provides an
Optuna tuning routine that is resilient to crashes. It matches the iForest
module's API shape, score-sign convention, Optuna/SQLite/YAML resume pattern,
save/load style and plotting-to-``reports/figures`` behaviour.

Design boundary
---------------
The detector is deliberately decoupled from the data / out-of-time (OOT)
logic. It consumes an already-preprocessed feature matrix ``X`` (a dense
:class:`numpy.ndarray` **or** a :mod:`scipy.sparse` matrix, exactly as produced
by :func:`src.preprocessing.pipeline.fit_transform_panel`) and returns a
per-row anomaly score. Because a VAE needs dense tensors, sparse input is
densified to ``float32`` internally. The OOT split and the join back to the
separate ground-truth file are the evaluation module's responsibility, not
this module's. Here we only ``fit`` on a given ``X_train`` and ``score`` any
``X``.

Algorithm note (Kingma & Welling, 2013)
---------------------------------------
An autoencoder compresses each input to a latent code and reconstructs it; a
model trained on (mostly) normal data reconstructs normal inputs well and
anomalous inputs poorly, so the **reconstruction error** is the anomaly
signal. A VAE is the generative extension: the encoder emits the parameters of
a Gaussian over the latent space -- a mean ``mu`` and a log-variance
``logvar`` -- a code ``z`` is sampled via the reparameterization trick
``z = mu + exp(0.5*logvar) * eps`` (``eps ~ N(0, I)``), and the decoder maps
``z`` back to input space. Training maximizes the ELBO (Evidence Lower Bound),
equivalently minimizing ``reconstruction + beta * KL(q(z|x) || N(0, I))``.
``beta = 1`` is the vanilla VAE; ``beta != 1`` is the beta-VAE, trading
reconstruction fidelity against latent disentanglement/regularization. (Concept
aligned with ``docs/geeksforgeeks_notes.md`` section 3; no verbatim
copy.)

Score-sign convention
----------------------
Throughout this project the anomaly score follows **higher = more anomalous**,
identical to the iForest module. For the VAE the score is the per-row
reconstruction error (see :meth:`VAEDetector.score_samples` for the exact
formula), which naturally increases with anomalousness, so no sign flip is
needed.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils import paths
from src.utils.atomic_io import atomic_replace
from src.utils.logging_config import log_phase, setup_logging

__all__ = [
    "VAEModel",
    "vae_loss",
    "collapse_verdict",
    "VAEDetector",
    "tune_vae",
    "plot_reconstruction_error",
    "plot_latent_space",
]

# Default on-disk locations. All artifact paths live in `src.utils.paths`, the
# single place that knows the `artifacts/` layout.
_DEFAULT_STORAGE_DB = paths.VAE_STUDY_DB
_DEFAULT_BEST_PARAMS = paths.VAE_BEST_PARAMS
_DEFAULT_MODEL_OUT = paths.VAE_MODEL
_DEFAULT_DETECTOR_OUT = paths.VAE_DETECTOR
_DEFAULT_CHECKPOINT_DIR = paths.VAE_CHECKPOINT_DIR
_DEFAULT_TUNING_CKPT_DIR = paths.VAE_TUNING_CHECKPOINT_DIR
_DEFAULT_FIG_DIR = paths.FIGURES_DIR

# KL annealing: epochs over which the KL weight ramps linearly from 0 to `beta`.
#
# TEORÍA (posterior collapse): the ELBO has a trivial optimum where the encoder
# ignores the input and returns the prior. Then KL = 0 and the decoder emits the
# dataset mean -- the loss looks respectable while the latent code carries no
# information, so every row reconstructs about equally badly and the anomaly
# score becomes noise. The KL term is what pulls the model there, and it does so
# fastest at the start when the decoder is still useless. Ramping the weight in
# gives the decoder time to learn a real reconstruction first, so by the time
# the full KL pressure arrives the latent code is already worth keeping.
_DEFAULT_KL_ANNEAL_EPOCHS = 10

# Active-unit threshold delta: a latent dimension counts as "active" when the
# variance of its encoder mean across the data exceeds this. 0.01 is the value
# used by Burda, Grosse & Salakhutdinov (IWAE, ICLR 2016) and since adopted as
# the reference. See `VAEDetector.latent_diagnostics`.
_ACTIVE_UNIT_DELTA = 0.01
# Below this fraction of active dimensions the latent space is judged
# collapsed. Not from the literature -- the papers report active-unit counts
# without prescribing a pass/fail line, because the acceptable fraction is
# task-dependent. Chosen here because the anomaly score is the reconstruction
# error: once fewer than a third of the dimensions carry signal, the decoder
# is reconstructing largely from the prior and the score stops discriminating.
# Documented as a project convention, not a published constant.
_COLLAPSE_ACTIVE_FRACTION = 1.0 / 3.0
# A per-dimension KL this small means the posterior for that coordinate is
# indistinguishable from the prior. Used only as corroborating evidence.
_COLLAPSE_KL_EPS = 0.01

# Early stopping: epochs without validation improvement before halting.
_DEFAULT_PATIENCE = 10

# Label-free objective names accepted by `tune_vae(objective_metric=...)`.
_UNSUPERVISED_METRICS: tuple[str, ...] = ("recon_p50",)

ArrayLike = Union[np.ndarray, "sp.spmatrix"]

# Named activations selectable as a plain string (Optuna-tunable, YAML-safe).
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

_OPTIMIZERS = {"adam", "adamw", "rmsprop"}


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _densify(X: ArrayLike) -> np.ndarray:
    """Return a dense contiguous ``float32`` 2D array (densifying sparse X)."""
    if sp.issparse(X):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resolve_hidden_dims(
    hidden_dims: Optional[Sequence[int]],
    n_layers: int,
    hidden_dim: int,
) -> list[int]:
    """Resolve encoder hidden layer widths.

    Explicit ``hidden_dims`` wins; otherwise build ``n_layers`` layers all of
    width ``hidden_dim``. The decoder mirrors these in reverse.
    """
    if hidden_dims is not None:
        dims = [int(h) for h in hidden_dims]
        if not dims:
            raise ValueError("hidden_dims must be non-empty when provided.")
        return dims
    n = max(1, int(n_layers))
    return [int(hidden_dim)] * n


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class VAEModel(nn.Module):
    """MLP Variational Autoencoder: encoder -> (mu, logvar) -> z -> decoder.

    Args:
        input_dim: Number of input features.
        latent_dim: Dimensionality of the latent Gaussian.
        hidden_dims: Explicit encoder hidden widths (decoder mirrors them). If
            ``None``, ``n_layers`` layers of width ``hidden_dim`` are used.
        n_layers: Number of hidden layers when ``hidden_dims`` is not given.
        hidden_dim: Hidden width when ``hidden_dims`` is not given.
        dropout: Dropout probability applied after each hidden activation.
        activation: Named activation (see ``_ACTIVATIONS``), default ``relu``.

    ``forward`` returns ``(x_recon, mu, logvar)``. The ELBO decomposes as
    ``ELBO = E[log p(x|z)] - beta * KL(q(z|x) || N(0, I))``; minimizing the
    negative ELBO is minimizing ``reconstruction + beta * KL`` (see
    :func:`vae_loss`).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Optional[Sequence[int]] = None,
        n_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        act_name = str(activation).lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from {sorted(_ACTIVATIONS)}."
            )
        dims = _resolve_hidden_dims(hidden_dims, n_layers, hidden_dim)

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = dims
        self.dropout = float(dropout)
        self.activation = act_name

        act = _ACTIVATIONS[act_name]

        # Encoder trunk: input_dim -> dims[0] -> ... -> dims[-1]
        enc_layers: list[nn.Module] = []
        prev = self.input_dim
        for h in dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(act())
            if self.dropout > 0:
                enc_layers.append(nn.Dropout(self.dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, self.latent_dim)
        self.fc_logvar = nn.Linear(prev, self.latent_dim)

        # Decoder trunk: latent_dim -> reversed(dims) -> input_dim
        dec_layers: list[nn.Module] = []
        prev = self.latent_dim
        for h in reversed(dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(act())
            if self.dropout > 0:
                dec_layers.append(nn.Dropout(self.dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard reparameterization trick: ``z = mu + std * eps``.

        Randomness is pushed into ``eps ~ N(0, I)`` so gradients flow through
        ``mu`` and ``logvar``. In eval mode we return ``mu`` deterministically
        (no sampling noise), which makes reconstruction-error scores stable.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def config(self) -> dict:
        """Return the architecture hyperparameters needed to rebuild this model."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "activation": self.activation,
        }


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = reconstruction (MSE) + ``beta`` * Gaussian KL.

    Reconstruction uses **mean squared error** because after preprocessing the
    features are standardized/continuous (not in [0, 1]), so an MSE / Gaussian
    likelihood is the appropriate reconstruction term (Bernoulli/BCE would be
    wrong here). The KL term uses the closed-form Gaussian divergence
    ``KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))``.

    **Both terms are per-sample sums**, then averaged over the batch: the MSE
    is summed over features and the KL over latent dimensions. That is the
    ELBO's own scaling, and it is what makes ``beta`` mean what the beta-VAE
    literature says it means (Higgins et al., 2017).

    .. warning::

       This was changed on 2026-08-22 and it **alters what every ``beta``
       value does**. Previously the MSE used ``F.mse_loss(..., 'mean')``,
       which divides by ``batch x n_features`` while the KL was only averaged
       over the batch. The KL therefore carried ``n_features`` times its
       intended weight, making the *effective* beta ``beta * n_features``:
       with 22 features the default ``beta=1.0`` trained as if it were 22, and
       the model collapsed (0 of 8 active latent units, measured). Any
       ``best_params_vae.yaml`` produced before this change encodes a ``beta``
       on the old scale and must be re-tuned, not reused.

    ``reduction='sum'`` returns batch totals instead of per-sample means; the
    recon/KL balance is identical either way, only the overall magnitude
    differs.

    Returns:
        ``(total, recon_term, kl_term)`` as scalar tensors, with
        ``total = recon_term + beta * kl_term``.
    """
    # Per-sample sums for both terms, so their ratio does not depend on the
    # feature count. `F.mse_loss(..., 'mean')` would divide the recon term by
    # n_features and silently inflate beta -- see the warning above.
    recon_per_sample = torch.sum((x_recon - x) ** 2, dim=1)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if reduction == "mean":
        recon = recon_per_sample.mean()
        kl = kl_per_sample.mean()
    elif reduction == "sum":
        recon = recon_per_sample.sum()
        kl = kl_per_sample.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'.")
    total = recon + float(beta) * kl
    return total, recon, kl


def collapse_verdict(diagnostics: dict) -> dict:
    """Judge whether ``diagnostics`` describes a collapsed latent space.

    Deliberately a function rather than a flag on the diagnostics dict: the
    measurement (what the encoder does) and the judgement (whether that is
    acceptable) are separate concerns, and only the judgement carries a
    project-specific threshold.

    **A low KL alone is not collapse.** Requiring corroboration is the whole
    point -- a single quiet dimension is normal in any trained VAE, and
    flagging on that would make the check noise. The verdict is ``collapsed``
    only when the active fraction is below
    :data:`_COLLAPSE_ACTIVE_FRACTION` *and* the mean KL is small enough that
    the posterior has genuinely fallen back to the prior. The two conditions
    fail together only in the real failure mode.

    ``degenerate`` covers the harder case: exactly one active dimension, or
    none. That is collapse regardless of KL, because a latent space with no
    surviving axes cannot carry the structure the reconstruction error is
    supposed to expose.

    Returns:
        ``{"collapsed": bool, "severity": "ok"|"warning"|"critical",
        "reason": str}``.
    """
    d = int(diagnostics.get("latent_dim", 0))
    active = int(diagnostics.get("active_units", 0))
    frac = float(diagnostics.get("active_fraction", float("nan")))
    mean_kl = float(diagnostics.get("mean_kl", float("nan")))

    if d == 0:
        return {"collapsed": False, "severity": "ok",
                "reason": "Sin dimensiones latentes que evaluar."}
    if active <= 1:
        return {
            "collapsed": True, "severity": "critical",
            "reason": (
                f"Solo {active} de {d} dimensiones latentes activas "
                f"(A_j > {diagnostics.get('delta')}). El decodificador no "
                "puede estar usando el código latente, así que el error de "
                "reconstrucción -- que ES el puntaje de anomalía -- ya no "
                "discrimina."
            ),
        }
    if frac < _COLLAPSE_ACTIVE_FRACTION and mean_kl < _COLLAPSE_KL_EPS * d:
        return {
            "collapsed": True, "severity": "critical",
            "reason": (
                f"{active} de {d} dimensiones activas ({frac:.0%}, por debajo "
                f"del {_COLLAPSE_ACTIVE_FRACTION:.0%}) y KL media {mean_kl:.4f} "
                "cercana a cero: el posterior colapsó al prior en la mayoría "
                "de las coordenadas."
            ),
        }
    if frac < _COLLAPSE_ACTIVE_FRACTION:
        return {
            "collapsed": False, "severity": "warning",
            "reason": (
                f"Solo {active} de {d} dimensiones activas ({frac:.0%}), pero "
                f"la KL media ({mean_kl:.4f}) no indica colapso al prior. "
                "Puede ser sobrecapacidad latente: considerar un latent_dim "
                "menor."
            ),
        }
    return {
        "collapsed": False, "severity": "ok",
        "reason": f"{active} de {d} dimensiones latentes activas ({frac:.0%}).",
    }


# --------------------------------------------------------------------------- #
# Detector                                                                     #
# --------------------------------------------------------------------------- #
class VAEDetector:
    """A project-consistent, sklearn-ish VAE anomaly detector.

    Mirrors :class:`~src.models.iforest.IsolationForestDetector`: standardized
    anomaly-score sign (higher = more anomalous), ``log_phase``-wrapped fit,
    transparent sparse/dense input (densified internally), and a torch-based
    ``save``/``load`` round-trip that scores identically.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
        beta: float = 1.0,
        lr: float = 1e-3,
        optimizer: str = "adam",
        batch_size: int = 256,
        epochs: int = 30,
        weight_decay: float = 0.0,
        activation: str = "relu",
        hidden_dims: Optional[Sequence[int]] = None,
        score_kl_weight: float = 0.0,
        kl_anneal_epochs: int = _DEFAULT_KL_ANNEAL_EPOCHS,
        early_stopping_patience: Optional[int] = _DEFAULT_PATIENCE,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.beta = float(beta)
        self.lr = float(lr)
        self.optimizer = str(optimizer).lower()
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.activation = str(activation).lower()
        self.hidden_dims = None if hidden_dims is None else [int(h) for h in hidden_dims]
        # score_kl_weight adds `score_kl_weight * per-row KL` to the anomaly
        # score. Default 0.0 => the score is the pure reconstruction error.
        self.score_kl_weight = float(score_kl_weight)
        self.kl_anneal_epochs = max(0, int(kl_anneal_epochs))
        self.early_stopping_patience = (
            None if early_stopping_patience is None else max(1, int(early_stopping_patience))
        )
        self.random_state = int(random_state)

        if self.optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}; choose from {sorted(_OPTIMIZERS)}."
            )

        self.device = self._resolve_device(device)
        self.model_: Optional[VAEModel] = None
        self.input_dim_: Optional[int] = None
        self.history_: list[dict] = []
        self.best_val_loss_: float = float("inf")
        # Negative ELBO (beta=1) at the best epoch. Distinct from
        # `best_val_loss_`, which is weighted by this fit's own `beta` and is
        # therefore only meaningful within a single fit.
        self.best_val_elbo_: float = float("inf")

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return str(device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _seed_everything(self) -> None:
        """Seed numpy/torch for CPU reproducibility (GPU not guaranteed)."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _arch_config(self) -> dict:
        """Architecture + training hyperparameters (for checkpoints/rebuild)."""
        return {
            "input_dim": self.input_dim_,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "beta": self.beta,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "score_kl_weight": self.score_kl_weight,
            "random_state": self.random_state,
        }

    def _build_model(self, input_dim: int) -> VAEModel:
        return VAEModel(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

    def _build_optimizer(self, model: VAEModel) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        if self.optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return torch.optim.RMSprop(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    # -- fit ---------------------------------------------------------------- #
    def fit(
        self,
        X: ArrayLike,
        checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR,
        resume: bool = True,
        val_fraction: float = 0.1,
        valid_mask: Optional[np.ndarray] = None,
    ) -> "VAEDetector":
        """Train the VAE with per-epoch checkpointing and crash recovery.

        A ``TensorDataset``/``DataLoader`` (shuffled train split) is built from
        a densified ``float32`` copy of ``X``. Training runs with a tqdm bar
        over epochs, logging per-epoch train/val loss and epoch duration, all
        wrapped in ``log_phase``.

        Resume semantics
        ----------------
        After **every** epoch, ``<checkpoint_dir>/checkpoint.pth`` is written
        atomically (``.tmp`` + ``os.replace``) containing: the model
        ``state_dict``, the optimizer ``state_dict``, the just-completed epoch
        index, the best val loss so far, the full architecture/hyperparameter
        config, the per-epoch training ``history``, and the numpy + torch RNG
        state. When ``resume=True`` and a *compatible* checkpoint exists (same
        architecture config as this detector -- ``input_dim``, ``latent_dim``,
        ``hidden_dim``/``hidden_dims``, ``n_layers``, ``dropout``,
        ``activation``), it is loaded and training continues from ``epoch + 1``
        with the optimizer, history, and RNG state restored, so a resumed run
        reproduces the trajectory it would have taken without the interruption.
        If the checkpoint is incompatible, a warning is logged and training
        starts fresh. ``<checkpoint_dir>/best_model.pth`` tracks the
        lowest-monitored-loss weights (validation loss when a val split exists,
        otherwise train loss) and is restored into the detector at the end of
        ``fit``. If the checkpoint is already at/after ``epochs``, training is
        skipped and only the best weights are restored.

        The internal ``torch.load`` calls pass ``weights_only=False`` on
        purpose: PyTorch >= 2.6 defaults ``weights_only=True``, which refuses to
        unpickle the config dict and numpy RNG state stored in these trusted
        project checkpoints.

        Args:
            X: Preprocessed matrix (dense ndarray or scipy sparse). Densified.
            checkpoint_dir: Directory for ``checkpoint.pth``/``best_model.pth``.
            resume: If True, resume from a compatible checkpoint when present.
            val_fraction: Fraction of rows held out (shuffled) for validation,
                used only when ``valid_mask`` is ``None``.
            valid_mask: Boolean row mask selecting the validation rows -- pass
                the chronological validation months here.

                TEORÍA: the shuffled fallback puts months 1-10 of a customer in
                train and month 7 of the same customer in validation, so early
                stopping is judged on rows interleaved with the training data
                and the model is rewarded for interpolating rather than
                forecasting. A time-based mask makes validation what deployment
                will be: later periods, never seen.
        """
        log = setup_logging()
        self._seed_everything()
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        best_path = os.path.join(checkpoint_dir, "best_model.pth")

        Xd = _densify(X)
        n_samples, n_features = Xd.shape
        self.input_dim_ = int(n_features)

        # Validation split: a caller-supplied temporal mask wins; otherwise a
        # deterministic shuffled split (see the `valid_mask` docstring for why
        # the shuffled one is a compromise, not the target).
        if valid_mask is not None:
            vm = np.asarray(valid_mask, dtype=bool).ravel()
            if vm.shape[0] != n_samples:
                raise ValueError(
                    f"valid_mask has {vm.shape[0]} entries but X has {n_samples} rows"
                )
            if not vm.any() or vm.all():
                raise ValueError("valid_mask must select some -- but not all -- rows")
            val_idx = np.flatnonzero(vm)
            train_idx = np.flatnonzero(~vm)
            split_kind = "temporal (valid_mask)"
        else:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(n_samples)
            n_val = int(round(val_fraction * n_samples)) if val_fraction and n_samples > 1 else 0
            n_val = min(max(n_val, 0), max(n_samples - 1, 0))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            split_kind = "shuffled"
        n_val = int(val_idx.size)

        X_t = torch.from_numpy(Xd)
        train_ds = TensorDataset(X_t[train_idx])
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        val_tensor = X_t[val_idx].to(self.device) if n_val > 0 else None

        self.model_ = self._build_model(self.input_dim_)
        optimizer = self._build_optimizer(self.model_)

        start_epoch = 0
        self.best_val_loss_ = float("inf")
        self.best_val_elbo_ = float("inf")
        self.history_ = []

        # -- resume from a compatible checkpoint ---------------------------- #
        if resume and os.path.isfile(ckpt_path):
            try:
                # weights_only=False: our own trusted checkpoint carries the
                # config dict and numpy/torch RNG state, which the PyTorch>=2.6
                # weights_only=True default refuses to unpickle.
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except Exception as exc:  # pragma: no cover - corrupt file guard
                log.warning("Could not read checkpoint %s (%s); starting fresh.", ckpt_path, exc)
                ckpt = None
            if ckpt is not None:
                if self._checkpoint_compatible(ckpt):
                    self.model_.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    start_epoch = int(ckpt["epoch"]) + 1
                    self.best_val_loss_ = float(ckpt.get("best_val_loss", float("inf")))
                    self.best_val_elbo_ = float(ckpt.get("best_val_elbo", float("inf")))
                    self.history_ = list(ckpt.get("history", []))
                    self._restore_rng_state(ckpt.get("rng_state"))
                    log.info(
                        "Resuming VAE training from %s at epoch %d (best_val_loss=%.6f).",
                        ckpt_path, start_epoch, self.best_val_loss_,
                    )
                else:
                    log.warning(
                        "Checkpoint %s is incompatible with the current config; "
                        "starting fresh.", ckpt_path,
                    )

        if start_epoch >= self.epochs:
            log.info(
                "Checkpoint already at/after target epochs (%d >= %d); "
                "restoring best weights and skipping training.",
                start_epoch, self.epochs,
            )
            self._restore_best(best_path, log)
            return self

        from tqdm.auto import tqdm

        with log_phase("vae.fit", log):
            log.info(
                "Training VAE on %d samples x %d features "
                "(latent_dim=%d, hidden=%s, beta=%.3f, optimizer=%s, lr=%.2e, "
                "batch_size=%d, epochs=%d, device=%s), val=%d rows (%s), "
                "kl_anneal=%d epochs, patience=%s.",
                n_samples, n_features, self.latent_dim,
                self.hidden_dims or [self.hidden_dim] * self.n_layers,
                self.beta, self.optimizer, self.lr, self.batch_size,
                self.epochs, self.device, n_val, split_kind,
                self.kl_anneal_epochs, self.early_stopping_patience,
            )
            epochs_without_improvement = 0
            progress = tqdm(
                range(start_epoch, self.epochs),
                total=self.epochs,
                initial=start_epoch,
                desc="vae[epochs]",
                unit="epoch",
            )
            for epoch in progress:
                t0 = time.perf_counter()
                self.model_.train()
                # Linear KL ramp 0 -> beta over the first `kl_anneal_epochs`.
                beta_epoch = self._annealed_beta(epoch)
                run_total = run_recon = run_kl = 0.0
                n_seen = 0
                for (xb,) in train_loader:
                    xb = xb.to(self.device)
                    optimizer.zero_grad()
                    x_recon, mu, logvar = self.model_(xb)
                    total, recon, kl = vae_loss(
                        xb, x_recon, mu, logvar, beta=beta_epoch, reduction="mean"
                    )
                    total.backward()
                    optimizer.step()
                    bs = xb.size(0)
                    run_total += float(total.item()) * bs
                    run_recon += float(recon.item()) * bs
                    run_kl += float(kl.item()) * bs
                    n_seen += bs
                train_loss = run_total / max(n_seen, 1)
                train_recon = run_recon / max(n_seen, 1)
                train_kl = run_kl / max(n_seen, 1)

                val_parts = self._evaluate(val_tensor)
                if val_parts is None:
                    val_loss = val_recon = val_kl = None
                    val_elbo = None
                else:
                    val_loss, val_recon, val_kl = val_parts
                    # Negative ELBO at beta=1: the beta-independent view of
                    # this model's fit, used for cross-trial comparison.
                    val_elbo = val_recon + val_kl
                # Model-selection metric: val loss if available, else train loss.
                monitor = val_loss if val_loss is not None else train_loss
                duration = time.perf_counter() - t0

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_recon": train_recon,
                    "train_kl": train_kl,
                    "val_loss": val_loss,
                    "val_recon": val_recon,
                    "val_kl": val_kl,
                    "val_elbo": val_elbo,
                    "beta": beta_epoch,
                    "duration_s": duration,
                }
                self.history_.append(record)
                log.info(
                    "Epoch %d/%d | beta=%.3f | train_loss=%.6f (recon=%.6f, kl=%.6f) | "
                    "val_loss=%s | %.2fs",
                    epoch + 1, self.epochs, beta_epoch, train_loss, train_recon,
                    train_kl, "n/a" if val_loss is None else f"{val_loss:.6f}", duration,
                )
                progress.set_postfix(
                    train=f"{train_loss:.4f}",
                    val="n/a" if val_loss is None else f"{val_loss:.4f}",
                )

                is_best = monitor < self.best_val_loss_
                if is_best:
                    self.best_val_loss_ = float(monitor)
                    # Captured at the SAME epoch the weights are saved from, so
                    # the cross-trial objective describes the model that is
                    # actually restored -- not whatever the last epoch happened
                    # to reach.
                    self.best_val_elbo_ = (
                        float(val_elbo) if val_elbo is not None else float(monitor)
                    )
                    self._save_state(best_path, epoch, optimizer)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                # Per-epoch crash-recovery checkpoint (atomic).
                self._save_state(ckpt_path, epoch, optimizer)

                # Early stopping. Held off until the KL ramp finishes: while beta
                # is still climbing the loss is being measured against a moving
                # objective, so a run of "no improvement" says nothing about
                # convergence.
                patience = self.early_stopping_patience
                if (
                    patience is not None
                    and epoch + 1 >= self.kl_anneal_epochs
                    and epochs_without_improvement >= patience
                ):
                    log.info(
                        "Early stopping at epoch %d/%d: no improvement in %d epochs "
                        "(best monitored loss=%.6f).",
                        epoch + 1, self.epochs, epochs_without_improvement,
                        self.best_val_loss_,
                    )
                    break
            progress.close()

        # Restore best-val weights before returning.
        self._restore_best(best_path, log)
        return self

    def _annealed_beta(self, epoch: int) -> float:
        """KL weight for ``epoch``: linear 0 -> ``beta`` over the ramp, then flat.

        See :data:`_DEFAULT_KL_ANNEAL_EPOCHS` for why the ramp exists. Epoch 0
        gets a non-zero weight (``beta/ramp``) rather than exactly 0 so the KL
        term is never fully switched off, which would let the encoder drift to
        an arbitrary scale that the sudden appearance of the penalty then has to
        undo.
        """
        ramp = self.kl_anneal_epochs
        if ramp <= 0:
            return self.beta
        return float(self.beta * min(1.0, (epoch + 1) / ramp))

    # -- checkpoint helpers ------------------------------------------------- #
    def _checkpoint_compatible(self, ckpt: dict) -> bool:
        cfg = ckpt.get("config", {})
        keys = ["input_dim", "latent_dim", "hidden_dim", "n_layers",
                "hidden_dims", "dropout", "activation"]
        cur = self._arch_config()
        for k in keys:
            if cfg.get(k) != cur.get(k):
                return False
        return True

    def _save_state(self, path: str, epoch: int, optimizer: torch.optim.Optimizer) -> None:
        payload = {
            "epoch": int(epoch),
            "best_val_loss": float(self.best_val_loss_),
            "best_val_elbo": float(self.best_val_elbo_),
            "model_state_dict": self.model_.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": self._arch_config(),
            "history": self.history_,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        atomic_replace(tmp, path)  # atomic swap, Windows-lock-safe

    @staticmethod
    def _restore_rng_state(rng_state: Optional[dict]) -> None:
        if not rng_state:
            return
        try:
            t_state = rng_state.get("torch")
            if t_state is not None:
                if not isinstance(t_state, torch.Tensor):
                    t_state = torch.as_tensor(t_state, dtype=torch.uint8)
                torch.set_rng_state(t_state.to(torch.uint8).cpu())
            np_state = rng_state.get("numpy")
            if np_state is not None:
                np.random.set_state(np_state)
        except Exception:  # pragma: no cover - best-effort RNG restore
            pass

    def _restore_best(self, best_path: str, log) -> None:
        if os.path.isfile(best_path):
            try:
                best = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model_.load_state_dict(best["model_state_dict"])
                self.best_val_loss_ = float(best.get("best_val_loss", self.best_val_loss_))
                self.best_val_elbo_ = float(best.get("best_val_elbo", self.best_val_elbo_))
                log.info("Restored best VAE weights (val_loss=%.6f) from %s.",
                         self.best_val_loss_, best_path)
            except Exception as exc:  # pragma: no cover
                log.warning("Could not restore best weights from %s (%s).", best_path, exc)

    def _evaluate(
        self, val_tensor: Optional[torch.Tensor]
    ) -> Optional[tuple[float, float, float]]:
        """Validation loss, broken into ``(total_at_beta, recon, kl)``.

        The three are returned separately rather than pre-combined because
        they answer different questions and get used for different things:

        * ``total_at_beta`` -- this model's own training objective. Correct
          for early stopping and best-epoch selection *within* a fit, where
          ``beta`` is fixed.
        * ``recon + kl`` -- the negative ELBO at ``beta = 1``. This is the
          only one comparable **across** fits with different ``beta``, which
          is what Optuna needs. See ``best_val_elbo_``.
        """
        if val_tensor is None or val_tensor.numel() == 0:
            return None
        self.model_.eval()
        with torch.no_grad():
            tot = rec = kld = 0.0
            n = 0
            for start in range(0, val_tensor.size(0), self.batch_size):
                xb = val_tensor[start:start + self.batch_size]
                x_recon, mu, logvar = self.model_(xb)
                loss, recon_t, kl_t = vae_loss(
                    xb, x_recon, mu, logvar, beta=self.beta, reduction="mean"
                )
                bs = xb.size(0)
                tot += float(loss.item()) * bs
                rec += float(recon_t.item()) * bs
                kld += float(kl_t.item()) * bs
                n += bs
        n = max(n, 1)
        return tot / n, rec / n, kld / n

    def _check_fitted(self) -> VAEModel:
        if self.model_ is None:
            raise RuntimeError("VAEDetector is not fitted; call fit(X) first.")
        return self.model_

    # -- scoring ------------------------------------------------------------ #
    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Per-row anomaly score where **higher = more anomalous**.

        Exact formula (eval mode, no grad, deterministic -- encoder mean is
        used, no sampling noise)::

            mu, logvar = encoder(x)
            x_recon    = decoder(mu)
            recon_err  = mean_j (x_j - x_recon_j)^2          # MSE over features
            kl         = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
            score      = recon_err + score_kl_weight * kl

        With the default ``score_kl_weight=0.0`` the score is exactly the
        per-row mean-squared reconstruction error. Reconstruction error grows
        for inputs the VAE (trained on the normal bulk) cannot reproduce, so
        the score already increases with anomalousness -- no sign flip needed.
        """
        model = self._check_fitted()
        Xd = _densify(X)
        model.eval()
        scores = np.empty(Xd.shape[0], dtype=np.float64)
        with torch.no_grad():
            for start in range(0, Xd.shape[0], self.batch_size):
                chunk = Xd[start:start + self.batch_size]
                xb = torch.from_numpy(chunk).to(self.device)
                mu, logvar = model.encode(xb)
                x_recon = model.decode(mu)
                recon_err = torch.mean((xb - x_recon) ** 2, dim=1)
                if self.score_kl_weight != 0.0:
                    kl = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                    )
                    batch_score = recon_err + self.score_kl_weight * kl
                else:
                    batch_score = recon_err
                scores[start:start + xb.size(0)] = batch_score.cpu().numpy()
        return scores

    def reconstruction_error(self, X: ArrayLike) -> np.ndarray:
        """Alias for :meth:`score_samples` (per-row reconstruction error)."""
        return self.score_samples(X)

    def encode(self, X: ArrayLike) -> np.ndarray:
        """Return latent means ``mu`` per row (for interpretability/plots)."""
        model = self._check_fitted()
        Xd = _densify(X)
        model.eval()
        out = np.empty((Xd.shape[0], model.latent_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, Xd.shape[0], self.batch_size):
                chunk = Xd[start:start + self.batch_size]
                xb = torch.from_numpy(chunk).to(self.device)
                mu, _ = model.encode(xb)
                out[start:start + xb.size(0)] = mu.cpu().numpy()
        return out

    def latent_diagnostics(
        self, X: ArrayLike, delta: float = _ACTIVE_UNIT_DELTA
    ) -> dict:
        """Posterior-collapse diagnostics for the fitted latent space.

        A collapsed VAE is the failure mode this project is *most* exposed to
        and *least* able to notice: the anomaly score IS the reconstruction
        error, so a decoder that has learned to ignore ``z`` still emits
        finite, plausible-looking scores and still writes a populated
        ``best_params.yaml``. Nothing downstream raises. The tuner can even
        select it, because a high ``beta`` (the search space allows up to 2.0)
        buys a lower KL term at the cost of the very latent structure the
        score depends on.

        The **active-units** statistic is the standard test (Burda, Grosse &
        Salakhutdinov, *Importance Weighted Autoencoders*, ICLR 2016)::

            A_j = Var_x( E_q[z_j | x] )        # variance of the encoder mean
            dimension j is active  <=>  A_j > delta

        ``delta = 0.01`` is the threshold used in that paper and since adopted
        as the reference value.

        Per-dimension KL is reported alongside it because neither number is
        sufficient alone: a low KL on one dimension is normal, and a low
        variance can also come from a genuinely low-information feature. The
        collapse claim needs both to point the same way across many dimensions
        -- see :func:`collapse_verdict`, which is where that judgement is made
        rather than being left to the caller.

        Args:
            X: Rows to measure over. Use the same block the model was scored
                on; the statistic is a property of the encoder's response to
                data, so measuring it on 10 rows is meaningless.
            delta: Activity threshold. Defaults to the literature's 0.01.

        Returns:
            ``dict`` with ``active_units``, ``inactive_units``, ``latent_dim``,
            ``active_fraction``, ``delta``, ``mean_kl``, ``n_rows``, plus the
            per-dimension arrays ``activity`` (A_j) and ``kl_per_dim`` as
            lists, and ``mean_mu_variance`` / ``max_activity`` summaries.
        """
        model = self._check_fitted()
        Xd = _densify(X)
        model.eval()
        n_rows, d = Xd.shape[0], model.latent_dim

        mus = np.empty((n_rows, d), dtype=np.float64)
        kl_sum = np.zeros(d, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, n_rows, self.batch_size):
                chunk = Xd[start:start + self.batch_size]
                xb = torch.from_numpy(chunk).to(self.device)
                mu, logvar = model.encode(xb)
                mus[start:start + xb.size(0)] = mu.cpu().numpy()
                # Same closed-form Gaussian KL as `vae_loss`, but kept
                # per-dimension instead of summed, so a single collapsed
                # coordinate is visible rather than averaged away.
                kl_d = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_sum += kl_d.sum(dim=0).cpu().numpy()

        # A_j: variance ACROSS rows of the per-row encoder mean. Population
        # variance (ddof=0) to match the literature's definition.
        activity = mus.var(axis=0) if n_rows > 1 else np.zeros(d)
        kl_per_dim = kl_sum / max(n_rows, 1)
        active = int((activity > float(delta)).sum())

        return {
            "latent_dim": int(d),
            "active_units": active,
            "inactive_units": int(d - active),
            "active_fraction": float(active / d) if d else float("nan"),
            "delta": float(delta),
            "activity": [float(v) for v in activity],
            "kl_per_dim": [float(v) for v in kl_per_dim],
            "mean_kl": float(kl_per_dim.sum()),
            "mean_mu_variance": float(activity.mean()) if d else float("nan"),
            "max_activity": float(activity.max()) if d else float("nan"),
            "n_rows": int(n_rows),
        }

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str = _DEFAULT_DETECTOR_OUT) -> str:
        """Persist the full detector (config + weights) via ``torch.save``.

        The written dict contains the architecture/training config, the model
        ``state_dict``, ``input_dim`` and training history. :meth:`load`
        reconstructs a working detector that scores identically.
        """
        self._check_fitted()
        _ensure_parent_dir(path)
        payload = {
            "format": "vae_detector",
            "version": 1,
            "config": self._arch_config(),
            "input_dim": self.input_dim_,
            "model_state_dict": self.model_.state_dict(),
            "best_val_loss": float(self.best_val_loss_),
            "best_val_elbo": float(self.best_val_elbo_),
            "history": self.history_,
        }
        torch.save(payload, path)
        setup_logging().info("Saved VAEDetector to %s", path)
        return path

    @classmethod
    def load(cls, path: str = _DEFAULT_DETECTOR_OUT, device: Optional[str] = None) -> "VAEDetector":
        """Load a detector previously written by :meth:`save`.

        Loads with ``weights_only=False`` because the payload is a trusted
        project artifact carrying a config dict (not just tensors); PyTorch >=
        2.6's ``weights_only=True`` default would reject it.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "vae_detector":
            raise TypeError(f"{path} does not contain a VAEDetector payload.")
        cfg = payload["config"]
        det = cls(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            beta=cfg["beta"],
            lr=cfg["lr"],
            optimizer=cfg["optimizer"],
            batch_size=cfg["batch_size"],
            epochs=0,
            weight_decay=cfg.get("weight_decay", 0.0),
            activation=cfg.get("activation", "relu"),
            hidden_dims=cfg.get("hidden_dims"),
            score_kl_weight=cfg.get("score_kl_weight", 0.0),
            device=device,
            random_state=cfg.get("random_state", 42),
        )
        det.input_dim_ = int(payload["input_dim"])
        det.model_ = det._build_model(det.input_dim_)
        det.model_.load_state_dict(payload["model_state_dict"])
        det.model_.eval()
        det.best_val_loss_ = float(payload.get("best_val_loss", float("inf")))
        det.best_val_elbo_ = float(payload.get("best_val_elbo", float("inf")))
        det.history_ = list(payload.get("history", []))
        return det


# --------------------------------------------------------------------------- #
# Optuna tuning helpers                                                        #
# --------------------------------------------------------------------------- #
def _default_storage_uri(db_path: str = _DEFAULT_STORAGE_DB) -> str:
    """Build a SQLite RDBStorage URI, creating the parent directory.

    SQLAlchemy needs forward slashes in the SQLite URL, so backslashes from
    Windows paths are normalised.
    """
    _ensure_parent_dir(db_path)
    return "sqlite:///" + db_path.replace("\\", "/")


def _detector_kwargs_from_params(params: dict) -> dict:
    """Translate an Optuna trial's params into ``VAEDetector`` kwargs."""
    return {
        "latent_dim": int(params["latent_dim"]),
        "hidden_dim": int(params["hidden_dim"]),
        "n_layers": int(params["n_layers"]),
        "dropout": float(params["dropout"]),
        "beta": float(params["beta"]),
        "lr": float(params["lr"]),
        "optimizer": str(params["optimizer"]),
        "batch_size": int(params["batch_size"]),
    }


def tune_vae(
    X: ArrayLike,
    n_trials: int = 30,
    y: Optional[np.ndarray] = None,
    storage: Optional[str] = None,
    study_name: str = "vae",
    direction: Optional[str] = None,
    objective_metric: Optional[Union[str, Callable[["VAEDetector", ArrayLike], float]]] = None,
    best_params_path: str = _DEFAULT_BEST_PARAMS,
    model_out: str = _DEFAULT_MODEL_OUT,
    checkpoint_dir: str = _DEFAULT_TUNING_CKPT_DIR,
    random_state: int = 42,
    max_epochs: int = 20,
    timeout: Optional[float] = None,
    valid_mask: Optional[np.ndarray] = None,
    early_stopping_patience: Optional[int] = 10,
    early_stopping_min_delta: float = 0.005,
    early_stopping_min_trials: int = 10,
):
    """Tune :class:`VAEDetector` with Optuna and crash recovery.

    Crash recovery
    --------------
    Same pattern as :func:`src.models.iforest.tune_iforest`. The study is
    created against a **persistent SQLite RDBStorage** (default
    ``sqlite:///configs/optuna_vae.db``) with
    ``optuna.create_study(..., load_if_exists=True)``: if the process dies
    mid-search and ``tune_vae`` is called again with the same
    ``study_name`` + ``storage``, Optuna reopens the existing study and
    continues from the completed trials. On top of that, the current best
    hyperparameters are written to ``best_params_path`` (YAML, atomic
    ``.tmp`` + ``os.replace``) after **every** completed trial. Each trial also
    trains in its own ``checkpoint_dir/trial_<n>`` subdir so per-trial VAE
    checkpoints never clobber each other (and so an interrupted trial can
    resume from its own epoch checkpoint).

    Search space
    ------------
    * ``latent_dim`` -- int in [4, 32]
    * ``lr`` -- float in [1e-4, 1e-3] (log scale)
    * ``optimizer`` -- {'adam', 'adamw', 'rmsprop'}
    * ``batch_size`` -- {128, 256, 512}
    * ``beta`` -- float in [0.1, 2.0]
    * ``dropout`` -- float in [0.1, 0.4]
    * ``n_layers`` -- int in [1, 3]
    * ``hidden_dim`` -- {32, 64, 128}
    * ``epochs`` -- int in [1, max_epochs] (small budget for tuning speed)

    Objective modes and direction handling
    ---------------------------------------
    * **Supervised** (``y`` given, aligned row-for-row to ``X``): default
      **average_precision_score (PR-AUC)** of the reconstruction-error scores
      vs. the 0/1 labels (switchable to ROC-AUC via
      ``objective_metric='roc_auc'``). When ``direction is None`` it is
      auto-set to ``'maximize'``.
    * **Unsupervised** (``y is None``): default objective is the **validation
      reconstruction loss** (lower is better). When ``direction is None`` it is
      auto-set to ``'minimize'``.

    ``objective_metric`` may also be a callable ``(detector, X) -> float`` for
    a fully custom objective; when a callable is used with ``direction is None``
    the direction defaults to ``'maximize'`` (override explicitly if needed).

    Args:
        X: Preprocessed feature matrix (dense ndarray or scipy sparse).
        n_trials: Number of *new* trials to run in this call.
        y: Optional 0/1 anomaly labels aligned to ``X`` rows (supervised mode).
        storage: Optuna storage URI; defaults to the SQLite DB above.
        study_name: Study name (reused for resume).
        direction: 'maximize'/'minimize'; auto-set from mode when ``None``.
        objective_metric: Metric name or custom callable (see above).
        best_params_path: YAML path for the incremental best-params checkpoint.
        model_out: torch path for the final refitted detector.
        checkpoint_dir: Root dir for per-trial VAE checkpoints.
        random_state: Seed for the sampler and every fitted VAE.
        max_epochs: Upper bound on the per-trial epoch budget.
        timeout: Optional wall-clock budget (seconds) for ``study.optimize``.
        valid_mask: Boolean row mask marking the chronological validation rows.
            Every trial trains on the complement and is scored on the mask, so
            hyperparameters (and early stopping) are selected on future periods.
            ``None`` falls back to a shuffled 10% split.

    Objective modes: a supervised metric when ``y`` is given, otherwise
    ``best_val_elbo_`` -- the negative ELBO at ``beta = 1``, *not* the
    beta-weighted training loss. Since ``beta`` is a search dimension, the
    weighted loss is a different function in every trial and cannot rank them
    (see the comment at the objective's return).
    ``objective_metric="recon_p50"`` selects the median validation
    reconstruction error (minimised) even when labels exist -- the label-free
    proxy for "the normal mass reconstructs well".

    Trial-level early stopping (distinct from the *per-epoch* early stopping
    inside each trial's own fit, controlled separately by
    ``VAEDetector.early_stopping_patience``): ``early_stopping_patience`` stops
    the *study* after that many consecutive trials with no
    ``early_stopping_min_delta``-relative improvement in ``study.best_value``,
    never before ``early_stopping_min_trials`` trials complete. ``None``
    disables it (all ``n_trials`` always run). See
    :class:`src.models._tuning_stop.TrialPatienceStopper`.

    Returns:
        The Optuna :class:`~optuna.study.Study` (completed + resumed trials).
    """
    import optuna
    from tqdm.auto import tqdm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log = setup_logging()
    if storage is None:
        storage = _default_storage_uri()
    _ensure_parent_dir(best_params_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    y_arr = None if y is None else np.asarray(y).ravel()
    supervised = y_arr is not None
    custom_objective = objective_metric if callable(objective_metric) else None

    # Auto-select the optimization direction from the objective mode.
    if direction is None:
        if custom_objective is not None:
            direction = "maximize"
        elif supervised:
            direction = "maximize"  # PR-AUC / ROC-AUC: higher is better
        else:
            direction = "minimize"  # validation reconstruction loss: lower better
    resolved_direction = direction

    metric_name = objective_metric if isinstance(objective_metric, str) else None
    label_free = metric_name in _UNSUPERVISED_METRICS
    use_supervised = supervised and not label_free
    if direction is None and label_free:
        resolved_direction = "minimize"  # both proxies are reconstruction errors
    mode = (
        "supervised(custom)" if (supervised and custom_objective is not None)
        else "custom" if custom_objective is not None
        else f"unsupervised({metric_name})" if label_free
        else "supervised" if supervised
        else "unsupervised"
    )

    vm = None if valid_mask is None else np.asarray(valid_mask, dtype=bool).ravel()
    if vm is not None:
        log.info(
            "tune_vae: temporal validation mask supplied (%d of %d rows held out)",
            int(vm.sum()), vm.size,
        )

    def objective(trial: "optuna.trial.Trial") -> float:
        # TEORÍA (bottleneck): the latent width is the capacity limit that makes
        # the VAE an anomaly detector at all. Too wide and it learns a
        # near-identity map -- anomalies reconstruct as well as normal rows and
        # the score flattens. The floor of 4 keeps enough room for the panel's
        # main factors; the ceiling stays generous but a *smoothed* feature mart
        # (rolling averages) wants the tight end of this range, 4-8.
        latent_dim = trial.suggest_int("latent_dim", 4, 32)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"])
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        # beta > 2 pushes hard toward the prior; combined with KL annealing the
        # useful band is the moderate one.
        beta = trial.suggest_float("beta", 0.1, 2.0)
        # Non-zero floor: dropout is the regulariser that stops the decoder from
        # memorising individual rows, which is what would let an anomaly
        # reconstruct perfectly.
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        # Floor at 2, not 1. A single-epoch trial has not trained: its latent
        # space is still at the prior, so its KL is ~0 and -- under the ELBO
        # objective -- it can *win* on the numbers while being useless. Seen
        # empirically: a 1-epoch trial was selected and produced a fully
        # collapsed model (0 of 9 active units). The floor scales with the
        # budget so a large `max_epochs` does not spend trials on stubs.
        epoch_floor = min(2, int(max_epochs)) if max_epochs >= 2 else 1
        epochs = trial.suggest_int(
            "epochs", max(epoch_floor, int(max_epochs) // 4), max(1, int(max_epochs))
        )
        # The KL ramp must fit inside this trial's own budget. The detector's
        # default ramp is 10 epochs; a trial that trains for 5 would spend its
        # entire life at a partial KL weight and never see the beta it is
        # being evaluated on -- making the trial's `beta` largely fictional.
        kl_anneal = min(_DEFAULT_KL_ANNEAL_EPOCHS, max(1, epochs // 2))

        detector = VAEDetector(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            beta=beta,
            lr=lr,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            kl_anneal_epochs=kl_anneal,
            random_state=random_state,
        )
        trial_ckpt = os.path.join(checkpoint_dir, f"trial_{trial.number}")
        detector.fit(
            X, checkpoint_dir=trial_ckpt, resume=True, val_fraction=0.1, valid_mask=vm,
        )

        if custom_objective is not None:
            return float(custom_objective(detector, X))
        if use_supervised:
            # Score the held-out block only. Scoring all of X would compare the
            # labels against rows the VAE was trained to reconstruct, which
            # measures memorisation rather than generalisation.
            if vm is not None:
                return _supervised_score(y_arr[vm], detector.score_samples(X[vm]), metric_name)
            return _supervised_score(y_arr, detector.score_samples(X), metric_name)
        if metric_name == "recon_p50":
            # TEORÍA: "reconstruction error of the mass". The detector's job is
            # to model *normal* behaviour well; anomalies are then whatever it
            # reconstructs badly. Optimising the mean error would let a model
            # win by shrinking the error on the outliers too -- exactly the
            # opposite of what is wanted, since that erases the signal. The
            # median (p50) is dominated by the normal bulk and is insensitive to
            # the tail, so minimising it sharpens the contrast the score relies on.
            eval_scores = detector.score_samples(X[vm] if vm is not None else X)
            return float(np.median(np.asarray(eval_scores, dtype=float)))
        # Default unsupervised: the negative ELBO (beta=1) at the best epoch.
        #
        # NOT `best_val_loss_`, which is `recon + beta*KL` with this trial's own
        # `beta`. Because `beta` is itself a search dimension, that value is a
        # different objective function in every trial: a low-beta trial scores
        # mechanically better than a high-beta one even when the two models are
        # equally good, so the study would be ranking the parameterisation
        # rather than the fit. Evaluating every trial at beta=1 -- the actual
        # ELBO -- restores a common yardstick while keeping the KL term, which
        # a reconstruction-only objective would drop (and thereby reward
        # overcapacity: a model that reconstructs everything, noise included,
        # has no anomaly signal left).
        #
        # Training still uses the trial's `beta`: that is the beta-VAE method
        # (Higgins et al. 2017) and is deliberate. Only the *selection* metric
        # is beta-free.
        if np.isfinite(detector.best_val_elbo_):
            return float(detector.best_val_elbo_)
        # No validation block (e.g. a caller passed neither mask nor fraction):
        # fall back to the training-objective value rather than returning inf,
        # which would make every such trial indistinguishable.
        return float(detector.best_val_loss_)

    from src.models._tuning_budget import tpe_startup_trials

    # Scaled to the budget: Optuna's fixed default of 10 would spend this
    # study's entire 10-trial default allowance on random exploration and
    # never actually optimise. See `_tuning_budget` for the numbers.
    sampler = optuna.samplers.TPESampler(
        seed=random_state, n_startup_trials=tpe_startup_trials(n_trials),
    )
    from src.models._optuna_storage import resolve_storage

    study = optuna.create_study(
        study_name=study_name,
        storage=resolve_storage(storage),
        direction=resolved_direction,
        sampler=sampler,
        load_if_exists=True,  # <-- crash-recovery / resume switch
    )

    log.info(
        "Optuna VAE tuning: study=%r storage=%r mode=%s direction=%s "
        "new_trials=%d existing_trials=%d",
        study_name, storage, mode, resolved_direction, n_trials, len(study.trials),
    )

    progress = tqdm(total=n_trials, desc=f"optuna[{study_name}]", unit="trial")

    def _progress_callback(study_, trial) -> None:
        progress.update(1)

    def _persist_best_callback(study_, trial) -> None:
        """Checkpoint the current best hyperparameters to YAML after each trial."""
        try:
            best = study_.best_trial
        except (ValueError, RuntimeError):
            return  # no completed trial yet
        payload = {
            "study_name": study_name,
            "direction": resolved_direction,
            "best_value": float(best.value) if best.value is not None else None,
            "best_trial_number": best.number,
            "n_trials_completed": len(
                [t for t in study_.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "objective_mode": mode,
            "random_state": random_state,
            "best_params": _detector_kwargs_from_params(best.params),
            "epochs": int(best.params.get("epochs", max_epochs)),
            "raw_optuna_params": dict(best.params),
        }
        tmp_path = best_params_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        atomic_replace(tmp_path, best_params_path)  # atomic swap, Windows-lock-safe
        log.info(
            "Checkpointed best VAE params (trial %d, value=%.6f) -> %s",
            best.number, best.value if best.value is not None else float("nan"),
            best_params_path,
        )

    callbacks = [_progress_callback, _persist_best_callback]
    stopper = None
    if early_stopping_patience is not None:
        from src.models._tuning_stop import TrialPatienceStopper

        stopper = TrialPatienceStopper(
            direction=resolved_direction, model_name="vae",
            n_trials_requested=n_trials + len(study.trials),
            patience=early_stopping_patience, min_delta=early_stopping_min_delta,
            min_trials=early_stopping_min_trials,
        )
        callbacks.append(stopper)

    with log_phase("vae.tune (optuna)", log):
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=callbacks,
                gc_after_trial=True,
            )
        finally:
            progress.close()
        if stopper is not None and stopper.stopped:
            log.info(
                "VAE tuning stopped early: %s (%d trial(s) skipped).",
                stopper.stop_reason, stopper.trials_skipped,
            )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.warning("No completed VAE trials; skipping final refit/save.")
        return study

    best_kwargs = _detector_kwargs_from_params(study.best_params)
    best_epochs = int(study.best_params.get("epochs", max_epochs))
    log.info(
        "Best VAE trial %d value=%.6f params=%s epochs=%d",
        study.best_trial.number, study.best_value, best_kwargs, best_epochs,
    )
    with log_phase("vae.refit_best", log):
        best_detector = VAEDetector(
            random_state=random_state, epochs=best_epochs, **best_kwargs
        )
        best_detector.fit(
            X,
            checkpoint_dir=os.path.join(checkpoint_dir, "best_refit"),
            resume=False,
            val_fraction=0.1,
            valid_mask=vm,
        )
        best_detector.save(model_out)

    return study


def _supervised_score(
    y: np.ndarray, scores: np.ndarray, objective_metric: Optional[str]
) -> float:
    """PR-AUC (default) or ROC-AUC of anomaly scores against binary labels."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    metric = (objective_metric or "average_precision").lower()
    if metric in ("average_precision", "pr_auc", "prauc", "ap"):
        return float(average_precision_score(y, scores))
    if metric in ("roc_auc", "rocauc", "auc", "roc"):
        return float(roc_auc_score(y, scores))
    raise ValueError(
        f"Unknown supervised objective_metric {objective_metric!r}; "
        "use 'average_precision' or 'roc_auc'."
    )


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def plot_reconstruction_error(
    scores: np.ndarray,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "vae_recon_error.png",
    y: Optional[np.ndarray] = None,
    max_points: int = 200_000,
    random_state: int = 42,
) -> str:
    """Histogram of per-row reconstruction-error scores, saved under figures.

    Uses the non-interactive ``Agg`` backend. When ``y`` (0/1 labels) is given
    the normal and anomaly score distributions are overlaid. Very large score
    vectors are randomly subsampled to ``max_points``. Figures always land
    under ``reports/figures/vae/`` per the project rule.

    Returns:
        The absolute path of the written PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float).ravel()
    y_arr = None if y is None else np.asarray(y).ravel()

    if scores.size > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(scores.size, size=max_points, replace=False)
        scores = scores[idx]
        if y_arr is not None:
            y_arr = y_arr[idx]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    if y_arr is not None and np.unique(y_arr).size > 1:
        normal = scores[y_arr == 0]
        anomaly = scores[y_arr == 1]
        bins = np.histogram_bin_edges(scores, bins=60)
        ax.hist(normal, bins=bins, alpha=0.6, density=True, label="normal", color="#4c72b0")
        ax.hist(anomaly, bins=bins, alpha=0.6, density=True, label="anomaly", color="#c44e52")
        ax.legend()
    else:
        ax.hist(scores, bins=60, alpha=0.8, color="#4c72b0")

    ax.set_title("VAE reconstruction-error distribution (higher = more anomalous)")
    ax.set_xlabel("reconstruction error")
    ax.set_ylabel("density" if y_arr is not None else "count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    setup_logging().info("Saved reconstruction-error figure to %s", out_path)
    return os.path.abspath(out_path)


def plot_latent_space(
    detector: "VAEDetector",
    X: ArrayLike,
    out_dir: str = _DEFAULT_FIG_DIR,
    filename: str = "vae_latent.png",
    y: Optional[np.ndarray] = None,
    max_points: int = 50_000,
    random_state: int = 42,
) -> str:
    """Encode ``X`` to latent means, PCA to 2D, and scatter (color by ``y``).

    Uses the non-interactive ``Agg`` backend. If the latent dimension is >2 a
    2-component PCA is applied; a 1D latent is padded with zeros; a 2D latent is
    plotted directly. Very large inputs are subsampled to ``max_points``.
    Figures land under ``reports/figures/vae/``.

    Returns:
        The absolute path of the written PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_arr = None if y is None else np.asarray(y).ravel()

    n_rows = _densify_nrows(X)
    idx = None
    if n_rows > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_rows, size=max_points, replace=False)
        idx.sort()
        X_sub = X[idx]
        if y_arr is not None:
            y_arr = y_arr[idx]
    else:
        X_sub = X

    latents = detector.encode(X_sub)
    if latents.shape[1] == 1:
        coords = np.hstack([latents, np.zeros((latents.shape[0], 1), dtype=latents.dtype)])
    elif latents.shape[1] == 2:
        coords = latents
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=random_state).fit_transform(latents)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(7, 6))
    if y_arr is not None and np.unique(y_arr).size > 1:
        normal = y_arr == 0
        anomaly = y_arr == 1
        ax.scatter(coords[normal, 0], coords[normal, 1], s=6, alpha=0.4,
                   label="normal", color="#4c72b0", linewidths=0)
        ax.scatter(coords[anomaly, 0], coords[anomaly, 1], s=10, alpha=0.7,
                   label="anomaly", color="#c44e52", linewidths=0)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.4,
                   color="#4c72b0", linewidths=0)

    ax.set_title("VAE latent space (PCA of encoder means)")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    setup_logging().info("Saved latent-space figure to %s", out_path)
    return os.path.abspath(out_path)


def _densify_nrows(X: ArrayLike) -> int:
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return int(np.asarray(X).shape[0])
