"""§9 diagnostic experiments that used to be NOT_REQUESTED, now run by default.

Six families, all label-free and descriptive (they never rank or "approve" a
configuration; there is no pass/fail threshold):

1. **Capacidad y dimensión latente (VAE)** -- retrain with a smaller/larger latent
   width and hidden width.
2. **Beta y programación KL (VAE)** -- retrain with other ``beta`` and other
   ``kl_anneal_epochs`` (the KL schedule was never swept before).
3. **Pérdidas por tipo de feature** -- where the reconstruction error and the
   diagnostic's ``recon_topk`` alert slots come from (categorical one-hot vs numeric
   vs panel history ...). No retraining.
4. **Ablación de familias de features** -- IF refit without each family; VAE refit
   without the most informative ones.
5. **Backtests temporales** -- rolling origins: each origin re-runs the preprocessing
   and refits on strictly earlier periods, then scores the origin period only.
6. **Estabilidad entre ventanas** -- rank agreement (Spearman, top-K Jaccard) of the
   entity scores between consecutive backtest windows. No extra fits.

Every alert set is built exactly the way production builds it: for the VAE the
*diagnostic's* ``recon_topk`` percentile against the train block; for the IF the
score percentile against the train block. (The previous VAE sweep compared a plain
MSE percentile against the production ``recon_topk`` percentile, so its Jaccards
were not comparable.)

Cost control: VAE retrains share one budget (``vae_fit_budget``, default 16), epochs
are capped (``epoch_cap``) and very large fits are subsampled (``max_fit_rows``);
what was capped is written into each row's ``detail``. When the budget runs out the
remaining VAE points are ``NOT_REQUESTED`` with that reason, never silently dropped.
Each refit is isolated (temporary checkpoint dir, ``resume=False``), seeded with the
run's seed, and a failing point becomes a ``FAILED`` row instead of stopping the run.
Capacity, beta/KL and ablation variants carry a **control**: the production
configuration refitted with another seed under the same experimental epoch/row
caps and compared with production through the same alert-set Jaccard. Backtest
origins deliberately do not use that comparison because their temporal geometry
differs from production. (An earlier version showed the stability section's
raw-score top-k Jaccard as a "floor"; it measures a different set.)

Data sources / inputs: the raw panel, fitted preprocessor, full IF/VAE matrices,
chronological masks, production detectors, and §9 settings from
``configs/pipeline.yaml``.
Created: 2026-09-25
Last modified: 2026-09-25
Changelog:
- 2026-09-25: Removed incomparable rolling-origin Jaccards, excluded validation
  rows from VAE score calibration, deduplicated capacity points, and enforced
  exact three-period aggregate stability windows.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluation.ifvae_contract import (
    STATUS_EXECUTED,
    STATUS_FAILED,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_REQUESTED,
    STATUS_UNAVAILABLE,
)

__all__ = ["ALL_FAMILIES", "FAMILY_LABELS", "build_experiment_context", "feature_families",
           "run_experiment_families"]

ALL_FAMILIES = ("capacity", "beta_kl", "loss_by_type", "ablation", "backtest", "window_stability")

FAMILY_LABELS = {
    "capacity": "Capacidad y dimensión latente (VAE)",
    "beta_kl": "Beta y programación KL (VAE)",
    "loss_by_type": "Pérdidas por tipo de feature",
    "ablation": "Ablación de familias de features",
    "backtest": "Backtests temporales",
    "window_stability": "Estabilidad entre ventanas",
}

_LOG = logging.getLogger("modelo.ifvae_experiments")
_QUIET = logging.getLogger("modelo.ifvae_experiments.quiet")
_QUIET.setLevel(logging.WARNING)

_MIN_FIT_ROWS = 200
_MIN_OOT_ROWS = 30

_RE_PANEL_HIST = re.compile(r"^num__.+_(lag|diff|ratio)\d+$|^num__.+_own_z$")


# --------------------------------------------------------------------------- #
# Feature families                                                            #
# --------------------------------------------------------------------------- #
def feature_families(
    names: Sequence[str], derived: Sequence[str] = (), ratio_names: Sequence[str] = (),
) -> dict[str, str]:
    """``{feature name: family}`` from the real column names.

    Families: ``derivada`` (stacked IF score, VAE only), ``cat`` (one-hot, VAE only),
    ``bool``, ``cyc`` (month sin/cos), ``panel_hist`` (lag/diff/ratio horizons and
    ``_own_z``), ``ratios_negocio`` (the engineer's ratio columns) and ``num_base``
    (every other numeric).
    """
    derived_set, ratio_set = set(derived), set(ratio_names)
    out: dict[str, str] = {}
    for name in names:
        if name in derived_set:
            fam = "derivada"
        elif name.startswith("cat__"):
            fam = "cat"
        elif name.startswith("bool__"):
            fam = "bool"
        elif name.startswith("cyc__"):
            fam = "cyc"
        elif _RE_PANEL_HIST.match(name):
            fam = "panel_hist"
        elif name.startswith("num__") and name[len("num__"):] in ratio_set:
            fam = "ratios_negocio"
        else:
            fam = "num_base"
        out[name] = fam
    return out


def build_experiment_context(
    *, df: pd.DataFrame, schema: Any, keys: pd.DataFrame, x_if_all: Any, if_feature_names: Sequence[str],
    x_vae_all: Any, vae_feature_names: Sequence[str], if_detector: Any, vae_detector: Any,
    train_mask: np.ndarray, in_mask: np.ndarray, oot_mask: np.ndarray, valid_local: np.ndarray,
    n_val_periods: int, derived_features: Sequence[str], prep_kwargs: dict,
    fitted_preprocessor: Any = None, x_onehot_all: Any = None,
    onehot_feature_names: Optional[Sequence[str]] = None, vae_builder: Any = None,
    mixed_config: Any = None, stack_scaler: Any = None,
) -> dict:
    """The arrays and fitted objects the experiment families need (single source for
    ``main.py`` and the tests): full matrices, row indices, the raw panel and the exact
    preprocessing settings, so each backtest origin can re-run the preprocessing on
    strictly earlier periods only."""
    time_col = schema.time_col or "period"
    entity_col = schema.entity_col or "entity_id"
    engineer = None
    try:
        engineer = fitted_preprocessor.named_steps["panel_features"]
    except Exception:  # noqa: BLE001 - family names are a nicety, not a dependency
        pass
    from src.preprocessing.mixed_view import categorical_sources

    cat_sources = categorical_sources(df, schema)
    return {
        "df": df, "schema": schema, "keys": keys,
        "x_if_all": _dense(x_if_all), "if_feature_names": list(if_feature_names),
        # `vae_feature_names` = the COLUMNS of the VAE input matrix; `vae_variable_names` = the ORIGINAL
        # variables that get one reconstruction contribution each (identical for the one-hot MLP).
        "x_vae_all": _dense(x_vae_all), "vae_feature_names": list(vae_feature_names),
        "vae_variable_names": (list(vae_detector.variable_names)
                               if getattr(vae_detector, "layout", None) is not None
                               else list(vae_feature_names)),
        "if_detector": if_detector, "vae_detector": vae_detector,
        "train_idx": np.flatnonzero(train_mask), "oot_idx": np.flatnonzero(oot_mask),
        "in_idx": np.flatnonzero(in_mask), "valid_local": np.asarray(valid_local, dtype=bool),
        "period_idx": np.unique(keys[time_col].to_numpy(), return_inverse=True)[1],
        "entity_ids": keys[entity_col].to_numpy(),
        "n_val_periods": int(n_val_periods),
        "derived_features": list(derived_features),
        "ratio_names": [n for _, _, n in getattr(engineer, "ratio_specs_", [])],
        "cat_sources": cat_sources, "prep_kwargs": dict(prep_kwargs),
        # Both VAE representations: the production one is `x_vae_all` / `vae_detector`; the one-hot
        # matrix, the builder of the mixed view and the stacking scaler let the experiments retrain
        # the control representation and rebuild the view at every backtest origin.
        "x_onehot_all": None if x_onehot_all is None else _dense(x_onehot_all),
        "onehot_feature_names": None if onehot_feature_names is None else list(onehot_feature_names),
        "vae_builder": vae_builder, "mixed_config": mixed_config, "stack_scaler": stack_scaler,
        "representation": "embedding" if getattr(vae_detector, "layout", None) is not None else "onehot",
    }


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _row(experiment: str, status: str, configuration: Optional[str] = None,
         detail: Optional[str] = None, reason: Optional[str] = None) -> dict:
    return {"experiment": experiment, "status": status, "configuration": configuration,
            "artifact": None, "detail": detail, "reason": reason}


def _jaccard(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else None


def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "n/d" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


class _Budget:
    def __init__(self, n: int):
        self.total = max(0, int(n))
        self.left = self.total

    def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        return True


def _budget_row(experiment: str, budget: _Budget, configuration: Optional[str] = None) -> dict:
    if budget.total == 0:
        why = "El presupuesto de reentrenos del VAE es 0 (experiments.vae_fit_budget): reentrenos desactivados."
    else:
        why = (f"Se agotó el presupuesto de reentrenos del VAE ({budget.total}); "
               "súbelo con experiments.vae_fit_budget en configs/pipeline.yaml.")
    return _row(experiment, STATUS_NOT_REQUESTED, configuration, reason=why)


def _step(name: str, label: str):
    from ifvae_diag import progress

    return progress.step(name, label=label)


# --------------------------------------------------------------------------- #
# Production-comparable alert sets                                            #
# --------------------------------------------------------------------------- #
def _vae_residuals(det: Any, x: Any) -> np.ndarray:
    """Per-term reconstruction residuals: ``|x - recon|`` per matrix column (one-hot architecture) or the
    per-ORIGINAL-variable contributions (mixed architecture)."""
    if getattr(det, "layout", None) is not None:
        return det.contributions(x)
    from src.evaluation.ifvae_diagnostic import _vae_forward

    xd = _dense(x)
    return np.abs(xd - _vae_forward(det, xd)[2])


def _recon_topk_alerts(det: Any, x_all: Any, train_idx: np.ndarray, oot_idx: np.ndarray,
                       top_k: int, threshold: float, scale_floor: float = 0.0,
                       center: bool = False) -> np.ndarray:
    """Alert flags for the OOT rows exactly as the diagnostic builds the VAE's:
    ``recon_topk`` of |residual|/MAD(train residual), percentile against train."""
    from ifvae_diag.scoring import anomaly_percentile, reconstruction_scores

    mixed = getattr(det, "layout", None) is not None                  # same normalisation rule as the bridge
    floor, ctr = (scale_floor, center) if mixed else (0.0, False)
    ref_res, oot_res = _vae_residuals(det, x_all[train_idx]), _vae_residuals(det, x_all[oot_idx])
    ref_s = reconstruction_scores(ref_res, ref_res, top_k, floor, ctr)["recon_topk"].to_numpy()
    oot_s = reconstruction_scores(ref_res, oot_res, top_k, floor, ctr)["recon_topk"].to_numpy()
    return anomaly_percentile(ref_s, oot_s) >= threshold


def _if_alerts(det: Any, x_all: Any, train_idx: np.ndarray, oot_idx: np.ndarray,
               threshold: float) -> tuple[np.ndarray, np.ndarray]:
    from ifvae_diag.scoring import anomaly_percentile

    ref = np.asarray(det.score_samples(x_all[train_idx]), dtype=float)
    oot = np.asarray(det.score_samples(x_all[oot_idx]), dtype=float)
    return anomaly_percentile(ref, oot) >= threshold, oot


def _fit_if(base: Any, x_fit: Any, seed: int) -> Any:
    from src.models import IsolationForestDetector
    from src.evaluation.ifvae_diagnostic import _IF_REFIT_PARAMS

    params = {n: getattr(base, n) for n in _IF_REFIT_PARAMS}
    det = IsolationForestDetector(random_state=seed, **params)
    det.fit(x_fit)
    return det


def _fit_vae(base: Any, x_fit: Any, valid_local: Optional[np.ndarray], overrides: dict,
             seed: int, epoch_cap: int, max_rows: int) -> tuple[Any, list[str]]:
    """Retrain the production VAE's configuration with ``overrides`` (isolated)."""
    from src.models import VAEDetector
    from src.evaluation.ifvae_diagnostic import _VAE_REFIT_PARAMS

    params = {n: getattr(base, n) for n in _VAE_REFIT_PARAMS}
    params.update(overrides)
    notes: list[str] = []
    if int(params["epochs"]) > epoch_cap:
        params["epochs"] = int(epoch_cap)
        if "kl_anneal_epochs" not in overrides:
            params["kl_anneal_epochs"] = min(int(params["kl_anneal_epochs"]), max(1, epoch_cap // 2))
        notes.append(f"épocas limitadas a {epoch_cap}")
    if int(params["kl_anneal_epochs"]) > int(params["epochs"]):
        notes.append(f"kl_anneal_epochs limitado a {int(params['epochs'])} (= épocas)")
    params["kl_anneal_epochs"] = min(int(params["kl_anneal_epochs"]), int(params["epochs"]))
    n = x_fit.shape[0]
    vm = None if valid_local is None else np.asarray(valid_local, bool)
    if n > max_rows:
        sel = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
        x_fit = x_fit[sel]
        vm = None if vm is None else vm[sel]
        notes.append(f"submuestra de {max_rows:,} de {n:,} filas")
    tmp = tempfile.mkdtemp(prefix="ifvae_exp_")
    try:
        det = VAEDetector(random_state=seed, **params)
        kwargs: dict = {"checkpoint_dir": tmp, "resume": False}
        if vm is not None and 0 < int(vm.sum()) < len(vm):
            kwargs["valid_mask"] = vm
        det.fit(x_fit, **kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return det, notes


# --------------------------------------------------------------------------- #
# Families 1 and 2: VAE capacity, beta and KL schedule                        #
# --------------------------------------------------------------------------- #
def _vae_variant_row(experiment: str, configuration: str, det: Any, notes: list[str],
                     x_all: Any, ctx: dict, prod_high: np.ndarray, floor: Optional[float],
                     show_elbo: bool = True) -> dict:
    from src.models import collapse_verdict

    high = _recon_topk_alerts(det, x_all, ctx["train_idx"], ctx["oot_idx"], ctx["top_k"], ctx["threshold"],
                              ctx.get("scale_floor", 0.0), ctx.get("center", False))
    sample = x_all[ctx["train_idx"][: 20_000]]
    diag = det.latent_diagnostics(sample)
    verdict = collapse_verdict(diag)
    elbo = getattr(det, "best_val_elbo_", float("inf"))
    parts = [
        f"{int(high.sum())} alertas de {len(high)}",
        f"Jaccard vs producción = {_fmt(_jaccard(prod_high, high))}"
        + (f" (control: misma configuración, otra semilla = {_fmt(floor)})" if floor is not None else ""),
        f"unidades activas {diag['active_units']}/{diag['latent_dim']}",
        f"veredicto = {verdict.get('severity', 'n/d')}",
        f"KL media = {_fmt(diag.get('mean_kl'))}",
    ]
    if show_elbo and np.isfinite(elbo):
        parts.append(f"-ELBO(β=1) val = {_fmt(elbo)}")
    if notes:
        parts.append("; ".join(notes))
    return _row(experiment, STATUS_EXECUTED, configuration, detail="; ".join(parts))


def _capacity_points(base: Any, n_features: int, override: Optional[Sequence[int]]) -> list[tuple[str, dict]]:
    d = int(base.latent_dim)
    pts: list[tuple[str, dict]] = []
    if override is None:
        cand = [max(2, d // 2), 2 * d, 4 * d]
    else:
        cand = [int(v) for v in override]
    seen = {d}
    for v in cand:
        if 1 <= v <= n_features and v not in seen:
            seen.add(v)
            pts.append((f"latent_dim={v}", {"latent_dim": v}))
    if override is None:
        for f in (0.5, 2.0):
            if base.hidden_dims:
                hd = [max(4, int(round(h * f))) for h in base.hidden_dims]
                if list(hd) != list(base.hidden_dims):
                    pts.append((f"hidden_dims={hd}", {"hidden_dims": hd}))
            else:
                h = max(4, int(round(base.hidden_dim * f)))
                if h != base.hidden_dim:
                    pts.append((f"hidden_dim={h}", {"hidden_dim": h}))
    return pts


def _beta_kl_points(base: Any, beta_grid: Optional[Sequence[float]],
                    kl_grid: Optional[Sequence[int]]) -> list[tuple[str, dict]]:
    beta, epochs, kl = float(base.beta), int(base.epochs), int(base.kl_anneal_epochs)
    pts: list[tuple[str, dict]] = []
    betas = [0.25 * beta, 4.0 * beta] if beta_grid is None else [float(b) for b in beta_grid]
    for b in betas:
        if b > 0 and not np.isclose(b, beta):
            pts.append((f"beta={b:g}", {"beta": float(b)}))
    kls = [0, epochs] if kl_grid is None else [int(k) for k in kl_grid]
    for k in dict.fromkeys(kls):
        if 0 <= k and k != kl:
            pts.append((f"kl_anneal_epochs={k}", {"kl_anneal_epochs": int(k)}))
    return pts


def _run_vae_sweep(family: str, points: list[tuple[str, dict]], ctx: dict, budget: _Budget,
                   prod_high: np.ndarray, floor: Optional[float]) -> list[dict]:
    label = FAMILY_LABELS[family]
    if ctx["x_vae_train_rows"] < _MIN_FIT_ROWS:
        return [_row(label, STATUS_UNAVAILABLE,
                     reason=f"Menos de {_MIN_FIT_ROWS} filas de ajuste para reentrenar el VAE.")]
    rows: list[dict] = []
    if not points:
        return [_row(label, STATUS_NOT_APPLICABLE,
                     reason="Ningún punto distinto de la configuración de producción.")]
    for cfg_text, overrides in points:
        name = f"{label}: {cfg_text}"
        if not budget.take():
            rows.append(_budget_row(name, budget, cfg_text))
            continue
        try:
            with _step(f"ifvae_experiments.{family}[{cfg_text}]", "Reajuste del VAE (experimento)"):
                det, notes = _fit_vae(ctx["vae_detector"], ctx["x_vae_fit"], ctx["valid_local"],
                                      overrides, ctx["seed"], ctx["epoch_cap"], ctx["max_fit_rows"])
                rows.append(_vae_variant_row(name, cfg_text, det, notes, ctx["x_vae_all"], ctx,
                                             prod_high, floor))
        except Exception as exc:  # noqa: BLE001 - one bad point must not stop the sweep
            rows.append(_row(name, STATUS_FAILED, cfg_text, reason=f"{type(exc).__name__}: {exc}"))
    return rows


# --------------------------------------------------------------------------- #
# Family 3: reconstruction loss by feature type                               #
# --------------------------------------------------------------------------- #
def _source_of(name: str, sources: Sequence[str]) -> str:
    body = name[len("cat__"):]
    best = ""
    for s in sources:
        if (body == s or body.startswith(s + "_")) and len(s) > len(best):
            best = s
    return best or body


def _loss_by_type(reference: pd.DataFrame, scored: pd.DataFrame, features: Sequence[str],
                  fam_of: dict[str, str], ctx: dict, prod_high: np.ndarray) -> tuple[list[dict], dict]:
    """Where the reconstruction loss and the ``recon_topk`` slots come from, per feature family.

    The unit is the *original variable*: for the mixed VAE each frame column is one variable (its own
    contribution), so the family counts, the average loss per variable, the share of the total loss and
    the expected top-k share are all computed over variables. For the one-hot MLP the unit is the matrix
    column (the historical, biased view this experiment exists to expose).
    """
    from ifvae_diag.scoring import anomaly_percentile, reconstruction_scores, residual_contributions

    label = FAMILY_LABELS["loss_by_type"]
    mixed = ctx.get("representation") == "embedding"
    unit = "variables originales" if mixed else "columnas"
    if len(scored) < _MIN_OOT_ROWS:
        return [_row(label, STATUS_UNAVAILABLE, reason=f"Menos de {_MIN_OOT_ROWS} filas evaluadas.")], {}
    floor = float(ctx.get("scale_floor", 0.0)) if mixed else 0.0
    center = bool(ctx.get("center", False)) if mixed else False
    feats = list(features)
    ref_x = reference[feats].to_numpy(float)
    ref_r = reference[[f"recon__{f}" for f in feats]].to_numpy(float)
    oot_x = scored[feats].to_numpy(float)
    oot_r = scored[[f"recon__{f}" for f in feats]].to_numpy(float)
    ref_res, oot_res = np.abs(ref_x - ref_r), np.abs(oot_x - oot_r)
    top_k = ctx["top_k"]
    contrib = residual_contributions(ref_res, oot_res, floor, center)
    kk = min(top_k, contrib.shape[1])
    slots = np.argpartition(contrib, contrib.shape[1] - kk, axis=1)[:, -kk:]
    fams = np.array([fam_of[f] for f in feats])
    med = np.nanmedian(ref_res, axis=0)
    mad = np.nanmedian(np.abs(ref_res - med), axis=0) * 1.4826
    power = 1 if mixed else 2                       # contributions are already losses; one-hot uses squared error
    tot_ref, tot_oot = (ref_res ** power).sum(), (oot_res ** power).sum()
    stats: dict = {"slots_share": {}, "col_share": {}}
    rows = []
    for fam in sorted(set(fams)):
        cols = np.flatnonzero(fams == fam)
        share_cols = len(cols) / len(feats)
        share_slots = float(np.isin(slots, cols).mean())
        stats["slots_share"][fam], stats["col_share"][fam] = share_slots, share_cols
        loss_name = "pérdida" if mixed else "error cuadrático"
        rows.append(_row(
            f"{label}: {fam}", STATUS_EXECUTED, f"{len(cols)} {unit} ({share_cols:.0%} del total)",
            detail=(f"pérdida media por variable = {_fmt(float((oot_res[:, cols] ** power).mean()))}; "
                    if mixed else "")
                   + f"{(oot_res[:, cols] ** power).sum() / tot_oot:.1%} de la {loss_name} OOT "
                     f"({(ref_res[:, cols] ** power).sum() / tot_ref:.1%} en train); "
                     f"ocupa {share_slots:.1%} de los {kk} lugares top-K por fila "
                     f"(esperado por número de {unit}: {share_cols:.1%}); "
                     f"MAD mediana del residuo = {_fmt(float(np.median(mad[cols])))}",
        ))

    def alerts_from(ref_cols: np.ndarray, oot_cols: np.ndarray) -> np.ndarray:
        rs = reconstruction_scores(ref_cols, ref_cols, top_k, floor, center)["recon_topk"].to_numpy()
        os_ = reconstruction_scores(ref_cols, oot_cols, top_k, floor, center)["recon_topk"].to_numpy()
        return anomaly_percentile(rs, os_) >= ctx["threshold"]

    is_cat = fams == "cat"
    if not is_cat.any():
        rows.append(_row(f"{label}: resumen de sesgo categórico", STATUS_NOT_APPLICABLE,
                         reason="El VAE no recibe variables categóricas en esta corrida."))
        return rows, stats
    parts = [f"{int(prod_high.sum())} alertas de producción (recon_topk)"]
    num_idx = np.flatnonzero(~is_cat)
    if len(num_idx) >= 1:
        a = alerts_from(ref_res[:, num_idx], oot_res[:, num_idx])
        parts.append(f"Jaccard vs solo-numéricas = {_fmt(_jaccard(prod_high, a))}")
    if mixed:
        parts.append("cada categórica aporta UNA contribución (NLL de la categoría observada), sin depender "
                     "de su cardinalidad")
    else:
        sources = ctx.get("cat_sources") or []
        if sources:
            groups: dict[str, list[int]] = {}
            for j in np.flatnonzero(is_cat):
                groups.setdefault(_source_of(feats[j], sources), []).append(j)
            agg_ref = np.column_stack([np.sqrt((ref_res[:, g] ** 2).sum(1)) for g in groups.values()])
            agg_oot = np.column_stack([np.sqrt((oot_res[:, g] ** 2).sum(1)) for g in groups.values()])
            a2 = alerts_from(np.hstack([ref_res[:, num_idx], agg_ref]), np.hstack([oot_res[:, num_idx], agg_oot]))
            parts.append(f"Jaccard vs one-hot agregado por variable de origen ({len(groups)}) = "
                         f"{_fmt(_jaccard(prod_high, a2))}")
        else:
            parts.append("no se dio la lista de columnas categóricas de origen: sin variante agregada")
        parts.append("el sesgo se expone; se corrige con vae.categorical_representation: embedding")
    rows.append(_row(f"{label}: resumen de sesgo categórico", STATUS_EXECUTED,
                     "recon_topk sin/agregando el bloque categórico", detail="; ".join(parts)))

    if mixed:
        # MISSING / UNKNOWN keep their own identifiable contribution (never a silent "normal" category).
        cat_cols = [f for f, k in zip(feats, fams) if k == "cat"]
        idx = scored[cat_cols].to_numpy(float)
        cnt = oot_res[:, [feats.index(c) for c in cat_cols]]
        normal = cnt[idx >= 2]
        for token, name in ((0, "MISSING"), (1, "UNKNOWN")):
            m = idx == token
            row_name = f"{label}: token {name}"
            if not m.any():
                rows.append(_row(row_name, STATUS_NOT_APPLICABLE,
                                 reason=f"No hay valores {name} en las variables categóricas de la ventana evaluada."))
                continue
            rows.append(_row(
                row_name, STATUS_EXECUTED, f"{int(m.sum())} pares (fila, variable) de {m.size}",
                detail=(f"contribución media = {_fmt(float(cnt[m].mean()))} vs {_fmt(float(normal.mean()))} en "
                        f"categorías vistas; {int(m.any(axis=1).sum())} filas afectadas; el token conserva su "
                        "contribución propia (no se reasigna a una categoría normal)"),
            ))
    return rows, stats


def _representation_control(frames_stats: dict, ctx: dict, budget: _Budget, prod_high: np.ndarray) -> list[dict]:
    """Retrain the OTHER representation (one-hot MLP) with the same configuration, seed, epochs, rows and
    alert budget, and compare where its top-k slots come from and how its alerts overlap production's."""
    from ifvae_diag.scoring import anomaly_percentile, residual_contributions

    label = f"{FAMILY_LABELS['loss_by_type']}: control one-hot"
    if ctx.get("representation") != "embedding":
        return [_row(label, STATUS_NOT_APPLICABLE,
                     reason="La producción ya usa one-hot; compara con tools/compare_vae_representations.py.")]
    if ctx.get("x_onehot_all") is None or not ctx.get("onehot_feature_names"):
        return [_row(label, STATUS_UNAVAILABLE, reason="No se proporcionó la matriz one-hot de control.")]
    if not budget.take():
        return [_budget_row(label, budget)]
    try:
        with _step("ifvae_experiments.control[one-hot]", "Control: VAE one-hot con la misma configuración"):
            det, notes = _fit_vae(ctx["vae_detector"], ctx["x_onehot_all"][ctx["in_idx"]], ctx["valid_local"],
                                  {"layout": None, "mixed_config": None}, ctx["seed"], ctx["epoch_cap"],
                                  ctx["max_fit_rows"])
            high = _recon_topk_alerts(det, ctx["x_onehot_all"], ctx["train_idx"], ctx["oot_idx"],
                                      ctx["top_k"], ctx["threshold"])
            ref_res = _vae_residuals(det, ctx["x_onehot_all"][ctx["train_idx"]])
            oot_res = _vae_residuals(det, ctx["x_onehot_all"][ctx["oot_idx"]])
        contrib = residual_contributions(ref_res, oot_res)
        kk = min(ctx["top_k"], contrib.shape[1])
        slots = np.argpartition(contrib, contrib.shape[1] - kk, axis=1)[:, -kk:]
        cat_cols = np.array([str(n).startswith("cat__") for n in ctx["onehot_feature_names"]])
        share_cols, share_slots = float(cat_cols.mean()), float(np.isin(slots, np.flatnonzero(cat_cols)).mean())
        prod_cols = frames_stats.get("col_share", {}).get("cat")
        prod_slots = frames_stats.get("slots_share", {}).get("cat")
        det_txt = (f"control one-hot: {share_cols:.0%} de las columnas son categóricas y ocupan "
                   f"{share_slots:.1%} de los {kk} lugares top-K; "
                   + (f"producción (embeddings): {prod_cols:.0%} de las variables son categóricas y ocupan "
                      f"{prod_slots:.1%}; " if prod_cols is not None else "")
                   + f"{int(high.sum())} alertas de {len(high)} en el control; Jaccard de alertas vs producción = "
                   f"{_fmt(_jaccard(prod_high, high))}"
                   + (f"; {'; '.join(notes)}" if notes else ""))
        return [_row(label, STATUS_EXECUTED, "misma configuración, semilla, épocas, filas y presupuesto de alertas",
                     detail=det_txt)]
    except Exception as exc:  # noqa: BLE001
        return [_row(label, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}")]


# --------------------------------------------------------------------------- #
# Family 4: feature-family ablation                                           #
# --------------------------------------------------------------------------- #
def _ablation(ctx: dict, fam_if: dict[str, str], fam_vae: dict[str, str], budget: _Budget,
              prod_if_high: np.ndarray, prod_if_scores: np.ndarray, prod_vae_high: np.ndarray,
              floors: dict) -> list[dict]:
    label = FAMILY_LABELS["ablation"]
    rows: list[dict] = []
    names_if = ctx["if_feature_names"]
    present_if = sorted(set(fam_if.values()))
    all_fams = ("cat", "bool", "cyc", "panel_hist", "ratios_negocio", "num_base", "derivada")
    for fam in all_fams:
        name = f"{label} (IF): sin {fam}"
        if fam in ("cat", "derivada"):
            rows.append(_row(name, STATUS_NOT_APPLICABLE, reason="El Isolation Forest no recibe esa familia."))
            continue
        if fam not in present_if:
            rows.append(_row(name, STATUS_NOT_APPLICABLE,
                             reason=f"La familia '{fam}' no existe en esta corrida (p. ej. "
                                    "panel_hist/cyc requieren --panel-features)."))
            continue
        keep = np.array([fam_if[n] != fam for n in names_if])
        if keep.sum() < 2:
            rows.append(_row(name, STATUS_UNAVAILABLE, reason="Quedarían menos de 2 columnas."))
            continue
        try:
            with _step(f"ifvae_experiments.ablation[IF sin {fam}]", "Reajuste del IF sin una familia"):
                x = ctx["x_if_all"][:, np.flatnonzero(keep)]
                det = _fit_if(ctx["if_detector"], x[ctx["in_idx"]], ctx["seed"])
                high, oot_scores = _if_alerts(det, x, ctx["train_idx"], ctx["oot_idx"], ctx["threshold"])
                rho = spearmanr(prod_if_scores, oot_scores).correlation
            rows.append(_row(
                name, STATUS_EXECUTED, f"{int((~keep).sum())} columnas quitadas",
                detail=(f"{int(high.sum())} alertas de {len(high)}; Jaccard vs producción = "
                        f"{_fmt(_jaccard(prod_if_high, high))}; Spearman de rangos = {_fmt(rho)}"
                        + (f" (control: misma configuración, otra semilla = {_fmt(floors.get('iforest'))})"
                           if floors.get("iforest") is not None else ""))))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(name, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))

    names_vae = ctx["vae_feature_names"]
    present_vae = set(fam_vae.values())
    for fam in ("cat", "derivada", "panel_hist"):
        name = f"{label} (VAE): sin {fam}"
        if fam not in present_vae:
            rows.append(_row(name, STATUS_NOT_APPLICABLE,
                             reason=f"La familia '{fam}' no existe en la entrada del VAE de esta corrida."))
            continue
        keep = np.array([fam_vae[n] != fam for n in names_vae])
        if keep.sum() < 2:
            rows.append(_row(name, STATUS_UNAVAILABLE, reason="Quedarían menos de 2 columnas."))
            continue
        if not budget.take():
            rows.append(_budget_row(name, budget))
            continue
        try:
            with _step(f"ifvae_experiments.ablation[VAE sin {fam}]", "Reajuste del VAE sin una familia"):
                overrides: dict = {}
                if getattr(ctx["vae_detector"], "layout", None) is not None:
                    # Mixed VAE: drop the family's columns from the layout as well (its variables lose
                    # their heads / embeddings), not just from the matrix.
                    sub_layout, keep_pos = ctx["vae_detector"].layout.subset(
                        [c for c, k in zip(names_vae, keep) if not k])
                    overrides["layout"] = sub_layout
                    x = ctx["x_vae_all"][:, keep_pos]
                else:
                    x = ctx["x_vae_all"][:, np.flatnonzero(keep)]
                det, notes = _fit_vae(ctx["vae_detector"], x[ctx["in_idx"]], ctx["valid_local"], overrides,
                                      ctx["seed"], ctx["epoch_cap"], ctx["max_fit_rows"])
                rows.append(_vae_variant_row(name, f"{int((~keep).sum())} columnas quitadas", det,
                                             notes, x, ctx, prod_vae_high, floors.get("vae"),
                                             show_elbo=False))   # other input dimension: not comparable
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(name, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))
    for fam in ("bool", "cyc", "ratios_negocio", "num_base"):
        if fam in present_vae:
            rows.append(_row(f"{label} (VAE): sin {fam}", STATUS_NOT_REQUESTED,
                             reason="Solo se reentrena el VAE sin cat, derivada y panel_hist "
                                    "(las familias más informativas) para acotar el costo."))
    return rows


# --------------------------------------------------------------------------- #
# Families 5 and 6: rolling-origin backtest and window stability              #
# --------------------------------------------------------------------------- #
def _backtest(ctx: dict, budget: _Budget, prod_if_high: np.ndarray, prod_vae_high: np.ndarray,
              settings: dict) -> tuple[list[dict], dict]:
    from ifvae_diag.scoring import anomaly_percentile
    from src.preprocessing import fit_transform_panel, split_matrix_for_model

    label = FAMILY_LABELS["backtest"]
    period_idx, entity = ctx["period_idx"], ctx["entity_ids"]
    n_periods = int(period_idx.max()) + 1
    min_fit, n_orig = int(settings["backtest_min_fit_periods"]), int(settings["backtest_origins"])
    if n_orig < 1:
        return [_row(label, STATUS_NOT_REQUESTED,
                     reason="experiments.backtest_origins = 0: sin orígenes de backtest.")], {}
    if n_periods < min_fit + 2:
        return [_row(label, STATUS_UNAVAILABLE,
                     reason=f"El panel tiene {n_periods} periodos; se necesitan al menos {min_fit + 2} "
                            f"({min_fit} de ajuste y 2 orígenes).")], {}
    first = max(min_fit, n_periods - n_orig)
    origins = list(range(first, n_periods))
    vae_origins = set(origins[-int(settings["backtest_vae_origins"]):]) if settings["backtest_vae_origins"] > 0 else set()
    rows: list[dict] = []
    windows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for t in origins:
        fit_mask, eval_mask = period_idx < t, period_idx == t
        name = f"{label} (IF): origen periodo {t + 1}/{n_periods}"
        try:
            with _step(f"ifvae_experiments.backtest[origen {t + 1}]", "Backtest: preprocesar y reajustar al origen"):
                X, _, names = fit_transform_panel(
                    ctx["df"], ctx["schema"], _QUIET, fit_mask=fit_mask, **ctx["prep_kwargs"])[:3]
                x_if_o, _ = split_matrix_for_model(X, names, "iforest")
                x_if_o = _dense(x_if_o)
                fit_idx, ev_idx = np.flatnonzero(fit_mask), np.flatnonzero(eval_mask)
                if len(fit_idx) < _MIN_FIT_ROWS or len(ev_idx) == 0:
                    rows.append(_row(name, STATUS_UNAVAILABLE, reason="Muy pocas filas de ajuste o de evaluación."))
                    continue
                fit_rows = fit_idx
                if len(fit_rows) > settings["max_fit_rows"]:
                    fit_rows = np.sort(np.random.default_rng(ctx["seed"]).choice(
                        fit_rows, size=settings["max_fit_rows"], replace=False))
                det = _fit_if(ctx["if_detector"], x_if_o[fit_rows], ctx["seed"])
                ref = np.asarray(det.score_samples(x_if_o[fit_idx]), float)
                sc = np.asarray(det.score_samples(x_if_o[ev_idx]), float)
                high = anomaly_percentile(ref, sc) >= ctx["threshold"]
            pct = anomaly_percentile(ref, sc)
            windows[t] = (entity[ev_idx], pct)
            detail = (f"{len(fit_idx):,} filas de ajuste (periodos estrictamente anteriores); "
                      f"{int(high.sum())} alertas de {len(ev_idx)} ({high.mean():.1%}) contra la "
                      "referencia del propio origen")
            rows.append(_row(name, STATUS_EXECUTED, "ventana expansiva, un paso adelante, sin gap", detail=detail))
            if t in vae_origins:
                vname = f"{label} (VAE): origen periodo {t + 1}/{n_periods}"
                if not budget.take():
                    rows.append(_budget_row(vname, budget))
                    continue
                try:
                    with _step(f"ifvae_experiments.backtest[VAE origen {t + 1}]", "Backtest: reajustar el VAE al origen"):
                        n_val = int(ctx["n_val_periods"])
                        vmask = ((period_idx >= t - n_val) & (period_idx < t))[fit_idx]
                        overrides = {}
                        if getattr(ctx["vae_detector"], "layout", None) is not None:
                            # Vocabularies of THIS origin: learned on its fit rows only, so a level that
                            # first appears at/after the origin is UNKNOWN there.
                            from src.preprocessing.mixed_view import MixedViewBuilder

                            builder = MixedViewBuilder(ctx["vae_builder"].config)
                            Xd = builder.fit_transform(ctx["df"], X, names, fit_mask, ctx["cat_sources"])
                            overrides["layout"] = builder.layout
                        else:
                            Xd = _dense(X)
                        det_v, notes = _fit_vae(ctx["vae_detector"], Xd[fit_idx], vmask, overrides, ctx["seed"],
                                                ctx["epoch_cap"], settings["max_fit_rows"])
                        # Match production: validation guides training, but only the
                        # earlier training block calibrates the score percentile.
                        ref_idx = fit_idx[~vmask]
                        high_v = _recon_topk_alerts(
                            det_v, Xd, ref_idx, ev_idx, ctx["top_k"], ctx["threshold"],
                            ctx.get("scale_floor", 0.0), ctx.get("center", False),
                        )
                    rows.append(_row(
                        vname, STATUS_EXECUTED,
                        ("matriz base sin la columna apilada del IF" if ctx.get("derived_features")
                         else "matriz completa del VAE"),
                        detail=(f"{len(fit_idx):,} filas de ajuste; {int(high_v.sum())} alertas de "
                                f"{len(ev_idx)} ({high_v.mean():.1%}) contra la referencia del propio origen"
                                + (f"; {'; '.join(notes)}" if notes else "")
                                + ("; no comparable con producción (sin apilado, otra dimensión de entrada)"
                                   if ctx.get("derived_features")
                                   else "; refit independiente por origen (sin Jaccard contra producción)"))))
                except Exception as exc:  # noqa: BLE001
                    rows.append(_row(vname, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(name, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))
    return rows, windows


def _chance_jaccard(n: int, k: int) -> float:
    """Expected Jaccard of two independent random top-k sets out of n items."""
    inter = k * k / n
    return float(inter / (2 * k - inter)) if k > 0 else 0.0


def _window_stability(windows: dict, top_k_cfg: int) -> list[dict]:
    label = FAMILY_LABELS["window_stability"]
    keys = sorted(windows)
    if len(keys) < 2:
        return [_row(label, STATUS_UNAVAILABLE,
                     reason="Se necesitan al menos 2 orígenes de backtest ejecutados.")]

    def compare(a: tuple, b: tuple) -> Optional[tuple[int, float, float, int]]:
        sa, sb = pd.Series(a[1], index=a[0]), pd.Series(b[1], index=b[0])
        sa, sb = sa[~sa.index.duplicated()], sb[~sb.index.duplicated()]
        common = sa.index.intersection(sb.index)
        if len(common) < 5:
            return None
        va, vb = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
        k = int(max(1, min(top_k_cfg, len(common) // 2)))
        ta, tb = set(common[np.argsort(-va, kind="stable")[:k]]), set(common[np.argsort(-vb, kind="stable")[:k]])
        return len(common), float(spearmanr(va, vb).correlation), len(ta & tb) / len(ta | tb), k

    rows, rhos, jacs = [], [], []
    for a, b in zip(keys[:-1], keys[1:]):
        res = compare(windows[a], windows[b])
        name = f"{label}: periodo {a + 1} → {b + 1}"
        if res is None:
            rows.append(_row(name, STATUS_UNAVAILABLE, reason="Menos de 5 entidades en común."))
            continue
        n, rho, jac, k = res
        rhos.append(rho)
        jacs.append(jac)
        rows.append(_row(name, STATUS_EXECUTED, f"{n:,} entidades en común; top-{k}",
                         detail=(f"Spearman de rangos = {_fmt(rho)}; Jaccard top-{k} = {_fmt(jac)} "
                                 f"(azar ≈ {_fmt(_chance_jaccard(n, k))})")))
    if rhos:
        rows.append(_row(f"{label}: resumen entre ventanas consecutivas", STATUS_EXECUTED,
                         f"{len(rhos)} pares",
                         detail=(f"Spearman medio = {_fmt(float(np.mean(rhos)))} (mín. {_fmt(float(np.min(rhos)))}); "
                                 f"Jaccard medio = {_fmt(float(np.mean(jacs)))} (mín. {_fmt(float(np.min(jacs)))}); "
                                 "descriptivo: las anomalías por fila no tienen por qué persistir entre periodos")))
    if len(keys) >= 6:
        # The declared entity-level view is two adjacent three-period windows.
        # Use the latest six origins when more history was requested.
        left_keys, right_keys = keys[-6:-3], keys[-3:]

        def agg(ks: list[int]) -> tuple:
            frame = pd.concat([pd.Series(windows[k][1], index=windows[k][0]) for k in ks])
            frame = frame.groupby(level=0).max()
            return frame.index.to_numpy(), frame.to_numpy()

        res = compare(agg(left_keys), agg(right_keys))
        if res is not None:
            n, rho, jac, k = res
            rows.append(_row(f"{label}: ventanas agregadas (máximo por entidad)", STATUS_EXECUTED,
                             f"periodos {left_keys[0] + 1}-{left_keys[-1] + 1} vs "
                             f"{right_keys[0] + 1}-{right_keys[-1] + 1}",
                             detail=(f"{n:,} entidades en común; Spearman = {_fmt(rho)}; Jaccard top-{k} = "
                                     f"{_fmt(jac)} (azar ≈ {_fmt(_chance_jaccard(n, k))})")))
    return rows


# --------------------------------------------------------------------------- #
# Controls: how much does an alert set move when only the seed changes?       #
# --------------------------------------------------------------------------- #
def _controls(ctx: dict, families: Sequence[str], budget: _Budget, prod_if_high: np.ndarray,
              prod_vae_high: np.ndarray) -> tuple[dict, list[dict]]:
    """Refit the PRODUCTION configuration with another seed and compare its alert set with
    production's, using the same alert construction, caps and Jaccard as every variant row.
    That number is the reference for "is this variant different or just refit noise"."""
    floors: dict = {}
    rows: list[dict] = []
    seed = int(ctx["seed"]) + 1
    if any(f in families for f in ("ablation", "backtest")):
        try:
            with _step("ifvae_experiments.control[IF]", "Control: IF de producción con otra semilla"):
                det = _fit_if(ctx["if_detector"], ctx["x_if_all"][ctx["in_idx"]], seed)
                high, _ = _if_alerts(det, ctx["x_if_all"], ctx["train_idx"], ctx["oot_idx"], ctx["threshold"])
            floors["iforest"] = _jaccard(prod_if_high, high)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("IF control failed: %s", exc)
    if any(f in families for f in ("capacity", "beta_kl", "ablation")) and ctx["x_vae_train_rows"] >= _MIN_FIT_ROWS:
        name = "Control de ruido del VAE (misma configuración, otra semilla)"
        if not budget.take():
            rows.append(_budget_row(name, budget))
            return floors, rows
        try:
            with _step("ifvae_experiments.control[VAE]", "Control: VAE de producción con otra semilla"):
                det, notes = _fit_vae(ctx["vae_detector"], ctx["x_vae_fit"], ctx["valid_local"], {}, seed,
                                      ctx["epoch_cap"], ctx["max_fit_rows"])
                high = _recon_topk_alerts(det, ctx["x_vae_all"], ctx["train_idx"], ctx["oot_idx"],
                                          ctx["top_k"], ctx["threshold"], ctx.get("scale_floor", 0.0),
                                          ctx.get("center", False))
            floors["vae"] = _jaccard(prod_vae_high, high)
            rows.append(_row(
                name, STATUS_EXECUTED, "producción reentrenada con la semilla +1 y los mismos topes",
                detail=(f"{int(high.sum())} alertas de {len(high)}; Jaccard vs producción = "
                        f"{_fmt(floors['vae'])} (referencia: una variante con Jaccard cercano a este valor "
                        "está dentro del ruido de reentrenamiento)" + (f"; {'; '.join(notes)}" if notes else ""))))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(name, STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))
    return floors, rows


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def run_experiment_families(
    *, ctx: dict, settings: dict, frames: dict, prod_if_high: np.ndarray, prod_vae_high: np.ndarray,
    prod_if_scores: np.ndarray, noise_floor: Optional[dict] = None,
) -> list[dict]:
    """Run every enabled family and return §9 rows (see the module docstring).

    ``ctx`` -- arrays and fitted objects from ``main.py`` (see
    ``ifvae_diagnostic.run_ifvae_diagnostic_suite``); ``settings`` -- the
    ``experiments:`` block of ``configs/pipeline.yaml``; ``frames`` -- the
    ``reference`` / ``scored`` diagnostic frames and the VAE feature list.
    """
    fam_setting = settings.get("families")
    families = ALL_FAMILIES if fam_setting is None else tuple(fam_setting)   # [] = run none
    budget = _Budget(settings["vae_fit_budget"])
    ctx = dict(ctx)
    ctx.update(seed=int(settings["seed"]), epoch_cap=int(settings["epoch_cap"]),
               max_fit_rows=int(settings["max_fit_rows"]), threshold=float(ctx["threshold"]))
    ctx["x_vae_train_rows"] = len(ctx["in_idx"])
    ctx["x_vae_fit"] = ctx["x_vae_all"][ctx["in_idx"]]
    ctx["valid_local"] = ctx["valid_local"]
    # Families of the VAE INPUT columns (ablation drops columns) and of its scored VARIABLES (loss by type).
    fam_vae = feature_families(ctx["vae_feature_names"], ctx.get("derived_features", ()), ctx.get("ratio_names", ()))
    fam_vars = feature_families(ctx["vae_variable_names"], ctx.get("derived_features", ()), ctx.get("ratio_names", ()))
    fam_if = feature_families(ctx["if_feature_names"], (), ctx.get("ratio_names", ()))
    rows: list[dict] = []
    floors, control_rows = _controls(ctx, families, budget, prod_if_high, prod_vae_high)
    rows.extend(control_rows)

    def disabled(fam: str) -> dict:
        return _row(FAMILY_LABELS[fam], STATUS_NOT_REQUESTED,
                    reason="Desactivada en configs/pipeline.yaml (experiments.families).")

    for fam in ("capacity", "beta_kl"):
        if fam not in families:
            rows.append(disabled(fam))
            continue
        base = ctx["vae_detector"]
        empty_capacity = fam == "capacity" and settings.get("capacity_grid") == ()
        empty_beta_kl = (fam == "beta_kl" and settings.get("beta_grid") == ()
                         and settings.get("kl_anneal_grid") == ())
        if empty_capacity or empty_beta_kl:
            rows.append(_row(FAMILY_LABELS[fam], STATUS_NOT_REQUESTED,
                             reason="Malla vacía en la configuración: el barrido se apagó a propósito."))
            continue
        n_limit = (len(ctx["vae_variable_names"]) if getattr(base, "layout", None) is not None
                   else len(ctx["vae_feature_names"]))
        pts = (_capacity_points(base, n_limit, settings.get("capacity_grid"))
               if fam == "capacity"
               else _beta_kl_points(base, settings.get("beta_grid"), settings.get("kl_anneal_grid")))
        rows.extend(_run_vae_sweep(fam, pts, ctx, budget, prod_vae_high, floors.get("vae")))

    if "loss_by_type" in families:
        try:
            lbt_rows, lbt_stats = _loss_by_type(frames["reference"], frames["scored"], frames["features"],
                                                fam_vars, ctx, prod_vae_high)
            rows.extend(lbt_rows)
            rows.extend(_representation_control(lbt_stats, ctx, budget, prod_vae_high))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(FAMILY_LABELS["loss_by_type"], STATUS_FAILED, reason=f"{type(exc).__name__}: {exc}"))
    else:
        rows.append(disabled("loss_by_type"))

    if "ablation" in families:
        rows.extend(_ablation(ctx, fam_if, fam_vae, budget, prod_if_high, prod_if_scores, prod_vae_high, floors))
    else:
        rows.append(disabled("ablation"))

    windows: dict = {}
    if "backtest" in families:
        bt_rows, windows = _backtest(ctx, budget, prod_if_high, prod_vae_high, settings)
        rows.extend(bt_rows)
    else:
        rows.append(disabled("backtest"))

    if "window_stability" in families:
        if "backtest" not in families:
            rows.append(_row(FAMILY_LABELS["window_stability"], STATUS_UNAVAILABLE,
                             reason="Reutiliza los scores del backtest, que está desactivado."))
        else:
            rows.extend(_window_stability(windows, int(ctx.get("alert_k", ctx["top_k"]))))
    else:
        rows.append(disabled("window_stability"))
    _LOG.info("§9 experiments: %d row(s); VAE retrains used %d of %d",
              len(rows), budget.total - budget.left, budget.total)
    return rows
