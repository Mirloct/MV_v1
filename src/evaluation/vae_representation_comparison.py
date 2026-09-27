"""Controlled A/B comparison of the two ways the VAE can see categorical variables.

A. **one-hot** -- the original MLP VAE over the one-hot-expanded matrix.
B. **embedding** -- the mixed-type VAE (one index per variable, learned embeddings, per-variable
   reconstruction).

Everything that is not the representation is held fixed: the same panel, the same chronological
splits, the same seeds, the same training rows and validation months, the same epoch budget, the same KL
ramp and architecture hyper-parameters, the same alert budget (top-K rows of the OOT window), and *no*
IF stacking (so the two arms differ only in how categoricals are represented). The reconstruction
losses are **never compared** with each other: they live on different scales (MSE over columns vs
Huber/BCE/NLL over variables); the winner is judged on detection quality, stability, how the top-k slots
are distributed, and cost.

Metrics per arm (mean over seeds, with the seed-to-seed sd):

* detection: average precision against the synthetic ground truth (a stand-in for reviewed labels,
  which need gate 4.5) and ``tail_separation`` of the OOT scores -- for the production score and for the
  diagnostic's ``recon_topk``;
* stability: alert-set Jaccard across seeds; rank agreement (Spearman) and top-K Jaccard between
  consecutive windows (the model scores each month of the test+OOT windows);
* alert overlap A vs B (same seed);
* where the top-k slots go: share of slots on categorical terms vs the share expected from the unit count
  (columns for A, original variables for B), the per-variable slot rate, and its rank correlation with
  the variable's cardinality (a representation whose top-k depends on cardinality has a large |rho|);
* latent health (active units), wall time and peak memory;
* unseen categories: rows whose categorical value is replaced by a level never seen in training -- the
  score lift and whether the perturbed variable becomes a top contributor.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

__all__ = ["ComparisonConfig", "run_comparison", "acceptance_criteria", "render_markdown"]

_LOG = logging.getLogger("modelo.vae_comparison")


class ComparisonConfig:
    def __init__(self, n_individuals: int = 400, n_periods: int = 14, data_seed: int = 42,
                 seeds: Sequence[int] = (11, 23, 37), epochs: int = 12, latent_dim: int = 8,
                 hidden_dim: int = 64, n_layers: int = 2, batch_size: int = 256,
                 alert_fraction: float = 0.05, top_k: int = 5, min_frequency: float = 0.001,
                 n_unseen_rows: int = 200, mixed_config: Any = None):
        self.n_individuals, self.n_periods, self.data_seed = n_individuals, n_periods, data_seed
        self.seeds, self.epochs, self.latent_dim = tuple(seeds), epochs, latent_dim
        self.hidden_dim, self.n_layers, self.batch_size = hidden_dim, n_layers, batch_size
        self.alert_fraction, self.top_k, self.min_frequency = alert_fraction, top_k, min_frequency
        self.n_unseen_rows = n_unseen_rows
        self.mixed_config = mixed_config


def _peak_rss_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:  # noqa: BLE001 - optional
        return None


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else float("nan")


def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(scores), dtype=bool)
    m[np.argsort(-scores, kind="stable")[:k]] = True
    return m


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))


def _prepare(cfg: ComparisonConfig, workdir: str) -> dict:
    """Panel, splits, the one-hot matrix and the mixed view, exactly like ``main.py`` builds them."""
    from src.data import load_or_generate_panel
    from src.evaluation import chronological_split, load_ground_truth_labels
    from src.preprocessing import fit_transform_panel
    from src.preprocessing.mixed_view import MixedViewBuilder, MixedViewConfig, categorical_sources

    df, schema = load_or_generate_panel(data_path=os.path.join(workdir, "panel.csv"),
                                        n_individuals=cfg.n_individuals, n_periods=cfg.n_periods,
                                        seed=cfg.data_seed)
    split = chronological_split(df, time_col=schema.time_col, n_val_periods=2, n_test_periods=3, n_oot_periods=3)
    prep = dict(numeric_transform="yeo-johnson", categorical_encoding="onehot", rare_min_frequency=cfg.min_frequency,
                impute_numeric="zero", add_panel_features=False, random_state=cfg.data_seed)
    X, keys, names, pipe = fit_transform_panel(df, schema, fit_mask=split.train_mask, return_pipeline=True, **prep)
    x_onehot = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    builder = MixedViewBuilder(MixedViewConfig(min_frequency=cfg.min_frequency))
    x_mixed = builder.fit_transform(df, X, names, split.train_mask, categorical_sources(df, schema))
    y = load_ground_truth_labels(schema, keys).astype(int)
    in_mask = split.train_mask | split.val_mask
    return dict(df=df, schema=schema, split=split, keys=keys, names=names, pipe=pipe, x_onehot=x_onehot,
                x_mixed=x_mixed, builder=builder, y=y, in_mask=in_mask, valid_local=split.val_mask[in_mask],
                sources=categorical_sources(df, schema))


def _fit(arm: str, data: dict, cfg: ComparisonConfig, seed: int, workdir: str):
    from src.models import MixedVAEConfig, VAEDetector

    kl = min(10, max(1, cfg.epochs // 2))
    common = dict(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers, epochs=cfg.epochs,
                  batch_size=cfg.batch_size, kl_anneal_epochs=kl, early_stopping_patience=None, random_state=seed)
    if arm == "embedding":
        det = VAEDetector(layout=data["builder"].layout, mixed_config=cfg.mixed_config or MixedVAEConfig(), **common)
        x = data["x_mixed"]
    else:
        det = VAEDetector(**common)
        x = data["x_onehot"]
    t0 = time.perf_counter()
    rss0 = _peak_rss_mb()
    det.fit(x[data["in_mask"]], valid_mask=data["valid_local"],
            checkpoint_dir=os.path.join(workdir, f"ck_{arm}_{seed}"), resume=False)
    return det, x, time.perf_counter() - t0, (None if rss0 is None else max(0.0, (_peak_rss_mb() or rss0) - rss0))


def _residuals(arm: str, det: Any, x: np.ndarray) -> np.ndarray:
    if arm == "embedding":
        return det.contributions(x)
    from src.evaluation.ifvae_diagnostic import _vae_forward

    return np.abs(x - _vae_forward(det, x)[2]).astype(float)


#: The two contribution normalisations applied to BOTH arms, so the representation is never confounded with
#: the normalisation: ``historical`` = |residual| / MAD (what the diagnostic did for the one-hot MLP) and
#: ``centered`` = centred at the reference median + scale floor (what the mixed VAE uses in the pipeline).
NORMS = {"historical": {"scale_floor_fraction": 0.0, "center": False},
         "centered": {"scale_floor_fraction": 0.1, "center": True}}


def _recon_topk(ref_res: np.ndarray, res: np.ndarray, top_k: int, norm: str) -> np.ndarray:
    from ifvae_diag.scoring import reconstruction_scores

    kw = NORMS[norm]
    return reconstruction_scores(ref_res, res, top_k, kw["scale_floor_fraction"], kw["center"])["recon_topk"].to_numpy()


def _slot_stats(arm: str, contrib_norm: np.ndarray, col_names: Sequence[str], sources: Sequence[str],
                cardinalities: dict, top_k: int) -> dict:
    """Categorical share of the top-k slots vs the share expected from the unit count, per-variable slot
    rates and their rank correlation with the variable's cardinality."""
    from src.preprocessing.pipeline import group_name_by_source

    n, d = contrib_norm.shape
    kk = min(top_k, d)
    slots = np.argpartition(contrib_norm, d - kk, axis=1)[:, -kk:]
    is_cat = np.array([str(c).startswith("cat__") for c in col_names])
    share_slots = float(np.isin(slots, np.flatnonzero(is_cat)).mean())
    share_units = float(is_cat.mean())
    group = [group_name_by_source(str(c), sources) if is_cat[i] else str(c) for i, c in enumerate(col_names)]
    rate: dict[str, float] = {}
    for i, g in enumerate(group):
        if is_cat[i]:
            rate[g] = rate.get(g, 0.0) + float((slots == i).sum()) / n      # slots the variable occupies per row
    cats = [c for c in rate if c in cardinalities]
    rho = float(spearmanr([cardinalities[c] for c in cats], [rate[c] for c in cats]).correlation) if len(cats) >= 3 else float("nan")
    rates = np.array([rate[c] for c in cats]) if cats else np.array([])
    return {"cat_share_of_slots": share_slots, "cat_share_of_units": share_units,
            "over_representation": share_slots / share_units if share_units else float("nan"),
            "slot_rate_by_variable": rate, "rho_slot_rate_vs_cardinality": rho,
            "slot_rate_spread": float(rates.max() / max(rates.min(), 1e-9)) if len(rates) else float("nan")}


def _cardinalities(data: dict) -> dict:
    df = data["df"]
    tr = data["split"].train_mask
    return {s: int(df.loc[tr, s].nunique()) for s in data["sources"]}


def _window_stability(scores: np.ndarray, entities: np.ndarray, periods: np.ndarray, k: int) -> tuple[float, float]:
    """Mean Spearman and top-k Jaccard of consecutive periods' entity scores."""
    rhos, jacs = [], []
    ordered = sorted(set(periods.tolist()))
    for a, b in zip(ordered[:-1], ordered[1:]):
        sa = pd.Series(scores[periods == a], index=entities[periods == a])
        sb = pd.Series(scores[periods == b], index=entities[periods == b])
        common = sa.index.intersection(sb.index)
        if len(common) < 10:
            continue
        va, vb = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
        rhos.append(float(spearmanr(va, vb).correlation))
        kk = min(k, len(common) // 2)
        jacs.append(_jaccard(_topk_mask(va, kk), _topk_mask(vb, kk)))
    return (float(np.nanmean(rhos)) if rhos else float("nan"), float(np.nanmean(jacs)) if jacs else float("nan"))


def _unseen_probe(arm: str, det: Any, data: dict, cfg: ComparisonConfig, seed: int) -> dict:
    """Replace one categorical value of OOT rows by a level never seen in training."""
    from ifvae_diag.scoring import residual_contributions  # noqa: F401 - import guard for the suite

    df, split, pipe = data["df"], data["split"], data["pipe"]
    src = data["sources"][0]
    rows = np.flatnonzero(split.oot_mask)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows, size=min(cfg.n_unseen_rows, len(rows)), replace=False)
    frame = df.iloc[pick].copy()
    base = frame.copy()
    frame[src] = "__level_never_seen_in_training__"
    out: dict = {"variable": src}
    try:
        if arm == "embedding":
            xb, xp = data["builder"].transform(base, pipe.transform(base)), data["builder"].transform(frame, pipe.transform(frame))
        else:
            xb, xp = np.asarray(_dense(pipe.transform(base))), np.asarray(_dense(pipe.transform(frame)))
        s0, s1 = det.score_samples(xb), det.score_samples(xp)
        out.update(ok=True, mean_score_before=float(s0.mean()), mean_score_after=float(s1.mean()),
                   score_lift=float((s1 - s0).mean()), share_rows_higher=float((s1 > s0).mean()))
        if arm == "embedding":
            names = det.variable_names
            c1 = det.contributions(xp)
            out["perturbed_variable_mean_rank"] = float(np.mean([(-c1[i]).argsort().argsort()[names.index(f"cat__{src}")] + 1
                                                                 for i in range(len(c1))]))
            out["token_used"] = "UNKNOWN (explicit)"
        else:
            out["token_used"] = "infrequent bucket / all-zero one-hot (implicit)"
    except Exception as exc:  # noqa: BLE001 - a failure here IS a finding
        out.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return out


def _dense(x: Any) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _arm_seed(arm: str, det: Any, x: np.ndarray, data: dict, cfg: ComparisonConfig, cards: dict) -> dict:
    from ifvae_diag.scoring import residual_contributions

    from src.models.iforest import _tail_separation

    split, y = data["split"], data["y"]
    tr, oot, ev = split.train_mask, split.oot_mask, (split.test_mask | split.oot_mask)
    ref_res, oot_res = _residuals(arm, det, x[tr]), _residuals(arm, det, x[oot])
    raw = det.score_samples(x[oot])
    K = max(5, int(round(cfg.alert_fraction * int(oot.sum()))))
    names = list(det.variable_names) if arm == "embedding" else list(data["names"])
    y_oot = y[oot]
    have_pos = 0 < y_oot.sum() < len(y_oot)
    ev_idx = np.flatnonzero(ev)
    ev_scores = det.score_samples(x[ev_idx])
    periods = data["keys"][data["schema"].time_col].astype(str).to_numpy()[ev_idx]
    entities = data["keys"][data["schema"].entity_col].to_numpy()[ev_idx]
    ev_res = _residuals(arm, det, x[ev_idx])
    wk = max(5, K // max(1, len(set(periods.tolist()))))
    is_num = np.array([not str(c).startswith("cat__") for c in names])
    diag = det.latent_diagnostics(x[tr][:20000])
    out: dict = {
        "K": K, "n_reference_rows": int(tr.sum()),
        "ap_raw": float(average_precision_score(y_oot, raw)) if have_pos else float("nan"),
        "tail_raw": float(_tail_separation(raw)), "alerts_raw": _topk_mask(raw, K),
        "win_spearman_raw": _window_stability(ev_scores, entities, periods, wk)[0],
        "win_jaccard_raw": _window_stability(ev_scores, entities, periods, wk)[1],
        "active_units": int(diag["active_units"]), "latent_dim": int(diag["latent_dim"]), "mean_kl": float(diag["mean_kl"]),
        "n_contributions": int(oot_res.shape[1]), "norm": {},
    }
    for norm, kw in NORMS.items():
        score = _recon_topk(ref_res, oot_res, cfg.top_k, norm)
        ev_topk = _recon_topk(ref_res, ev_res, cfg.top_k, norm)
        ev_topk_num = _recon_topk(ref_res[:, is_num], ev_res[:, is_num], cfg.top_k, norm)
        C_oot = residual_contributions(ref_res, oot_res, kw["scale_floor_fraction"], kw["center"])
        # Slot statistics: on OOT (may hold real drift) and on the REFERENCE block (train: no drift, no injected
        # anomaly), where a dependence on cardinality would be pure bias.
        s_oot = _slot_stats(arm, C_oot, names, data["sources"], cards, cfg.top_k)
        s_ref = _slot_stats(arm, residual_contributions(ref_res, ref_res, kw["scale_floor_fraction"], kw["center"]),
                            names, data["sources"], cards, cfg.top_k)
        out["norm"][norm] = {
            "ap": float(average_precision_score(y_oot, score)) if have_pos else float("nan"),
            "tail": float(_tail_separation(score)), "alerts": _topk_mask(score, K),
            "win_spearman": _window_stability(ev_topk, entities, periods, wk)[0],
            "win_jaccard": _window_stability(ev_topk, entities, periods, wk)[1],
            "win_spearman_numeric_only": _window_stability(ev_topk_num, entities, periods, wk)[0],
            "cat_share_of_slots": s_oot["cat_share_of_slots"], "cat_share_of_units": s_oot["cat_share_of_units"],
            "over_representation": s_oot["over_representation"], "slot_rate_by_variable": s_oot["slot_rate_by_variable"],
            "rho_oot": s_oot["rho_slot_rate_vs_cardinality"],
            "rho_reference": s_ref["rho_slot_rate_vs_cardinality"], "spread_reference": s_ref["slot_rate_spread"],
            "share_reference": s_ref["cat_share_of_slots"],
        }
    return out


def run_comparison(cfg: Optional[ComparisonConfig] = None, workdir: Optional[str] = None) -> dict:
    """Run both arms over every seed and return the measurements (see the module docstring)."""
    cfg = cfg or ComparisonConfig()
    logging.getLogger("modelo").setLevel(logging.ERROR)
    own = workdir is None
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True) if own else None
    workdir = tmp.name if own else workdir
    prev_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        data = _prepare(cfg, workdir)
        cards = _cardinalities(data)
        arms: dict[str, list[dict]] = {"onehot": [], "embedding": []}
        dets: dict[tuple, Any] = {}
        for seed in cfg.seeds:
            for arm in ("onehot", "embedding"):
                det, x, secs, mem = _fit(arm, data, cfg, seed, workdir)
                res = _arm_seed(arm, det, x, data, cfg, cards)
                res.update(seed=seed, fit_seconds=secs, fit_rss_delta_mb=mem,
                           n_input_columns=int(x.shape[1]),
                           unseen=_unseen_probe(arm, det, data, cfg, seed))
                arms[arm].append(res)
                dets[(arm, seed)] = det
        summary = _summarise(arms, cfg, data, cards)
        summary["checkpoint_rejection"] = _checkpoint_rejection(dets, workdir)
        lay = data["builder"].layout
        summary["layout"] = {"columns": lay.n_columns, "onehot_columns": int(data["x_onehot"].shape[1]),
                             "categorical_variables": len(lay.names("cat")),
                             "scored_variables": len(lay.variables),
                             "numeric_and_binary": len(lay.names("num")) + len(lay.names("bool")),
                             "onehot_in_layout": int(sum(c.startswith("cat__") and r != "cat" for c, r in zip(lay.columns, lay.roles))),
                             "cardinalities": cards, "fingerprint": lay.fingerprint()}
        summary["calibration_leak_check"] = _leak_check(data)
        summary["config"] = {k: v for k, v in vars(cfg).items() if k != "mixed_config"}
        return summary
    finally:
        os.chdir(prev_cwd)
        if tmp is not None:
            tmp.cleanup()


def _summarise(arms: dict, cfg: ComparisonConfig, data: dict, cards: dict) -> dict:
    out: dict = {"arms": {}}
    flat = ("ap_raw", "tail_raw", "win_spearman_raw", "win_jaccard_raw", "active_units", "latent_dim", "mean_kl",
            "fit_seconds", "fit_rss_delta_mb", "n_input_columns", "n_contributions", "n_reference_rows")
    per_norm = ("ap", "tail", "win_spearman", "win_jaccard", "win_spearman_numeric_only", "cat_share_of_slots",
                "cat_share_of_units", "over_representation", "rho_oot", "rho_reference", "spread_reference",
                "share_reference")
    for arm, runs in arms.items():
        m: dict = {k: _mean_sd([r.get(k) for r in runs]) for k in flat}
        pairs = [_jaccard(a["alerts_raw"], b["alerts_raw"]) for i, a in enumerate(runs) for b in runs[i + 1:]]
        m["seed_stability_jaccard_raw"] = _mean_sd(pairs)
        m["norm"] = {}
        for norm in NORMS:
            n = {k: _mean_sd([r["norm"][norm][k] for r in runs]) for k in per_norm}
            n["seed_stability_jaccard"] = _mean_sd([_jaccard(a["norm"][norm]["alerts"], b["norm"][norm]["alerts"])
                                                    for i, a in enumerate(runs) for b in runs[i + 1:]])
            n["slot_rate_by_variable"] = {v: float(np.mean([r["norm"][norm]["slot_rate_by_variable"].get(v, 0.0) for r in runs]))
                                          for v in runs[0]["norm"][norm]["slot_rate_by_variable"]}
            m["norm"][norm] = n
        m["unseen"] = [r["unseen"] for r in runs]
        out["arms"][arm] = m
    out["alert_jaccard_A_vs_B"] = {
        "raw": _mean_sd([_jaccard(a["alerts_raw"], b["alerts_raw"]) for a, b in zip(arms["onehot"], arms["embedding"])]),
        "recon_topk (pipeline: A historical, B centred)": _mean_sd(
            [_jaccard(a["norm"]["historical"]["alerts"], b["norm"]["centered"]["alerts"]) for a, b in zip(arms["onehot"], arms["embedding"])]),
    }
    out["alert_budget_rows"] = arms["onehot"][0]["K"]
    return out


def _checkpoint_rejection(dets: dict, workdir: str) -> dict:
    from src.models import IncompatibleCheckpointError, VAEDetector

    seed = next(s for (a, s) in dets if a == "embedding")
    res = {}
    for arm, expect in (("onehot", "mixed_v1"), ("embedding", "onehot_v1")):
        path = dets[(arm, seed)].save(os.path.join(workdir, f"probe_{arm}.pt"))
        try:
            VAEDetector.load(path, expect_architecture=expect)
            res[f"{arm}_payload_as_{expect}"] = "LOADED (defect)"
        except IncompatibleCheckpointError:
            res[f"{arm}_payload_as_{expect}"] = "rejected"
    return res


def _leak_check(data: dict) -> dict:
    """Vocabularies come from the train rows only; the percentile / median / MAD reference of every score is the
    train block (its measured row count is compared with ``train_mask.sum()`` in ``acceptance_criteria``)."""
    df, split = data["df"], data["split"]
    spec_ok = True
    for spec in data["builder"].layout.cat_specs():
        train_levels = set(df.loc[split.train_mask, spec.name].dropna().astype(str))
        spec_ok &= set(spec.vocabulary) <= train_levels
    return {"vocabularies_subset_of_train_levels": bool(spec_ok), "train_rows": int(split.train_mask.sum())}


def acceptance_criteria(summary: dict) -> list[dict]:
    """Evaluate the objective criteria that the measurements can decide (the rest are covered by tests).

    Every recon_topk comparison is made on BOTH pairings: the *pipeline-as-is* pair (A: historical
    ``|residual|/MAD``, B: centred + floor, what each arm would actually run) and the *like-for-like* pair (both
    centred + floor), so the representation is never confounded with the normalisation. Window stability is judged
    literally (A complete) and like-for-like (numeric variables only). Each item: ``{"criterion", "passed", "evidence"}``.
    """
    A, B = summary["arms"]["onehot"], summary["arms"]["embedding"]
    items: list[dict] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        items.append({"criterion": name, "passed": bool(passed), "evidence": evidence})

    lay = summary["layout"]
    add("Ninguna columna one-hot entra al VAE (arm B)", lay["columns"] < lay["onehot_columns"] and lay["onehot_in_layout"] == 0,
        f"{lay['columns']} columnas de entrada frente a {lay['onehot_columns']} one-hot; columnas `cat__*` con rol distinto de índice: "
        f"{lay['onehot_in_layout']}")
    ncat = lay["categorical_variables"]
    add("Cada categórica produce una sola contribución al score",
        B["n_contributions"][0] == lay["scored_variables"] and lay["scored_variables"] == lay["numeric_and_binary"] + ncat,
        f"{ncat} variables categóricas -> {ncat} contribuciones; contribuciones totales por fila = {B['n_contributions'][0]:.0f} "
        f"(= variables originales {lay['scored_variables']}); el one-hot tiene {A['n_contributions'][0]:.0f} términos")
    unseen = B["unseen"]
    add("Categorías nuevas no causan errores", all(u.get("ok") for u in unseen),
        f"{sum(bool(u.get('ok')) for u in unseen)}/{len(unseen)} corridas con nivel nunca visto sin error; "
        f"elevación media del score {np.nanmean([u.get('score_lift', np.nan) for u in unseen]):.4f}")
    rej = summary["checkpoint_rejection"]
    add("Los checkpoints incompatibles se rechazan", all(v == "rejected" for v in rej.values()), str(rej))
    leak = summary["calibration_leak_check"]
    ref_ok = B["n_reference_rows"][0] == leak["train_rows"] and A["n_reference_rows"][0] == leak["train_rows"]
    add("Sin fuga temporal en la calibración", leak["vocabularies_subset_of_train_levels"] and ref_ok,
        f"vocabularios ⊂ niveles de train; filas de referencia de la normalización = {B['n_reference_rows'][0]:.0f} "
        f"(train = {leak['train_rows']})")

    hA, cA, cB = A["norm"]["historical"], A["norm"]["centered"], B["norm"]["centered"]
    hB = B["norm"]["historical"]
    # -- cardinality: like-for-like (both centred) ------------------------------------------------------------
    sp_a, sp_b = cA["spread_reference"][0], cB["spread_reference"][0]
    add("El top-k categórico deja de depender de la cardinalidad (referencia sin deriva; MISMA normalización centrada en A y B)",
        bool(sp_b <= 1.5 and sp_b <= sp_a) and cB["cat_share_of_slots"][0] < cA["cat_share_of_slots"][0],
        f"dispersión máx/mín de la tasa de lugares entre categóricas: A={sp_a:.2f}x B={sp_b:.2f}x (límite 1.5x y B<=A); lugares top-k "
        f"categóricos: A={cA['cat_share_of_slots'][0]:.0%} B={cB['cat_share_of_slots'][0]:.0%} (peso de las categóricas entre las variables "
        f"originales: {cB['cat_share_of_units'][0]:.0%}). Referencia: normalización histórica A={hA['spread_reference'][0]:.2f}x B={hB['spread_reference'][0]:.2f}x")
    # -- detection: both pairings + the production score ---------------------------------------------------------
    def tol_of(a_pair, key):
        return max(a_pair[1], 0.05 * abs(a_pair[0]) if np.isfinite(a_pair[0]) else 0.0, 0.005 if key == "ap" else 0.0)

    for key, label in (("ap", "AP"), ("tail", "tail separation")):
        for pair_name, a_n, b_n in (("pipeline actual (A histórica, B centrada)", hA, cB), ("misma normalización centrada", cA, cB)):
            a, b = a_n[key], b_n[key]
            tol = tol_of(a, key)
            add(f"No empeora materialmente {label} del score de alertas recon_topk — {pair_name}",
                bool(b[0] >= a[0] - tol) if np.isfinite(a[0]) and np.isfinite(b[0]) else True,
                f"A={a[0]:.4f}±{a[1]:.4f}  B={b[0]:.4f}±{b[1]:.4f} (tolerancia {tol:.4f})")
    for key, label in (("ap_raw", "AP"), ("tail_raw", "tail separation")):
        a, b = A[key], B[key]
        tol = tol_of(a, "ap" if key == "ap_raw" else "tail")
        add(f"No empeora materialmente {label} del score de producción del Excel",
            bool(b[0] >= a[0] - tol) if np.isfinite(a[0]) and np.isfinite(b[0]) else True,
            f"A={a[0]:.4f}±{a[1]:.4f}  B={b[0]:.4f}±{b[1]:.4f} (tolerancia {tol:.4f})")
    # -- stability ------------------------------------------------------------------------------------------------
    for pair_name, a_n, b_n in (("pipeline actual", hA, cB), ("misma normalización", cA, cB)):
        a, b = a_n["seed_stability_jaccard"], b_n["seed_stability_jaccard"]
        floor = 0.8 * a[0]
        add(f"La estabilidad entre semillas no cae bajo el piso del control — {pair_name}", bool(b[0] >= floor),
            f"A={a[0]:.3f}  B={b[0]:.3f}  (piso = 80 % del control = {floor:.3f})")
    for label, a_val, b_val in (
        ("Spearman, control literal: A completo (pipeline actual)", hA["win_spearman"], cB["win_spearman"]),
        ("Jaccard top-K, control literal: A completo (pipeline actual)", hA["win_jaccard"], cB["win_jaccard"]),
        ("Spearman, control equivalente: ambos solo variables numéricas", cA["win_spearman_numeric_only"], cB["win_spearman_numeric_only"]),
    ):
        floor = 0.8 * a_val[0]
        add(f"La estabilidad entre ventanas no cae bajo el piso del control — {label}", bool(b_val[0] >= floor),
            f"A={a_val[0]:.3f}  B={b_val[0]:.3f}  (piso = 80 % del control = {floor:.3f})"
            + ("; el control literal de A está inflado por atributos categóricos que no cambian entre meses" if "literal" in label else ""))
    return items


def promotion_decision(criteria: list[dict]) -> dict:
    """Embedding becomes the production default only when EVERY measurable criterion passes."""
    failed = [c["criterion"] for c in criteria if not c["passed"]]
    return {"promote": not failed, "failed_criteria": failed}


def _fmt(pair: tuple, nd: int = 3) -> str:
    m, s = pair
    return "n/d" if not np.isfinite(m) else (f"{m:.{nd}f} ± {s:.{nd}f}" if np.isfinite(s) else f"{m:.{nd}f}")


def render_markdown(summary: dict, criteria: Optional[list[dict]] = None) -> str:
    A, B = summary["arms"]["onehot"], summary["arms"]["embedding"]
    cfg = summary["config"]
    hA, cA = A["norm"]["historical"], A["norm"]["centered"]
    hB, cB = B["norm"]["historical"], B["norm"]["centered"]
    lines = [
        "# Comparación controlada VAE one-hot (A) vs embeddings (B)",
        "",
        f"Mismo panel sintético ({cfg['n_individuals']} entidades × {cfg['n_periods']} periodos), mismos splits temporales, "
        f"semillas {list(cfg['seeds'])}, {cfg['epochs']} épocas, misma rampa KL, sin stacking, presupuesto de alertas = "
        f"{summary['alert_budget_rows']} filas OOT. **No se compara la pérdida de reconstrucción** (escalas distintas). "
        "El score de alertas `recon_topk` se calcula con **dos normalizaciones aplicadas a ambos brazos** (histórica = |residuo|/MAD; "
        "centrada = centrada en la mediana de referencia con piso), para no confundir la representación con la normalización; "
        "el pipeline actual corresponde a A histórica y B centrada.",
        "",
        "| Métrica | A: one-hot | B: embeddings |", "|---|---|---|",
        f"| Columnas de entrada del VAE | {_fmt(A['n_input_columns'])} | {_fmt(B['n_input_columns'])} |",
        f"| Términos de reconstrucción por fila | {_fmt(A['n_contributions'])} | {_fmt(B['n_contributions'])} |",
        f"| AP vs verdad sintética — score de producción (Excel) | {_fmt(A['ap_raw'])} | {_fmt(B['ap_raw'])} |",
        f"| Tail separation — score de producción (Excel) | {_fmt(A['tail_raw'])} | {_fmt(B['tail_raw'])} |",
    ]
    for label, key in (("AP recon_topk", "ap"), ("Tail separation recon_topk", "tail"),
                       ("Estabilidad entre semillas (Jaccard alertas recon_topk)", "seed_stability_jaccard"),
                       ("Estabilidad entre ventanas — Spearman (recon_topk)", "win_spearman"),
                       ("Estabilidad entre ventanas — Jaccard top-K (recon_topk)", "win_jaccard"),
                       ("Estabilidad entre ventanas — Spearman, solo variables numéricas", "win_spearman_numeric_only"),
                       ("Lugares top-k ocupados por categóricas (OOT)", "cat_share_of_slots"),
                       ("Dispersión máx/mín de la tasa de lugares entre categóricas (referencia sin deriva)", "spread_reference")):
        lines.append(f"| {label} — normalización histórica | {_fmt(hA[key])} | {_fmt(hB[key])} |")
        lines.append(f"| {label} — normalización centrada | {_fmt(cA[key])} | {_fmt(cB[key])} |")
    lines += [
        f"| Peso de las categóricas entre las unidades (columnas en A, variables en B) | {_fmt(cA['cat_share_of_units'])} | {_fmt(cB['cat_share_of_units'])} |",
        f"| Estabilidad entre semillas (Jaccard alertas, score de producción) | {_fmt(A['seed_stability_jaccard_raw'])} | {_fmt(B['seed_stability_jaccard_raw'])} |",
        f"| Unidades latentes activas | {_fmt(A['active_units'])} | {_fmt(B['active_units'])} |",
        f"| Tiempo de ajuste (s) | {_fmt(A['fit_seconds'])} | {_fmt(B['fit_seconds'])} |",
        f"| Memoria pico Δ RSS (MB) | {_fmt(A['fit_rss_delta_mb'])} | {_fmt(B['fit_rss_delta_mb'])} |",
        "",
        "Jaccard de alertas A vs B (misma semilla): " + "; ".join(f"{k}: {_fmt(v)}" for k, v in summary["alert_jaccard_A_vs_B"].items()) + ".",
        "", "**Categorías no vistas** (un nivel que el entrenamiento nunca vio en una variable categórica de filas OOT):", "",
    ]
    for arm, label in (("onehot", "A"), ("embedding", "B")):
        u = summary["arms"][arm]["unseen"]
        ok = sum(bool(x.get("ok")) for x in u)
        lift = np.nanmean([x.get("score_lift", np.nan) for x in u])
        lines.append(f"- {label}: {ok}/{len(u)} sin error; elevación media del score = {lift:.4f}; tratamiento: {u[0].get('token_used', 'n/d')}.")
    if criteria:
        lines += ["", "## Criterios de aceptación evaluables con estas mediciones", "", "| Criterio | Resultado | Evidencia |", "|---|---|---|"]
        for c in criteria:
            lines.append(f"| {c['criterion']} | {'✅ cumple' if c['passed'] else '❌ NO cumple'} | {c['evidence']} |")
        dec = promotion_decision(criteria)
        lines += ["", "**Decisión de promoción a default:** "
                  + ("todos los criterios medibles se cumplen." if dec["promote"] else
                     f"NO se promueve; {len(dec['failed_criteria'])} criterio(s) no se cumplen: " + "; ".join(dec["failed_criteria"]) + ".")]
    lines += ["", "Datos sintéticos: valida el cableado y la metodología, no el desempeño sobre datos reales."]
    return "\n".join(lines)
