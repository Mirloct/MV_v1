"""Validation suite for `src.interpretability` and `src.reporting`.

Reuses the conftest session sandbox. Tiny models; all artifacts under tmp_path.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from src.data import load_or_generate_panel
from src.preprocessing import fit_transform_panel
from src.models import IsolationForestDetector, VAEDetector
from src.evaluation import oot_split, load_ground_truth_labels, supervised_metrics, \
    build_scored_frame, export_oot_top_decile
from src.interpretability import (
    shap_summary_iforest,
    path_length_analysis,
    latent_space_plot,
    reconstruction_error_by_feature,
)
from src.reporting import build_report

N_ENTITIES = 250
N_PERIODS = 5
SEED = 20260724


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    dest = tmp_path_factory.mktemp("ir_panel")
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
    ifd = IsolationForestDetector(n_estimators=120, random_state=0).fit(X[in_mask])
    vae = VAEDetector(latent_dim=4, hidden_dim=16, n_layers=1, epochs=4,
                      batch_size=128, random_state=0)
    vae.fit(X[in_mask], checkpoint_dir=str(dest / "vck"), resume=False)
    return dict(df=df, schema=schema, X=X, keys=keys, names=names, y=y,
                oot_mask=oot_mask, ifd=ifd, vae=vae)


class TestInterpretability:
    def test_shap_summary_returns_importances_and_figure(self, fitted, tmp_path):
        out = tmp_path / "reports" / "figures" / "interpretability"
        imp = shap_summary_iforest(fitted["ifd"], fitted["X"], fitted["names"],
                                   out_dir=str(out), max_samples=500)
        assert isinstance(imp, dict) and len(imp) > 0
        assert all(np.isfinite(list(imp.values())))
        pngs = list(out.glob("*.png"))
        assert pngs, "no SHAP figure written"

    def test_path_length_analysis_stats_and_figure(self, fitted, tmp_path):
        out = tmp_path / "reports" / "figures" / "interpretability"
        stats = path_length_analysis(fitted["ifd"], fitted["X"], out_dir=str(out))
        for k in ("path_length_mean", "score_mean", "score_pathlen_corr"):
            assert k in stats
        assert os.path.exists(stats["figure_path"])
        assert "reports" in Path(stats["figure_path"]).parts

    def test_latent_space_plot_written(self, fitted, tmp_path):
        out = tmp_path / "reports" / "figures" / "interpretability"
        p = latent_space_plot(fitted["vae"], fitted["X"], out_dir=str(out), y=fitted["y"])
        assert os.path.exists(p) and p.lower().endswith(".png")

    def test_recon_error_by_feature(self, fitted, tmp_path):
        out = tmp_path / "reports" / "figures" / "interpretability"
        rbf = reconstruction_error_by_feature(fitted["vae"], fitted["X"],
                                              fitted["names"], out_dir=str(out), max_samples=500)
        assert isinstance(rbf, dict) and len(rbf) == len(fitted["names"])
        assert all(v >= 0 for v in rbf.values())
        # sorted descending
        vals = list(rbf.values())
        assert vals == sorted(vals, reverse=True)


class TestReporting:
    def _context(self, fitted, tmp_path):
        import pandas as pd
        sc = fitted["ifd"].score_samples(fitted["X"])
        sm = supervised_metrics(fitted["y"][fitted["oot_mask"]], sc[fitted["oot_mask"]])
        sdf = build_scored_frame(fitted["df"], fitted["keys"], sc, fitted["schema"])
        xlsx, _ = export_oot_top_decile(
            sdf, fitted["schema"],
            out_path=str(tmp_path / "reports" / "oot_top10_iforest.xlsx"),
            model_name="iforest")
        return {
            "title": "Test Report",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": {"n_rows": len(fitted["df"]),
                        "n_entities": int(fitted["df"][fitted["schema"].entity_col].nunique()),
                        "n_periods": N_PERIODS, "anomaly_rate": float(fitted["y"].mean()),
                        "oot_period": str(pd.to_datetime(fitted["keys"][fitted["schema"].time_col]).max().date())},
            "models": {"iforest": {"best_params": {"n_estimators": 120}, "metrics": sm}},
            "figures": [],
            "oot_excel": xlsx,
            "notes": "unit test",
        }

    def test_build_report_both_formats(self, fitted, tmp_path):
        ctx = self._context(fitted, tmp_path)
        res = build_report(ctx, out_dir=str(tmp_path / "reports"),
                           basename="anomaly_report", formats=("html", "md"))
        assert res["html"] and os.path.exists(res["html"])
        assert res["md"] and os.path.exists(res["md"])
        assert "pdf" not in res, "no PDF format exists -- build_report must not emit one"
        # HTML is self-contained (no external CDN <script src=http...>)
        html = Path(res["html"]).read_text(encoding="utf-8", errors="ignore")
        assert "<html" in html.lower()
        assert "src=\"http" not in html and "src='http" not in html
        # Markdown carries the OOT deliverable explanation
        md = Path(res["md"]).read_text(encoding="utf-8", errors="ignore")
        assert "SCORE" in md.upper()

    def test_no_pdf_is_ever_written_even_when_explicitly_requested(self, fitted, tmp_path):
        """PDF generation was removed from the project by explicit decision.

        Guards both halves of that: the returned dict has no ``pdf`` slot, and
        passing ``"pdf"`` in ``formats`` (as older callers did) is silently
        ignored rather than resurrecting a renderer or raising.
        """
        ctx = self._context(fitted, tmp_path)
        out_dir = tmp_path / "nopdf"
        res = build_report(ctx, out_dir=str(out_dir), basename="r",
                           formats=("html", "md", "pdf", "model_doc"))
        assert "pdf" not in res, res
        assert not list(out_dir.rglob("*.pdf")), list(out_dir.rglob("*.pdf"))
        # The other formats still work -- removal must not have broken them.
        assert res["html"] and res["md"] and res["model_doc"]

    def test_report_renders_multi_model_oot_dict_with_badges_and_tiles(self, fitted, tmp_path):
        """Regression test for the dashboard-style report redesign:
        `oot_excel` accepts {model: path} (not just a single path) and both
        entries must render; headline metric tiles get a visible good/warning/
        serious text chip (never color alone); dataset KPI tiles render.
        """
        import pandas as pd
        from src.reporting.report import _badge_for, _oot_excel_items

        sc = fitted["ifd"].score_samples(fitted["X"])
        sm = supervised_metrics(fitted["y"][fitted["oot_mask"]], sc[fitted["oot_mask"]])
        sdf = build_scored_frame(fitted["df"], fitted["keys"], sc, fitted["schema"])
        out_dir = tmp_path / "reports"
        xlsx_if, _ = export_oot_top_decile(
            sdf, fitted["schema"], out_path=str(out_dir / "oot_top10_iforest.xlsx"),
            model_name="iforest")
        xlsx_vae, _ = export_oot_top_decile(
            sdf, fitted["schema"], out_path=str(out_dir / "oot_top10_vae.xlsx"),
            model_name="vae")

        # metrics prefixed oot_* so the headline-tile path (not the unsupervised
        # fallback) is exercised, matching what main.py actually produces.
        oot_metrics = {f"oot_{k}": v for k, v in sm.items()}
        ctx = {
            "title": "Multi-model Test Report",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": {"rows": len(fitted["df"]), "entities": N_ENTITIES,
                        "periods": N_PERIODS, "anomaly_rate": float(fitted["y"].mean()),
                        "n_anomalies": int(fitted["y"].sum())},
            "models": {
                "iforest": {"best_params": {"n_estimators": 120}, "metrics": oot_metrics},
                "vae": {"best_params": {"latent_dim": 4}, "metrics": oot_metrics},
            },
            "figures": [],
            "oot_excel": {"iforest": xlsx_if, "vae": xlsx_vae},
            "notes": "multi-model test",
        }

        res = build_report(ctx, out_dir=str(out_dir), basename="multi", formats=("html", "md"))
        html = Path(res["html"]).read_text(encoding="utf-8", errors="ignore")
        md = Path(res["md"]).read_text(encoding="utf-8", errors="ignore")

        # Both OOT deliverables are linked, not just the first.
        assert "oot_top10_iforest.xlsx" in html and "oot_top10_vae.xlsx" in html
        assert "oot_top10_iforest.xlsx" in md and "oot_top10_vae.xlsx" in md

        # Two model cards rendered, both accent-colored (fixed categorical order).
        assert html.count("model-card") >= 2
        assert "var(--series-1)" in html and "var(--series-2)" in html

        # Every status-colored tile ships a visible text chip, never color alone.
        # (Search for the class actually applied to a div, not just the CSS
        # selector text -- the stylesheet mentions "badge-warning" etc. even
        # when no tile uses it.) The chip's CSS class stays in English (it is
        # the badge value / stylesheet hook); the visible label is Spanish.
        from src.reporting.report import _BADGE_LABELS

        for badge in ("good", "warning", "serious"):
            if f"tile badge-{badge}'" in html:
                assert f"<span class='chip {badge}'>{_BADGE_LABELS[badge]}</span>" in html, (
                    f"'{badge}' tile has no visible text chip -- color-alone status"
                )

        # Dataset KPI tiles show comma-formatted, human-readable numbers.
        assert f"{len(fitted['df']):,}" in html

        # Helper functions used by the redesign behave as documented.
        assert _oot_excel_items({"a": "x", "b": "y"}) == [("a", "x"), ("b", "y")]
        assert _oot_excel_items("solo.xlsx") == [("", "solo.xlsx")]
        assert _badge_for("roc_auc", 0.9) == "good"
        assert _badge_for("roc_auc", float("nan")) is None

    def test_build_report_robust_to_missing_pieces(self, tmp_path):
        ctx = {"title": "Minimal", "generated_at": "2026-07-24T00:00:00",
               "dataset": {"n_rows": 10}, "models": {}, "figures": [],
               "oot_excel": None, "notes": ""}
        res = build_report(ctx, out_dir=str(tmp_path / "r2"),
                           basename="min", formats=("html", "md"))
        assert res["html"] and os.path.exists(res["html"])
        assert res["md"] and os.path.exists(res["md"])
