"""Controlled comparison of the VAE's categorical representations: one-hot (A) vs embeddings (B).

    py tools/compare_vae_representations.py --individuals 600 --periods 14 --epochs 15 --seeds 11 23 37 \
        --out reports/vae_representation_comparison.md

Runs in a temporary working directory (the project's ``artifacts/`` is never touched) on the synthetic
panel. See ``src/evaluation/vae_representation_comparison.py`` for exactly what is held fixed and measured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "if_vae_diagnostic_suite", "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--individuals", type=int, default=400)
    p.add_argument("--periods", type=int, default=14)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--alert-fraction", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None, help="Markdown report path (also writes <out>.json).")
    args = p.parse_args(argv)

    from src.evaluation.vae_representation_comparison import (
        ComparisonConfig, acceptance_criteria, render_markdown, run_comparison,
    )

    out = os.path.abspath(args.out) if args.out else None      # resolve BEFORE the run chdirs to a temp dir
    summary = run_comparison(ComparisonConfig(
        n_individuals=args.individuals, n_periods=args.periods, data_seed=args.data_seed, seeds=args.seeds,
        epochs=args.epochs, alert_fraction=args.alert_fraction))
    criteria = acceptance_criteria(summary)
    text = render_markdown(summary, criteria)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text)
        with open(out + ".json", "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "criteria": criteria}, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
                      ensure_ascii=False, indent=2)
        print(f"\nWritten: {out}")
    return 0 if all(c["passed"] for c in criteria) else 2


if __name__ == "__main__":
    raise SystemExit(main())
