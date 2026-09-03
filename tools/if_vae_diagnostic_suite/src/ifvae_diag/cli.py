from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, save_config
from .pipeline import run_diagnostic
from .simulation import generate_synthetic_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ifvae-diagnose",
        description="Diagnose Isolation Forest / VAE disagreement without test leakage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run diagnostics on exported model artifacts")
    run.add_argument("--reference", required=True, type=Path)
    run.add_argument("--scored", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    simulate = subparsers.add_parser("simulate", help="Exercise the suite on synthetic failure modes")
    simulate.add_argument("--out", required=True, type=Path)
    simulate.add_argument("--seed", type=int, default=7)
    return parser


def _run_files(args: argparse.Namespace) -> None:
    reference = pd.read_csv(args.reference)
    scored = pd.read_csv(args.scored)
    config = load_config(args.config)
    result = run_diagnostic(reference, scored, config, args.out)
    print(f"Diagnostic completed: {args.out / 'report.md'}")
    print(f"Known positives: {result.summary['known_positives']}")


def _simulate(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    reference, scored, config = generate_synthetic_case(args.seed)
    reference.to_csv(args.out / "synthetic_reference.csv", index=False)
    scored.to_csv(args.out / "synthetic_scored.csv", index=False)
    save_config(config, args.out / "synthetic_config.yaml")
    result = run_diagnostic(reference, scored, config, args.out)
    print(f"Synthetic diagnostic completed: {args.out / 'report.md'}")
    print(f"Positive quadrants: {result.summary['positive_quadrants']}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        _run_files(args)
    else:
        _simulate(args)

