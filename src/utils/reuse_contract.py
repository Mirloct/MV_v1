"""Validate preprocessing identity and model integrity before trained reuse.

Data sources / inputs: in-time preprocessed matrix, ordered features, fitted
preprocessor and local detector files; writes a versioned SHA-256 manifest.
Created: 2026-11-25
Last modified: 2026-11-25
Changelog: 2026-11-25 — added atomic manifests and fail-closed reuse validation.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from os import PathLike, fsync
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib

from src.utils.atomic_io import atomic_replace

__all__ = [
    "input_fingerprint", "save_contract", "validate_contract",
    "validate_if_reusing", "save_if_training",
]

_RETRAIN = "Retrain the models and save a new reuse contract before reuse."
_VERSION = 1


def input_fingerprint(X, feature_names: Sequence[str], preprocessor=None) -> str:
    names = tuple(feature_names)
    if len(X.shape) != 2 or X.shape[1] != len(names):
        raise ValueError("Reuse input matrix width must match ordered feature names. " + _RETRAIN)
    return joblib.hash((X, names, preprocessor))


def _check_fingerprint(fingerprint: str) -> None:
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("Reuse input fingerprint must be a non-empty string. " + _RETRAIN)


def _model_hashes(model_paths: Mapping[str, str | PathLike]) -> dict[str, str]:
    if not isinstance(model_paths, Mapping) or not model_paths:
        raise ValueError("Reuse model_paths must be a non-empty mapping of model names to files. " + _RETRAIN)
    hashes = {}
    for name, path in model_paths.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Reuse model names must be non-empty strings. " + _RETRAIN)
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Reuse model {name!r} cannot be read at {path!r}. " + _RETRAIN) from exc
        hashes[name] = digest.hexdigest()
    return hashes


def save_contract(
    path: str | PathLike, fingerprint: str, model_paths: Mapping[str, str | PathLike],
) -> None:
    _check_fingerprint(fingerprint)
    manifest = {"version": _VERSION, "fingerprint": fingerprint, "models": _model_hashes(model_paths)}
    destination = Path(path)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            fsync(stream.fileno())
        atomic_replace(str(temporary), str(destination))
    except OSError as exc:
        raise ValueError(f"Reuse contract could not be saved at {destination}. " + _RETRAIN) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_contract(
    path: str | PathLike, fingerprint: str, model_paths: Mapping[str, str | PathLike],
) -> None:
    _check_fingerprint(fingerprint)
    try:
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Reuse contract is missing or unreadable at {path!r}. " + _RETRAIN) from exc
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
        raise ValueError("Reuse contract format/version is invalid. " + _RETRAIN)
    if manifest["version"] != _VERSION:
        raise ValueError("Reuse contract format/version is invalid. " + _RETRAIN)
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("Reuse input fingerprint changed (matrix, ordered features or fitted preprocessor). " + _RETRAIN)
    if manifest.get("models") != _model_hashes(model_paths):
        raise ValueError("Reuse model SHA-256 hashes or model names changed. " + _RETRAIN)


def validate_if_reusing(enabled, path, fingerprint, models) -> None:
    if enabled:
        validate_contract(path, fingerprint, models)


def save_if_training(reuse, path, fingerprint, models) -> None:
    if reuse:
        return
    save_contract(path, fingerprint, models)
