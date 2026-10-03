"""The single, user-editable configuration file for the diagnostic knobs.

One place -- ``configs/pipeline.yaml`` next to ``main.py`` (or ``--config PATH``) --
holds the settings a user edits most often (the segment column, the analyst
identity column, identification-only columns excluded from every modelling
phase, the section-9 experiment grids, ...). Before this file the same
setting lived in three places (the ``PipelineConfig`` default, a CLI flag and the
vendored suite's own default), so editing the wrong one was silently a no-op.

Precedence, highest first::

    explicit CLI flag  >  value set in code (differs from the built-in default)
                       >  configs/pipeline.yaml  >  built-in default

The file only overrides a field that still holds its built-in default and was not
given on the command line, so a deliberate value is never overwritten. Unknown keys
and malformed values are an **error that lists the valid keys**: a typo must not
turn into an ignored setting.

Data sources / inputs: ``configs/pipeline.yaml`` or the path supplied by
``--config``; values are applied to ``main.PipelineConfig``.
Created: 2026-09-25
Last modified: 2026-10-01
Changelog:
- 2026-09-25: Made YAML validation reject duplicate/unknown empty keys,
  non-finite or out-of-range grids, and inconsistent backtest origin counts.
- 2026-09-27: Added ``data.identification_columns`` (list of column names).
- 2026-10-01: ``dashboard.identity_column`` (one field) replaced by
  ``dashboard.identity_columns`` (a list) -- the analyst dashboard can now
  show several identification fields per case, not only one.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Callable, Optional

import yaml

__all__ = ["CONFIG_KEYS", "ConfigFileError", "DEFAULT_CONFIG_PATH", "configured_value", "find_config_file",
           "load_config_file", "apply_config_file"]

#: ``configs/pipeline.yaml`` next to ``main.py`` (not the CWD: project paths are
#: CWD-relative for artifacts, but the config belongs to the code base).
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "pipeline.yaml",
)


class ConfigFileError(ValueError):
    """The configuration file is unreadable or holds an unknown key / bad value."""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any) -> int:
    if isinstance(value, bool) or int(value) != float(value):
        raise ValueError("expected an integer")
    return int(value)


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("expected true/false")
    return value


def _floats(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    vals = tuple(float(v) for v in value)
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("expected finite numbers")
    return vals


def _ints(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    return tuple(_int(v) for v in value)


def _opt_floats(value: Any) -> Optional[tuple]:
    """``None`` (key present but empty / null) = automatic points; ``[]`` = sweep off."""
    return None if value is None else _floats(value)


def _opt_ints(value: Any) -> Optional[tuple]:
    return None if value is None else _ints(value)


def _opt_pos_ints(value: Any) -> Optional[tuple]:
    vals = _opt_ints(value)
    if vals is not None and any(v <= 0 for v in vals):
        raise ValueError("expected positive integers")
    return vals


def _opt_nonneg_ints(value: Any) -> Optional[tuple]:
    vals = _opt_ints(value)
    if vals is not None and any(v < 0 for v in vals):
        raise ValueError("expected non-negative integers")
    return vals


def _opt_pos_floats(value: Any) -> Optional[tuple]:
    vals = _opt_floats(value)
    if vals is not None and any(v <= 0 for v in vals):
        raise ValueError("expected positive finite numbers")
    return vals


def _choice(*options: str) -> Callable[[Any], str]:
    def coerce(value: Any) -> str:
        v = _text(value)
        if v not in options:
            raise ValueError(f"expected one of {list(options)}")
        return v
    return coerce


def _pos_float(value: Any) -> float:
    v = float(value)
    if isinstance(value, bool) or not v > 0:
        raise ValueError("expected a number > 0")
    return v


def _opt_at_least(minimum: int) -> Callable[[Any], Optional[int]]:
    def coerce(value: Any) -> Optional[int]:
        return None if value is None else _at_least(minimum)(value)
    return coerce


def _families(value: Any) -> tuple:
    from src.evaluation.ifvae_experiments import ALL_FAMILIES

    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    fams = tuple(str(v).strip() for v in value)
    bad = [f for f in fams if f not in ALL_FAMILIES]
    if bad:
        raise ValueError(f"unknown famil{'y' if len(bad) == 1 else 'ies'} {bad}; valid: {list(ALL_FAMILIES)}")
    return fams


def _pos_int(value: Any) -> int:
    v = _int(value)
    if v < 0:
        raise ValueError("expected a non-negative integer")
    return v


def _at_least(minimum: int) -> Callable[[Any], int]:
    def coerce(value: Any) -> int:
        v = _int(value)
        if v < minimum:
            raise ValueError(f"expected an integer >= {minimum}")
        return v
    return coerce


def _unit_floats(value: Any) -> tuple:
    vals = _floats(value)
    if any(not 0.0 < v < 1.0 for v in vals):
        raise ValueError("expected values strictly between 0 and 1")
    return vals


def _half_floats(value: Any) -> tuple:
    vals = _floats(value)
    if any(not 0.0 < v < 0.5 for v in vals):
        raise ValueError("expected values strictly between 0 and 0.5")
    return vals


def _strings(value: Any) -> tuple:
    """A column name, or a list of them; blanks are dropped, order preserved, no duplicates."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    out: list = []
    for v in value:
        text = str(v).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)

#: ``"section.key"`` -> (``PipelineConfig`` attribute, coercer). Extend here only.
CONFIG_KEYS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "diagnostic.run_suite": ("run_diagnostic_suite", _bool),
    "diagnostic.auto_install_suite": ("diagnostic_auto_install_suite", _bool),
    "diagnostic.segment_column": ("diagnostic_segment_column", _text),
    "diagnostic.entity_view": ("diagnostic_entity_view", _bool),
    "diagnostic.stability_refits": ("diagnostic_stability_refits", _pos_int),
    "diagnostic.sensitivity_grid": ("diagnostic_sensitivity_grid", _unit_floats),
    "dashboard.identity_columns": ("analyst_identity_columns", _strings),
    "data.identification_columns": ("identification_columns", _strings),
    # -- Section 9 experiments (src/evaluation/ifvae_experiments.py) ------------
    "experiments.families": ("diagnostic_experiment_families", _families),
    "experiments.vae_fit_budget": ("diagnostic_experiment_vae_fit_budget", _pos_int),
    "experiments.epoch_cap": ("diagnostic_experiment_epoch_cap", _at_least(1)),
    "experiments.max_fit_rows": ("diagnostic_experiment_max_fit_rows", _at_least(200)),
    "experiments.capacity_grid": ("diagnostic_experiment_capacity_grid", _opt_pos_ints),
    "experiments.beta_grid": ("diagnostic_experiment_beta_grid", _opt_pos_floats),
    "experiments.kl_anneal_grid": ("diagnostic_experiment_kl_grid", _opt_nonneg_ints),
    "experiments.contamination_grid": ("diagnostic_experiment_contamination_grid", _half_floats),
    "experiments.backtest_origins": ("diagnostic_backtest_origins", _pos_int),
    "experiments.backtest_vae_origins": ("diagnostic_backtest_vae_origins", _pos_int),
    # -- VAE representation of categoricals (src/models/mixed_vae.py) -------------------
    "vae.categorical_representation": ("vae_categorical_representation", _choice("onehot", "embedding")),
    "vae.categorical_embedding.dimension_strategy": ("vae_embedding_dimension_strategy", _choice("auto", "fixed")),
    "vae.categorical_embedding.dimension": ("vae_embedding_dimension", _opt_at_least(1)),
    "vae.categorical_embedding.min_dimension": ("vae_embedding_min_dimension", _at_least(1)),
    "vae.categorical_embedding.max_dimension": ("vae_embedding_max_dimension", _at_least(1)),
    "vae.reconstruction_loss.numeric": ("vae_numeric_loss", _choice("huber", "mse")),
    "vae.reconstruction_loss.boolean": ("vae_boolean_loss", _choice("binary_cross_entropy")),
    "vae.reconstruction_loss.categorical": ("vae_categorical_loss", _choice("cross_entropy")),
    "vae.reconstruction_loss.aggregate_by_original_feature": ("vae_aggregate_by_original_feature", _bool),
    "vae.loss_weights.numeric": ("vae_weight_numeric", _pos_float),
    "vae.loss_weights.boolean": ("vae_weight_boolean", _pos_float),
    "vae.loss_weights.categorical": ("vae_weight_categorical", _pos_float),
    "vae.unknown_category_policy": ("vae_unknown_category_policy", _choice("explicit_token")),
    "vae.missing_category_policy": ("vae_missing_category_policy", _choice("explicit_token")),
    "experiments.backtest_min_fit_periods": ("diagnostic_backtest_min_fit_periods", _at_least(2)),
    # -- Isolation Forest post-tuning selection (src/models/iforest.py::tune_iforest) -----
    "iforest.selection_top_k": ("iforest_selection_top_k", _pos_int),
    "iforest.noise_seeds": ("iforest_noise_seeds", _at_least(2)),
}


def find_config_file(explicit: Optional[str] = None) -> Optional[str]:
    """The file to read: ``explicit`` (must exist) or the default if it exists."""
    if explicit:
        if not os.path.isfile(explicit):
            raise ConfigFileError(f"--config {explicit!r} does not exist.")
        return explicit
    return DEFAULT_CONFIG_PATH if os.path.isfile(DEFAULT_CONFIG_PATH) else None


def load_config_file(path: str) -> dict[str, Any]:
    """Read and validate ``path``; returns ``{"section.key": raw value}``."""
    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _unique_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"duplicate key {key!r}", key_node.start_mark,
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping,
    )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=_UniqueKeyLoader)
    except UnicodeDecodeError as exc:
        raise ConfigFileError(f"Cannot read {path}: it is not UTF-8 (save it as UTF-8, e.g. from "
                              f"Notepad 'Guardar como' > Codificación UTF-8): {exc}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigFileError(f"Cannot read {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigFileError(f"{path}: the top level must be a mapping of sections.")
    flat: dict[str, Any] = {}

    def _walk(prefix: str, body: dict, out: dict) -> None:
        for key, value in body.items():
            dotted = f"{prefix}.{key}"
            # A mapping is a nested group unless the dotted key itself is a setting.
            if isinstance(value, dict) and dotted not in CONFIG_KEYS:
                _walk(dotted, value, out)
            else:
                out[dotted] = value

    valid_sections = {key.split(".", 1)[0] for key in CONFIG_KEYS}
    for section, body in data.items():
        if section not in valid_sections:
            raise ConfigFileError(
                f"{path}: unknown section {section!r}. Valid sections: {sorted(valid_sections)}."
            )
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigFileError(f"{path}: section {section!r} must be a mapping of keys.")
        _walk(str(section), body, flat)
    unknown = sorted(set(flat) - set(CONFIG_KEYS))
    if unknown:
        raise ConfigFileError(
            f"{path}: unknown key(s) {unknown}. Valid keys: {sorted(CONFIG_KEYS)}."
        )
    return flat


def _field_default(config: Any, attr: str) -> Any:
    for f in dataclasses.fields(config):
        if f.name == attr:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                return f.default_factory()  # type: ignore[misc]
    raise AttributeError(attr)


def apply_config_file(config: Any, path: Optional[str], protected: frozenset = frozenset()) -> dict:
    """Apply ``path`` onto ``config`` and return a report of where each value came from.

    ``protected`` holds the attributes given explicitly on the command line: they
    are never touched. Every other attribute is overridden only while it still
    equals its built-in default. The report (``sources``) maps each file-managed
    attribute to ``cli`` / ``code`` / ``file`` / ``default``.
    """
    sources = {attr: "default" for attr, _ in CONFIG_KEYS.values()}
    values = load_config_file(path) if path else {}
    for dotted, (attr, coerce) in CONFIG_KEYS.items():
        current, default = getattr(config, attr), _field_default(config, attr)
        if attr in protected:
            sources[attr] = "cli"
            continue
        if dotted in values:
            try:
                new = coerce(values[dotted])
            except (TypeError, ValueError) as exc:
                raise ConfigFileError(f"{path}: {dotted}={values[dotted]!r} is invalid ({exc}).") from exc
            if current == default:
                setattr(config, attr, new)
                sources[attr] = "file"
                continue
        sources[attr] = "code" if current != default else "default"
    if config.diagnostic_backtest_vae_origins > config.diagnostic_backtest_origins:
        raise ConfigFileError(
            f"{path}: experiments.backtest_vae_origins "
            f"({config.diagnostic_backtest_vae_origins}) cannot exceed "
            f"experiments.backtest_origins ({config.diagnostic_backtest_origins})."
        )
    return {"path": path, "sources": sources, "keys_in_file": sorted(values)}


def configured_value(dotted: str, default: Any = None, path: Optional[str] = None) -> Any:
    """One value of the config file (coerced), or ``default`` when file/key is absent.

    For stand-alone tools (e.g. ``tools/export_diagnostic_suite_inputs.py``) that do
    not build a ``PipelineConfig`` but must read the same single source.
    """
    found = find_config_file(path)
    if not found:
        return default
    values = load_config_file(found)
    if dotted not in values:
        return default
    return CONFIG_KEYS[dotted][1](values[dotted])
