"""Loader for ``config/settings.yaml``.

Import this instead of reading the YAML yourself, so that every service resolves the
same file and a missing key fails loudly in one place.

CLAUDE.md: no magic numbers in service code. If you need a number that is not in
settings.yaml, add it there rather than defaulting it here — a default in code is a
magic number wearing a disguise, and it will not be found when someone goes looking for
the dial to turn.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Raised when a required setting is absent or a placeholder is still unresolved."""


@functools.lru_cache(maxsize=None)
def load(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and cache the settings file. ``SPARK_SETTINGS`` overrides the location."""
    resolved = Path(path or os.environ.get("SPARK_SETTINGS") or DEFAULT_SETTINGS_PATH)
    if not resolved.is_file():
        raise ConfigError(f"settings file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"settings file is not a mapping: {resolved}")
    return data


def get(dotted: str, default: Any = _MISSING) -> Any:
    """Fetch a nested setting, e.g. ``get("ingest.gate.target_skip_rate")``.

    Raises rather than returning None for an absent key when no default is given.
    Silent Nones from config surface three modules away as an unrelated TypeError.
    """
    node: Any = load()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise ConfigError(f"missing setting: {dotted}")
            return default
        node = node[part]
    return node


def require(dotted: str) -> Any:
    """Fetch a setting that must be present *and* non-null.

    Use for values marked UNRESOLVED in settings.yaml — model names gated on SPEC §10
    decisions. Failing here with a readable message beats a 404 from an inference
    endpoint that was never told which model to serve.
    """
    value = get(dotted)
    if value is None:
        raise ConfigError(
            f"setting {dotted!r} is null — it depends on an unresolved decision in "
            f"SPEC §10. Resolve the decision and set it in config/settings.yaml."
        )
    return value


def repo_path(dotted: str) -> Path:
    """Resolve a ``paths.*`` setting to an absolute path under the repo root."""
    return (REPO_ROOT / str(get(dotted))).resolve()
