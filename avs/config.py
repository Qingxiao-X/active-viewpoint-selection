"""YAML config loading with dotted-key command-line overrides."""

from __future__ import annotations

import copy
from typing import Any, Dict, Sequence

import yaml


def deep_update(base: Dict, upd: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: Sequence[str] = ()) -> Dict[str, Any]:
    """Load YAML; apply overrides like ``collect.n_scenes=20`` (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for o in overrides or ():
        key, _, val = o.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {o!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg
