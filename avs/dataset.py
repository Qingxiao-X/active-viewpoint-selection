"""Offline transition dataset: format, IO, grouped splits, feature matrices.

Directory layout (``DATASET_SCHEMA_VERSION`` = 1)::

    <dir>/meta.json         schema version, config, action space, counts, failures
    <dir>/scenes.jsonl      one scene config per line (PRIVILEGED: lighting/material/identity)
    <dir>/transitions.npz   column arrays, one row per (state, candidate action)

Column prefixes in ``transitions.npz`` (the prefix IS the access policy):

    group.*   scene_id, state_id, identity      grouping only — never features
    obs.*     current PolicyObservation fields  (NaN = missing)   -> features
    act.*     candidate action fields, plus action_index           -> features
    status    'ok' | 'invalid_action' | 'render_failure'
    label.*   q_id, label, aux_glare, aux_depth, next_detected     targets only

Only ``status == 'ok'`` rows are used for training. Invalid actions and
render failures are kept as explicit records and counted, never converted
into low-quality labels.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import FeatureSchema

DATASET_SCHEMA_VERSION = 1
STATUS_OK, STATUS_INVALID, STATUS_RENDER_FAIL = "ok", "invalid_action", "render_failure"


class Dataset:
    def __init__(self, cols: Dict[str, np.ndarray], meta: Dict, scenes: List[Dict]):
        self.cols = cols
        self.meta = meta
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.cols["status"])

    # ------------------------------------------------------------------ IO
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "transitions.npz"), **self.cols)
        with open(os.path.join(out_dir, "scenes.jsonl"), "w") as f:
            for s in self.scenes:
                f.write(json.dumps(s) + "\n")
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Dataset":
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if meta.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(f"{path}: dataset schema {meta.get('schema_version')} != {DATASET_SCHEMA_VERSION}")
        with np.load(os.path.join(path, "transitions.npz"), allow_pickle=False) as z:
            cols = {k: z[k] for k in z.files}
        with open(os.path.join(path, "scenes.jsonl")) as f:
            scenes = [json.loads(line) for line in f if line.strip()]
        return cls(cols, meta, scenes)

    @classmethod
    def load_many(cls, paths: Sequence[str]) -> "Dataset":
        parts = [cls.load(p) for p in paths]
        if len(parts) == 1:
            return parts[0]
        ids = [s["scene_id"] for p in parts for s in p.scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id collision across datasets; use distinct --start-scene-id per shard")
        a0 = parts[0].meta["action_space"]
        for p in parts[1:]:
            if p.meta["action_space"] != a0:
                raise ValueError("datasets were collected with different action spaces")
        # state ids are local to each shard; offset them
        cols: Dict[str, list] = {k: [] for k in parts[0].cols}
        off = 0
        for p in parts:
            for k in cols:
                v = p.cols[k]
                cols[k].append(v + off if k == "group.state_id" else v)
            off += int(p.cols["group.state_id"].max()) + 1 if len(p) else 0
        meta = dict(parts[0].meta)
        meta["merged_from"] = list(paths)
        return cls({k: np.concatenate(v) for k, v in cols.items()}, meta, [s for p in parts for s in p.scenes])

    # ------------------------------------------------------------ accessors
    def ok_mask(self) -> np.ndarray:
        return self.cols["status"] == STATUS_OK

    def features_raw(self, schema: FeatureSchema, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Raw feature matrix built ONLY from obs.* and act.* columns."""
        sel = slice(None) if rows is None else rows
        obs = {f: self.cols[f"obs.{f}"][sel] for f in schema.obs_fields}
        act = {f: self.cols[f"act.{f}"][sel] for f in schema.action_fields}
        return schema.raw_matrix(obs, act)

    def labels(self, rows=None, key: str = "label.label") -> np.ndarray:
        sel = slice(None) if rows is None else rows
        return self.cols[key][sel].astype(np.float64)

    def status_counts(self) -> Dict[str, int]:
        u, c = np.unique(self.cols["status"], return_counts=True)
        return {str(a): int(b) for a, b in zip(u, c)}


# --------------------------------------------------------------------------
# Grouped splits
# --------------------------------------------------------------------------

def grouped_split(scenes: Sequence[Dict], fractions=(0.7, 0.15, 0.15), seed: int = 0,
                  heldout_identities: Optional[Sequence[str]] = None,
                  n_heldout_identities: int = 0) -> Dict[str, List[int]]:
    """Split scene ids into train/val/test (+ optional test_heldout_identity).

    The unit of splitting is the scene configuration: every start viewpoint
    and every candidate transition of a scene lands in the same split.
    Held-out identities (explicit list or ``n`` sampled ones) send *all* their
    scenes to ``test_heldout_identity`` so those identities are never seen in
    train/val.
    """
    rng = np.random.default_rng(seed)
    ids_all = sorted({s["identity"] for s in scenes})
    held = list(heldout_identities or [])
    if not held and n_heldout_identities > 0:
        if n_heldout_identities >= len(ids_all):
            raise ValueError("cannot hold out all identities")
        held = sorted(rng.choice(ids_all, size=n_heldout_identities, replace=False).tolist())
    held_set = set(held)
    held_scenes = sorted(s["scene_id"] for s in scenes if s["identity"] in held_set)
    rest = np.array(sorted(s["scene_id"] for s in scenes if s["identity"] not in held_set))
    rng.shuffle(rest)
    f = np.asarray(fractions, dtype=float)
    f = f / f.sum()
    n = len(rest)
    n_tr = int(round(f[0] * n))
    n_va = int(round(f[1] * n))
    if n >= 3:  # keep every split non-empty when possible
        n_tr = min(max(n_tr, 1), n - 2)
        n_va = min(max(n_va, 1), n - n_tr - 1)
    splits = {
        "train": sorted(rest[:n_tr].tolist()),
        "val": sorted(rest[n_tr:n_tr + n_va].tolist()),
        "test": sorted(rest[n_tr + n_va:].tolist()),
    }
    if held:
        splits["test_heldout_identity"] = held_scenes
    check_no_leakage(splits, scenes)
    return {k: [int(x) for x in v] for k, v in splits.items()}


def check_no_leakage(splits: Dict[str, List[int]], scenes: Optional[Sequence[Dict]] = None) -> None:
    """Raise if any scene appears in two splits, or a held-out identity leaks."""
    seen: Dict[int, str] = {}
    for name, ids in splits.items():
        for sid in ids:
            if sid in seen:
                raise AssertionError(f"scene {sid} in both {seen[sid]} and {name}")
            seen[sid] = name
    if scenes is not None and "test_heldout_identity" in splits:
        ident = {s["scene_id"]: s["identity"] for s in scenes}
        held = {ident[s] for s in splits["test_heldout_identity"]}
        for name in ("train", "val", "test"):
            leak = {ident[s] for s in splits.get(name, [])} & held
            if leak:
                raise AssertionError(f"held-out identities {sorted(leak)} appear in {name}")


def rows_for_scenes(ds: Dataset, scene_ids: Sequence[int], ok_only: bool = True) -> np.ndarray:
    m = np.isin(ds.cols["group.scene_id"], np.asarray(list(scene_ids), dtype=np.int64))
    if ok_only:
        m &= ds.ok_mask()
    return np.nonzero(m)[0]
