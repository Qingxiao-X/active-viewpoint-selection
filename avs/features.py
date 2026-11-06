"""Feature schema shared by dataset building, training and inference.

Raw features are float64 with ``NaN`` for missing values. ``Normalizer`` (fit
on the TRAIN split only) z-scores each column using statistics of the present
values, replaces missing entries by 0 (= the train mean) and appends a binary
``<name>__missing`` indicator for every optional column.

Observation features (from ``PolicyObservation`` only):
  azimuth, elevation [deg]; radius [m]; face_detected {0,1};
  det_score [0,1]; bbox_cx, bbox_cy, bbox_w, bbox_h [fraction of image];
  pose_yaw, pose_pitch, pose_roll [deg, estimated];
  top1_similarity [cosine]; top2_margin [cosine difference];
  glare_fraction [fraction of face box]; glare_cx, glare_cy [-1,1 in face box];
  depth_valid_fraction [fraction]; depth_std [m]
Candidate-action features:
  d_az, d_el [deg]; next_az, next_el [deg]; move_cost [m, arc length]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .actions import Action, ActionSpace, Viewpoint
from .types import PolicyObservation

FEATURE_SCHEMA_VERSION = 1

ALL_OBS_FIELDS = [
    "azimuth", "elevation", "radius", "face_detected", "det_score",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h", "pose_yaw", "pose_pitch", "pose_roll",
    "top1_similarity", "top2_margin", "glare_fraction", "glare_cx", "glare_cy",
    "depth_valid_fraction", "depth_std",
]
REQUIRED_OBS_FIELDS = {"azimuth", "elevation", "radius", "face_detected"}
ALL_ACTION_FIELDS = ["d_az", "d_el", "next_az", "next_el", "move_cost"]

# Names that must never appear in a policy feature schema.
FORBIDDEN_FEATURE_NAMES = {
    "identity", "gt_identity", "predicted_identity", "scene_id", "light_az", "light_el",
    "light_intensity", "env_intensity", "roughness", "q_id", "label", "aux_glare", "aux_depth",
}


def observation_record(obs: PolicyObservation, fields: Sequence[str] = ALL_OBS_FIELDS) -> Dict[str, float]:
    """Numeric raw observation values (NaN = missing). Only reads PolicyObservation."""
    d = obs.to_dict()
    out = {}
    for f in fields:
        v = d[f]
        out[f] = float("nan") if v is None else float(v)
    return out


def action_record(vp: Viewpoint, action: Action, space: ActionSpace) -> Dict[str, float]:
    nxt = space.apply(vp, action)
    return {"d_az": action.d_az, "d_el": action.d_el, "next_az": nxt.azimuth,
            "next_el": nxt.elevation, "move_cost": space.movement_cost(vp, action)}


class FeatureSchema:
    def __init__(self, obs_fields: Optional[Sequence[str]] = None, action_fields: Optional[Sequence[str]] = None):
        self.obs_fields = list(obs_fields or ALL_OBS_FIELDS)
        self.action_fields = list(action_fields or ALL_ACTION_FIELDS)
        bad = (set(self.obs_fields) | set(self.action_fields)) & FORBIDDEN_FEATURE_NAMES
        if bad:
            raise ValueError(f"privileged/oracle fields not allowed as features: {sorted(bad)}")
        unknown = (set(self.obs_fields) - set(ALL_OBS_FIELDS)) | (set(self.action_fields) - set(ALL_ACTION_FIELDS))
        if unknown:
            raise ValueError(f"unknown feature fields: {sorted(unknown)}")

    @property
    def raw_names(self) -> List[str]:
        return [f"obs.{f}" for f in self.obs_fields] + [f"act.{f}" for f in self.action_fields]

    @property
    def optional_mask(self) -> np.ndarray:
        return np.array([f not in REQUIRED_OBS_FIELDS for f in self.obs_fields] + [False] * len(self.action_fields))

    def raw_matrix(self, obs_cols: Dict[str, np.ndarray], act_cols: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack column dicts (each array of length N) into an N x F raw matrix."""
        cols = [np.asarray(obs_cols[f], dtype=np.float64) for f in self.obs_fields]
        cols += [np.asarray(act_cols[f], dtype=np.float64) for f in self.action_fields]
        return np.stack(cols, axis=1)

    def candidates_raw(self, obs: PolicyObservation, space: ActionSpace, actions: Sequence[Action]) -> np.ndarray:
        """Raw features for every candidate action from one observation (inference)."""
        o = observation_record(obs, self.obs_fields)
        vp = obs.viewpoint
        rows = []
        for a in actions:
            r = action_record(vp, a, space)
            rows.append([o[f] for f in self.obs_fields] + [r[f] for f in self.action_fields])
        return np.asarray(rows, dtype=np.float64)

    def to_json(self) -> Dict:
        return {"version": FEATURE_SCHEMA_VERSION, "obs_fields": self.obs_fields,
                "action_fields": self.action_fields, "raw_names": self.raw_names}

    @classmethod
    def from_json(cls, d: Dict) -> "FeatureSchema":
        if d.get("version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema version {d.get('version')} != {FEATURE_SCHEMA_VERSION}")
        return cls(d["obs_fields"], d["action_fields"])


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, optional_mask: np.ndarray, names: List[str]):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.optional_mask = np.asarray(optional_mask, dtype=bool)
        self.names = list(names)

    @classmethod
    def fit(cls, X_raw: np.ndarray, schema: FeatureSchema) -> "Normalizer":
        mean = np.zeros(X_raw.shape[1])
        std = np.ones(X_raw.shape[1])
        for j in range(X_raw.shape[1]):
            col = X_raw[:, j]
            col = col[~np.isnan(col)]
            if col.size:
                mean[j] = col.mean()
                s = col.std()
                std[j] = s if s > 1e-8 else 1.0
        req = ~schema.optional_mask
        if np.isnan(X_raw[:, req]).any():
            raise ValueError("required feature has missing values")
        return cls(mean, std, schema.optional_mask, schema.raw_names)

    @property
    def out_names(self) -> List[str]:
        return self.names + [f"{n}__missing" for n, o in zip(self.names, self.optional_mask) if o]

    @property
    def out_dim(self) -> int:
        return len(self.names) + int(self.optional_mask.sum())

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        X_raw = np.atleast_2d(X_raw)
        missing = np.isnan(X_raw)
        Z = (X_raw - self.mean) / self.std
        Z[missing] = 0.0
        ind = missing[:, self.optional_mask].astype(np.float64)
        return np.concatenate([Z, ind], axis=1).astype(np.float32)

    def to_json(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "optional_mask": self.optional_mask.tolist(), "names": self.names}

    @classmethod
    def from_json(cls, d: Dict) -> "Normalizer":
        return cls(d["mean"], d["std"], d["optional_mask"], d["names"])
