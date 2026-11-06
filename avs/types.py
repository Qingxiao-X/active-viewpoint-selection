"""Data types that define the information boundary.

``PolicyObservation`` is the ONLY thing a policy receives. It holds the current
camera viewpoint plus perception outputs computed from the *current* frame and
the shared enrollment gallery. It deliberately has no ground-truth identity,
no lighting/material parameters and nothing from unvisited viewpoints.

``OracleInfo`` holds privileged scene metadata (ground-truth identity,
lighting, material). Only the data collector (for labels) and the evaluator
(for scoring / condition grouping) may read it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .actions import Viewpoint


class RenderError(RuntimeError):
    """Infrastructure failure while rendering or measuring a view.

    Never converted into a low-quality label: collectors record it as a
    failure status and evaluators record it as an infrastructure error.
    """


@dataclass
class Frame:
    """Raw sensor output from the environment for one viewpoint."""

    rgb: Optional[np.ndarray]  # HxWx3 uint8, or None for the mock env
    depth: Optional[np.ndarray]  # HxW float32 meters (0 / nan / inf = invalid)
    viewpoint: Viewpoint
    extras: Dict[str, Any] = field(default_factory=dict)  # env-specific payload


@dataclass
class PerceptionResult:
    """Per-frame perception output. Missing quantities are ``None``.

    Units: bbox in pixels (x1, y1, x2, y2); pose in degrees (yaw, pitch, roll);
    glare centroid offsets in [-1, 1] relative to the face box center
    (+x = image right, +y = image down); depth values in meters.
    """

    face_detected: bool
    det_score: Optional[float] = None
    bbox: Optional[tuple] = None
    image_size: Optional[tuple] = None  # (width, height) px
    pose: Optional[tuple] = None  # (yaw, pitch, roll) deg
    embedding: Optional[np.ndarray] = None  # L2-normalized
    glare_fraction: Optional[float] = None  # fraction of face-box pixels flagged specular
    glare_cx: Optional[float] = None
    glare_cy: Optional[float] = None
    depth_valid_fraction: Optional[float] = None  # fraction of face-box pixels with valid depth
    depth_std: Optional[float] = None  # std of valid face-box depth (m)


@dataclass
class PolicyObservation:
    """Decision-time observation shared by every policy (see module doc)."""

    azimuth: float  # deg
    elevation: float  # deg
    radius: float  # m
    face_detected: bool
    det_score: Optional[float]
    bbox_cx: Optional[float]  # face box center / image width, [0, 1]
    bbox_cy: Optional[float]  # face box center / image height, [0, 1]
    bbox_w: Optional[float]  # face box width / image width
    bbox_h: Optional[float]  # face box height / image height
    pose_yaw: Optional[float]  # deg, estimated by perception
    pose_pitch: Optional[float]
    pose_roll: Optional[float]
    top1_similarity: Optional[float]  # cosine sim. to best-matching enrolled identity
    top2_margin: Optional[float]  # top1 - top2 similarity (None if < 2 enrolled)
    predicted_identity: Optional[str]  # best-matching enrolled identity (observable)
    glare_fraction: Optional[float]
    glare_cx: Optional[float]
    glare_cy: Optional[float]
    depth_valid_fraction: Optional[float]
    depth_std: Optional[float]
    step: int = 0  # number of movements executed so far in the episode

    @property
    def viewpoint(self) -> Viewpoint:
        return Viewpoint(self.azimuth, self.elevation, self.radius)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleInfo:
    """Privileged per-scene metadata. Never passed to policies."""

    scene_id: int
    identity: str
    params: Dict[str, Any]  # lighting / material / asset parameters
