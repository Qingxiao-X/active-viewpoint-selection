"""Viewpoint representation, discrete action space, masking and movement cost.

Conventions (all angles in degrees, distances in meters):
  * The head center is the origin. The face looks along +x at azimuth 0,
    elevation 0. Positive azimuth rotates the camera toward the head's left
    (+y), positive elevation raises the camera (+z).
  * The camera stays on a sphere of fixed ``radius`` and always looks at the
    head center, so a viewpoint is fully described by (azimuth, elevation).
  * An action is a bounded displacement (d_az, d_el). Actions that would leave
    the allowed azimuth/elevation box are masked as invalid (never clipped),
    so every executed action moves the camera exactly as commanded.
  * Movement cost is the great-circle arc length (meters) travelled by the
    camera on the sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class Viewpoint:
    azimuth: float  # deg
    elevation: float  # deg
    radius: float  # m

    def direction(self) -> np.ndarray:
        """Unit vector from head center toward the camera."""
        return sph_to_vec(self.azimuth, self.elevation)

    def position(self) -> np.ndarray:
        return self.radius * self.direction()


@dataclass(frozen=True)
class Action:
    index: int
    d_az: float  # deg
    d_el: float  # deg


def sph_to_vec(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def arc_length(a: Viewpoint, b: Viewpoint) -> float:
    """Great-circle distance (m) between two viewpoints on the same sphere."""
    cosang = float(np.clip(np.dot(a.direction(), b.direction()), -1.0, 1.0))
    return float(a.radius * np.arccos(cosang))


class ActionSpace:
    """Discrete grid of (d_az, d_el) moves with box bounds on the viewpoint."""

    def __init__(
        self,
        az_steps: Sequence[float] = (-30.0, -15.0, 0.0, 15.0, 30.0),
        el_steps: Sequence[float] = (-15.0, 0.0, 15.0),
        az_limits: Sequence[float] = (-80.0, 80.0),
        el_limits: Sequence[float] = (-30.0, 40.0),
        radius: float = 0.6,
        include_noop: bool = False,
    ):
        self.az_limits = (float(az_limits[0]), float(az_limits[1]))
        self.el_limits = (float(el_limits[0]), float(el_limits[1]))
        self.radius = float(radius)
        acts: List[Action] = []
        for da in az_steps:
            for de in el_steps:
                if da == 0 and de == 0 and not include_noop:
                    continue
                acts.append(Action(len(acts), float(da), float(de)))
        if not acts:
            raise ValueError("empty action space")
        self.actions = acts

    @classmethod
    def from_config(cls, cfg: dict) -> "ActionSpace":
        return cls(**cfg)

    def to_config(self) -> dict:
        return {
            "az_steps": sorted({a.d_az for a in self.actions}),
            "el_steps": sorted({a.d_el for a in self.actions}),
            "az_limits": list(self.az_limits),
            "el_limits": list(self.el_limits),
            "radius": self.radius,
            "include_noop": any(a.d_az == 0 and a.d_el == 0 for a in self.actions),
        }

    def __len__(self) -> int:
        return len(self.actions)

    def in_bounds(self, vp: Viewpoint, eps: float = 1e-6) -> bool:
        return (
            self.az_limits[0] - eps <= vp.azimuth <= self.az_limits[1] + eps
            and self.el_limits[0] - eps <= vp.elevation <= self.el_limits[1] + eps
        )

    def apply(self, vp: Viewpoint, action: Action) -> Viewpoint:
        return Viewpoint(vp.azimuth + action.d_az, vp.elevation + action.d_el, self.radius)

    def valid_mask(self, vp: Viewpoint) -> np.ndarray:
        return np.array([self.in_bounds(self.apply(vp, a)) for a in self.actions], dtype=bool)

    def movement_cost(self, vp: Viewpoint, action: Action) -> float:
        return arc_length(vp, self.apply(vp, action))

    def sample_viewpoint(self, rng: np.random.Generator, az_range=None, el_range=None) -> Viewpoint:
        """Uniform start viewpoint inside (optionally narrower) bounds."""
        lo_a, hi_a = az_range if az_range is not None else self.az_limits
        lo_e, hi_e = el_range if el_range is not None else self.el_limits
        lo_a, hi_a = max(lo_a, self.az_limits[0]), min(hi_a, self.az_limits[1])
        lo_e, hi_e = max(lo_e, self.el_limits[0]), min(hi_e, self.el_limits[1])
        return Viewpoint(float(rng.uniform(lo_a, hi_a)), float(rng.uniform(lo_e, hi_e)), self.radius)
