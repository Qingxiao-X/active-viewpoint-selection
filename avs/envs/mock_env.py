"""Deterministic mock environment for SOFTWARE VALIDATION ONLY.

It imitates the *structure* of the real problem (identity embeddings degraded
by off-frontal pose and by a specular highlight whose position depends on the
light direction, roughness and camera direction) so the full
collect -> train -> evaluate pipeline can be exercised without Isaac Sim.
Numbers produced with this backend say nothing about real face perception.

Determinism: every measurement is a pure function of (scene, viewpoint), so
``set_state`` restores the environment exactly. Optional transient render
failures (``render_failure_prob``) exercise the failure-recording path.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

from ..actions import Viewpoint, sph_to_vec
from ..types import Frame, OracleInfo, PerceptionResult, RenderError
from .base import ViewpointEnv


def _seed(*parts) -> int:
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


class MockViewpointEnv(ViewpointEnv):
    exact_restore = True

    def __init__(self, radius: float = 0.6, n_identities: int = 8, emb_dim: int = 64,
                 seed: int = 0, render_failure_prob: float = 0.0, image_size=(640, 480)):
        self.radius = float(radius)
        self.dim = int(emb_dim)
        self.seed = int(seed)
        self.render_failure_prob = float(render_failure_prob)
        self.image_size = tuple(image_size)
        self._ids = [f"id_{i:02d}" for i in range(int(n_identities))]
        rng = np.random.default_rng(_seed("identities", self.seed))
        refs = rng.normal(size=(len(self._ids), self.dim))
        self._ref = refs / np.linalg.norm(refs, axis=1, keepdims=True)
        self._scene: Dict | None = None
        self._vp = Viewpoint(0.0, 0.0, self.radius)
        self._fail_rng = np.random.default_rng(_seed("fail", self.seed))
        self.render_calls = 0

    # ---------------------------------------------------------------- interface
    def identities(self) -> List[str]:
        return list(self._ids)

    def load_scene(self, scene: Dict) -> None:
        self._scene = dict(scene)

    def move_to(self, vp: Viewpoint) -> None:
        self._vp = Viewpoint(float(vp.azimuth), float(vp.elevation), self.radius)

    def current_viewpoint(self) -> Viewpoint:
        return self._vp

    def get_state(self):
        return {"scene": dict(self._scene), "vp": self._vp}

    def set_state(self, state) -> None:
        self._scene = dict(state["scene"])
        self._vp = state["vp"]

    def oracle(self) -> OracleInfo:
        s = self._scene
        params = {k: v for k, v in s.items() if k not in ("scene_id", "identity")}
        return OracleInfo(scene_id=s["scene_id"], identity=s["identity"], params=params)

    def enrollment_frame(self, identity: str) -> Frame:
        i = self._ids.index(identity)
        rng = np.random.default_rng(_seed("enroll", self.seed, identity))
        e = self._ref[i] + 0.15 * rng.normal(size=self.dim) / np.sqrt(self.dim)
        W, H = self.image_size
        ex = dict(detected=True, det_score=0.95, bbox=(W / 2 - 100, H / 2 - 120, W / 2 + 100, H / 2 + 120),
                  image_size=self.image_size, pose=(0.0, 0.0, 0.0), embedding=e / np.linalg.norm(e),
                  glare=0.0, glare_cx=None, glare_cy=None, depth_valid=0.97, depth_std=0.02)
        return Frame(None, None, Viewpoint(0.0, 0.0, self.radius), extras=ex)

    def render(self) -> Frame:
        self.render_calls += 1
        if self.render_failure_prob > 0 and self._fail_rng.random() < self.render_failure_prob:
            raise RenderError("mock transient render failure")
        return Frame(None, None, self._vp, extras=self._measure(self._scene, self._vp))

    # ---------------------------------------------------------------- physics
    def _measure(self, s: Dict, vp: Viewpoint) -> Dict:
        rng = np.random.default_rng(_seed("view", self.seed, s["scene_id"], round(vp.azimuth, 3), round(vp.elevation, 3)))
        c = sph_to_vec(vp.azimuth, vp.elevation)
        n = np.array([1.0, 0.0, 0.0])
        light = sph_to_vec(s["light_az"], s["light_el"])
        ndl = float(n @ light)
        r = 2 * ndl * n - light  # mirror direction of the key light about the face normal
        width = np.deg2rad(22.0 + 40.0 * s["roughness"])
        strength = s["light_intensity"] * (1.5 - 1.1 * s["roughness"]) / (1.0 + 0.4 * s["env_intensity"])
        theta = float(np.arccos(np.clip(c @ r, -1, 1)))
        glare = 0.0 if ndl <= 0 else float(np.clip(strength * np.exp(-(theta / width) ** 2), 0.0, 1.0))
        glare = float(np.clip(glare + abs(rng.normal(0, 0.005)), 0, 1))

        # pose factor: how frontal the face appears
        az, el = np.deg2rad(vp.azimuth), np.deg2rad(vp.elevation)
        p = float(max(np.cos(az), 0.0) ** 2 * max(np.cos(el), 0.0) ** 1.5)

        # glare centroid in the face box, from mirror-direction offset in image axes
        gcx = gcy = None
        if glare > 0.01:
            f = -c
            right = np.cross(f, [0, 0, 1.0])
            right /= max(np.linalg.norm(right), 1e-9)
            up = np.cross(right, f)
            d = r - c
            scale = 1.0 / (2.0 * np.sin(width) + 1e-6)
            gcx = float(np.clip(np.tanh(scale * d @ right) + rng.normal(0, 0.05), -1, 1))
            gcy = float(np.clip(np.tanh(-scale * d @ up) + rng.normal(0, 0.05), -1, 1))

        det_score = float(np.clip(0.3 + 0.7 * p - 0.8 * max(0.0, glare - 0.35) + rng.normal(0, 0.03), 0, 1))
        detected = det_score >= 0.5

        # embedding: identity signal shrinks with pose and glare; glare pushes
        # toward a scene-specific confuser identity (=> incorrect acceptances)
        gi = self._ids.index(s["identity"])
        srng = np.random.default_rng(_seed("scene", self.seed, s["scene_id"]))
        confuser = int(srng.integers(len(self._ids) - 1))
        confuser = confuser + (confuser >= gi)
        scene_offset = srng.normal(size=self.dim) / np.sqrt(self.dim)
        alpha = p * (1.0 - 0.9 * glare)
        beta = 1.0 * glare
        e = (alpha * self._ref[gi] + beta * self._ref[confuser]
             + 0.6 * scene_offset + 0.8 * rng.normal(size=self.dim) / np.sqrt(self.dim))
        e /= np.linalg.norm(e)

        W, H = self.image_size
        k = 0.6 / self.radius
        bw = 200 * k * (0.6 + 0.4 * np.cos(az))
        bh = 240 * k * (0.7 + 0.3 * np.cos(el))
        cx, cy = W / 2 + 40 * np.sin(az), H / 2 - 30 * np.sin(el)
        return dict(
            detected=bool(detected), det_score=det_score,
            bbox=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), image_size=self.image_size,
            pose=(vp.azimuth + rng.normal(0, 3), vp.elevation + rng.normal(0, 3), rng.normal(0, 2)),
            embedding=e, glare=glare, glare_cx=gcx, glare_cy=gcy,
            depth_valid=float(np.clip(0.97 - 0.9 * glare - 0.15 * (1 - p) + rng.normal(0, 0.02), 0, 1)),
            depth_std=float(max(0.005, 0.02 + 0.03 * (1 - p) + 0.04 * glare + rng.normal(0, 0.003))),
        )


class MockPerception:
    """Reads the mock env's synthetic measurements (stands in for InsightFace)."""

    def __call__(self, frame: Frame) -> PerceptionResult:
        x = frame.extras
        if not x["detected"]:
            # no face: box-level quantities are unavailable; depth over the
            # image center is still measured
            return PerceptionResult(False, image_size=x["image_size"],
                                    depth_valid_fraction=x["depth_valid"], depth_std=x["depth_std"])
        return PerceptionResult(
            True, x["det_score"], x["bbox"], x["image_size"], x["pose"], np.asarray(x["embedding"]),
            x["glare"], x["glare_cx"], x["glare_cy"], x["depth_valid"], x["depth_std"],
        )
