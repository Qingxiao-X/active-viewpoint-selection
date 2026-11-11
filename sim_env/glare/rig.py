"""Camera rig on a horizontal orbit around the head center. No Isaac Sim imports.

Coordinate convention (world frame, same as the converted head assets):
  * meters, +Z up, the head faces +X, head center = orbit center = look-at target c
  * camera position p = c + r * [cos(el)cos(az), cos(el)sin(az), sin(el)]
  * az = 0 is directly in front of the face; az > 0 moves toward +Y
  * the camera always looks at c (look-at orientation is recomputed after every move)

Actions (discrete, the 4-way motion used on slide 4):
  LEFT     az -= step_az   (camera moves to ITS OWN left; at az=0 the camera's left is -Y,
                            which is the subject's right-hand side)
  RIGHT    az += step_az
  FORWARD  r  -= step_r    (toward the head)
  BACKWARD r  += step_r    (away from the head)
Moves that would leave [az_min, az_max] or [r_min, r_max] are rejected (masked), never clipped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ACTIONS = ("LEFT", "RIGHT", "FORWARD", "BACKWARD")


def sph_to_vec(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


@dataclass(frozen=True)
class Pose:
    az: float  # deg
    el: float  # deg
    r: float   # m

    def position(self, center) -> np.ndarray:
        return np.asarray(center, dtype=float) + self.r * sph_to_vec(self.az, self.el)

    def to_dict(self):
        return asdict(self)


@dataclass
class Limits:
    az_min: float = -45.0
    az_max: float = 45.0
    r_min: float = 0.4
    r_max: float = 0.9


class OrbitRig:
    def __init__(self, center, start: Pose, step_az: float = 15.0, step_r: float = 0.1,
                 limits: Limits = Limits()):
        self.center = np.asarray(center, dtype=float)
        self.pose = start
        self.step_az, self.step_r = float(step_az), float(step_r)
        self.limits = limits

    def propose(self, action: str) -> Pose:
        p = self.pose
        if action == "LEFT":
            return Pose(p.az - self.step_az, p.el, p.r)
        if action == "RIGHT":
            return Pose(p.az + self.step_az, p.el, p.r)
        if action == "FORWARD":
            return Pose(p.az, p.el, round(p.r - self.step_r, 6))
        if action == "BACKWARD":
            return Pose(p.az, p.el, round(p.r + self.step_r, 6))
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")

    def is_valid(self, pose: Pose) -> bool:
        L, eps = self.limits, 1e-9
        return (L.az_min - eps <= pose.az <= L.az_max + eps) and (L.r_min - eps <= pose.r <= L.r_max + eps)

    def valid_actions(self):
        return [a for a in ACTIONS if self.is_valid(self.propose(a))]

    def step(self, action: str):
        """Returns (accepted, new_pose). A rejected action leaves the pose unchanged."""
        nxt = self.propose(action)
        if not self.is_valid(nxt):
            return False, self.pose
        self.pose = nxt
        return True, nxt


def look_at_quat_wxyz(cam_pos, target) -> np.ndarray:
    """Orientation (w, x, y, z) in Isaac's 'world' camera axes (+X forward, +Z up)."""
    fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    return mat_to_quat_wxyz(np.stack([fwd, left, up], axis=1))


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_check(cam_pos, quat_wxyz, center) -> dict:
    """Recover (az, el, r) and the look-at error (deg) from a camera pose read back from the stage."""
    cam_pos, center = np.asarray(cam_pos, float), np.asarray(center, float)
    d = cam_pos - center
    r = float(np.linalg.norm(d))
    az = float(np.degrees(np.arctan2(d[1], d[0])))
    el = float(np.degrees(np.arcsin(d[2] / r)))
    fwd = quat_wxyz_to_mat(quat_wxyz)[:, 0]  # world-axes camera: +X is the viewing direction
    want = -d / r
    err = float(np.degrees(np.arccos(np.clip(np.dot(fwd, want), -1.0, 1.0))))
    return {"az_deg": az, "el_deg": el, "r_m": r, "look_at_error_deg": err}
