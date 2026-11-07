"""Isaac Sim backend — UNTESTED TEMPLATE.

Isaac Sim was not available where this was written, so this adapter has never
been run. It targets the Isaac Sim 4.5+/5.x Python API (``isaacsim.*``
namespaces). Check every ``TODO(integration)`` against the team's actual
stage layout before trusting results.

What it does:
  * one USD head asset per identity (``assets: {identity: usd_path}``),
    referenced under ``/World/Heads/<identity>`` (shown one at a time) with the face looking along +x,
  * a DistantLight (directional glare source) and a DomeLight (environment),
  * a roughness override on the head's skin shader,
  * an RGB-D camera teleported on the viewing sphere and aimed at the head
    center. If the team has an arm controller, pass ``arm_controller`` as
    ``"module:function"``; it is called as ``fn(env, viewpoint)`` and must
    leave the camera at the requested pose.

State restoration: there is no full renderer snapshot. ``get_state`` returns
(scene params, viewpoint) and ``set_state`` rebuilds lights/material/camera
from them (deterministic reconstruction). Restoration is exact only if the
renderer is deterministic for identical inputs (use RTX real-time mode with
fixed settings, warm-up frames, no temporal accumulation artifacts; path
tracing with a fixed seed/spp). Verify by rendering the same state twice and
comparing images (see README "Integration checklist").
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional

import numpy as np

from ..actions import Viewpoint
from ..types import Frame, OracleInfo, RenderError
from .base import ViewpointEnv


def _look_at_quat_wxyz(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) for Isaac 'world' camera axes (+X fwd, +Z up)."""
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    if np.linalg.norm(left) < 1e-6:  # looking straight up/down
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    R = np.stack([fwd, left, up], axis=1)
    return _mat_to_quat_wxyz(R)


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method: stable for every rotation, including angles near 180 deg
    (a camera at azimuth 0 looking back along -x). The copysign shortcut used before
    picked wrong signs there and mis-aimed frontal views by up to 90 deg."""
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


class IsaacViewpointEnv(ViewpointEnv):
    exact_restore = False  # deterministic reconstruction, see module doc

    def __init__(self, assets: Dict[str, str], radius: float = 0.6, head_center=(0.0, 0.0, 1.6),
                 resolution=(640, 480), headless: bool = True, warmup_frames: int = 10,
                 key_light_base_intensity: float = 3000.0, dome_base_intensity: float = 1000.0,
                 skin_shader_path: Optional[str] = None, arm_controller: Optional[str] = None,
                 enroll_light: Optional[Dict] = None, focal_length: float = 28.0,
                 horizontal_aperture: float = 20.955, clipping_range=(0.01, 100.0)):
        if not assets:
            raise ValueError("isaac backend needs env.isaac.assets: {identity: usd_path}")
        from isaacsim import SimulationApp  # noqa: lazy, must precede other omni imports

        self._app = SimulationApp({"headless": headless})
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera

        self.assets = dict(assets)
        self.radius = float(radius)
        self.center = np.asarray(head_center, dtype=float)
        self.warmup = int(warmup_frames)
        self.key_base, self.dome_base = float(key_light_base_intensity), float(dome_base_intensity)
        self.skin_shader_path = skin_shader_path
        self.enroll_light = enroll_light or {"light_intensity": 0.0, "env_intensity": 1.0,
                                             "light_az": 0.0, "light_el": 0.0, "roughness": None}
        self._arm = None
        if arm_controller:
            mod, fn = arm_controller.split(":")
            self._arm = getattr(importlib.import_module(mod), fn)

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        self._make_lights()
        self.camera = Camera(prim_path="/World/RGBDCamera", resolution=tuple(resolution))
        self._set_optics(resolution, focal_length, horizontal_aperture, clipping_range)
        self.world.reset()
        self.world.play()  # camera annotators only produce data while the timeline runs
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        self._scene: Optional[Dict] = None
        self._loaded_identity: Optional[str] = None
        self._vp = Viewpoint(0.0, 0.0, self.radius)
        self._needs_long_settle = True

    # ------------------------------------------------------------ scene setup
    def _set_optics(self, resolution, focal_length, horizontal_aperture, clipping_range):
        """Explicit pinhole optics (FOV depends only on focal/aperture ratio).

        Default 28 / 20.955 gives ~41 deg horizontal FOV: at 0.6 m the whole head fits.
        Vertical aperture follows the image aspect so pixels stay square."""
        from pxr import Gf, UsdGeom

        cam = UsdGeom.Camera(self.stage.GetPrimAtPath("/World/RGBDCamera"))
        w, h = resolution
        cam.GetFocalLengthAttr().Set(float(focal_length))
        cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
        cam.GetVerticalApertureAttr().Set(float(horizontal_aperture) * float(h) / float(w))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(*clipping_range))

    def _make_lights(self):
        from pxr import UsdLux

        self.key = UsdLux.DistantLight.Define(self.stage, "/World/KeyLight")
        self.key.CreateAngleAttr(0.5)  # small angular size -> sharp specular highlight
        self.dome = UsdLux.DomeLight.Define(self.stage, "/World/EnvLight")

    def _head_root(self, identity: str) -> str:
        return "/World/Heads/" + "".join(c if c.isalnum() else "_" for c in identity)

    def _load_asset(self, identity: str):
        """Show one head. Every head is referenced once under its own prim and switched by
        visibility. (Deleting /World/Head and re-referencing another asset at the same path
        left the RTX renderer drawing the first head's mesh with the new texture.)"""
        if identity == self._loaded_identity:
            return
        from isaacsim.core.utils.prims import is_prim_path_valid
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, UsdGeom

        root = self._head_root(identity)
        if not is_prim_path_valid(root):
            add_reference_to_stage(self.assets[identity], root)
            xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(root))
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*self.center))
            # Assets are converted to Z-up, meters, face along +x, origin at the head center
            # (see assets_tools/convert_heads.py), so no rotate op is needed.
        for other in self.assets:
            r = self._head_root(other)
            if is_prim_path_valid(r):
                img = UsdGeom.Imageable(self.stage.GetPrimAtPath(r))
                img.MakeVisible() if other == identity else img.MakeInvisible()
        self._loaded_identity = identity
        self._needs_long_settle = True  # shaders/textures of a new asset take many frames to stream in

    def _set_lighting(self, s: Dict):
        from pxr import Gf, UsdGeom

        self.key.GetIntensityAttr().Set(self.key_base * float(s["light_intensity"]))
        self.dome.GetIntensityAttr().Set(self.dome_base * float(s["env_intensity"]))
        # DistantLight shines along its local -Z. Tilt about Y by (90 - el), then yaw about Z by az,
        # so light travels along -sph_to_vec(az, el), i.e. it comes FROM (light_az, light_el).
        # (The original rotateZYX(0, -(90-el), az) lit the back of the head and ignored az;
        # checked numerically with pxr.)
        xf = UsdGeom.Xformable(self.key.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0 - float(s["light_el"]), float(s["light_az"])))

    def _set_roughness(self, roughness):
        if roughness is None or not self.skin_shader_path:
            return
        from pxr import Sdf, UsdShade

        path = self.skin_shader_path
        if path.startswith("/World/Head/") and self._loaded_identity is not None:
            path = self._head_root(self._loaded_identity) + path[len("/World/Head"):]
        shader = UsdShade.Shader(self.stage.GetPrimAtPath(path))
        # OmniPBR uses 'reflection_roughness_constant'; UsdPreviewSurface uses 'roughness'.
        for name in ("reflection_roughness_constant", "roughness"):
            inp = shader.GetInput(name)
            if inp:
                inp.Set(float(roughness))
                return
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))

    def _settle(self):
        n = self.warmup * (8 if self._needs_long_settle else 1)
        for _ in range(n):
            self.world.step(render=True)
        self._needs_long_settle = False

    def _grab(self):
        """Read the camera buffers. The annotators return empty arrays for the first frames
        (and while new materials compile), so keep stepping until data arrives."""
        for _ in range(300):
            rgba = self.camera.get_rgba()
            depth = self.camera.get_depth()
            if rgba is not None and depth is not None and np.size(rgba) > 0 and np.size(depth) > 0:
                return rgba, depth
            self.world.step(render=True)
        return None, None

    # -------------------------------------------------------------- interface
    def identities(self) -> List[str]:
        return list(self.assets)

    def load_scene(self, scene: Dict) -> None:
        self._load_asset(scene["identity"])
        self._set_lighting(scene)
        self._set_roughness(scene.get("roughness"))
        self._scene = dict(scene)

    def move_to(self, vp: Viewpoint) -> None:
        vp = Viewpoint(float(vp.azimuth), float(vp.elevation), self.radius)
        if self._arm is not None:
            self._arm(self, vp)
        else:
            pos = self.center + vp.position()
            self.camera.set_world_pose(pos, _look_at_quat_wxyz(pos, self.center), camera_axes="world")
        self._vp = vp

    def current_viewpoint(self) -> Viewpoint:
        return self._vp

    def render(self) -> Frame:
        try:
            self._settle()
            rgba, depth = self._grab()
        except Exception as e:  # infrastructure failure, never a quality label
            raise RenderError(f"isaac render failed: {e}") from e
        if rgba is None or depth is None or np.size(rgba) == 0:
            raise RenderError("isaac returned an empty frame after 300 extra steps")
        return Frame(np.asarray(rgba)[..., :3].astype(np.uint8), np.asarray(depth, dtype=np.float32), self._vp)

    def get_state(self):
        return {"scene": dict(self._scene), "vp": self._vp}

    def set_state(self, state) -> None:
        if state["scene"] != self._scene:
            self.load_scene(state["scene"])
        self.move_to(state["vp"])

    def oracle(self) -> OracleInfo:
        s = self._scene
        return OracleInfo(s["scene_id"], s["identity"], {k: v for k, v in s.items() if k not in ("scene_id", "identity")})

    def enrollment_frame(self, identity: str) -> Frame:
        prev_scene, prev_vp = self._scene, self._vp
        self.load_scene({"scene_id": -1, "identity": identity, **self.enroll_light})
        self.move_to(Viewpoint(0.0, 0.0, self.radius))
        frame = self.render()
        if prev_scene is not None:
            self.set_state({"scene": prev_scene, "vp": prev_vp})
        return frame

    def close(self) -> None:
        # SimulationApp.close() can terminate the interpreter, and callers (collect/evaluate)
        # write their outputs after env.close(). Defer the real shutdown to interpreter exit.
        if not getattr(self, "_close_registered", False):
            import atexit

            atexit.register(self._app.close)
            self._close_registered = True
