"""Minimal Isaac Sim evidence capture for two presentation slides:

  Slide 1 "Motivation & Baseline: Face Perception under Specular Glare"
  Slide 4 "Simulation Setup: Controllable Lighting & 4-Way Camera Motion"

No model training. Section 5 (ArcFace) is attempted only with --with-arcface,
and only reports real detector/similarity output -- never an invented score.

Reuses the project's existing Isaac adapter (avs/envs/isaac_env.py) for scene,
camera and lighting setup instead of re-deriving it, and the asset converted by
tools/assets/convert_heads.py (see assets_usd/assets_manifest.json for license).

Run inside Isaac Sim's python (see hprc/evidence_job.sbatch):
    python.sh scripts/evidence_capture.py --usd <lee_perry_smith.usd> \
        --manifest <assets_usd/assets_manifest.json> --out results/evidence_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IsaacViewpointEnv.__init__ creates the one-and-only SimulationApp for this process
# (and already applies the overscan-fitOutputToDataWindow fix). A second SimulationApp
# instance here crashes Kit, so nothing in this module may import isaacsim/pxr eagerly.
from PIL import Image  # noqa: E402

from avs.actions import sph_to_vec  # noqa: E402
from avs.envs.isaac_env import IsaacViewpointEnv, _look_at_quat_wxyz  # noqa: E402

RESOLUTION = (640, 480)
RADIUS0 = 0.6          # m, matches configs/isaac_heads.yaml
RADIUS_LIMITS = (0.35, 0.9)   # m, design choice: head bbox half-width ~0.13m: 0.35m keeps
                               # the whole head in frame, 0.9m is an arbitrary "far" bound.
AZ_LIMITS = (-75.0, 75.0)     # deg, matches configs/isaac_heads.yaml action_space
EL_LIMITS = (-30.0, 45.0)
ORBIT_STEP = 15.0      # deg
RADIAL_STEP = 0.1      # m

# Lighting presets. "glare" reuses scripts/isaac_preflight.py's validated
# glossy_glare preset; "clean" is a diffuse, well-exposed preset for contrast.
LIGHT_CLEAN = dict(light_intensity=0.6, env_intensity=0.6, light_az=25.0, light_el=20.0, roughness=0.6)
LIGHT_GLARE = dict(light_intensity=1.4, env_intensity=0.3, light_az=25.0, light_el=20.0, roughness=0.08)


def log(msg):
    print(f"[evidence] {msg}", flush=True)


def set_pose(env, az, el, radius):
    """Move the camera on the viewing sphere, keeping look-at on the head center.

    Bypasses IsaacViewpointEnv.move_to on purpose: that method hardcodes
    env.radius (the main project never moves radially), but this script needs
    FORWARD/BACKWARD too. Reuses the same look-at math (_look_at_quat_wxyz) and
    the same convention: +azimuth = toward the head's left (+y), +elevation = up.
    """
    pos = env.center + radius * sph_to_vec(az, el)
    quat = _look_at_quat_wxyz(pos, env.center)
    env.camera.set_world_pose(pos, quat, camera_axes="world")


def capture(env, long_settle=False, retries=5):
    """Step the sim to let rendering catch up, then grab RGB + depth.

    Reuses IsaacViewpointEnv's own settle/grab loops (same ones render() uses)
    instead of re-deriving the "wait for non-empty annotator buffers" logic.

    The very first render in a process is known-flaky in Isaac Sim 5.1 (camera
    annotator buffers can report empty for longer than one settle+grab cycle
    covers; isaac_preflight.py's exposure-calibration loop happens to absorb
    this by making many real renders before its first real shot, but this
    script's first capture() call IS the first-ever render). Retry with a
    fresh full-length settle instead of failing on the first empty frame.
    """
    last_err = None
    for attempt in range(retries):
        env._needs_long_settle = True  # force the full warmup*8 settle every retry
        env._settle()
        rgba, depth = env._grab()
        if rgba is not None and depth is not None and np.size(rgba) > 0 and np.size(depth) > 0:
            return np.asarray(rgba)[..., :3].astype(np.uint8), np.asarray(depth, dtype=np.float32)
        last_err = f"attempt {attempt + 1}/{retries}: empty frame after settle+grab"
        log(last_err)
    raise RuntimeError(f"isaac returned an empty frame after {retries} full settle+grab retries ({last_err})")


def depth_to_vis(depth, radius_hint):
    """Grayscale visualization: near=bright, far=dark, invalid=red.

    Invalid per avs/types.py Frame docstring: 0 / nan / inf.
    """
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = radius_hint - 0.35, radius_hint + 0.35  # clip window around the head
        norm = np.clip((hi - depth) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        for c in range(3):
            vis[..., c] = np.where(valid, gray, 0)
    vis[~valid] = (200, 30, 30)  # red = invalid pixel
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True, help="converted head USD, e.g. assets_usd/lee_perry_smith/lee_perry_smith.usd")
    ap.add_argument("--identity", default="lee_perry_smith")
    ap.add_argument("--manifest", required=True, help="assets_usd/assets_manifest.json, for license/source reporting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-arcface", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "rgbd_sample"), exist_ok=True)

    manifest = json.load(open(args.manifest))
    asset_info = next((m for m in manifest if m["identity"] == args.identity), None)

    report = {
        "environment": {},
        "asset": asset_info,
        "coordinate_convention": (
            "Head center is the world origin. Camera stays on a sphere of radius `radius` "
            "(m) around the head and always looks at the head center. Azimuth 0 / elevation 0 "
            "looks at the face from straight ahead (face points along +x). +azimuth rotates the "
            "camera toward the head's own left (+y); +elevation raises the camera (+z). "
            "LEFT = azimuth += +15 deg, RIGHT = azimuth -= 15 deg (orbit, radius unchanged). "
            "FORWARD = radius -= 0.1 m, BACKWARD = radius += 0.1 m (radial, azimuth/elevation unchanged). "
            f"Orbit limits az={AZ_LIMITS} el={EL_LIMITS} deg, radius limits={RADIUS_LIMITS} m "
            "(design choices, not validated as optimal)."
        ),
        "lighting_presets": {"clean": LIGHT_CLEAN, "glare": LIGHT_GLARE},
        "poses": {},
        "depth": {
            "definition": "distance from the camera's image plane to the surface, in meters "
                           "(IsaacViewpointEnv uses add_distance_to_image_plane_to_frame, not "
                           "Euclidean distance-to-camera).",
            "units": "meters",
            "invalid_value": "0, NaN, or Inf (no geometry hit / outside clip range)",
        },
    }

    try:
        import subprocess
        gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        gpu_name = f"unknown ({e})"
    report["environment"] = {
        "isaac_sim": "5.1.0 (isaacsim.exp.base-5.1.0, from Simulation App Starting log)",
        "gpu": gpu_name,
        "rendering_mode": "headless (SimulationApp({'headless': True}))",
        "resolution": list(RESOLUTION),
    }

    log(f"loading {args.usd}")
    env = IsaacViewpointEnv(
        assets={args.identity: args.usd},
        radius=RADIUS0,
        head_center=(0.0, 0.0, 0.0),
        resolution=RESOLUTION,
        warmup_frames=8,
        key_light_base_intensity=3000.0,
        dome_base_intensity=1000.0,
        skin_shader_path="/World/Head/Looks/Skin/Shader",
        focal_length=28.0,
        horizontal_aperture=20.955,
    )
    report["environment"]["gpu_rendering_works"] = "unknown (will be set after first render)"

    # In isaac_preflight.py, constructing InsightFacePerception (ONNX model load) happens
    # between env construction and the first render, giving Kit's async render/replicator
    # pipeline real wall-clock time to finish initializing. This script has no perception
    # stack, so the first-ever render is attempted almost immediately -- which empirically
    # comes back empty every time (not just occasionally), regardless of step-count retries.
    # Wait for real time to pass instead of just stepping the sim.
    import time
    log("warming up 30s before first render (async render pipeline init)")
    time.sleep(30)

    def shot(name, light, az, el, radius, long_settle=False):
        env.load_scene({"identity": args.identity, **light})
        set_pose(env, az, el, radius)
        rgb, depth = capture(env, long_settle=long_settle)
        Image.fromarray(rgb).save(os.path.join(args.out, f"{name}.png"))
        valid = np.isfinite(depth) & (depth > 0)
        report["poses"][name] = {
            "azimuth_deg": az, "elevation_deg": el, "radius_m": radius,
            "lighting": light,
            "depth_valid_fraction": float(valid.mean()),
            "depth_min_m": float(depth[valid].min()) if valid.any() else None,
            "depth_max_m": float(depth[valid].max()) if valid.any() else None,
        }
        log(f"{name}: az={az} el={el} r={radius} depth_valid_frac={report['poses'][name]['depth_valid_fraction']:.3f}")
        return rgb, depth

    # ---- Slide 1 evidence: clean vs. glare, same pose; then left/right under glare
    shot("clean_front", LIGHT_CLEAN, 0.0, 0.0, RADIUS0, long_settle=True)
    shot("glare_front", LIGHT_GLARE, 0.0, 0.0, RADIUS0, long_settle=True)
    shot("glare_left", LIGHT_GLARE, +ORBIT_STEP, 0.0, RADIUS0)
    shot("glare_right", LIGHT_GLARE, -ORBIT_STEP, 0.0, RADIUS0)
    report["environment"]["gpu_rendering_works"] = True  # reached here only if 4 real renders succeeded

    # ---- Slide 4 evidence: four discrete actions from a fresh start pose
    log("four-way motion check")
    az, el, radius = 0.0, 0.0, RADIUS0
    moves = []

    def clamp_and_log(action, new_az, new_el, new_radius):
        caz = float(np.clip(new_az, *AZ_LIMITS))
        cel = float(np.clip(new_el, *EL_LIMITS))
        crad = float(np.clip(new_radius, *RADIUS_LIMITS))
        hit_limit = (caz != new_az) or (cel != new_el) or (crad != new_radius)
        moves.append({"action": action, "requested": [new_az, new_el, new_radius],
                       "applied": [caz, cel, crad], "hit_limit": hit_limit})
        return caz, cel, crad

    env.load_scene({"identity": args.identity, **LIGHT_GLARE})
    for action, (daz, del_, drad) in [
        ("LEFT", (+ORBIT_STEP, 0.0, 0.0)),
        ("RIGHT", (-ORBIT_STEP, 0.0, 0.0)),   # single -15 deg action, back to az=0
        ("FORWARD", (0.0, 0.0, -RADIAL_STEP)),
        ("BACKWARD", (0.0, 0.0, +RADIAL_STEP)),  # single +0.1 m action, back to radius0
    ]:
        az, el, radius = clamp_and_log(action, az + daz, el + del_, radius + drad)
        set_pose(env, az, el, radius)
        rgb, depth = capture(env)
        log(f"{action} -> az={az:.1f} el={el:.1f} r={radius:.2f}")
        if action == "FORWARD":  # save the synchronized RGB-D pair here
            Image.fromarray(rgb).save(os.path.join(args.out, "rgbd_sample", "rgb.png"))
            np.save(os.path.join(args.out, "rgbd_sample", "depth_raw.npy"), depth)
            valid = np.isfinite(depth) & (depth > 0)
            Image.fromarray(depth_to_vis(depth, radius)).save(
                os.path.join(args.out, "rgbd_sample", "depth_vis.png"))
            report["rgbd_sample"] = {
                "pose": {"azimuth_deg": az, "elevation_deg": el, "radius_m": radius},
                "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
                "depth_valid_fraction": float(valid.mean()),
                "depth_min_m": float(depth[valid].min()) if valid.any() else None,
                "depth_max_m": float(depth[valid].max()) if valid.any() else None,
            }
    report["four_way_motion"] = moves

    # scene_overview.png is NOT produced here: an in-engine render with small marker
    # spheres at the camera positions was tried and discarded (they render as a
    # near-skin-color smudge under the scene's lighting -- not legible). See
    # tools/assets/scene_overview_diagram.py for a labeled schematic built from this
    # script's own manifest.json (real poses, real lighting, no invented numbers) plus
    # the captured clean_front.png, run separately with a modern Python (no GPU needed).

    # ---- Section 5 (optional, real-or-nothing)
    report["arcface"] = {"attempted": bool(args.with_arcface)}
    if args.with_arcface:
        try:
            from avs.perception import InsightFacePerception
            perc = InsightFacePerception(model_name="buffalo_l",
                                          root="/scratch/user/chenzc2001/qx/689/hprc/insightface",
                                          providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            # Reference = clean_front pose/lighting, recomputed now that perception is loaded.
            env.load_scene({"identity": args.identity, **LIGHT_CLEAN})
            set_pose(env, 0.0, 0.0, RADIUS0)
            ref_rgb, _ = capture(env, long_settle=True)
            from avs.types import Frame
            from avs.actions import Viewpoint as VP
            ref_frame = Frame(ref_rgb, None, VP(0.0, 0.0, RADIUS0))
            ref_perc = perc(ref_frame)
            if not ref_perc.face_detected or ref_perc.embedding is None:
                report["arcface"]["result"] = "reference frontal view: face NOT detected"
            else:
                ref_emb = ref_perc.embedding
                results = {}
                for name, (az_, el_, rad_, light) in {
                    "clean_front": (0.0, 0.0, RADIUS0, LIGHT_CLEAN),
                    "glare_front": (0.0, 0.0, RADIUS0, LIGHT_GLARE),
                    "glare_left": (+ORBIT_STEP, 0.0, RADIUS0, LIGHT_GLARE),
                    "glare_right": (-ORBIT_STEP, 0.0, RADIUS0, LIGHT_GLARE),
                }.items():
                    env.load_scene({"identity": args.identity, **light})
                    set_pose(env, az_, el_, rad_)
                    rgb, _ = capture(env)
                    frame = Frame(rgb, None, VP(az_, el_, rad_))
                    pc = perc(frame)
                    if not pc.face_detected or pc.embedding is None:
                        results[name] = {"detected": False}
                    else:
                        cos_sim = float(np.dot(ref_emb, pc.embedding))
                        results[name] = {"detected": True, "det_score": pc.det_score,
                                          "cosine_similarity_to_reference": cos_sim}
                report["arcface"]["result"] = results
        except Exception as e:
            report["arcface"]["result"] = f"pipeline unavailable or failed: {type(e).__name__}: {e}"

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"wrote {args.out}/manifest.json")
    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
