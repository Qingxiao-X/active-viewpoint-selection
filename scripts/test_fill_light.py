"""One-off sanity check for the new fill-light code (avs/envs/isaac_env.py,
avs/scenes.py) before committing GPU time to a full re-collect: render
single-light vs dual-light and save both for visual inspection.

    python.sh scripts/test_fill_light.py --usd <lee_perry_smith.usd> --out /tmp/fill_test
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from avs.actions import Viewpoint  # noqa: E402
from avs.envs.isaac_env import IsaacViewpointEnv  # noqa: E402


def log(msg):
    print(f"[test_fill_light] {msg}", flush=True)


def capture(env, retries=5):
    for attempt in range(retries):
        env._needs_long_settle = True
        env._settle()
        rgba, depth = env._grab()
        if rgba is not None and depth is not None and np.size(rgba) > 0 and np.size(depth) > 0:
            return np.asarray(rgba)[..., :3].astype(np.uint8)
        log(f"attempt {attempt + 1}/{retries}: empty frame")
    raise RuntimeError("empty frame after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = IsaacViewpointEnv(
        assets={"lee_perry_smith": args.usd}, radius=0.6, head_center=(0.0, 0.0, 0.0),
        resolution=(640, 480), warmup_frames=8,
        key_light_base_intensity=3000.0, dome_base_intensity=1000.0,
        skin_shader_path="/World/Head/Looks/Skin/Shader",
        focal_length=28.0, horizontal_aperture=20.955,
    )
    time.sleep(30)  # known Isaac Sim 5.1 first-render flakiness, see scripts/evidence_capture.py
    env.move_to(Viewpoint(0.0, 0.0, 0.6))  # __init__ never positions the camera; must do it explicitly

    env.load_scene({"identity": "lee_perry_smith", "light_intensity": 1.0, "env_intensity": 0.4,
                     "light_az": -60.0, "light_el": 20.0, "roughness": 0.08,
                     "fill_light_intensity": 0.0})
    Image.fromarray(capture(env)).save(os.path.join(args.out, "single_light.png"))
    log("saved single_light.png")

    env.load_scene({"identity": "lee_perry_smith", "light_intensity": 1.0, "env_intensity": 0.4,
                     "light_az": -60.0, "light_el": 20.0, "roughness": 0.08,
                     "fill_light_intensity": 1.0, "fill_light_az": 180.0, "fill_light_el": 20.0})
    Image.fromarray(capture(env)).save(os.path.join(args.out, "dual_light.png"))
    log("saved dual_light.png")

    env.load_scene({"identity": "lee_perry_smith", "light_intensity": 0.0, "env_intensity": 0.4,
                     "light_az": -60.0, "light_el": 20.0, "roughness": 0.08,
                     "fill_light_intensity": 1.0, "fill_light_az": 180.0, "fill_light_el": 20.0})
    Image.fromarray(capture(env)).save(os.path.join(args.out, "fill_only.png"))
    log("saved fill_only.png")

    env.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
