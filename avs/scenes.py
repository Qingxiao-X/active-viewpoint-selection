"""Scene randomization and condition binning.

A scene config is a plain dict (JSON-serializable)::

    {scene_id, identity, light_az [deg], light_el [deg], light_intensity [rel.],
     env_intensity [rel.], roughness [0..1]}

All of these except ``scene_id`` are privileged: they reach the environment
(to build the scene) and the evaluator (for condition grouping), never a policy.
Sampling is deterministic per (seed, scene_id).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

DEFAULT_RANGES = {
    "light_az": [-80.0, 80.0],  # directional (glare) light azimuth, deg, head frame
    "light_el": [-10.0, 50.0],
    "light_intensity": [0.3, 1.5],  # relative to a nominal key light
    "env_intensity": [0.2, 1.0],  # dome/environment light, relative
    "roughness": [0.05, 0.8],  # skin material roughness override
    # Second directional light, independently randomized over roughly the SAME azimuth/
    # elevation range the camera can actually reach (action_space az/el limits are
    # +-75/-30..45) -- a first attempt biased this to 90..270 (the head's back), which
    # never overlaps what the camera can see and made single- vs dual-light scenes nearly
    # indistinguishable in practice. Independently sampled from light_az/light_el, so it's
    # sometimes near the key light (reinforcing) and sometimes on the opposite side within
    # view (a second, competing highlight) -- breaking "move away from the one glare blob."
    # fill_light_intensity can be 0 (single-light, same as the original setup) up to
    # roughly as strong as the key light.
    "fill_light_az": [-75.0, 75.0],
    "fill_light_el": [-25.0, 45.0],
    "fill_light_intensity": [0.0, 1.3],
}


def sample_scene(scene_id: int, identities: Sequence[str], ranges: Dict, seed: int) -> Dict:
    rng = np.random.default_rng([int(seed), int(scene_id)])
    r = {**DEFAULT_RANGES, **(ranges or {})}
    scene = {"scene_id": int(scene_id), "identity": str(identities[int(rng.integers(len(identities)))])}
    for k in ("light_az", "light_el", "light_intensity", "env_intensity", "roughness",
              "fill_light_az", "fill_light_el", "fill_light_intensity"):
        lo, hi = r[k]
        scene[k] = float(rng.uniform(lo, hi))
    return scene


def sample_scenes(n: int, identities: Sequence[str], ranges: Dict, seed: int, start_id: int = 0) -> List[Dict]:
    return [sample_scene(start_id + i, identities, ranges, seed) for i in range(n)]


def scene_conditions(scene: Dict, bins: Dict | None = None) -> Dict[str, str]:
    """Discrete condition labels for grouped reporting (evaluation only)."""
    b = {"light_intensity": 0.9, "roughness": [0.3, 0.55], "light_offaxis": 35.0,
         "fill_light": 0.3, **(bins or {})}
    rough = scene["roughness"]
    r_lo, r_hi = b["roughness"]
    off = float(np.hypot(scene["light_az"], scene["light_el"]))
    out = {
        "light_intensity": "high" if scene["light_intensity"] >= b["light_intensity"] else "low",
        "roughness": "glossy" if rough < r_lo else ("medium" if rough < r_hi else "matte"),
        "light_offaxis": "frontal" if off < b["light_offaxis"] else "oblique",
    }
    if "fill_light_intensity" in scene:
        out["fill_light"] = "dual" if scene["fill_light_intensity"] >= b["fill_light"] else "single"
    return out
