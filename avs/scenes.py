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
}


def sample_scene(scene_id: int, identities: Sequence[str], ranges: Dict, seed: int) -> Dict:
    rng = np.random.default_rng([int(seed), int(scene_id)])
    r = {**DEFAULT_RANGES, **(ranges or {})}
    scene = {"scene_id": int(scene_id), "identity": str(identities[int(rng.integers(len(identities)))])}
    for k in ("light_az", "light_el", "light_intensity", "env_intensity", "roughness"):
        lo, hi = r[k]
        scene[k] = float(rng.uniform(lo, hi))
    return scene


def sample_scenes(n: int, identities: Sequence[str], ranges: Dict, seed: int, start_id: int = 0) -> List[Dict]:
    return [sample_scene(start_id + i, identities, ranges, seed) for i in range(n)]


def scene_conditions(scene: Dict, bins: Dict | None = None) -> Dict[str, str]:
    """Discrete condition labels for grouped reporting (evaluation only)."""
    b = {"light_intensity": 0.9, "roughness": [0.3, 0.55], "light_offaxis": 35.0, **(bins or {})}
    rough = scene["roughness"]
    r_lo, r_hi = b["roughness"]
    off = float(np.hypot(scene["light_az"], scene["light_el"]))
    return {
        "light_intensity": "high" if scene["light_intensity"] >= b["light_intensity"] else "low",
        "roughness": "glossy" if rough < r_lo else ("medium" if rough < r_hi else "matte"),
        "light_offaxis": "frontal" if off < b["light_offaxis"] else "oblique",
    }
