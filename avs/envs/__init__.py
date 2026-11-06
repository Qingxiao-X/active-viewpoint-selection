"""Environment backends. ``make_env`` builds one from a config dict."""

from __future__ import annotations

from typing import Dict, Tuple

from ..actions import ActionSpace
from ..perception import Perception
from .base import ViewpointEnv


def make_env(cfg: Dict, action_space: ActionSpace) -> Tuple[ViewpointEnv, Perception]:
    """Return (env, perception) for ``cfg['env']['backend']`` in {mock, isaac}."""
    env_cfg = dict(cfg.get("env", {}))
    backend = env_cfg.pop("backend", "mock")
    if backend == "mock":
        from .mock_env import MockPerception, MockViewpointEnv

        env = MockViewpointEnv(radius=action_space.radius, **env_cfg.get("mock", {}))
        return env, MockPerception()
    if backend == "isaac":
        from ..perception import InsightFacePerception
        from .isaac_env import IsaacViewpointEnv

        env = IsaacViewpointEnv(radius=action_space.radius, **env_cfg.get("isaac", {}))
        return env, InsightFacePerception(**cfg.get("perception", {}))
    raise ValueError(f"unknown env backend {backend!r}")
