"""Simulator-independent environment interface used by collector and evaluator.

A teammate's environment can be plugged in by subclassing ``ViewpointEnv``
(or wrapping their existing class) — nothing else in ``avs`` depends on the
simulator.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..actions import Viewpoint
from ..types import Frame, OracleInfo


class ViewpointEnv:
    #: True if ``set_state`` restores the exact renderer state; False if it
    #: reconstructs the scene deterministically from parameters.
    exact_restore: bool = True

    def identities(self) -> List[str]:
        """Enrollable identities (asset ids) available in this environment."""
        raise NotImplementedError

    def load_scene(self, scene: Dict) -> None:
        """Build/configure the scene (asset, lighting, material)."""
        raise NotImplementedError

    def move_to(self, vp: Viewpoint) -> None:
        """Command the camera (via arm/base controller if any) to ``vp``."""
        raise NotImplementedError

    def current_viewpoint(self) -> Viewpoint:
        raise NotImplementedError

    def render(self) -> Frame:
        """Acquire RGB-D at the current viewpoint. May raise RenderError."""
        raise NotImplementedError

    def get_state(self) -> Any:
        raise NotImplementedError

    def set_state(self, state: Any) -> None:
        raise NotImplementedError

    def oracle(self) -> OracleInfo:
        """Privileged metadata of the loaded scene (collector/evaluator only)."""
        raise NotImplementedError

    def enrollment_frame(self, identity: str) -> Frame:
        """Reference capture used to enroll ``identity`` (frontal, neutral light)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
