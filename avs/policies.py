"""Viewpoint-selection policies with a common interface::

    policy.reset(seed)
    action = policy.act(observation)        # observation: PolicyObservation

Every policy gets the same ``ActionSpace`` (and so the same valid-action
mask and movement cost) and only ``PolicyObservation`` inputs.

Baselines here are MINIMAL stand-ins written because the repository had no
Random/Greedy implementation. If teammates provide theirs, wrap them in this
interface and register them in ``POLICIES``.

* ``random``            uniform over valid actions.
* ``greedy_heuristic``  no previews; observable-only rule (toward frontal, away
                        from the observed glare centroid).
* ``greedy_preview``    renders every valid neighbor before moving and picks the
                        best *observable* confidence. These previews are extra
                        acquisitions: the evaluator counts them and reports this
                        policy as a preview (oracle-access) baseline.
* ``learned``           argmax_a  q_hat(obs, a) - movement_cost_weight * cost(a).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .actions import Action, ActionSpace
from .features import FeatureSchema
from .types import PolicyObservation


class Policy:
    name = "base"
    uses_preview = False

    def __init__(self, space: ActionSpace, revisit_penalty: float = 0.0):
        self.space = space
        self.revisit_penalty = float(revisit_penalty)
        self._visited: set = set()
        self.last_inference_seconds = 0.0

    def reset(self, seed: int = 0) -> None:
        self._visited = set()

    def _key(self, vp):
        return (round(vp.azimuth, 3), round(vp.elevation, 3))

    def valid_actions(self, obs: PolicyObservation) -> List[Action]:
        mask = self.space.valid_mask(obs.viewpoint)
        return [a for a in self.space.actions if mask[a.index]]

    def _revisit(self, obs: PolicyObservation, acts: List[Action]) -> np.ndarray:
        """1.0 where the action returns to a viewpoint already visited this episode."""
        return np.array([float(self._key(self.space.apply(obs.viewpoint, a)) in self._visited) for a in acts])

    def act(self, obs: PolicyObservation) -> Action:
        self._visited.add(self._key(obs.viewpoint))
        acts = self.valid_actions(obs)
        if not acts:
            raise RuntimeError("no valid action from this viewpoint")
        t = time.perf_counter()
        a = self._choose(obs, acts)
        self.last_inference_seconds = time.perf_counter() - t
        return a

    def _choose(self, obs: PolicyObservation, acts: List[Action]) -> Action:  # pragma: no cover
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def reset(self, seed: int = 0) -> None:
        super().reset(seed)
        self.rng = np.random.default_rng(seed)

    def _choose(self, obs, acts):
        return acts[int(self.rng.integers(len(acts)))]


class GreedyHeuristicPolicy(Policy):
    """Observable-only heuristic. Image-plane glare centroid convention: +cx =
    image right ~ +azimuth side, +cy = image down ~ highlight below the camera.
    Moving the camera away from the highlight side shifts it off the face."""

    name = "greedy_heuristic"

    def __init__(self, space, revisit_penalty=0.0, frontal_weight=1.0, glare_weight=2.0):
        super().__init__(space, revisit_penalty)
        self.fw, self.gw = float(frontal_weight), float(glare_weight)

    def _choose(self, obs, acts):
        vp = obs.viewpoint
        g = obs.glare_fraction or 0.0
        cx = obs.glare_cx or 0.0
        cy = obs.glare_cy or 0.0
        scores = []
        for a in acts:
            nxt = self.space.apply(vp, a)
            frontal = -(abs(nxt.azimuth) / 90.0 + abs(nxt.elevation) / 90.0)
            away = -np.sign(cx) * a.d_az / 30.0 + np.sign(cy) * a.d_el / 15.0
            scores.append(self.fw * frontal + self.gw * g * away)
        scores = np.asarray(scores) - self.revisit_penalty * self._revisit(obs, acts)
        return acts[int(np.argmax(scores))]


class GreedyPreviewPolicy(Policy):
    """One-step lookahead using preview renders of every valid neighbor.

    ``previewer(action) -> PolicyObservation`` is injected by the evaluator,
    which renders the candidate, restores the state and counts the acquisition.
    The score uses only observable confidence (never ground truth).
    """

    name = "greedy_preview"
    uses_preview = True

    def __init__(self, space, revisit_penalty=0.0, cost_weight=0.0):
        super().__init__(space, revisit_penalty)
        self.cost_weight = float(cost_weight)
        self.previewer: Optional[Callable[[Action], PolicyObservation]] = None

    @staticmethod
    def confidence(o: PolicyObservation) -> float:
        if not o.face_detected or o.top1_similarity is None:
            return -1.0
        return o.top2_margin if o.top2_margin is not None else o.top1_similarity

    def _choose(self, obs, acts):
        if self.previewer is None:
            raise RuntimeError("greedy_preview needs an evaluator-provided previewer")
        t_policy = time.perf_counter()
        conf, t_prev = [], 0.0
        for a in acts:
            t = time.perf_counter()
            o = self.previewer(a)
            t_prev += time.perf_counter() - t
            conf.append(self.confidence(o))
        cost = np.array([self.space.movement_cost(obs.viewpoint, a) for a in acts])
        s = np.asarray(conf) - self.cost_weight * cost - self.revisit_penalty * self._revisit(obs, acts)
        choice = acts[int(np.argmax(s))]
        # exclude preview rendering time from the decision latency
        self._preview_seconds = t_prev
        self._decision_seconds = time.perf_counter() - t_policy - t_prev
        return choice

    def act(self, obs):
        a = super().act(obs)
        self.last_inference_seconds = self._decision_seconds
        return a


class LearnedPolicy(Policy):
    name = "learned"

    def __init__(self, space, model, movement_cost_weight: float = 0.0, revisit_penalty: float = 0.0):
        super().__init__(space, revisit_penalty)
        self.model = model
        self.schema: FeatureSchema = model.schema
        self.w = float(movement_cost_weight)
        self.last_scores: Optional[np.ndarray] = None

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, space: ActionSpace, **kw) -> "LearnedPolicy":
        from .model import QModel  # torch import only when needed

        return cls(space, QModel.load(ckpt_dir), **kw)

    def score(self, obs: PolicyObservation, acts: List[Action]) -> np.ndarray:
        q_hat = self.model.predict_raw(self.schema.candidates_raw(obs, self.space, acts))
        cost = np.array([self.space.movement_cost(obs.viewpoint, a) for a in acts])
        return q_hat - self.w * cost

    def _choose(self, obs, acts):
        s = self.score(obs, acts) - self.revisit_penalty * self._revisit(obs, acts)
        self.last_scores = s
        return acts[int(np.argmax(s))]


POLICIES = {"random", "greedy_heuristic", "greedy_preview", "learned"}


def build_policy(name: str, space: ActionSpace, pcfg: Dict, checkpoint: Optional[str] = None) -> Policy:
    c = dict(pcfg.get(name, {}) or {})
    common_revisit = float(pcfg.get("revisit_penalty", 0.0))
    if name == "random":
        return RandomPolicy(space)
    if name == "greedy_heuristic":
        return GreedyHeuristicPolicy(space, c.get("revisit_penalty", common_revisit),
                                     c.get("frontal_weight", 1.0), c.get("glare_weight", 2.0))
    if name == "greedy_preview":
        return GreedyPreviewPolicy(space, c.get("revisit_penalty", common_revisit), c.get("cost_weight", 0.0))
    if name == "learned":
        ckpt = c.get("checkpoint") or checkpoint
        if not ckpt:
            raise ValueError("learned policy needs a checkpoint directory")
        return LearnedPolicy.from_checkpoint(ckpt, space, movement_cost_weight=c.get("movement_cost_weight", 0.0),
                                             revisit_penalty=c.get("revisit_penalty", common_revisit))
    raise ValueError(f"unknown policy {name!r}; known: {sorted(POLICIES)}")
