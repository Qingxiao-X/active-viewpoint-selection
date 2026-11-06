"""Unit tests: actions, labels, image metrics, splits, information boundary."""

import dataclasses

import numpy as np
import pytest

from avs.actions import ActionSpace, Viewpoint
from avs.dataset import check_no_leakage, grouped_split
from avs.envs.mock_env import MockPerception, MockViewpointEnv
from avs.enroll import enroll_gallery
from avs.features import FORBIDDEN_FEATURE_NAMES, FeatureSchema, Normalizer
from avs.perception import Gallery, build_observation, compute_label, depth_metrics, glare_metrics
from avs.scenes import sample_scenes
from avs.types import PerceptionResult, PolicyObservation


# ------------------------------------------------------------------ actions
def test_valid_mask_and_bounded_motion():
    sp = ActionSpace(az_limits=(-60, 60), el_limits=(-20, 30))
    corner = Viewpoint(60, 30, sp.radius)
    mask = sp.valid_mask(corner)
    for a, ok in zip(sp.actions, mask):
        assert ok == (a.d_az <= 0 and a.d_el <= 0)
        if ok:
            assert sp.in_bounds(sp.apply(corner, a))
    # random walks with masking never leave the box
    rng = np.random.default_rng(0)
    vp = Viewpoint(0, 0, sp.radius)
    for _ in range(500):
        valid = np.flatnonzero(sp.valid_mask(vp))
        vp = sp.apply(vp, sp.actions[int(rng.choice(valid))])
        assert sp.in_bounds(vp)


def test_noop_excluded_and_cost_is_arc_length():
    sp = ActionSpace(radius=0.5)
    assert all(not (a.d_az == 0 and a.d_el == 0) for a in sp.actions)
    a = next(a for a in sp.actions if a.d_az == 30 and a.d_el == 0)
    assert sp.movement_cost(Viewpoint(0, 0, 0.5), a) == pytest.approx(0.5 * np.deg2rad(30))


# ------------------------------------------------------------------ labels
def _gallery(n=3):
    e = np.eye(n, 8)
    return Gallery([f"p{i}" for i in range(n)], e)


def test_identity_margin_label_and_detection_failure():
    g = _gallery(3)
    emb = np.array([0.8, 0.5, 0.1, 0, 0, 0, 0, 0])
    res = PerceptionResult(True, embedding=emb / np.linalg.norm(emb), glare_fraction=0.2, depth_valid_fraction=0.9)
    s = g.similarities(res.embedding)
    lab = compute_label(res, g, "p0", {"mode": "auto"})
    assert lab["q_id"] == pytest.approx(s[0] - s[1])
    assert lab["label"] == pytest.approx(lab["q_id"])  # aux weights default to 0
    lab2 = compute_label(res, g, "p0", {"aux_glare_weight": 1.0})
    assert lab2["label"] == pytest.approx(lab["q_id"] - 0.2)
    fail = compute_label(PerceptionResult(False), g, "p0", {"detection_failure_value": -0.7})
    assert fail["q_id"] == -0.7


def test_single_identity_falls_back_to_similarity():
    g = _gallery(1)
    res = PerceptionResult(True, embedding=np.eye(1, 8)[0])
    assert compute_label(res, g, "p0", {"mode": "auto"})["q_id"] == pytest.approx(1.0)
    assert g.match(res.embedding)[2] is None  # no margin observable with one identity
    with pytest.raises(ValueError):
        g.identity_quality(res.embedding, "p0", mode="margin")


# ------------------------------------------------------------------ image metrics
def test_glare_and_depth_metrics():
    rgb = np.full((100, 100, 3), 120, np.uint8)
    rgb[10:30, 60:90] = 255  # white highlight, upper right of the box
    frac, cx, cy = glare_metrics(rgb, (0, 0, 100, 100))
    assert frac == pytest.approx(600 / 10000)
    assert cx > 0 and cy < 0
    depth = np.full((100, 100), 0.6, np.float32)
    depth[:50] = np.nan
    vf, std = depth_metrics(depth, (0, 0, 100, 100))
    assert vf == pytest.approx(0.5) and std == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ splits
def test_grouped_split_no_leakage_and_heldout_identity():
    ids = [f"id_{i}" for i in range(6)]
    scenes = sample_scenes(120, ids, {}, seed=3)
    sp = grouped_split(scenes, (0.7, 0.15, 0.15), seed=0, n_heldout_identities=2)
    all_ids = [s for v in sp.values() for s in v]
    assert len(all_ids) == len(set(all_ids)) == 120
    held = {s["identity"] for s in scenes if s["scene_id"] in sp["test_heldout_identity"]}
    assert len(held) == 2
    for name in ("train", "val", "test"):
        assert not ({s["identity"] for s in scenes if s["scene_id"] in sp[name]} & held)
    with pytest.raises(AssertionError):
        check_no_leakage({"train": [1, 2], "test": [2]})


# ------------------------------------------------------------------ information boundary
def test_policy_observation_has_no_privileged_fields():
    names = {f.name for f in dataclasses.fields(PolicyObservation)}
    assert not (names & {"identity", "gt_identity", "scene_id", "light_az", "light_el", "light_intensity",
                         "env_intensity", "roughness", "label", "q_id"})
    with pytest.raises(ValueError):
        FeatureSchema(obs_fields=["azimuth", "roughness"])
    assert not (set(FeatureSchema().raw_names) & {f"obs.{n}" for n in FORBIDDEN_FEATURE_NAMES})


def test_observation_does_not_depend_on_ground_truth_metadata():
    """Same rendered frame + different privileged metadata -> identical features."""
    env, per = MockViewpointEnv(), MockPerception()
    g = enroll_gallery(env, per, env.identities())
    scene = sample_scenes(1, env.identities(), {}, seed=0)[0]
    env.load_scene(scene)
    env.move_to(Viewpoint(20, 10, env.radius))
    frame = env.render()
    obs1 = build_observation(frame.viewpoint, per(frame), g)
    env._scene = {**scene, "identity": "id_07" if scene["identity"] != "id_07" else "id_00",
                  "light_intensity": 99.0}  # tamper with oracle metadata only
    obs2 = build_observation(frame.viewpoint, per(frame), g)
    schema = FeatureSchema()
    sp = ActionSpace()
    X1 = schema.candidates_raw(obs1, sp, sp.actions)
    X2 = schema.candidates_raw(obs2, sp, sp.actions)
    np.testing.assert_array_equal(X1, X2)


def test_normalizer_missing_values_and_train_only_stats():
    schema = FeatureSchema(obs_fields=["azimuth", "elevation", "radius", "face_detected", "glare_cx"],
                           action_fields=["d_az"])
    X = np.array([[0, 0, .6, 1, 0.5, 15], [10, 5, .6, 1, np.nan, -15], [20, 0, .6, 0, -0.5, 30]], float)
    n = Normalizer.fit(X, schema)
    Z = n.transform(X)
    assert Z.shape == (3, 6 + 1)  # one optional column -> one missing indicator
    assert Z[1, 4] == 0.0 and Z[1, -1] == 1.0 and Z[0, -1] == 0.0
    assert n.mean[4] == pytest.approx(0.0)  # mean over present values only
    with pytest.raises(ValueError):  # required field missing
        Normalizer.fit(np.array([[np.nan, 0, .6, 1, 0, 0]]), schema)
