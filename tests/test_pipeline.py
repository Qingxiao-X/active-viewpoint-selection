"""Integration tests on the mock env: collector, training, checkpoint, evaluator."""

import copy
import json
import os

import numpy as np
import pytest

from avs.actions import ActionSpace, Viewpoint
from avs.collect import collect
from avs.config import load_config
from avs.dataset import Dataset, STATUS_INVALID, STATUS_OK, STATUS_RENDER_FAIL, rows_for_scenes
from avs.enroll import enroll_gallery
from avs.envs.mock_env import MockPerception, MockViewpointEnv
from avs.evaluate import build_episodes, evaluate, run_episode
from avs.features import FeatureSchema
from avs.model import QModel
from avs.policies import GreedyPreviewPolicy, LearnedPolicy, Policy
from avs.scenes import sample_scenes
from avs.train import train
from avs.types import PolicyObservation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("avs")
    cfg = load_config(os.path.join(ROOT, "configs", "mock.yaml"), [
        f"collect.out_dir={d}/data", "collect.n_scenes=40", "collect.starts_per_scene=3",
        "collect.verify_restore_every=5", "env.mock.render_failure_prob=0.02",
        f"train.out_dir={d}/run", f"train.datasets=[{d}/data]", "train.epochs=8",
        f"eval.checkpoint={d}/run", f"eval.gallery_file={d}/data/gallery.npz", f"eval.out_dir={d}/eval",
        "eval.episodes_per_scene=2", "eval.budget=4",
    ])
    ds = collect(cfg, verbose=False)
    metrics = train(cfg, verbose=False)
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval["env"]["mock"]["render_failure_prob"] = 0.0
    result = evaluate(cfg_eval, verbose=False)
    return {"dir": d, "cfg": cfg_eval, "ds": ds, "metrics": metrics, "result": result}


def test_collector_records_failures_and_invalid_actions_explicitly(run):
    ds = Dataset.load(f"{run['dir']}/data")
    st = ds.cols["status"]
    assert set(np.unique(st)) <= {STATUS_OK, STATUS_INVALID, STATUS_RENDER_FAIL}
    bad = st != STATUS_OK
    assert (st == STATUS_INVALID).any()
    assert np.isnan(ds.cols["label.label"][bad]).all()  # never turned into low-quality labels
    assert not np.isnan(ds.cols["label.label"][~bad]).any()
    n_fail = int((st == STATUS_RENDER_FAIL).sum())
    assert n_fail == sum(f["stage"] == "candidate" for f in ds.meta["failures"])
    assert ds.meta["schema_version"] == 1 and ds.meta["software_validation_only"]
    # each state has exactly one row per action (restore before every candidate)
    n_act = len(ActionSpace.from_config(ds.meta["action_space"]))
    _, counts = np.unique(ds.cols["group.state_id"], return_counts=True)
    assert (counts == n_act).all()
    # exact restoration check in the mock
    assert ds.meta["restore_checks"]
    assert all(c.get("max_emb_diff") in (0.0, None) for c in ds.meta["restore_checks"] if "error" not in c)


def test_invalid_actions_match_mask(run):
    ds = run["ds"]
    sp = ActionSpace.from_config(ds.meta["action_space"])
    c = ds.cols
    for i in range(0, len(ds), 37):
        vp = Viewpoint(c["obs.azimuth"][i], c["obs.elevation"][i], c["obs.radius"][i])
        valid = sp.valid_mask(vp)[c["act.action_index"][i]]
        assert valid == (c["status"][i] != STATUS_INVALID)


def test_splits_saved_grouped_without_leakage(run):
    with open(f"{run['dir']}/run/splits.json") as f:
        sp = json.load(f)["splits"]
    ds = run["ds"]
    seen = {}
    for name, ids in sp.items():
        for r in rows_for_scenes(ds, ids, ok_only=False):
            st = int(ds.cols["group.state_id"][r])
            assert seen.setdefault(st, name) == name  # a state never spans two splits


def test_features_use_only_obs_and_act_columns(run):
    ds = Dataset.load(f"{run['dir']}/data")
    schema = FeatureSchema()
    X = ds.features_raw(schema)
    for k in list(ds.cols):
        if k.startswith(("label.", "next.", "group.")):
            ds.cols[k] = np.zeros_like(ds.cols[k]) if ds.cols[k].dtype.kind in "fi" else ds.cols[k]
    np.testing.assert_array_equal(np.nan_to_num(X, nan=-9), np.nan_to_num(ds.features_raw(schema), nan=-9))


def test_checkpoint_roundtrip_and_inference(run):
    ck = f"{run['dir']}/run"
    for f in ("model.pt", "normalizer.json", "feature_schema.json", "model_config.json", "config.json",
              "splits.json", "metrics.json"):
        assert os.path.exists(os.path.join(ck, f)), f
    m1, m2 = QModel.load(ck), QModel.load(ck)
    ds = run["ds"]
    X = ds.features_raw(m1.schema, np.arange(50))
    np.testing.assert_allclose(m1.predict_raw(X), m2.predict_raw(X))
    sp = ActionSpace.from_config(ds.meta["action_space"])
    pol = LearnedPolicy(sp, m1, movement_cost_weight=0.1)
    pol.reset(0)
    obs = PolicyObservation(sp.az_limits[1], sp.el_limits[1], sp.radius, False, *([None] * 16))
    a = pol.act(obs)  # all perception fields missing, at a corner of the box
    assert sp.valid_mask(obs.viewpoint)[a.index]
    assert np.isfinite(pol.last_scores).all()


def test_selected_checkpoint_is_best_validation(run):
    m = run["metrics"]
    vals = [h["val_loss"] for h in m["history"]]
    assert m["best_val_loss"] == pytest.approx(min(vals))
    assert m["val"]["loss"] == pytest.approx(min(vals), rel=1e-5)


class _SpyPolicy(Policy):
    name = "spy"

    def reset(self, seed=0):
        super().reset(seed)
        self.seen = []

    def _choose(self, obs, acts):
        self.seen.append(obs)
        return acts[0]


def test_evaluator_shares_episodes_and_passes_only_observations(run):
    cfg = run["cfg"]
    env, per = MockViewpointEnv(), MockPerception()
    g = enroll_gallery(env, per, env.identities())
    sp = ActionSpace.from_config(cfg["action_space"])
    eps1 = build_episodes(cfg, env.identities())
    eps2 = build_episodes(cfg, env.identities())
    assert [(e["scene"]["scene_id"], e["start"]) for e in eps1] == [(e["scene"]["scene_id"], e["start"]) for e in eps2]
    rule = {"min_top1_similarity": 2.0}  # never accept -> runs full budget
    spy = _SpyPolicy(sp)
    rec = run_episode(env, per, g, sp, spy, eps1[0], rule, budget=3)
    assert rec["outcome"] == "budget_exhausted" and rec["steps"] == 3 and rec["renders"] == 4
    assert all(type(o) is PolicyObservation for o in spy.seen)
    # preview baseline: previews are counted and state is restored after each
    gp = GreedyPreviewPolicy(sp)
    rec = run_episode(env, per, g, sp, gp, eps1[0], rule, budget=2)
    assert rec["previews"] > 0 and rec["acquisitions"] == rec["renders"] + rec["previews"]
    assert rec["path_length"] > 0


def test_evaluation_outputs(run):
    out = f"{run['dir']}/eval"
    for f in ("summary.json", "summary.csv", "episodes.csv", "episodes.jsonl", "success_vs_budget.csv",
              "by_condition.csv", "paired.csv", "success_vs_budget.png", "success_by_condition.png", "path_length.png"):
        assert os.path.exists(os.path.join(out, f)), f
    r = run["result"]
    assert "MOCK" in r["note"]
    assert r["preview_baselines"] == ["greedy_preview"]
    for p, s in r["summary"].items():
        curve = s["success_vs_budget"]
        assert all(b >= a for a, b in zip(curve, curve[1:]))  # monotone in budget
        assert s["success_rate"] == pytest.approx(curve[-1])
        assert s["success_rate"] + s["incorrect_accept_rate"] + s["budget_exhausted_rate"] == pytest.approx(1.0)
    # step-0 outcome is identical across policies (same start views, same rule)
    eps = [json.loads(l) for l in open(os.path.join(out, "episodes.jsonl"))]
    step0 = {}
    for e in eps:
        k = (e["episode_id"], e["accepted_step"] == 0)
        step0.setdefault(e["episode_id"], set()).add(k[1])
    assert all(len(v) == 1 for v in step0.values())
