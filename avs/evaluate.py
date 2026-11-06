"""Shared evaluator: paired episodes, one stopping rule, one budget for all policies.

    python -m avs.evaluate --config configs/mock.yaml [key=value ...]

Protocol (identical for every policy):
  * Same scenes, start viewpoints, action space, perception, enrollment gallery.
  * Step 0 = initial view. After each observation the shared acceptance rule
    is checked on OBSERVABLE confidence (detection, top-1 similarity, top-1/2
    margin). If accepted, the episode stops and ground truth is used only to
    score the accepted prediction as correct or incorrect.
  * Each executed movement consumes one action; the episode ends unaccepted
    after ``budget`` movements (budget exhaustion).
  * ``greedy_preview`` renders candidates before moving; those renders are
    counted as ``previews`` and the policy is reported as a preview baseline.
  * Render failures end the episode as ``infra_error``; they are counted and
    excluded from rates, never scored as policy failures.

Because the stopping rule does not depend on the budget, success at every
smaller budget b is read off one run with the maximum budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Dict, List

import numpy as np

from .actions import ActionSpace
from .config import load_config
from .dataset import Dataset
from .enroll import enroll_gallery
from .envs import make_env
from .perception import Gallery, build_observation
from .policies import build_policy
from .scenes import sample_scenes, scene_conditions
from .types import PolicyObservation, RenderError

OUTCOMES = ("correct_accept", "incorrect_accept", "budget_exhausted", "infra_error")


def accept(obs: PolicyObservation, rule: Dict) -> bool:
    """Shared observable stopping criterion."""
    if not obs.face_detected or obs.top1_similarity is None:
        return False
    if obs.det_score is not None and obs.det_score < float(rule.get("min_det_score", 0.0)):
        return False
    if obs.top1_similarity < float(rule.get("min_top1_similarity", 0.0)):
        return False
    if obs.top2_margin is not None and obs.top2_margin < float(rule.get("min_margin", 0.0)):
        return False
    return True


def build_episodes(cfg: Dict, identities: List[str]) -> List[Dict]:
    ecfg = cfg["eval"]
    space = ActionSpace.from_config(cfg["action_space"])
    src = ecfg.get("scene_source", "test_split")
    scene_sets = []
    if src == "test_split":
        ckpt = ecfg.get("checkpoint") or cfg["train"]["out_dir"]
        with open(os.path.join(ckpt, "splits.json")) as f:
            sp = json.load(f)
        ds = Dataset.load_many(sp["datasets"] if isinstance(sp["datasets"], list) else [sp["datasets"]])
        by_id = {s["scene_id"]: s for s in ds.scenes}
        for name in ecfg.get("splits", ["test"]):
            for sid in sp["splits"].get(name, []):
                scene_sets.append((name, by_id[sid]))
    elif src == "fresh":
        scfg = cfg.get("scenes", {})
        for s in sample_scenes(int(ecfg["n_fresh_scenes"]), identities, scfg.get("ranges", {}),
                               int(ecfg.get("scene_seed", 12345)), int(ecfg.get("fresh_start_scene_id", 1_000_000))):
            scene_sets.append(("fresh", s))
    else:
        raise ValueError(f"unknown eval.scene_source {src!r}")
    max_scenes = ecfg.get("max_scenes")
    if max_scenes:
        scene_sets = scene_sets[: int(max_scenes)]
    eps, seed = [], int(ecfg.get("seed", 0))
    for split, s in scene_sets:
        rng = np.random.default_rng([seed, s["scene_id"]])
        for k in range(int(ecfg.get("episodes_per_scene", 1))):
            vp = space.sample_viewpoint(rng, ecfg.get("start_az_range"), ecfg.get("start_el_range"))
            eps.append({"episode_id": len(eps), "split": split, "scene": s, "start": vp,
                        "policy_seed": int(np.random.default_rng([seed, s["scene_id"], k, 99]).integers(2**31))})
    return eps


def run_episode(env, perception, gallery: Gallery, space: ActionSpace, policy, ep: Dict,
                rule: Dict, budget: int, cond_bins: Dict | None = None) -> Dict:
    scene = ep["scene"]
    env.load_scene(scene)
    env.move_to(ep["start"])
    gt = env.oracle().identity  # scoring only
    policy.reset(ep["policy_seed"])
    rec = {"policy": policy.name, "episode_id": ep["episode_id"], "split": ep["split"],
           "scene_id": scene["scene_id"], "identity": scene["identity"],
           "start_az": ep["start"].azimuth, "start_el": ep["start"].elevation,
           "renders": 0, "previews": 0, "path_length": 0.0, "steps": 0,
           "accepted_step": None, "predicted_identity": None}
    sense_t, infer_t, preview_t, trace = [], [], [], []

    def sense(step: int) -> PolicyObservation:
        t = time.perf_counter()
        frame = env.render()
        o = build_observation(env.current_viewpoint(), perception(frame), gallery, step)
        sense_t.append(time.perf_counter() - t)
        return o

    def previewer(action):
        state = env.get_state()
        try:
            env.move_to(space.apply(env.current_viewpoint(), action))
            t = time.perf_counter()
            o = build_observation(env.current_viewpoint(), perception(env.render()), gallery, rec["steps"])
            preview_t.append(time.perf_counter() - t)
            rec["previews"] += 1
            return o
        finally:
            env.set_state(state)

    if getattr(policy, "uses_preview", False):
        policy.previewer = previewer
    outcome = "budget_exhausted"
    try:
        obs = sense(0)
        rec["renders"] += 1
        for step in range(budget + 1):
            ok = accept(obs, rule)
            trace.append({"step": step, "az": obs.azimuth, "el": obs.elevation, "detected": obs.face_detected,
                          "top1": obs.top1_similarity, "margin": obs.top2_margin, "glare": obs.glare_fraction,
                          "accepted": ok})
            if ok:
                rec["accepted_step"] = step
                rec["predicted_identity"] = obs.predicted_identity
                outcome = "correct_accept" if obs.predicted_identity == gt else "incorrect_accept"
                break
            if step == budget:
                break
            vp = obs.viewpoint
            action = policy.act(obs)
            infer_t.append(policy.last_inference_seconds)
            if not space.valid_mask(vp)[action.index]:
                raise AssertionError(f"{policy.name} returned invalid action {action} at {vp}")
            rec["path_length"] += space.movement_cost(vp, action)
            env.move_to(space.apply(vp, action))
            rec["steps"] += 1
            obs = sense(step + 1)
            rec["renders"] += 1
    except RenderError as e:
        outcome = "infra_error"
        rec["error"] = str(e)
    rec["outcome"] = outcome
    rec["success"] = outcome == "correct_accept"
    rec["acquisitions"] = rec["renders"] + rec["previews"]
    rec["sense_ms_mean"] = 1e3 * float(np.mean(sense_t)) if sense_t else None
    rec["preview_ms_total"] = 1e3 * float(np.sum(preview_t)) if preview_t else 0.0
    rec["inference_ms_mean"] = 1e3 * float(np.mean(infer_t)) if infer_t else None
    rec["inference_ms_list"] = [1e3 * x for x in infer_t]
    rec["trace"] = trace
    for k, v in scene_conditions(scene, cond_bins).items():
        rec[f"cond.{k}"] = v
    return rec


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _rate(x):
    return float(np.mean(x)) if len(x) else float("nan")


def summarize(recs: List[Dict], budget: int) -> Dict:
    out = {}
    valid = [r for r in recs if r["outcome"] != "infra_error"]
    succ = [r for r in valid if r["success"]]
    lat = [x for r in valid for x in r["inference_ms_list"]]
    sense = [r["sense_ms_mean"] for r in valid if r["sense_ms_mean"] is not None]
    out.update({
        "n_episodes": len(recs), "n_valid": len(valid), "infra_errors": len(recs) - len(valid),
        "success_rate": _rate([r["success"] for r in valid]),
        "failure_rate": 1.0 - _rate([r["success"] for r in valid]) if valid else float("nan"),
        "incorrect_accept_rate": _rate([r["outcome"] == "incorrect_accept" for r in valid]),
        "budget_exhausted_rate": _rate([r["outcome"] == "budget_exhausted" for r in valid]),
        "mean_steps_to_success": _rate([r["accepted_step"] for r in succ]),
        "mean_path_length_m": _rate([r["path_length"] for r in valid]),
        "mean_path_length_success_m": _rate([r["path_length"] for r in succ]),
        "mean_previews": _rate([r["previews"] for r in valid]),
        "mean_acquisitions": _rate([r["acquisitions"] for r in valid]),
        "inference_ms_mean": _rate(lat), "inference_ms_p95": float(np.percentile(lat, 95)) if lat else float("nan"),
        "sense_ms_mean": _rate(sense),
        "success_vs_budget": [
            _rate([r["success"] and r["accepted_step"] <= b for r in valid]) for b in range(budget + 1)],
        "incorrect_accept_vs_budget": [
            _rate([r["outcome"] == "incorrect_accept" and r["accepted_step"] <= b for r in valid]) for b in range(budget + 1)],
    })
    return out


def paired_comparison(by_pol: Dict[str, List[Dict]], ref: str, n_boot: int = 2000, seed: int = 0) -> List[Dict]:
    rows = []
    if ref not in by_pol:
        return rows
    rng = np.random.default_rng(seed)
    a_map = {r["episode_id"]: r for r in by_pol[ref]}
    for other, recs in by_pol.items():
        if other == ref:
            continue
        pairs = [(a_map[r["episode_id"]], r) for r in recs
                 if r["episode_id"] in a_map and "infra_error" not in (r["outcome"], a_map[r["episode_id"]]["outcome"])]
        if not pairs:
            continue
        a = np.array([p[0]["success"] for p in pairs], dtype=float)
        b = np.array([p[1]["success"] for p in pairs], dtype=float)
        d = a - b
        boots = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n_boot)]
        rows.append({"policy": ref, "vs": other, "n_pairs": len(pairs), "success_diff": float(d.mean()),
                     "ci95_low": float(np.percentile(boots, 2.5)), "ci95_high": float(np.percentile(boots, 97.5)),
                     "both_success": int(((a == 1) & (b == 1)).sum()), "only_" + ref: int(((a == 1) & (b == 0)).sum()),
                     "only_other": int(((a == 0) & (b == 1)).sum()), "neither": int(((a == 0) & (b == 0)).sum())})
    return rows


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def evaluate(cfg: Dict, out_dir: str | None = None, verbose: bool = True) -> Dict:
    ecfg = cfg["eval"]
    out_dir = out_dir or ecfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    space = ActionSpace.from_config(cfg["action_space"])
    budget = int(ecfg["budget"])
    rule = dict(ecfg.get("accept", {}))
    if ecfg.get("accept_file"):
        with open(ecfg["accept_file"]) as f:
            rule.update(json.load(f)["accept"])
    cond_bins = cfg.get("scenes", {}).get("condition_bins")
    env, perception = make_env(cfg, space)
    try:
        identities = env.identities()
        gfile = ecfg.get("gallery_file")
        if gfile:
            gallery = Gallery.load(gfile)
            if gallery.ids != identities:
                raise ValueError("gallery identities do not match environment identities")
        else:
            gallery = enroll_gallery(env, perception, identities)
        episodes = build_episodes(cfg, identities)
        pcfg = cfg.get("policy", {})
        ckpt = ecfg.get("checkpoint") or cfg["train"]["out_dir"]
        policies = [build_policy(n, space, pcfg, ckpt) for n in ecfg["policies"]]
        recs: List[Dict] = []
        t0 = time.time()
        for p in policies:
            for ep in episodes:
                recs.append(run_episode(env, perception, gallery, space, p, ep, rule, budget, cond_bins))
            if verbose:
                pr = [r for r in recs if r["policy"] == p.name and r["outcome"] != "infra_error"]
                print(f"[eval] {p.name:17s} {len(pr)} valid episodes  success={_rate([r['success'] for r in pr]):.3f}  "
                      f"({time.time() - t0:.1f}s)")
    finally:
        env.close()

    by_pol: Dict[str, List[Dict]] = {}
    for r in recs:
        by_pol.setdefault(r["policy"], []).append(r)
    summary = {p: summarize(rs, budget) for p, rs in by_pol.items()}
    cond_keys = sorted({k for r in recs for k in r if k.startswith("cond.")})
    by_cond = []
    for p, rs in by_pol.items():
        for ck in cond_keys:
            for val in sorted({r[ck] for r in rs}):
                sub = [r for r in rs if r[ck] == val and r["outcome"] != "infra_error"]
                by_cond.append({"policy": p, "condition": ck[5:], "value": val, "n": len(sub),
                                "success_rate": _rate([r["success"] for r in sub]),
                                "incorrect_accept_rate": _rate([r["outcome"] == "incorrect_accept" for r in sub]),
                                "mean_path_length_m": _rate([r["path_length"] for r in sub])})
    splits_seen = sorted({r["split"] for r in recs})
    by_split = {s: {p: summarize([r for r in rs if r["split"] == s], budget) for p, rs in by_pol.items()}
                for s in splits_seen} if len(splits_seen) > 1 else {}
    paired = paired_comparison(by_pol, "learned", seed=int(ecfg.get("seed", 0))) if "learned" in by_pol else []

    backend = cfg.get("env", {}).get("backend", "mock")
    result = {
        "note": ("MOCK ENVIRONMENT: software validation only, not evidence of real face-perception performance"
                 if backend == "mock" else "isaac backend"),
        "budget": budget, "accept_rule": rule, "n_episodes_per_policy": len(episodes),
        "preview_baselines": [p.name for p in policies if p.uses_preview],
        "summary": summary, "by_split": by_split, "paired_vs_learned": paired,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    flat = [{"policy": p, **{k: v for k, v in s.items() if not isinstance(v, list)},
             "preview_baseline": p in result["preview_baselines"]} for p, s in summary.items()]
    _write_csv(os.path.join(out_dir, "summary.csv"), flat)
    _write_csv(os.path.join(out_dir, "success_vs_budget.csv"),
               [{"policy": p, "budget": b, "success_rate": s["success_vs_budget"][b],
                 "incorrect_accept_rate": s["incorrect_accept_vs_budget"][b]}
                for p, s in summary.items() for b in range(budget + 1)])
    _write_csv(os.path.join(out_dir, "by_condition.csv"), by_cond)
    _write_csv(os.path.join(out_dir, "paired.csv"), paired)
    _write_csv(os.path.join(out_dir, "episodes.csv"),
               [{k: v for k, v in r.items() if k not in ("trace", "inference_ms_list")} for r in recs])
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    try:
        from .plots import make_plots

        make_plots(out_dir, result, by_cond, recs)
    except ImportError as e:
        print(f"[eval] plots skipped: {e}")
    if verbose:
        print(f"[eval] {result['note']}")
        hdr = f"{'policy':17s} {'succ':>6s} {'wrong':>6s} {'exh':>6s} {'steps':>6s} {'path_m':>7s} {'prev':>6s} {'inf_ms':>7s} {'sense_ms':>8s}"
        print(hdr)
        for p, s in summary.items():
            print(f"{p:17s} {s['success_rate']:6.3f} {s['incorrect_accept_rate']:6.3f} {s['budget_exhausted_rate']:6.3f} "
                  f"{s['mean_steps_to_success']:6.2f} {s['mean_path_length_m']:7.3f} {s['mean_previews']:6.2f} "
                  f"{s['inference_ms_mean']:7.3f} {s['sense_ms_mean']:8.3f}")
        print(f"[eval] wrote {out_dir}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    evaluate(load_config(args.config, args.overrides), args.out)


if __name__ == "__main__":
    main()
