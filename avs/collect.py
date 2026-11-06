"""Offline data collection: label every valid candidate action from saved states.

For each randomized scene and sampled start viewpoint:
  1. render + perceive the current view -> PolicyObservation (obs.*)
  2. save env state
  3. for each candidate action: restore state, move, render, perceive, label
  4. restore state again before the next candidate

Candidate renders are an offline data-collection cost (counted in meta.json);
no policy ever sees them at evaluation time.

    python -m avs.collect --config configs/mock.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time
from collections import defaultdict
from typing import Dict

import numpy as np

from .actions import ActionSpace
from .config import load_config
from .dataset import (DATASET_SCHEMA_VERSION, STATUS_INVALID, STATUS_OK, STATUS_RENDER_FAIL, Dataset)
from .enroll import enroll_gallery
from .envs import make_env
from .features import ALL_ACTION_FIELDS, ALL_OBS_FIELDS, action_record, observation_record
from .perception import build_observation, compute_label
from .scenes import sample_scenes
from .types import RenderError

NAN = float("nan")


def collect(cfg: Dict, out_dir: str | None = None, verbose: bool = True) -> Dataset:
    ccfg = cfg["collect"]
    out_dir = out_dir or ccfg["out_dir"]
    space = ActionSpace.from_config(cfg["action_space"])
    env, perception = make_env(cfg, space)
    label_cfg = cfg.get("label", {})
    try:
        identities = env.identities()
        gallery = enroll_gallery(env, perception, identities)
        if len(gallery) < 2 and label_cfg.get("mode", "auto") != "similarity":
            print("[collect] only one identity enrolled: label falls back to reference similarity; "
                  "identity discrimination cannot be evaluated")
        scfg = cfg.get("scenes", {})
        scenes = sample_scenes(int(ccfg["n_scenes"]), identities, scfg.get("ranges", {}),
                               int(scfg.get("seed", 0)), int(ccfg.get("start_scene_id", 0)))
        cols: Dict[str, list] = defaultdict(list)
        failures, restore_checks = [], []
        n_render = n_cand_render = 0
        t_render = 0.0
        verify_every = int(ccfg.get("verify_restore_every", 0))
        state_id = 0
        t0 = time.time()
        for si, scene in enumerate(scenes):
            env.load_scene(scene)
            gt = env.oracle().identity  # label-only
            rng = np.random.default_rng([int(scfg.get("seed", 0)), scene["scene_id"], 7919])
            for _ in range(int(ccfg["starts_per_scene"])):
                vp = space.sample_viewpoint(rng, ccfg.get("start_az_range"), ccfg.get("start_el_range"))
                env.move_to(vp)
                try:
                    tr = time.perf_counter()
                    frame = env.render()
                    perc_cur = perception(frame)
                    t_render += time.perf_counter() - tr
                    n_render += 1
                except RenderError as e:
                    failures.append({"stage": "current_view", "scene_id": scene["scene_id"],
                                     "azimuth": vp.azimuth, "elevation": vp.elevation, "error": str(e)})
                    continue
                obs = build_observation(vp, perc_cur, gallery, step=0)
                orec = observation_record(obs, ALL_OBS_FIELDS)
                state = env.get_state()
                mask = space.valid_mask(vp)
                for a in space.actions:
                    cols["group.scene_id"].append(scene["scene_id"])
                    cols["group.state_id"].append(state_id)
                    cols["group.identity"].append(scene["identity"])
                    for f in ALL_OBS_FIELDS:
                        cols[f"obs.{f}"].append(orec[f])
                    arec = action_record(vp, a, space)
                    for f in ALL_ACTION_FIELDS:
                        cols[f"act.{f}"].append(arec[f])
                    cols["act.action_index"].append(a.index)
                    lab = {"q_id": NAN, "label": NAN, "aux_glare": NAN, "aux_depth": NAN}
                    nxt = {"detected": NAN, "top1": NAN, "margin": NAN, "correct": NAN}
                    if not mask[a.index]:
                        status = STATUS_INVALID
                    else:
                        env.set_state(state)
                        env.move_to(space.apply(vp, a))
                        try:
                            tr = time.perf_counter()
                            p2 = perception(env.render())
                            t_render += time.perf_counter() - tr
                            n_render += 1
                            n_cand_render += 1
                            status = STATUS_OK
                            lab = compute_label(p2, gallery, gt, label_cfg)
                            pred, top1, margin = gallery.match(p2.embedding if p2.face_detected else None)
                            nxt = {"detected": float(p2.face_detected),
                                   "top1": NAN if top1 is None else top1,
                                   "margin": NAN if margin is None else margin,
                                   "correct": NAN if pred is None else float(pred == gt)}
                        except RenderError as e:
                            status = STATUS_RENDER_FAIL
                            failures.append({"stage": "candidate", "scene_id": scene["scene_id"],
                                             "state_id": state_id, "action_index": a.index, "error": str(e)})
                    cols["status"].append(status)
                    for k, v in lab.items():
                        cols[f"label.{k}"].append(v)
                    # next-view observable confidence + correctness: for threshold
                    # calibration on validation scenes only, never features
                    for k, v in nxt.items():
                        cols[f"next.{k}"].append(v)
                    env.set_state(state)
                if verify_every and state_id % verify_every == 0:
                    try:
                        env.set_state(state)
                        p3 = perception(env.render())
                        n_render += 1
                        same_det = p3.face_detected == perc_cur.face_detected
                        diff = (float(np.max(np.abs(p3.embedding - perc_cur.embedding)))
                                if same_det and p3.embedding is not None and perc_cur.embedding is not None else None)
                        restore_checks.append({"state_id": state_id, "same_detection": same_det, "max_emb_diff": diff})
                    except RenderError as e:
                        restore_checks.append({"state_id": state_id, "error": str(e)})
                state_id += 1
            if verbose and ((si + 1) % max(1, len(scenes) // 10) == 0 or si + 1 == len(scenes)):
                print(f"[collect] scene {si + 1}/{len(scenes)}  states={state_id}  renders={n_render}  "
                      f"failures={len(failures)}  {time.time() - t0:.1f}s")
    finally:
        env.close()

    arrays = {}
    for k, v in cols.items():
        if k in ("group.identity", "status"):
            arrays[k] = np.array(v, dtype=str)
        elif k in ("group.scene_id", "group.state_id", "act.action_index"):
            arrays[k] = np.array(v, dtype=np.int64)
        else:
            arrays[k] = np.array(v, dtype=np.float64)
    meta = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "backend": cfg.get("env", {}).get("backend", "mock"),
        "software_validation_only": cfg.get("env", {}).get("backend", "mock") == "mock",
        "exact_restore": bool(getattr(env, "exact_restore", False)),
        "action_space": space.to_config(),
        "label": label_cfg,
        "n_identities": len(gallery),
        "identities": gallery.ids,
        "n_scenes": len(scenes),
        "n_states": state_id,
        "n_rows": int(len(arrays.get("status", []))),
        "status_counts": {s: int(np.sum(arrays["status"] == s)) for s in (STATUS_OK, STATUS_INVALID, STATUS_RENDER_FAIL)}
        if "status" in arrays else {},
        "collection_cost": {"renders_total": n_render, "candidate_renders": n_cand_render,
                            "render_perception_seconds": round(t_render, 3),
                            "wall_seconds": round(time.time() - t0, 3)},
        "failures": failures,
        "restore_checks": restore_checks,
        "config": cfg,
    }
    ds = Dataset(arrays, meta, scenes)
    ds.save(out_dir)
    gallery.save(os.path.join(out_dir, "gallery.npz"))
    if verbose:
        print(f"[collect] wrote {out_dir}: {meta['n_rows']} rows, status={meta['status_counts']}")
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dataset dir (default: collect.out_dir)")
    ap.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    args = ap.parse_args(argv)
    collect(load_config(args.config, args.overrides), args.out)


if __name__ == "__main__":
    main()
