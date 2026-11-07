"""Isaac Sim preflight: exposure calibration + integration checks before data collection.

Run inside the Isaac Sim python (hprc/gpu_main.sh does this):
    python.sh scripts/isaac_preflight.py --config configs/isaac_heads.yaml --out runs/isaac_heads_v1/preflight

Writes to --out:
  exposure.json    calibrated key/dome base intensities (used as overrides by collection/eval)
  report.json      all checks, timing, enrollment similarity matrix
  grid_*.png       annotated renders per identity (viewpoints x lighting)
Exit code 2 if a blocking check fails (no face at the frontal enrollment view, light not
reaching the face), so the batch job stops before spending GPU time on collection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avs.actions import ActionSpace, Viewpoint  # noqa: E402
from avs.config import load_config  # noqa: E402
from avs.enroll import enroll_gallery  # noqa: E402
from avs.envs import make_env  # noqa: E402
from avs.perception import build_observation  # noqa: E402

TARGET_DOME_LUMA = 0.45   # face median (sRGB, 0..1) under environment light only
TARGET_KEY_LUMA = 0.55    # face median under the key light only, matte skin, intensity 1.0


def luma(rgb):
    x = rgb.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def face_region(perc, shape):
    h, w = shape[:2]
    if perc.face_detected and perc.bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in perc.bbox]
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return int(w * 0.35), int(h * 0.3), int(w * 0.65), int(h * 0.7)  # center crop fallback


def shoot(env, perception, scene, az=0.0, el=0.0):
    env.load_scene(scene)
    env.move_to(Viewpoint(az, el, env.radius))
    t = time.perf_counter()
    frame = env.render()
    t_render = time.perf_counter() - t
    t = time.perf_counter()
    perc = perception(frame)
    t_perc = time.perf_counter() - t
    return frame, perc, t_render, t_perc


def face_luma(frame, perc):
    x1, y1, x2, y2 = face_region(perc, frame.rgb.shape)
    return float(np.median(luma(frame.rgb[y1:y2, x1:x2])))


def scene(identity, **kw):
    s = {"scene_id": -100, "identity": identity, "light_intensity": 0.0, "env_intensity": 1.0,
         "light_az": 0.0, "light_el": 20.0, "roughness": 0.5}
    s.update(kw)
    return s


def calibrate_exposure(env, perception, ids, log):
    out = {}
    # 1) environment (dome) light only
    for it in range(6):
        ls = [face_luma(*shoot(env, perception, scene(i, light_intensity=0.0, env_intensity=1.0))[:2]) for i in ids]
        L = float(np.mean(ls))
        log.append({"stage": "dome", "iter": it, "dome_base": env.dome_base, "face_luma": ls})
        print(f"[preflight] dome_base={env.dome_base:.4g} face luma={np.round(ls, 3)}", flush=True)
        if abs(L - TARGET_DOME_LUMA) < 0.04:
            break
        env.dome_base *= float(np.clip((TARGET_DOME_LUMA / max(L, 1e-3)) ** 2.2, 0.05, 20.0))
    out["dome_base_intensity"], out["dome_face_luma"] = env.dome_base, L
    # 2) key light only (matte skin, frontal-ish light)
    for it in range(6):
        ls = [face_luma(*shoot(env, perception, scene(i, light_intensity=1.0, env_intensity=0.0,
                                                       light_az=0.0, light_el=20.0, roughness=0.8))[:2]) for i in ids]
        L = float(np.mean(ls))
        log.append({"stage": "key", "iter": it, "key_base": env.key_base, "face_luma": ls})
        print(f"[preflight] key_base={env.key_base:.4g} face luma={np.round(ls, 3)}", flush=True)
        if abs(L - TARGET_KEY_LUMA) < 0.04:
            break
        env.key_base *= float(np.clip((TARGET_KEY_LUMA / max(L, 1e-3)) ** 2.2, 0.05, 20.0))
    out["key_light_base_intensity"], out["key_face_luma"] = env.key_base, L
    return out


def annotate(rgb, perc, text):
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(im)
    if perc.face_detected and perc.bbox is not None:
        d.rectangle(perc.bbox, outline=(0, 255, 0), width=2)
    d.rectangle((0, 0, im.width, 34), fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    os.makedirs(args.out, exist_ok=True)
    space = ActionSpace.from_config(cfg["action_space"])
    t0 = time.time()
    env, perception = make_env(cfg, space)
    report = {"startup_seconds": round(time.time() - t0, 1), "checks": {}, "blocking_failures": []}
    print(f"[preflight] Isaac + perception up in {report['startup_seconds']} s", flush=True)
    try:
        sess = perception.app.models["recognition"].session
        report["onnx_providers"] = sess.get_providers()
    except Exception as e:  # pragma: no cover - informational
        report["onnx_providers"] = f"unknown: {e}"
    ids = env.identities()
    t_render, t_perc, log = [], [], []

    # ---- first frames: discard warm-up / shader compilation time
    t = time.time()
    shoot(env, perception, scene(ids[0]))
    report["first_frame_seconds"] = round(time.time() - t, 1)

    # ---- exposure
    report["exposure"] = calibrate_exposure(env, perception, ids, log)
    report["exposure_log"] = log
    with open(os.path.join(args.out, "exposure.json"), "w") as f:
        json.dump({k: report["exposure"][k] for k in ("key_light_base_intensity", "dome_base_intensity")}, f, indent=2)
    if report["exposure"]["key_face_luma"] < 0.2:
        report["blocking_failures"].append("key light does not reach the face (luma < 0.2 at max gain)")

    # ---- enrollment (shared gallery) and frontal detection
    gallery = enroll_gallery(env, perception, ids)
    S = gallery.emb @ gallery.emb.T
    report["enrollment_similarity"] = {"ids": ids, "matrix": np.round(S, 4).tolist()}
    print("[preflight] enrollment similarity\n" + str(np.round(S, 3)), flush=True)

    # ---- light direction: light from the head's left (+y, az>0) must brighten image RIGHT half
    #      (camera at +x looking back at the face, so +y appears on the image right)
    ld = {}
    for az_l in (60.0, -60.0):
        fr, pc, *_ = shoot(env, perception, scene(ids[0], light_intensity=1.0, env_intensity=0.1,
                                                   light_az=az_l, light_el=10.0, roughness=0.8))
        x1, y1, x2, y2 = face_region(pc, fr.rgb.shape)
        L = luma(fr.rgb[y1:y2, x1:x2])
        mid = L.shape[1] // 2
        ld[str(az_l)] = {"left_half": float(L[:, :mid].mean()), "right_half": float(L[:, mid:].mean())}
    ok = ld["60.0"]["right_half"] > ld["60.0"]["left_half"] and ld["-60.0"]["left_half"] > ld["-60.0"]["right_half"]
    report["checks"]["light_direction"] = {"pass": bool(ok), **ld}

    # ---- viewpoint grid per identity; pose sign / identification / glare / timing
    from PIL import Image

    views = [(-60, 0), (-30, 0), (0, 0), (30, 0), (60, 0), (0, 30), (0, -20)]
    lights = {"glossy_glare": dict(light_intensity=1.4, env_intensity=0.3, light_az=25.0, light_el=20.0, roughness=0.08),
              "matte": dict(light_intensity=1.0, env_intensity=0.5, light_az=25.0, light_el=20.0, roughness=0.7)}
    grid_stats = []
    for ident in ids:
        rows = []
        for lname, lkw in lights.items():
            tiles = []
            for az, el in views:
                fr, pc, tr, tp = shoot(env, perception, scene(ident, **lkw), az, el)
                t_render.append(tr); t_perc.append(tp)
                obs = build_observation(Viewpoint(az, el, env.radius), pc, gallery)
                st = {"identity": ident, "light": lname, "az": az, "el": el, "detected": pc.face_detected,
                      "det_score": pc.det_score, "pose_yaw": obs.pose_yaw, "pose_pitch": obs.pose_pitch,
                      "pred": obs.predicted_identity, "top1": obs.top1_similarity, "margin": obs.top2_margin,
                      "glare": pc.glare_fraction, "bbox_cx": obs.bbox_cx, "bbox_cy": obs.bbox_cy, "bbox_w": obs.bbox_w,
                      "depth_valid": pc.depth_valid_fraction, "mean_luma": float(luma(fr.rgb).mean())}
                grid_stats.append(st)
                fmt = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
                txt = (f"{ident[:14]} az{az:+d} el{el:+d} {lname}\n"
                       f"det {fmt(pc.det_score)} yaw {fmt(obs.pose_yaw, 0)} pred {str(obs.predicted_identity)[:10]} "
                       f"s {fmt(obs.top1_similarity)} m {fmt(obs.top2_margin)} g {fmt(pc.glare_fraction, 3)}")
                tiles.append(annotate(fr.rgb, pc, txt))
            rows.append(np.concatenate(tiles, axis=1))
        img = Image.fromarray(np.concatenate(rows, axis=0))
        img = img.resize((img.width // 2, img.height // 2))
        img.save(os.path.join(args.out, f"grid_{ident}.png"))
    report["grid"] = grid_stats
    front = [g for g in grid_stats if g["az"] == 0 and g["el"] == 0]
    report["checks"]["frontal_detection"] = {
        "pass": all(g["detected"] for g in front),
        "per_identity": {g["identity"] + "/" + g["light"]: g["detected"] for g in front}}
    if not any(g["detected"] for g in front):
        report["blocking_failures"].append("no face detected in any frontal view")
    # camera at +az sees the face turned the other way: yaw sign must flip between az=-30 and az=+30
    yaws = {(g["identity"], g["light"], g["az"]): g["pose_yaw"] for g in grid_stats if g["el"] == 0}
    flips = [np.sign(yaws[(i, l, 30)]) != np.sign(yaws[(i, l, -30)]) for i in ids for l in lights
             if yaws.get((i, l, 30)) is not None and yaws.get((i, l, -30)) is not None]
    report["checks"]["pose_yaw_flips_with_azimuth"] = {"pass": bool(flips) and bool(np.all(flips)), "n": len(flips)}
    cen = [(g["bbox_cx"], g["bbox_cy"]) for g in front if g["bbox_cx"] is not None]
    report["checks"]["frontal_face_centered"] = {"mean_center": np.round(np.mean(cen, axis=0), 3).tolist() if cen else None}
    det = [g for g in grid_stats if g["detected"]]
    report["checks"]["identification_on_grid"] = {
        "n_detected": len(det), "n_views": len(grid_stats),
        "accuracy_detected": float(np.mean([g["pred"] == g["identity"] for g in det])) if det else None}
    gl = {l: float(np.mean([g["glare"] or 0.0 for g in grid_stats if g["light"] == l])) for l in lights}
    report["checks"]["glare_fraction_mean"] = gl
    report["checks"]["glossy_has_more_glare"] = {"pass": gl["glossy_glare"] > gl["matte"]}

    # ---- restore determinism: same state rendered again after moving away
    s0 = scene(ids[0], **lights["glossy_glare"])
    fa, pa, *_ = shoot(env, perception, s0, 15, 0)
    state = env.get_state()
    shoot(env, perception, scene(ids[-1], **lights["matte"]), -45, 15)
    env.set_state(state)
    fb = env.render()
    pb = perception(fb)
    diff = np.abs(fa.rgb.astype(int) - fb.rgb.astype(int))
    report["checks"]["restore_determinism"] = {
        "max_pixel_diff": int(diff.max()), "mean_pixel_diff": float(diff.mean()),
        "max_emb_diff": (float(np.max(np.abs(pa.embedding - pb.embedding)))
                         if pa.embedding is not None and pb.embedding is not None else None)}

    # ---- timing (per render / per perception call, after warm-up)
    report["timing"] = {"render_s_mean": float(np.mean(t_render)), "perception_s_mean": float(np.mean(t_perc)),
                        "render_plus_perception_s": float(np.mean(t_render) + np.mean(t_perc)),
                        "n": len(t_render)}
    t = time.perf_counter()
    for i in ids:  # asset switch cost
        shoot(env, perception, scene(i))
    report["timing"]["asset_switch_s"] = (time.perf_counter() - t) / len(ids)

    # Each identity must render its own geometry: compare side-view depth silhouettes.
    # (A stale-mesh bug once rendered every identity with the first head's mesh.)
    masks = {}
    for i in ids:
        fr, _, _, _ = shoot(env, perception, scene(i), az=60.0)
        d = np.asarray(fr.depth, np.float32)
        masks[i] = np.isfinite(d) & (d > 0) & (d < 2.0 * env.radius)
    ious = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            m1, m2 = masks[ids[a]], masks[ids[b]]
            ious[f"{ids[a]}|{ids[b]}"] = round(float((m1 & m2).sum() / max((m1 | m2).sum(), 1)), 4)
    same = [k for k, v in ious.items() if v > 0.97]
    report["checks"]["distinct_geometry"] = {"pass": not same, "side_silhouette_iou": ious}
    print("[preflight] side-silhouette IoU between identities:", ious, flush=True)
    if same:
        report["blocking_failures"].append(f"identities render the same geometry: {same}")
    report["total_seconds"] = round(time.time() - t0, 1)
    # write everything before closing: SimulationApp.close() may end the process
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in report.items() if k not in ("grid", "exposure_log")}, indent=2, default=float))
    blocking = bool(report["blocking_failures"])
    if blocking:
        print("[preflight] BLOCKING:", report["blocking_failures"], flush=True)
    with open(os.path.join(args.out, "STATUS"), "w") as f:
        f.write("BLOCKED\n" if blocking else "OK\n")
    env.close()
    sys.exit(2 if blocking else 0)


if __name__ == "__main__":
    main()
