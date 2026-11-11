"""Face-glare scene capture for Isaac Sim 5.1 (executable code, run with /isaac-sim/python.sh).

    python.sh scripts/run_capture.py --config configs/glare_scene.json --out outputs

Steps: build scene -> (optional) calibrate key-light intensity for the clean view ->
roughness x intensity sweep -> reference / clean / glare front / left / right captures ->
roughness diagnostic -> 4-way action demo with RGB-D -> scene overview -> ArcFace -> USD export.
Everything measured is written to <out>/run_record.json; figures are built by glare/figures.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from glare.rig import Limits, OrbitRig, Pose, look_at_quat_wxyz, pose_check, sph_to_vec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(ROOT, "configs/glare_scene.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
ap.add_argument("--skip-sweep", action="store_true")
args = ap.parse_args()
CFG = json.load(open(args.config))
# relative paths in the config are relative to this environment folder
CFG["head_usd"] = os.path.join(ROOT, os.path.expandvars(CFG["head_usd"]))
CFG["arcface"]["root"] = os.path.join(ROOT, os.path.expandvars(CFG["arcface"]["root"]))
OUT = os.path.abspath(args.out)
for d in ("", "depth", "actions", "sweep"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
T0 = time.time()
REC = {"config": CFG, "config_path": os.path.abspath(args.config), "captures": {}, "actions": [],
       "sweep": [], "errors": []}


def log(*a):
    print(f"[glare {time.time() - T0:7.1f}s]", *a, flush=True)


def save_record():
    with open(os.path.join(OUT, "run_record.json"), "w") as f:
        json.dump(REC, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ------------------------------------------------------------------ Isaac Sim start
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
REC["startup_seconds"] = round(time.time() - T0, 1)

import carb  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx/post/histogram/enabled", False)  # no auto exposure: fixed exposure for every capture
EXPOSURE_KEYS = ["/rtx/rendermode", "/rtx/post/histogram/enabled", "/rtx/post/tonemap/op",
                 "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber",
                 "/rtx/post/tonemap/whitepoint", "/rtx/post/aa/op", "/rtx/directLighting/sampledLighting/enabled"]

world = World(stage_units_in_meters=1.0)
stage = world.stage
C = np.asarray(CFG["head_center"], float)

# lights
dome = UsdLux.DomeLight.Define(stage, "/World/EnvLight")
dome.CreateIntensityAttr(float(CFG["dome_light"]["intensity"]))
dome.CreateColorAttr(Gf.Vec3f(*CFG["dome_light"]["color"]))
KL = CFG["key_light"]
key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
key.CreateRadiusAttr(float(KL["radius"]))
key.CreateTreatAsPointAttr(False)
key.CreateIntensityAttr(0.0)
key_translate = UsdGeom.Xformable(key.GetPrim()).AddTranslateOp()


def light_pos(spec):
    return C + float(spec["dist"]) * sph_to_vec(float(spec["az"]), float(spec["el"]))


def set_key(spec, intensity):
    key_translate.Set(Gf.Vec3d(*light_pos(spec)))
    key.GetIntensityAttr().Set(float(intensity))


# camera
CAM = CFG["camera"]
W, H = CAM["resolution"]
camera = Camera(prim_path="/World/RGBDCamera", resolution=(W, H))


def set_optics(prim_path, res, focal, haperture, clip):
    cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    cam.GetFocalLengthAttr().Set(float(focal))
    cam.GetHorizontalApertureAttr().Set(float(haperture))
    cam.GetVerticalApertureAttr().Set(float(haperture) * res[1] / res[0])
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(*clip))


set_optics("/World/RGBDCamera", (W, H), CAM["focal_length_mm"], CAM["horizontal_aperture_mm"], CAM["clipping_range"])
world.reset()
world.play()  # annotators only deliver data while the timeline runs
camera.initialize()
camera.add_distance_to_image_plane_to_frame()

# head (fixed): asset is meters, Z-up, face along +X, origin at the head center
add_reference_to_stage(CFG["head_usd"], "/World/Head")
hx = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Head"))
hx.ClearXformOpOrder()
hx.AddTranslateOp().Set(Gf.Vec3d(*C))
SHADER = UsdShade.Shader(stage.GetPrimAtPath(CFG["skin_shader"]))
REC["shader_id"] = str(SHADER.GetIdAttr().Get())


def set_material(roughness=None, metallic=None, ior=None):
    m = CFG["material"]
    vals = {"roughness": m["roughness"] if roughness is None else roughness,
            "metallic": m["metallic"] if metallic is None else metallic,
            "ior": m["ior"] if ior is None else ior}
    for k, v in vals.items():
        inp = SHADER.GetInput(k) or SHADER.CreateInput(k, Sdf.ValueTypeNames.Float)
        inp.Set(float(v))
    return vals


REC["material_applied"] = set_material()
REC["material_readback"] = {i.GetBaseName(): (i.Get() if not hasattr(i.Get(), "path") else str(i.Get().path))
                            for i in SHADER.GetInputs() if i.Get() is not None}
for k in list(REC["material_readback"]):
    v = REC["material_readback"][k]
    REC["material_readback"][k] = v if isinstance(v, (int, float, str)) else str(v)

rig = OrbitRig(C, Pose(**CAM["start"]), CAM["step_az_deg"], CAM["step_r_m"], Limits(**CAM["limits"]))


def place_camera(pose: Pose):
    pos = pose.position(C)
    camera.set_world_pose(pos, look_at_quat_wxyz(pos, C), camera_axes="world")


def read_pose():
    p, q = camera.get_world_pose(camera_axes="world")
    p, q = np.asarray(p, float), np.asarray(q, float)
    return {"position_m": p.round(6).tolist(), "quat_wxyz_world_axes": q.round(6).tolist(), **pose_check(p, q, C)}


# ------------------------------------------------------------------ render helpers
RENDER = CFG["render"]
_first = [True]


def step(n):
    for _ in range(int(n)):
        world.step(render=True)


def grab(cam=None):
    cam = cam or camera
    for _ in range(400):
        rgba = cam.get_rgba()
        depth = cam.get_depth() if cam is camera else np.zeros((1, 1))
        if rgba is not None and depth is not None and np.size(rgba) > 0 and np.size(depth) > 0:
            return np.asarray(rgba)[..., :3].astype(np.uint8).copy(), np.asarray(depth, np.float32).copy()
        world.step(render=True)
    raise RuntimeError("camera returned empty buffers after 400 extra frames")


def render():
    """Let the renderer update after a change, then read RGB + depth from the SAME frame.
    A second read a few frames later measures whether the image had settled."""
    step(RENDER["first_settle_frames"] if _first[0] else RENDER["settle_frames"])
    _first[0] = False
    rgb1, _ = grab()
    step(RENDER["convergence_extra_frames"])
    rgb, depth = grab()
    conv = float(np.abs(rgb.astype(np.int16) - rgb1.astype(np.int16)).mean())
    return rgb, depth, conv


def luma(rgb):
    x = rgb.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def specular_mask(rgb):
    g = CFG["glare_pixel"]
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(2), x.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (mx >= g["value_thresh"]) & (sat <= g["sat_thresh"])


def depth_valid(depth):
    return np.isfinite(depth) & (depth > 0) & (depth < CAM["clipping_range"][1])


# ------------------------------------------------------------------ ArcFace (pretrained, optional)
FACE = None
if CFG["arcface"]["enabled"]:
    try:
        from insightface.app import FaceAnalysis

        A = CFG["arcface"]
        FACE = FaceAnalysis(name=A["model"], root=A["root"], allowed_modules=["detection", "recognition"],
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        FACE.prepare(ctx_id=0, det_size=tuple(A["det_size"]))
        import onnxruntime

        REC["arcface_runtime"] = {"insightface_model": A["model"], "onnxruntime": onnxruntime.__version__,
                                  "providers": onnxruntime.get_available_providers()}
    except Exception as e:
        REC["errors"].append(f"ArcFace unavailable: {e!r}")
        FACE = None


def detect(rgb):
    if FACE is None:
        return None
    faces = FACE.get(np.ascontiguousarray(rgb[:, :, ::-1]))  # InsightFace wants BGR
    if not faces:
        return {"detected": False}
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return {"detected": True, "det_score": float(f.det_score), "bbox": [float(v) for v in f.bbox],
            "embedding": np.asarray(f.normed_embedding, np.float32)}


def face_box(det, shape):
    h, w = shape[:2]
    if det and det.get("detected"):
        x1, y1, x2, y2 = det["bbox"]
        return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)), "detector"
    return int(w * 0.35), int(h * 0.25), int(w * 0.65), int(h * 0.75), "fixed_center_box"


def image_stats(rgb, depth, det):
    fg = depth_valid(depth)
    spec = specular_mask(rgb)
    x1, y1, x2, y2, src = face_box(det, rgb.shape)
    box = np.zeros_like(fg)
    box[y1:y2, x1:x2] = True
    roi = box & fg
    clipped = rgb.max(2) >= 254
    L = luma(rgb)
    return {"face_box": [x1, y1, x2, y2], "face_box_source": src,
            "specular_frac_face_box": float(spec[roi].mean()) if roi.any() else None,
            "specular_frac_head_fg": float(spec[fg].mean()) if fg.any() else None,
            "clipped_frac_face_box": float(clipped[roi].mean()) if roi.any() else None,
            "face_median_luma": float(np.median(L[roi])) if roi.any() else None,
            "face_p99_luma": float(np.percentile(L[roi], 99)) if roi.any() else None}


def depth_report(rgb, depth):
    fg = depth_valid(depth)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.int16)
    bg = np.median(border, axis=0)
    rgb_fg = np.abs(rgb.astype(np.int16) - bg).max(2) > 6
    inter, union = (fg & rgb_fg).sum(), (fg | rgb_fg).sum()
    h, w = depth.shape
    c = depth[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    c = c[np.isfinite(c)]
    v = depth[fg]
    return {"shape": list(depth.shape), "dtype": str(depth.dtype),
            "n_valid": int(fg.sum()), "n_inf": int(np.isinf(depth).sum()), "n_nan": int(np.isnan(depth).sum()),
            "n_nonpositive": int((depth <= 0).sum()),
            "min_valid_m": float(v.min()) if v.size else None, "max_valid_m": float(v.max()) if v.size else None,
            "center_20px_median_m": float(np.median(c)) if c.size else None,
            "rgb_depth_foreground_iou": float(inter / union) if union else None,
            "background_rgb": bg.tolist()}


def save_capture(name, rgb, depth, conv, extra=None, subdir="", with_depth=True):
    det = detect(rgb)
    rec = {"file": os.path.join(subdir, f"{name}.png"), "pose_cmd": rig.pose.to_dict(),
           "pose_readback": read_pose(), "key_light": {
               "position_m": list(np.asarray(key_translate.Get(), float).round(4)),
               "intensity": float(key.GetIntensityAttr().Get())},
           "dome_intensity": float(dome.GetIntensityAttr().Get()),
           "material": {k: float(SHADER.GetInput(k).Get()) for k in ("roughness", "metallic", "ior")},
           "render_convergence_mean_abs_diff_u8": conv, "stats": image_stats(rgb, depth, det)}
    Image.fromarray(rgb).save(os.path.join(OUT, subdir, f"{name}.png"))
    if with_depth:
        dname = os.path.join("depth" if not subdir else subdir, f"{name}_depth.npy")
        np.save(os.path.join(OUT, dname), depth)
        rec["depth_file"] = dname
        rec["depth"] = depth_report(rgb, depth)
    if det is not None:
        rec["arcface"] = {k: v for k, v in det.items() if k != "embedding"}
        if det.get("detected"):
            np.save(os.path.join(OUT, subdir or "", f"{name}_arcface_emb.npy"), det["embedding"])
    if extra:
        rec.update(extra)
    return rec, det


def shoot(name, pose, light_spec, intensity, subdir="", with_depth=True, extra=None):
    rig.pose = pose
    place_camera(pose)
    set_key(light_spec, intensity)
    rgb, depth, conv = render()
    rec, det = save_capture(name, rgb, depth, conv, extra=extra, subdir=subdir, with_depth=with_depth)
    log(f"{name}: pose={pose} I={intensity:.4g} conv={conv:.3f} stats={rec['stats']}")
    return rec, det, rgb


FRONT = Pose(**CAM["start"])
try:
    # ---------------------------------------------------------------- warm-up + reference
    REC["exposure_settings"] = {k: settings.get(k) for k in EXPOSURE_KEYS}
    try:
        REC["intrinsics_K"] = np.asarray(camera.get_intrinsics_matrix()).round(4).tolist()
    except Exception as e:
        REC["errors"].append(f"get_intrinsics_matrix: {e!r}")
    t = time.time()
    ref_rec, ref_det, _ = shoot("arcface_reference_front", FRONT, KL["clean"], 0.0,
                                extra={"note": "ArcFace reference: dome light only (key light off), frontal"})
    REC["first_frame_seconds"] = round(time.time() - t, 1)
    REC["captures"]["arcface_reference_front"] = ref_rec
    save_record()

    # ---------------------------------------------------------------- key intensity
    I0 = KL["intensity"]
    if I0 is None:  # calibrate: clean frontal face median luma -> target (monotone in intensity)
        tgt = CFG["clean_target_face_luma"]
        box = ref_rec["stats"]["face_box"]
        place_camera(FRONT)
        hist = []

        def f(I):
            set_key(KL["clean"], I * KL["clean"]["gain"])
            rgb, depth, _ = render()
            x1, y1, x2, y2 = box
            m = depth_valid(depth)[y1:y2, x1:x2]
            val = float(np.median(luma(rgb)[y1:y2, x1:x2][m]))
            hist.append({"intensity": I, "face_median_luma": val})
            log(f"calib I={I:.4g} luma={val:.3f}")
            return val

        lo = hi = None
        I = 1.0e4
        for _ in range(14):
            v = f(I)
            if v < tgt:
                lo, I = I, I * 4
            else:
                hi, I = I, I / 4
            if lo is not None and hi is not None:
                break
        if lo is None or hi is None:
            raise RuntimeError(f"could not bracket key intensity: {hist}")
        for _ in range(7):
            mid = float(np.sqrt(lo * hi))
            (lo, hi) = (mid, hi) if f(mid) < tgt else (lo, mid)
        I0 = float(np.sqrt(lo * hi))
        REC["key_calibration"] = {"target_face_median_luma": tgt, "face_box": box, "history": hist,
                                  "resolved_base_intensity": I0}
    REC["key_base_intensity"] = float(I0)
    I_clean = I0 * KL["clean"]["gain"]
    I_glare = I0 * KL["glare"]["gain"]
    save_record()

    # ---------------------------------------------------------------- sweep (supplementary)
    if not args.skip_sweep:
        for rgh in CFG["sweep"]["roughness"]:
            set_material(roughness=rgh)
            for g in CFG["sweep"]["gain"]:
                nm = f"sweep_r{rgh:.2f}_g{g:g}"
                rec, _, _ = shoot(nm, FRONT, KL["glare"], I0 * g, subdir="sweep", with_depth=False)
                rec.update({"roughness": rgh, "gain": g})
                REC["sweep"].append(rec)
        set_material()
        save_record()

    # ---------------------------------------------------------------- settling probe
    # How many frames does the image need after the key light moves (clean -> glare)?
    rig.pose = FRONT
    place_camera(FRONT)
    set_key(KL["clean"], I_clean)
    step(240)
    set_key(KL["glare"], I_glare)
    probe, done = [], 0
    for n in (1, 2, 5, 10, 20, 30, 60, 120, 240, 480):
        step(n - done)
        done = n
        rgb, depth = grab()
        st = image_stats(rgb, depth, None)
        probe.append({"frames_after_light_change": n, "specular_frac_center_box": st["specular_frac_face_box"],
                      "median_luma_center_box": st["face_median_luma"]})
        log(f"probe n={n} spec={st['specular_frac_face_box']:.4f} luma={st['face_median_luma']:.3f}")
    REC["settle_probe"] = probe
    save_record()

    # ---------------------------------------------------------------- main comparison
    REC["captures"]["clean_front"], _, _ = shoot("clean_front", FRONT, KL["clean"], I_clean)
    REC["captures"]["glare_front"], _, _ = shoot("glare_front", FRONT, KL["glare"], I_glare)
    for act, nm in (("LEFT", "glare_left"), ("RIGHT", "glare_right")):
        rig.pose = FRONT
        ok, p = rig.step(act)
        assert ok, act
        REC["captures"][nm], _, _ = shoot(nm, p, KL["glare"], I_glare, extra={"action_from_front": act})
    # diagnostic: same glare lighting, rough skin -> highlights should mostly vanish if they are specular
    rd = CFG["diagnostic_rough_roughness"]
    set_material(roughness=rd)
    REC["captures"][f"diag_glare_front_rough{rd:.2f}"], _, _ = shoot(
        f"diag_glare_front_rough{rd:.2f}", FRONT, KL["glare"], I_glare,
        extra={"note": "diagnostic only: glare lighting with roughness raised; not part of the 4-view comparison"})
    set_material()
    save_record()

    # ---------------------------------------------------------------- 4-way action demo
    rig.pose = FRONT
    rec, _, _ = shoot("step_00_START", FRONT, KL["glare"], I_glare, subdir="actions")
    REC["actions"].append({"step": 0, "action": "START", "accepted": True, **rec})
    for i, a in enumerate(CFG["action_demo"], start=1):
        before = rig.pose
        valid = rig.valid_actions()
        ok, p = rig.step(a)
        entry = {"step": i, "action": a, "accepted": ok, "pose_before": before.to_dict(),
                 "valid_actions_before": valid}
        if ok:
            rec, _, _ = shoot(f"step_{i:02d}_{a}", p, KL["glare"], I_glare, subdir="actions")
            entry.update(rec)
        else:
            entry["note"] = "rejected by limits; camera not moved, nothing rendered"
            log(f"step {i} {a}: rejected at {before}")
        REC["actions"].append(entry)
    save_record()

    # ---------------------------------------------------------------- overview with proxy markers
    markers = UsdGeom.Xform.Define(stage, "/World/Markers")

    def material(path, rgb, emissive=0.0):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(np.asarray(rgb) * emissive)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def finish(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        pv = UsdGeom.PrimvarsAPI(prim)
        pv.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        pv.CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)

    def sphere(path, pos, radius, mat):
        s = UsdGeom.Sphere.Define(stage, path)
        s.CreateRadiusAttr(float(radius))
        s.AddTranslateOp().Set(Gf.Vec3d(*pos))
        finish(s.GetPrim(), mat)

    def cube(path, pos, size, mat):
        s = UsdGeom.Cube.Define(stage, path)
        s.CreateSizeAttr(float(size))
        s.AddTranslateOp().Set(Gf.Vec3d(*pos))
        finish(s.GetPrim(), mat)

    def segment(path, a, b, radius, mat):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(L)
        cyl.CreateAxisAttr("Z")
        cyl.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2)))
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(d / L)))
        cyl.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(rot.GetQuat()))
        finish(cyl.GetPrim(), mat)

    def stop_short(p, dist=0.13):
        d = C - p
        return C - dist * d / np.linalg.norm(d)

    M = {"glare": material("/World/Markers/Looks/glare", (1.0, 0.75, 0.1), 3.0),
         "clean": material("/World/Markers/Looks/clean", (0.55, 0.65, 0.8), 0.5),
         "front": material("/World/Markers/Looks/front", (0.15, 0.45, 1.0), 0.8),
         "left": material("/World/Markers/Looks/left", (0.1, 0.8, 0.2), 0.8),
         "right": material("/World/Markers/Looks/right", (0.95, 0.15, 0.15), 0.8),
         "dot": material("/World/Markers/Looks/dot", (0.9, 0.9, 0.9), 0.3)}
    LP = {"glare": light_pos(KL["glare"]), "clean": light_pos(KL["clean"])}
    sphere("/World/Markers/glare_light", LP["glare"], KL["radius"], M["glare"])
    segment("/World/Markers/glare_ray", LP["glare"], stop_short(LP["glare"]), 0.004, M["glare"])
    sphere("/World/Markers/clean_light", LP["clean"], KL["radius"] * 0.8, M["clean"])
    segment("/World/Markers/clean_ray", LP["clean"], stop_short(LP["clean"]), 0.003, M["clean"])
    CP = {}
    for nm in ("front", "left", "right"):
        cap = REC["captures"][{"front": "glare_front", "left": "glare_left", "right": "glare_right"}[nm]]
        p = np.asarray(cap["pose_readback"]["position_m"])
        CP[nm] = p
        cube(f"/World/Markers/cam_{nm}", p, 0.045, M[nm])
        segment(f"/World/Markers/cam_{nm}_ray", p, stop_short(p), 0.003, M[nm])
    L_ = CAM["limits"]
    for i, az in enumerate(np.arange(L_["az_min"], L_["az_max"] + 1e-6, CAM["step_az_deg"])):
        sphere(f"/World/Markers/orbit_{i}", Pose(float(az), 0.0, FRONT.r).position(C), 0.008, M["dot"])
    for i, r in enumerate(np.arange(L_["r_min"], L_["r_max"] + 1e-6, CAM["step_r_m"])):
        sphere(f"/World/Markers/radial_{i}", Pose(0.0, 0.0, float(r)).position(C), 0.008, M["dot"])
    set_key(KL["glare"], I_glare)
    rig.pose = FRONT
    place_camera(FRONT)

    OV = CFG["overview_camera"]
    ov_res = tuple(OV["resolution"])
    ov = Camera(prim_path="/World/OverviewCamera", resolution=ov_res)
    set_optics("/World/OverviewCamera", ov_res, OV["focal_length_mm"], 20.955, (0.01, 100.0))
    ov.initialize()
    ov_target = C + np.asarray(OV["target_offset"], float)
    ov_pos = ov_target + float(OV["dist"]) * sph_to_vec(float(OV["az"]), float(OV["el"]))
    ov.set_world_pose(ov_pos, look_at_quat_wxyz(ov_pos, ov_target), camera_axes="world")
    step(RENDER["first_settle_frames"])
    ov_rgb, _ = grab(ov)
    Image.fromarray(ov_rgb).save(os.path.join(OUT, "scene_overview_raw.png"))
    pts = {"head center": C, "glare key light": LP["glare"], "clean key light": LP["clean"],
           "cam FRONT (az 0)": CP["front"], "cam LEFT (az -15)": CP["left"], "cam RIGHT (az +15)": CP["right"]}
    try:
        uv = np.asarray(ov.get_image_coords_from_world_points(np.stack(list(pts.values()))), float)
        REC["overview_label_px"] = {k: uv[i].round(1).tolist() for i, k in enumerate(pts)}
    except Exception as e:
        REC["errors"].append(f"overview projection: {e!r}")
    REC["overview_camera"] = {"position_m": ov_pos.round(4).tolist(), "target_m": ov_target.round(4).tolist(),
                              "resolution": list(ov_res), "focal_length_mm": OV["focal_length_mm"]}
    REC["marker_positions_m"] = {k: np.asarray(v).round(4).tolist() for k, v in pts.items()}
    save_record()

    # ---------------------------------------------------------------- ArcFace scores vs the reference
    if FACE is not None:
        ref = REC["captures"]["arcface_reference_front"].get("arcface", {})
        out = {"reference": "arcface_reference_front.png", "reference_detected": ref.get("detected", False),
               "views": {}}
        ref_emb = None
        if ref.get("detected"):
            ref_emb = np.load(os.path.join(OUT, "arcface_reference_front_arcface_emb.npy"))
        for nm in ("clean_front", "glare_front", "glare_left", "glare_right", f"diag_glare_front_rough{rd:.2f}"):
            a = REC["captures"][nm].get("arcface", {})
            v = {"detected": bool(a.get("detected")), "det_score": a.get("det_score"), "cosine_to_reference": None}
            if a.get("detected") and ref_emb is not None:
                e = np.load(os.path.join(OUT, f"{nm}_arcface_emb.npy"))
                v["cosine_to_reference"] = float(np.dot(e, ref_emb))
            out["views"][nm] = v
        REC["arcface_scores"] = out
        log("arcface", json.dumps(out))

    # ---------------------------------------------------------------- export reusable scene
    UsdGeom.Imageable(markers.GetPrim()).MakeInvisible()
    scene_path = os.path.join(ROOT, "scene", "face_glare_scene.usda")
    stage.GetRootLayer().Export(scene_path)
    REC["scene_usd"] = scene_path
except Exception as e:
    REC["errors"].append("FATAL: " + "".join(traceback.format_exception(e)))
    log("FATAL", e)
    traceback.print_exc()
finally:
    REC["total_seconds"] = round(time.time() - T0, 1)
    save_record()
    log("record written", os.path.join(OUT, "run_record.json"))
    sys.stdout.flush()
    app.close()
