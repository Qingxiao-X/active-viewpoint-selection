"""Print Markdown tables from run_record.json (no Isaac Sim needed).

    python scripts/summarize.py outputs [outputs_run1_calibrated]
"""

import json
import os
import sys

import numpy as np


def load(d):
    return json.load(open(os.path.join(d, "run_record.json")))


def pct(v):
    return "n/a" if v is None else f"{100 * v:.2f}%"


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


out = sys.argv[1]
R = load(out)
V = ["clean_front", "glare_front", "glare_left", "glare_right"]
rd = R["config"]["diagnostic_rough_roughness"]
DIAG = f"diag_glare_front_rough{rd:.2f}"

print("### Views\n")
print("| image | camera az / el / r (read back) | look-at err | key light pos (m) | key I | roughness | spec px face box | spec px head | clipped px face box | face median luma | render settle diff |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for nm in V + [DIAG]:
    c = R["captures"][nm]
    p, s = c["pose_readback"], c["stats"]
    print(f"| `{nm}.png` | {p['az_deg']:+.2f}° / {p['el_deg']:+.2f}° / {p['r_m']:.4f} m | {p['look_at_error_deg']:.3f}° | "
          f"{tuple(round(x, 3) for x in c['key_light']['position_m'])} | {c['key_light']['intensity']:.4g} | {c['material']['roughness']:.2f} | "
          f"{pct(s['specular_frac_face_box'])} | {pct(s['specular_frac_head_fg'])} | {pct(s['clipped_frac_face_box'])} | "
          f"{f3(s['face_median_luma'])} | {c['render_convergence_mean_abs_diff_u8']:.3f} |")

print("\n### Camera poses (world frame)\n")
print("| image | position (m) | quaternion w,x,y,z (Isaac 'world' camera axes: +X fwd, +Z up) |")
print("|---|---|---|")
for nm in V:
    p = R["captures"][nm]["pose_readback"]
    print(f"| `{nm}` | {tuple(round(x, 4) for x in p['position_m'])} | {tuple(round(x, 4) for x in p['quat_wxyz_world_axes'])} |")

if "arcface_scores" in R:
    A = R["arcface_scores"]
    print(f"\n### ArcFace (reference `{A['reference']}`, detected: {A['reference_detected']})\n")
    print("| image | face detected | det score | cosine to reference |")
    print("|---|---|---|---|")
    for nm, v in A["views"].items():
        print(f"| `{nm}` | {v['detected']} | {f3(v['det_score'])} | {f3(v['cosine_to_reference'])} |")

print("\n### Action demo\n")
print("| step | action | accepted | az (deg) | r (m) | look-at err | min valid depth (m) | center 20px median depth (m) | RGB/depth fg IoU | spec px face box |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in R["actions"]:
    if a["accepted"]:
        p, d = a["pose_readback"], a["depth"]
        print(f"| {a['step']} | {a['action']} | yes | {p['az_deg']:+.2f} | {p['r_m']:.4f} | {p['look_at_error_deg']:.3f}° | "
              f"{f3(d['min_valid_m'])} | {f3(d['center_20px_median_m'])} | {f3(d['rgb_depth_foreground_iou'])} | {pct(a['stats']['specular_frac_face_box'])} |")
    else:
        b = a["pose_before"]
        print(f"| {a['step']} | {a['action']} | **rejected** (limit) | {b['az']:+.2f} | {b['r']:.4f} | – | – | – | – | – |")

d = R["captures"]["glare_front"]["depth"]
print("\n### Depth array (glare_front)\n")
print("```\n" + json.dumps(d, indent=1) + "\n```")
print("\nexposure settings:", json.dumps(R["exposure_settings"]))
print("intrinsics K:", R.get("intrinsics_K"))
print("key base intensity:", R["key_base_intensity"], "| startup s:", R["startup_seconds"],
      "| first frame s:", R["first_frame_seconds"], "| total s:", R["total_seconds"], "| errors:", R["errors"])

if len(sys.argv) > 2:
    R1 = load(sys.argv[2])
    print("\n### Repeatability vs", sys.argv[2], "\n")
    print("| image | mean abs RGB diff (0-255) | spec px face box run1 → this run | ArcFace cos run1 → this run |")
    print("|---|---|---|---|")
    from PIL import Image
    for nm in V:
        a = np.asarray(Image.open(os.path.join(sys.argv[2], f"{nm}.png")), np.int16)
        b = np.asarray(Image.open(os.path.join(out, f"{nm}.png")), np.int16)
        s1 = R1["captures"][nm]["stats"]["specular_frac_face_box"]
        s2 = R["captures"][nm]["stats"]["specular_frac_face_box"]
        c1 = R1.get("arcface_scores", {}).get("views", {}).get(nm, {}).get("cosine_to_reference")
        c2 = R.get("arcface_scores", {}).get("views", {}).get(nm, {}).get("cosine_to_reference")
        print(f"| `{nm}` | {np.abs(a - b).mean():.3f} | {pct(s1)} → {pct(s2)} | {f3(c1)} → {f3(c2)} |")
