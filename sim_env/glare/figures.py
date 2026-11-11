"""Build figures from <out>/run_record.json and the saved captures. Needs numpy, PIL, matplotlib
(no Isaac Sim). Raw depth (.npy) is only read, never modified.

    python glare/figures.py --out outputs
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
FONT_B = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def valid(depth, far=100.0):
    return np.isfinite(depth) & (depth > 0) & (depth < far)


def depth_vis(npy_path, png_path, title=None):
    d = np.load(npy_path)
    m = valid(d)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    if m.any():
        lo, hi = np.percentile(d[m], 0.5), np.percentile(d[m], 99.5)
        show = np.ma.array(d, mask=~m)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("black")
        im = ax.imshow(show, cmap=cmap, vmin=lo, vmax=hi)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("distance to image plane (m)")
    ax.set_title(title or os.path.basename(npy_path) + "  (black = no hit / invalid)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def depth_panel(npy_path, size):
    """Plain colorized depth (no axes) for contact sheets."""
    d = np.load(npy_path)
    m = valid(d)
    rgb = np.zeros(d.shape + (3,), np.uint8)
    if m.any():
        lo, hi = np.percentile(d[m], 0.5), np.percentile(d[m], 99.5)
        x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = (plt.get_cmap("turbo")(x)[..., :3] * 255).astype(np.uint8)
        rgb[~m] = 0
    return Image.fromarray(rgb).resize(size)


def spec_overlay(rgb, vt=0.92, st=0.25):
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(2), x.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    m = (mx >= vt) & (sat <= st)
    out = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    out[m] = (255, 0, 255)
    return Image.fromarray(out)


def pose_txt(c):
    p = c["pose_readback"]
    return f"az {p['az_deg']:+.1f}°  el {p['el_deg']:+.1f}°  r {p['r_m']:.3f} m  (look-at err {p['look_at_error_deg']:.2f}°)"


def fmt(v, f="{:.4f}"):
    return "n/a" if v is None else f.format(v)


def contact_sheet(rec, out):
    names = [("clean_front", "CLEAN · front"), ("glare_front", "GLARE · front"),
             ("glare_left", "GLARE · LEFT 15° orbit"), ("glare_right", "GLARE · RIGHT 15° orbit")]
    W, H, pad, top, bot = 640, 480, 12, 64, 118
    sheet = Image.new("RGB", (pad + 4 * (W + pad), 56 + top + 2 * H + pad + bot), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    rd = rec["config"]["material"]["roughness"]
    d.text((pad, 10), "Isaac Sim 5.1 · Lee Perry-Smith head (CC BY 3.0) · same head, material, intrinsics, exposure; "
           "glare views differ only in camera pose", font=font(22, True), fill=(20, 20, 20))
    d.text((pad, 38), f"skin UsdPreviewSurface roughness {rd}, ior {rec['config']['material']['ior']}, "
           "640×480, f=28 mm / 20.955 mm aperture (HFOV ≈ 41°). Row 2: pixels with HSV V≥0.92 & S≤0.25 in magenta.",
           font=font(16), fill=(60, 60, 60))
    arc = rec.get("arcface_scores", {}).get("views", {})
    for i, (nm, label) in enumerate(names):
        c = rec["captures"][nm]
        x0, y0 = pad + i * (W + pad), 56 + top
        rgb = np.asarray(Image.open(os.path.join(out, c["file"])).convert("RGB"))
        sheet.paste(Image.fromarray(rgb), (x0, y0))
        sheet.paste(spec_overlay(rgb), (x0, y0 + H + 4))
        kl = c["key_light"]
        d.text((x0, 56 + 6), label, font=font(22, True), fill=(10, 10, 10))
        d.text((x0, 56 + 36), pose_txt(c), font=font(14), fill=(40, 40, 40))
        s = c["stats"]
        a = arc.get(nm, {})
        if a:
            atxt = (f"ArcFace: no face detected" if not a["detected"] else
                    f"ArcFace det {a['det_score']:.2f} · cos to ref {fmt(a['cosine_to_reference'], '{:.3f}')}")
        else:
            atxt = "ArcFace: not run"
        lines = [f"key light at {tuple(round(v, 2) for v in kl['position_m'])} m, I={kl['intensity']:.3g}",
                 f"specular px: face box {100 * (s['specular_frac_face_box'] or 0):.2f}% · "
                 f"head {100 * (s['specular_frac_head_fg'] or 0):.2f}%",
                 f"clipped (≥254) in face box {100 * (s['clipped_frac_face_box'] or 0):.2f}% · "
                 f"median luma {fmt(s['face_median_luma'], '{:.2f}')}",
                 atxt]
        for j, t in enumerate(lines):
            d.text((x0, y0 + 2 * H + 10 + j * 25), t, font=font(15), fill=(30, 30, 30))
    p = os.path.join(out, "contact_sheet_face_views.png")
    sheet.save(p)
    return p


def action_sheet(rec, out):
    steps = rec["actions"]
    W, H = 320, 240
    cols = 5
    cells = []
    for s in steps:
        cells.append(s)
    rows = int(np.ceil(len(cells) / cols))
    ch = 2 * H + 70
    sheet = Image.new("RGB", (cols * (W + 8) + 8, 50 + rows * (ch + 8)), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    lim = rec["config"]["camera"]["limits"]
    d.text((8, 10), f"4-way action demo (glare lighting) · RGB + depth after each action · limits az "
           f"[{lim['az_min']}, {lim['az_max']}]°, r [{lim['r_min']}, {lim['r_max']}] m · rejected moves are not rendered",
           font=font(17, True), fill=(20, 20, 20))
    for k, s in enumerate(cells):
        x0 = 8 + (k % cols) * (W + 8)
        y0 = 50 + (k // cols) * (ch + 8)
        title = f"{s['step']:02d} {s['action']}"
        if s["accepted"]:
            sheet.paste(Image.open(os.path.join(out, s["file"])).convert("RGB").resize((W, H)), (x0, y0 + 40))
            sheet.paste(depth_panel(os.path.join(out, s["depth_file"]), (W, H)), (x0, y0 + 40 + H))
            p = s["pose_readback"]
            dr = s["depth"]
            sub = (f"az {p['az_deg']:+.1f}° r {p['r_m']:.3f} m · min depth {fmt(dr['min_valid_m'], '{:.3f}')} m")
            d.text((x0, y0), title, font=font(16, True), fill=(10, 10, 10))
        else:
            d.rectangle([x0, y0 + 40, x0 + W, y0 + 40 + 2 * H], fill=(225, 225, 225))
            d.text((x0 + 20, y0 + 40 + H - 20), "REJECTED by limits\ncamera not moved", font=font(18, True),
                   fill=(170, 30, 30))
            b = s["pose_before"]
            sub = f"stays at az {b['az']:+.0f}°, r {b['r']:.1f} m"
            d.text((x0, y0), title, font=font(16, True), fill=(170, 30, 30))
        d.text((x0, y0 + 20), sub, font=font(12), fill=(50, 50, 50))
    p = os.path.join(out, "contact_sheet_actions_rgbd.png")
    sheet.save(p)
    return p


def sweep_grid(rec, out):
    sw = rec.get("sweep") or []
    if not sw:
        return None
    rs = sorted({s["roughness"] for s in sw})
    gs = sorted({s["gain"] for s in sw})
    W, H = 320, 240
    sheet = Image.new("RGB", (90 + len(gs) * (W + 6), 70 + len(rs) * (H + 30)), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"Sweep at the glare light position, frontal camera (base intensity "
           f"{rec['key_base_intensity']:.3g}). Supplementary; not used for any claim.", font=font(15, True), fill=(20, 20, 20))
    for j, g in enumerate(gs):
        d.text((90 + j * (W + 6), 40), f"intensity × {g:g}", font=font(15, True), fill=(20, 20, 20))
    for i, r in enumerate(rs):
        d.text((6, 70 + i * (H + 30) + H // 2), f"rough\n{r:.2f}", font=font(15, True), fill=(20, 20, 20))
        for j, g in enumerate(gs):
            s = next(s for s in sw if s["roughness"] == r and s["gain"] == g)
            x0, y0 = 90 + j * (W + 6), 70 + i * (H + 30)
            sheet.paste(Image.open(os.path.join(out, s["file"])).convert("RGB").resize((W, H)), (x0, y0))
            st = s["stats"]
            d.text((x0, y0 + H + 2), f"spec px face {100 * (st['specular_frac_face_box'] or 0):.2f}% · "
                   f"median luma {fmt(st['face_median_luma'], '{:.2f}')}", font=font(13), fill=(30, 30, 30))
    p = os.path.join(out, "sweep_roughness_intensity.png")
    sheet.save(p)
    return p


def overview(rec, out):
    raw = os.path.join(out, "scene_overview_raw.png")
    if not os.path.exists(raw):
        return None
    img = Image.open(raw).convert("RGB")
    W, H = img.size
    canvas = Image.new("RGB", (W, H + 220), (250, 250, 250))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    colors = {"glare key light": (255, 190, 20), "clean key light": (140, 165, 205), "cam FRONT (az 0)": (40, 115, 255),
              "cam LEFT (az -15)": (25, 200, 50), "cam RIGHT (az +15)": (240, 40, 40), "head center": (255, 255, 255)}
    for k, (u, v) in (rec.get("overview_label_px") or {}).items():
        if 0 <= u < W and 0 <= v < H:
            d.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(0, 0, 0), width=2)
            tx, ty = u + 10, v - 26
            box = d.textbbox((tx, ty), k, font=font(18, True))
            d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
            d.text((tx, ty), k, font=font(18, True), fill=colors.get(k, (255, 255, 255)))
    kl = rec["config"]["key_light"]
    cam = rec["config"]["camera"]
    lines = [
        "Scene overview (rendered in Isaac Sim with proxy markers that exist only for this view; they are not present in the face captures).",
        f"Glare key light: SphereLight r={kl['radius']} m at az {kl['glare']['az']}°, el {kl['glare']['el']}°, {kl['glare']['dist']} m from head center. "
        f"Clean key light: az {kl['clean']['az']}°, el {kl['clean']['el']}°, {kl['clean']['dist']} m. Dome light gives uniform ambient.",
        f"Cameras (cubes + view rays): FRONT az 0°, LEFT az -15°, RIGHT az +15°, all r={cam['start']['r']} m, look-at head center. "
        f"White dots: orbit stops (15° steps, ±{cam['limits']['az_max']:.0f}°) and radial stops ({cam['limits']['r_min']}-{cam['limits']['r_max']} m).",
        "World frame: meters, +Z up, head faces +X, az measured from +X toward +Y.",
    ]
    import textwrap
    y = H + 10
    for i, t in enumerate(lines):
        for part in textwrap.wrap(t, 150):
            d.text((12, y), part, font=font(14, i == 0), fill=(25, 25, 25))
            y += 20
    canvas = canvas.crop((0, 0, W, y + 8))
    p = os.path.join(out, "scene_overview.png")
    canvas.save(p)
    return p


def topdown(rec, out):
    C = np.asarray(rec["config"]["head_center"], float)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.add_patch(plt.Circle((0, 0), 0.1, color="#e8c6a8", zorder=2))
    ax.annotate("", xy=(0.17, 0), xytext=(0.0, 0), arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(0.0, -0.035, "head\n(faces +X)", ha="center", va="top", fontsize=8, zorder=3)
    lim = rec["config"]["camera"]["limits"]
    r0 = rec["config"]["camera"]["start"]["r"]
    th = np.deg2rad(np.linspace(lim["az_min"], lim["az_max"], 100))
    ax.plot(r0 * np.cos(th), r0 * np.sin(th), color="0.6", ls="--", lw=1, label=f"orbit r={r0} m, ±{lim['az_max']:.0f}°")
    ax.plot([lim["r_min"], lim["r_max"]], [0, 0], color="0.3", lw=3, alpha=0.3, label=f"radial range {lim['r_min']}-{lim['r_max']} m")
    cols = {"glare_front": "#2873ff", "glare_left": "#19c832", "glare_right": "#f02828"}
    for nm, col in cols.items():
        p = np.asarray(rec["captures"][nm]["pose_readback"]["position_m"]) - C
        pr = rec["captures"][nm]["pose_readback"]
        ax.plot(*p[:2], "s", color=col, ms=9, zorder=4)
        ax.annotate("", xy=(0, 0), xytext=p[:2], arrowprops=dict(arrowstyle="->", color=col, lw=1))
        ax.text(p[0] + 0.03, p[1], f"{nm.split('_')[1].upper()}\naz {pr['az_deg']:+.1f}°", color=col, fontsize=8)
    for nm, col in (("glare", "#ffbe14"), ("clean", "#8ca5cd")):
        p = np.asarray(rec["marker_positions_m"][f"{nm} key light"]) - C
        ax.plot(*p[:2], "o", color=col, ms=12, mec="k", zorder=4)
        ax.text(p[0] + 0.04, p[1] + 0.03, f"{nm} key light\n(height {p[2]:+.2f} m)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)  → face direction")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-down layout from recorded poses (diagram, not a render)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.75, 1.0)
    fig.tight_layout()
    p = os.path.join(out, "scene_topdown_diagram.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
    out = ap.parse_args().out
    rec = json.load(open(os.path.join(out, "run_record.json")))
    made = []
    for npy in sorted(glob.glob(os.path.join(out, "depth", "*_depth.npy")) + glob.glob(os.path.join(out, "actions", "*_depth.npy"))):
        png = npy[:-4] + "_vis.png"
        depth_vis(npy, png)
        made.append(png)
    for fn in (contact_sheet, action_sheet, sweep_grid, overview, topdown):
        try:
            p = fn(rec, out)
            if p:
                made.append(p)
        except Exception as e:  # keep building the other figures
            print(f"[figures] {fn.__name__} failed: {e!r}")
    print("[figures] wrote", len(made), "files")


if __name__ == "__main__":
    main()
