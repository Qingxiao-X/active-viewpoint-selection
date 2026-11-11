"""Convert the raw head scans into USD assets for avs.envs.isaac_env.

Every output asset follows the same layout so one config works for all identities:
  * defaultPrim /Head, meters, Z up, face looking along +X,
  * origin at the head center (mid-cranium laterally / front-back, about eye height),
  * skin material at /Head/Looks/Skin/Shader (UsdPreviewSurface, `roughness` input
    is overridden per scene by the env).

Usage: python convert_heads.py <assets_raw_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_meshes import load_glb, load_obj  # noqa: E402

HEAD_WIDTH_M = 0.155   # typical adult cranial breadth incl. some hair, used to set scale
EYE_BELOW_CROWN_M = 0.115
TEX_MAX = 2048

# raw axes -> output axes (rows: output x(forward), y(left), z(up) in raw coordinates)
ROT_YUP_FACE_PZ = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)     # Lee, Emily
ROT_YDOWN_FACE_NZ = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)  # FaceVerse

HEADS = {
    "lee_perry_smith": {
        "src": "leeperrysmith/LeePerrySmith.glb", "rot": ROT_YUP_FACE_PZ,
        "skin_tex": "leeperrysmith/Map-COL.jpg", "normal_tex": "leeperrysmith/Infinite-Level_02_Tangent_SmoothUV.jpg",
        "license": "CC BY 3.0, Lee Perry-Smith / Infinite-Realities (www.triplegangers.com, ir-ltd.net)",
    },
    "faceverse_001": {
        "src": "faceverse_001/001_01.obj", "rot": ROT_YDOWN_FACE_NZ, "skin_tex": "faceverse_001/001_01.jpg",
        "license": "FaceVerse dataset public sample (Tsinghua Univ.), non-commercial research only",
    },
    "digital_emily_2": {
        "src": "emily2/Emily_2_1.obj", "rot": ROT_YUP_FACE_PZ,
        "skin_tex": "emily2/Textures/Color_unpainted/00_diffuse_unlit_unpainted.exr",
        "materials": {  # OBJ material -> (texture or rgb, keep)
            "Skin_Blend_01": "skin",
            "Eye_Outter_blend_01": "emily2/Textures/Color_unpainted/Eye_Outter_Color_01.png",
            "Eye_Inner_Iris_SSS_01": "emily2/Textures/Color_unpainted/Eye_Inner_Iris_Color_01.png",
            "Eye_Lashes_mtl": (0.03, 0.02, 0.02),
            "Eye_Wetness_01": None,  # transparent tear film: would render as an opaque shell, drop
        },
        "license": "Digital Emily 2 (USC ICT / Wikihuman), free for non-commercial use",
    },
}


def head_frame(P):
    """Scale and center from the cranium: iterate a top band whose depth is half its width."""
    top = P[:, 2].max()
    w = (P[:, 1].max() - P[:, 1].min())
    for _ in range(20):
        band = P[P[:, 2] > top - 0.5 * w]
        w_new = band[:, 1].max() - band[:, 1].min()
        if abs(w_new - w) < 1e-6 * max(w, 1):
            break
        w = w_new
    band = P[P[:, 2] > top - 0.5 * w]
    scale = HEAD_WIDTH_M / w
    cx = 0.5 * (band[:, 0].max() + band[:, 0].min())
    cy = 0.5 * (band[:, 1].max() + band[:, 1].min())
    cz = top - EYE_BELOW_CROWN_M / scale
    return scale, np.array([cx, cy, cz])


def to_srgb_u8(lin):
    lin = np.clip(lin, 0, None)
    lin = lin / max(np.percentile(lin, 99.5), 1e-6)  # unlit diffuse EXR: normalize to albedo range
    lin = np.clip(lin * 0.85, 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (srgb * 255 + 0.5).astype(np.uint8)


def write_texture(src, dst):
    if src.lower().endswith(".exr"):
        import OpenEXR
        with OpenEXR.File(src) as f:
            ch = f.channels()
            if "RGB" in ch:
                rgb = np.asarray(ch["RGB"].pixels, np.float32)
            else:
                rgb = np.stack([np.asarray(ch[c].pixels, np.float32) for c in ("R", "G", "B")], -1)
        img = Image.fromarray(to_srgb_u8(rgb[..., :3]))
    else:
        img = Image.open(src).convert("RGB")
    if max(img.size) > TEX_MAX:
        img.thumbnail((TEX_MAX, TEX_MAX), Image.LANCZOS)
    img.save(dst, quality=93)
    return os.path.basename(dst)


def vertex_normals(P, tris):
    n = np.zeros_like(P)
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])  # area weighted
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def make_material(stage, path, tex=None, rgb=None, normal_tex=None, roughness=0.4):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.4)  # skin-like dielectric
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if tex or normal_tex:
        st = UsdShade.Shader.Define(stage, f"{path}/st_reader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    if tex:
        t = UsdShade.Shader.Define(stage, f"{path}/diffuse_tex")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex}")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(rgb or (0.6, 0.45, 0.4))))
    if normal_tex:
        n = UsdShade.Shader.Define(stage, f"{path}/normal_tex")
        n.CreateIdAttr("UsdUVTexture")
        n.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{normal_tex}")
        n.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        n.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        n.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        n.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            n.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    return mat


def convert(name, spec, raw_dir, out_dir):
    src = os.path.join(raw_dir, spec["src"])
    V, T, F = load_glb(src) if src.endswith(".glb") else load_obj(src)
    P = V @ spec["rot"].T
    mats = spec.get("materials", {k: "skin" for k in F})
    keep = {m: f for m, f in F.items() if mats.get(m, "skin") is not None}
    used = np.unique(np.concatenate([f[..., 0].ravel() for f in keep.values()]))
    scale, c = head_frame(P[used])
    P = (P - c) * scale

    asset_dir = os.path.join(out_dir, name)
    os.makedirs(os.path.join(asset_dir, "textures"), exist_ok=True)
    usd_path = os.path.join(asset_dir, f"{name}.usd")
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Head")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Head/Looks")

    skin_tex = write_texture(os.path.join(raw_dir, spec["skin_tex"]), os.path.join(asset_dir, "textures", f"{name}_skin.jpg"))
    normal_tex = None
    if spec.get("normal_tex"):
        normal_tex = write_texture(os.path.join(raw_dir, spec["normal_tex"]),
                                   os.path.join(asset_dir, "textures", f"{name}_skin_normal.jpg"))
    skin = make_material(stage, "/Head/Looks/Skin", tex=skin_tex, normal_tex=normal_tex)

    stats = {}
    for mi, (m, f) in enumerate(keep.items()):
        vi, ti = f[..., 0], f[..., 1]
        has_uv = T is not None and (ti >= 0).all()
        # One point per unique (position, uv) pair, uv as a plain per-vertex primvar.
        # (Isaac/RTX mis-mapped the earlier face-varying indexed uvs on two of the heads.)
        key = np.stack([vi.ravel(), ti.ravel() if has_uv else np.zeros(vi.size, np.int64)], 1)
        ukey, inv = np.unique(key, axis=0, return_inverse=True)
        tris = inv.reshape(-1, 3)
        pv_idx = ukey[:, 0]
        # normals from the unsplit mesh so uv seams stay smooth
        upos, pinv = np.unique(vi.ravel(), return_inverse=True)
        n_u = vertex_normals(P[upos], pinv.reshape(-1, 3))
        nrm = n_u[np.searchsorted(upos, pv_idx)]
        pts = P[pv_idx]
        mesh = UsdGeom.Mesh.Define(stage, f"/Head/mesh_{mi:02d}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.ravel().astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))])
        if has_uv:
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(T[ukey[:, 1]].astype(np.float32)))
        what = mats.get(m, "skin")
        if what == "skin":
            mat = skin
        elif isinstance(what, str):
            tex = write_texture(os.path.join(raw_dir, what), os.path.join(asset_dir, "textures", f"{name}_{m}.jpg"))
            mat = make_material(stage, f"/Head/Looks/{m}", tex=tex, roughness=0.15)
        else:
            mat = make_material(stage, f"/Head/Looks/{m}", rgb=what, roughness=0.6)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        stats[m] = {"triangles": int(len(tris)), "vertices": int(len(pts))}
    stage.GetRootLayer().Save()
    lo, hi = P[used].min(0), P[used].max(0)
    info = {"identity": name, "usd": usd_path, "scale_raw_to_m": scale, "license": spec["license"],
            "source": spec["src"], "bbox_min_m": lo.round(4).tolist(), "bbox_max_m": hi.round(4).tolist(),
            "meshes": stats}
    print(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    only = sys.argv[3:] or list(HEADS)
    infos = [convert(n, HEADS[n], raw_dir, out_dir) for n in only]
    with open(os.path.join(out_dir, "assets_manifest.json"), "w") as f:
        json.dump(infos, f, indent=1)
