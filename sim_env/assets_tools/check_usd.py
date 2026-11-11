"""Textured point splat of converted USD heads seen from +x (camera az=0) and az=+45 (z up)."""
import sys, os, numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade
tiles = []
for path in sys.argv[2:]:
    st = Usd.Stage.Open(path); d = os.path.dirname(path)
    for az in (0, 45, -45):
        a = np.radians(az); S = 256; canvas = np.zeros((S, S, 3), np.uint8); allp = []; allc = []
        for prim in st.Traverse():
            if not prim.IsA(UsdGeom.Mesh): continue
            m = UsdGeom.Mesh(prim); P = np.array(m.GetPointsAttr().Get()); idx = np.array(m.GetFaceVertexIndicesAttr().Get())
            pv = UsdGeom.PrimvarsAPI(m).GetPrimvar('st'); mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            tex = [s for s in mat.GetPrim().GetChildren() if s.GetName() == 'diffuse_tex']
            if pv and tex:
                uv = np.array(pv.Get()); uv = uv[idx] if pv.GetInterpolation() == 'vertex' else (uv[np.array(pv.GetIndices())] if pv.IsIndexed() else uv)
                img = np.asarray(Image.open(os.path.join(d, tex[0].GetAttribute('inputs:file').Get().path)).convert('RGB')); h, w = img.shape[:2]
                col = img[np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1), np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)]
            else:
                col = np.full((len(idx), 3), 40, np.uint8)
            allp.append(P[idx]); allc.append(col)
        P = np.concatenate(allp); C = np.concatenate(allc)
        fwd = np.array([np.cos(a), np.sin(a), 0]); left = np.array([-np.sin(a), np.cos(a), 0])
        depth = P @ fwd; u = -(P @ left); v = P[:, 2]   # viewer at +fwd looking back at origin
        o = np.argsort(depth); half = 0.2
        px = ((u[o] / half + 1) / 2 * (S - 1)).astype(int); py = ((1 - (v[o] / half + 1) / 2) * (S - 1)).astype(int)
        ok = (px >= 0) & (px < S) & (py >= 0) & (py < S); canvas[py[ok], px[ok]] = C[o][ok]
        canvas[S // 2, :] = (0, 255, 0); canvas[:, S // 2] = (0, 255, 0)  # crosshair = look-at center
        tiles.append(canvas)
Image.fromarray(np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0)).save(sys.argv[1])
