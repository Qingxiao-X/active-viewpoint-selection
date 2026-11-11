"""Orthographic textured point splats of a raw head along +-x, +-z (y assumed up) for orientation checks."""
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from load_meshes import load_obj, load_glb

path, tex, out = sys.argv[1], sys.argv[2], sys.argv[3]
V, T, F = load_glb(path) if path.endswith('.glb') else load_obj(path)
tri = np.concatenate(list(F.values()))
vi, ti = tri[..., 0].ravel(), tri[..., 1].ravel()
img = np.asarray(Image.open(tex).convert('RGB')) if tex != '-' else None
pos = V[vi]
if img is not None and T is not None:
    uv = T[ti]; h, w = img.shape[:2]
    col = img[np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1), np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)]
else:
    col = np.full((len(pos), 3), 200, np.uint8)
S = 256; tiles = []
lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2; s = (hi - lo).max() / 2
for name, (ax_h, ax_d, sign_h) in {'+z': (0, 2, 1), '-z': (0, 2, -1), '+x': (2, 0, -1), '-x': (2, 0, 1)}.items():
    view_sign = 1 if name[0] == '+' else -1
    u = sign_h * (pos[:, ax_h] - c[ax_h]) / s; v = (pos[:, 1] - c[1]) / s; d = view_sign * pos[:, ax_d]
    order = np.argsort(d)  # painter: far first, near overwrites
    canvas = np.zeros((S, S, 3), np.uint8)
    px = ((u[order] + 1) / 2 * (S - 1)).astype(int); py = ((1 - (v[order] + 1) / 2) * (S - 1)).astype(int)
    canvas[py, px] = col[order]
    tiles.append(canvas)
Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
print('views left->right: from +z, -z, +x, -x  (y up)')
