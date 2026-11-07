"""Load the raw head scans (OBJ / GLB) into numpy arrays: positions, uvs, triangles per material."""
import numpy as np


def load_obj(path):
    vs, vts, faces = [], [], {}   # faces[material] = list of (vi, ti) triangles
    mat = "default"; group = None
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vts.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("usemtl"):
            mat = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                parts = tok.split("/")
                vi = int(parts[0]); ti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                idx.append((vi - 1 if vi > 0 else len(vs) + vi, (ti - 1 if ti > 0 else len(vts) + ti) if ti else -1))
            for k in range(1, len(idx) - 1):  # fan triangulation
                faces.setdefault(mat, []).append((idx[0], idx[k], idx[k + 1]))
    V = np.asarray(vs, np.float64); T = np.asarray(vts, np.float64) if vts else None
    F = {m: np.asarray(f, np.int64) for m, f in faces.items()}  # (n, 3, 2)
    return V, T, F


def load_glb(path):
    import pygltflib
    g = pygltflib.GLTF2().load(path)
    blob = g.binary_blob()

    def acc(i):
        a = g.accessors[i]; bv = g.bufferViews[a.bufferView]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a.componentType]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        arr = np.frombuffer(blob, dtype=dt, count=a.count * n, offset=off)
        return arr.reshape(a.count, n) if n > 1 else arr

    p = g.meshes[0].primitives[0]
    V = acc(p.attributes.POSITION).astype(np.float64)
    T = acc(p.attributes.TEXCOORD_0).astype(np.float64).copy()
    # this GLB (three.js example, Blender export) already stores OBJ-style v; verified visually
    I = acc(p.indices).astype(np.int64).reshape(-1, 3)
    F = {"skin": np.stack([I, I], axis=-1)}  # same index for position and uv
    return V, T, F
