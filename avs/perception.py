"""Perception: per-frame measurements, enrollment gallery, observation building.

* ``Perception`` turns a ``Frame`` into a ``PerceptionResult``.
* ``InsightFacePerception`` wraps a pretrained InsightFace/ArcFace model (no
  training) and computes glare/depth statistics inside the detected face box.
* ``Gallery`` holds enrolled reference embeddings. It produces *observable*
  quantities (top-1 similarity, top-1/top-2 margin, predicted identity) for
  policies, and separately the *label* quantity ``identity_quality`` that uses
  ground truth and must only be called by collectors/evaluators.

Note: valid depth fraction is a sensing-quality measurement. It is NOT a
liveness / anti-spoofing signal and should not be described as one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .types import Frame, PerceptionResult, PolicyObservation


class Perception:
    def __call__(self, frame: Frame) -> PerceptionResult:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------
# Image-based measurements (used by the real pipeline; pure numpy)
# --------------------------------------------------------------------------

def _crop_box(img: np.ndarray, bbox) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def glare_metrics(rgb: np.ndarray, bbox, value_thresh: float = 0.92, sat_thresh: float = 0.25):
    """Specular-glare statistics inside the face box.

    A pixel is flagged as specular if it is near-saturated in brightness
    (HSV value >= value_thresh) and weakly colored (saturation <= sat_thresh).
    Returns (fraction, cx, cy) where (cx, cy) is the glare centroid in [-1, 1]
    relative to the box center, or (fraction, None, None) if no glare pixels.
    """
    crop = _crop_box(rgb, bbox).astype(np.float32) / 255.0
    if crop.size == 0:
        return None, None, None
    mx = crop.max(axis=2)
    mn = crop.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    mask = (mx >= value_thresh) & (sat <= sat_thresh)
    frac = float(mask.mean())
    if not mask.any():
        return frac, None, None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx = float((xs.mean() + 0.5) / w * 2.0 - 1.0)
    cy = float((ys.mean() + 0.5) / h * 2.0 - 1.0)
    return frac, cx, cy


def depth_metrics(depth: np.ndarray, bbox, min_depth: float = 0.05, max_depth: float = 5.0):
    """Fraction of valid depth pixels in the face box and their std (m)."""
    crop = _crop_box(depth, bbox).astype(np.float64)
    if crop.size == 0:
        return None, None
    valid = np.isfinite(crop) & (crop > min_depth) & (crop < max_depth)
    frac = float(valid.mean())
    std = float(crop[valid].std()) if valid.sum() >= 2 else None
    return frac, std


class InsightFacePerception(Perception):
    """Pretrained InsightFace (ArcFace) detector + recognizer. Not trained here.

    Requires ``pip install insightface onnxruntime`` and the model pack
    (default ``buffalo_l``) to be downloadable or present in ~/.insightface.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size=(640, 640), ctx_id: int = -1,
                 glare_value_thresh: float = 0.92, glare_sat_thresh: float = 0.25,
                 min_depth: float = 0.05, max_depth: float = 5.0):
        from insightface.app import FaceAnalysis  # lazy import

        self.app = FaceAnalysis(name=model_name, allowed_modules=["detection", "recognition", "landmark_3d_68"])
        self.app.prepare(ctx_id=ctx_id, det_size=tuple(det_size))
        self.gv, self.gs = glare_value_thresh, glare_sat_thresh
        self.dmin, self.dmax = min_depth, max_depth

    def __call__(self, frame: Frame) -> PerceptionResult:
        if frame.rgb is None:
            raise ValueError("InsightFacePerception needs an RGB image")
        h, w = frame.rgb.shape[:2]
        # InsightFace expects BGR
        faces = self.app.get(np.ascontiguousarray(frame.rgb[:, :, ::-1]))
        if not faces:
            res = PerceptionResult(face_detected=False, image_size=(w, h))
            if frame.depth is not None:  # whole-image depth quality is still observable
                res.depth_valid_fraction, res.depth_std = depth_metrics(frame.depth, (0, 0, w, h), self.dmin, self.dmax)
            return res
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = tuple(float(v) for v in face.bbox)
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        pose = None
        if getattr(face, "pose", None) is not None:
            pitch, yaw, roll = [float(v) for v in face.pose]  # insightface order: pitch, yaw, roll
            pose = (yaw, pitch, roll)
        gf, gcx, gcy = glare_metrics(frame.rgb, bbox, self.gv, self.gs)
        dvf, dstd = (None, None)
        if frame.depth is not None:
            dvf, dstd = depth_metrics(frame.depth, bbox, self.dmin, self.dmax)
        return PerceptionResult(True, float(face.det_score), bbox, (w, h), pose, emb, gf, gcx, gcy, dvf, dstd)


# --------------------------------------------------------------------------
# Enrollment gallery
# --------------------------------------------------------------------------

class Gallery:
    def __init__(self, ids: Sequence[str], embeddings: np.ndarray):
        emb = np.asarray(embeddings, dtype=np.float64)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.ids: List[str] = list(ids)
        self.emb = emb
        self._index = {k: i for i, k in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def similarities(self, embedding: np.ndarray) -> np.ndarray:
        e = np.asarray(embedding, dtype=np.float64)
        e = e / max(np.linalg.norm(e), 1e-12)
        return self.emb @ e

    # ---- observable (allowed in policy inputs) ----
    def match(self, embedding: Optional[np.ndarray]):
        """Return (predicted_id, top1_similarity, top2_margin or None)."""
        if embedding is None:
            return None, None, None
        s = self.similarities(embedding)
        order = np.argsort(-s)
        top1 = float(s[order[0]])
        margin = float(s[order[0]] - s[order[1]]) if len(s) >= 2 else None
        return self.ids[order[0]], top1, margin

    # ---- label / scoring only (uses ground truth) ----
    def identity_quality(self, embedding: np.ndarray, gt_identity: str, mode: str = "auto") -> float:
        """q_id: sim(correct) - max sim(incorrect); falls back to sim(correct)
        when only one identity is enrolled (``mode='auto'``)."""
        s = self.similarities(embedding)
        i = self._index[gt_identity]
        if mode == "similarity" or (mode == "auto" and len(s) < 2):
            return float(s[i])
        if len(s) < 2:
            raise ValueError("margin mode needs >= 2 enrolled identities")
        return float(s[i] - np.max(np.delete(s, i)))

    def save(self, path: str) -> None:
        np.savez(path, ids=np.array(self.ids), emb=self.emb)

    @classmethod
    def load(cls, path: str) -> "Gallery":
        d = np.load(path, allow_pickle=False)
        return cls([str(x) for x in d["ids"]], d["emb"])


def build_observation(frame_vp, perc: PerceptionResult, gallery: Gallery, step: int = 0) -> PolicyObservation:
    """Combine viewpoint + perception + gallery match into a policy observation."""
    bcx = bcy = bw = bh = None
    if perc.face_detected and perc.bbox is not None and perc.image_size is not None:
        W, H = perc.image_size
        x1, y1, x2, y2 = perc.bbox
        bcx, bcy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
    yaw = pitch = roll = None
    if perc.pose is not None:
        yaw, pitch, roll = perc.pose
    pred, top1, margin = gallery.match(perc.embedding if perc.face_detected else None)
    return PolicyObservation(
        azimuth=frame_vp.azimuth, elevation=frame_vp.elevation, radius=frame_vp.radius,
        face_detected=bool(perc.face_detected), det_score=perc.det_score,
        bbox_cx=bcx, bbox_cy=bcy, bbox_w=bw, bbox_h=bh,
        pose_yaw=yaw, pose_pitch=pitch, pose_roll=roll,
        top1_similarity=top1, top2_margin=margin, predicted_identity=pred,
        glare_fraction=perc.glare_fraction, glare_cx=perc.glare_cx, glare_cy=perc.glare_cy,
        depth_valid_fraction=perc.depth_valid_fraction, depth_std=perc.depth_std, step=step,
    )


def compute_label(perc: PerceptionResult, gallery: Gallery, gt_identity: str, label_cfg: Dict) -> Dict[str, float]:
    """Training target for one resulting view (uses ground truth: oracle only).

    Returns separate components so auxiliary terms stay explicit:
      q_id    identity quality (margin, or similarity fallback); on detection
              failure = ``label_cfg['detection_failure_value']``
      aux_glare, aux_depth   raw auxiliary measurements (nan if missing)
      label   q_id + w_glare * (-glare) + w_depth * depth_valid  (weights default 0)
    """
    mode = label_cfg.get("mode", "auto")
    fail_val = float(label_cfg.get("detection_failure_value", -1.0))
    if perc.face_detected and perc.embedding is not None:
        q_id = gallery.identity_quality(perc.embedding, gt_identity, mode)
    else:
        q_id = fail_val
    g = perc.glare_fraction if perc.glare_fraction is not None else np.nan
    d = perc.depth_valid_fraction if perc.depth_valid_fraction is not None else np.nan
    wg = float(label_cfg.get("aux_glare_weight", 0.0))
    wd = float(label_cfg.get("aux_depth_weight", 0.0))
    label = q_id
    if wg:
        label -= wg * (0.0 if np.isnan(g) else g)
    if wd:
        label += wd * (0.0 if np.isnan(d) else d)
    return {"q_id": float(q_id), "aux_glare": float(g), "aux_depth": float(d), "label": float(label)}
