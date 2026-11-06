"""Shared enrollment: one reference embedding per identity, same for all policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .perception import Gallery, Perception


def enroll_gallery(env, perception: Perception, identities: Sequence[str]) -> Gallery:
    embs = []
    for ident in identities:
        res = perception(env.enrollment_frame(ident))
        if not res.face_detected or res.embedding is None:
            raise RuntimeError(f"enrollment failed for {ident!r}: no face detected in reference capture")
        embs.append(np.asarray(res.embedding, dtype=np.float64))
    return Gallery(list(identities), np.stack(embs))
