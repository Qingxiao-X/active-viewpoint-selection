"""Calibrate the shared acceptance thresholds on VALIDATION scenes only.

Uses the observable next-view confidence (``next.top1``, ``next.margin``) and
correctness (``next.correct``) recorded by the collector for candidate views of
validation scenes. Picks the (min_top1_similarity, min_margin) pair that
accepts the most views while keeping precision >= ``--target-precision``.

    python -m avs.calibrate --config configs/mock.yaml --out runs/mock/accept.json

The output file is then passed to the evaluator with ``eval.accept_file=...``
and applies to every policy equally.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import load_config
from .dataset import Dataset, rows_for_scenes


def calibrate(cfg, target_precision: float = 0.95, min_accepts: int = 20, split: str = "val"):
    ckpt = cfg.get("eval", {}).get("checkpoint") or cfg["train"]["out_dir"]
    with open(os.path.join(ckpt, "splits.json")) as f:
        sp = json.load(f)
    if split.startswith("test"):
        raise ValueError("refusing to calibrate thresholds on a test split")
    ds = Dataset.load_many(sp["datasets"])
    rows = rows_for_scenes(ds, sp["splits"][split])
    det = ds.cols["next.detected"][rows] == 1
    top1 = ds.cols["next.top1"][rows]
    margin = ds.cols["next.margin"][rows]
    correct = ds.cols["next.correct"][rows] == 1
    margin = np.where(np.isnan(margin), np.inf, margin)  # single identity: margin not applicable
    best = None
    for ts in np.round(np.arange(0.0, 0.95, 0.025), 3):
        for tm in np.round(np.arange(0.0, 0.6, 0.01), 3):
            acc = det & (top1 >= ts) & (margin >= tm)
            n = int(acc.sum())
            if n < min_accepts:
                continue
            prec = float(correct[acc].mean())
            cov = n / len(rows)
            if prec >= target_precision and (best is None or cov > best["coverage"]):
                best = {"min_top1_similarity": float(ts), "min_margin": float(tm), "precision": prec,
                        "coverage": cov, "n_accepted": n}
    if best is None:
        raise RuntimeError(f"no threshold pair reaches precision {target_precision} on {split}")
    return {"accept": {"min_top1_similarity": best["min_top1_similarity"], "min_margin": best["min_margin"],
                       "min_det_score": float(cfg.get("eval", {}).get("accept", {}).get("min_det_score", 0.0))},
            "calibration": {**best, "split": split, "n_views": int(len(rows)), "target_precision": target_precision}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--min-accepts", type=int, default=20)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    res = calibrate(load_config(args.config, args.overrides), args.target_precision, args.min_accepts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
