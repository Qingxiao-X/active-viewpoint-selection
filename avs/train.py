"""Train the action-quality MLP on grouped splits; select checkpoint on validation.

    python -m avs.train --config configs/mock.yaml [key=value ...]

Outputs in ``train.out_dir``: model files (see avs.model), ``splits.json``,
``config.json``, ``metrics.json`` (training history, validation metrics of the
selected checkpoint, and test metrics computed once for reporting — never
used for selection or tuning).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .config import load_config
from .dataset import Dataset, grouped_split, rows_for_scenes, check_no_leakage
from .features import FeatureSchema, Normalizer
from .model import MLP, QModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ranking_metrics(ds: Dataset, rows: np.ndarray, pred: np.ndarray, cost_weight: float = 0.0) -> Dict[str, float]:
    """Per-state decision quality: for each state, pick argmax(pred - w*cost)
    among labeled candidates; regret = best true label - chosen true label."""
    if len(rows) == 0:
        return {}
    st = ds.cols["group.state_id"][rows]
    y = ds.labels(rows)
    cost = ds.cols["act.move_cost"][rows]
    score = pred - cost_weight * cost
    regrets, rand_regrets, top1 = [], [], []
    order = np.argsort(st, kind="stable")
    st_s, y_s, sc_s = st[order], y[order], score[order]
    bounds = np.flatnonzero(np.diff(st_s)) + 1
    for yy, ss in zip(np.split(y_s, bounds), np.split(sc_s, bounds)):
        if len(yy) < 2:
            continue
        best = yy.max()
        chosen = yy[np.argmax(ss)]
        regrets.append(best - chosen)
        rand_regrets.append(best - yy.mean())
        top1.append(float(chosen >= best - 1e-12))
    return {"n_states": len(regrets), "mean_regret": float(np.mean(regrets)),
            "random_mean_regret": float(np.mean(rand_regrets)), "top1_action_acc": float(np.mean(top1))}


def train(cfg: Dict, out_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    tcfg = cfg["train"]
    out_dir = out_dir or tcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = int(tcfg.get("seed", 0))
    set_seed(seed)
    paths = tcfg.get("datasets") or [cfg["collect"]["out_dir"]]
    ds = Dataset.load_many(paths if isinstance(paths, list) else [paths])
    label_key = tcfg.get("label_key", "label.label")

    # ---- grouped splits (scene-level); reuse an existing split file if given
    scfg = cfg.get("split", {})
    if scfg.get("splits_file"):
        with open(scfg["splits_file"]) as f:
            splits = json.load(f)["splits"]
        check_no_leakage(splits, ds.scenes)
    else:
        splits = grouped_split(ds.scenes, scfg.get("fractions", (0.7, 0.15, 0.15)), int(scfg.get("seed", 0)),
                               scfg.get("heldout_identities"), int(scfg.get("n_heldout_identities", 0)))
    rows = {k: rows_for_scenes(ds, v) for k, v in splits.items()}
    if len(rows["train"]) == 0 or len(rows["val"]) == 0:
        raise RuntimeError("empty train or val split; collect more scenes")

    # ---- features: normalization statistics from TRAIN rows only
    fcfg = cfg.get("features", {}) or {}
    schema = FeatureSchema(fcfg.get("obs_fields"), fcfg.get("action_fields"))
    X_raw = {k: ds.features_raw(schema, r) for k, r in rows.items()}
    Y = {k: ds.labels(r, label_key) for k, r in rows.items()}
    norm = Normalizer.fit(X_raw["train"], schema)
    X = {k: torch.from_numpy(norm.transform(v)) if len(v) else torch.zeros(0, norm.out_dim) for k, v in X_raw.items()}
    Yt = {k: torch.from_numpy(v.astype(np.float32)) for k, v in Y.items()}

    mcfg = {"in_dim": norm.out_dim, "hidden": list(cfg.get("model", {}).get("hidden", [128, 128])),
            "dropout": float(cfg.get("model", {}).get("dropout", 0.0))}
    net = MLP(mcfg["in_dim"], mcfg["hidden"], mcfg["dropout"])
    loss_name = tcfg.get("loss", "huber")
    loss_fn = nn.HuberLoss(delta=float(tcfg.get("huber_delta", 0.1))) if loss_name == "huber" else nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    bs, epochs = int(tcfg.get("batch_size", 256)), int(tcfg.get("epochs", 50))
    patience = int(tcfg.get("patience", 0))
    select = tcfg.get("select_metric", "val_loss")  # or val_regret
    cost_w = float(cfg.get("policy", {}).get("learned", {}).get("movement_cost_weight", 0.0))
    gen = torch.Generator().manual_seed(seed)

    def evaluate(split):
        net.eval()
        with torch.no_grad():
            p = net(X[split])
            l = float(loss_fn(p, Yt[split]))
        return l, p.numpy().astype(np.float64)

    history, best, best_state, since = [], None, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Yt["train"]), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(X["train"][idx]), Yt["train"][idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vl, vp = evaluate("val")
        vm = ranking_metrics(ds, rows["val"], vp, cost_w)
        rec = {"epoch": ep, "train_loss": tot / len(perm), "val_loss": vl, "val_regret": vm["mean_regret"]}
        history.append(rec)
        key = rec[select]
        if best is None or key < best - 1e-9:
            best, since = key, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            rec["selected"] = True
        else:
            since += 1
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"[train] ep {ep:3d} train {rec['train_loss']:.4f} val {vl:.4f} val_regret {vm['mean_regret']:.4f}")
        if patience and since >= patience:
            if verbose:
                print(f"[train] early stop at epoch {ep}")
            break
    net.load_state_dict(best_state)
    model = QModel(net, schema, norm, mcfg)
    model.save(out_dir)

    metrics = {"history": history, "selected_by": select, "best_" + select: best, "label_key": label_key,
               "n_rows": {k: int(len(r)) for k, r in rows.items()},
               "software_validation_only": bool(ds.meta.get("software_validation_only", False))}
    for split in splits:
        if len(rows[split]):
            l, p = evaluate(split)
            metrics[split] = {"loss": l, **ranking_metrics(ds, rows[split], p, cost_w)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump({"seed": int(scfg.get("seed", 0)), "datasets": paths, "splits": splits,
                   "heldout_identities": sorted({s["identity"] for s in ds.scenes
                                                 if s["scene_id"] in set(splits.get("test_heldout_identity", []))})},
                  f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if verbose:
        for split in splits:
            if split in metrics:
                m = metrics[split]
                print(f"[train] {split:22s} loss {m['loss']:.4f} regret {m.get('mean_regret', float('nan')):.4f} "
                      f"(random {m.get('random_mean_regret', float('nan')):.4f}) top1 {m.get('top1_action_acc', float('nan')):.3f}")
        print(f"[train] saved checkpoint to {out_dir}")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    train(load_config(args.config, args.overrides), args.out)


if __name__ == "__main__":
    main()
