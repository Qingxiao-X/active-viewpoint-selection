"""Action-quality predictor q_hat = f(observation features, candidate action features).

``QModel`` bundles the network with its feature schema and train-split
normalizer, so a checkpoint directory is self-contained:

    model.pt              torch state_dict
    model_config.json     {"in_dim", "hidden", "dropout"}
    feature_schema.json   FeatureSchema.to_json()
    normalizer.json       Normalizer.to_json() (train-split statistics)

To try another model (e.g. with image features), implement the same
``predict_raw(X_raw) -> np.ndarray`` method.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn

from .features import FeatureSchema, Normalizer


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (128, 128), dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QModel:
    def __init__(self, net: MLP, schema: FeatureSchema, normalizer: Normalizer, model_cfg: Dict):
        self.net = net.eval()
        self.schema = schema
        self.normalizer = normalizer
        self.model_cfg = model_cfg

    @torch.no_grad()
    def predict_raw(self, X_raw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(self.normalizer.transform(X_raw))
        return self.net(x).numpy().astype(np.float64)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(out_dir, "model.pt"))
        for name, obj in (("model_config.json", self.model_cfg), ("feature_schema.json", self.schema.to_json()),
                          ("normalizer.json", self.normalizer.to_json())):
            with open(os.path.join(out_dir, name), "w") as f:
                json.dump(obj, f, indent=2)

    @classmethod
    def load(cls, ckpt_dir: str) -> "QModel":
        def rd(n):
            with open(os.path.join(ckpt_dir, n)) as f:
                return json.load(f)

        mcfg = rd("model_config.json")
        schema = FeatureSchema.from_json(rd("feature_schema.json"))
        norm = Normalizer.from_json(rd("normalizer.json"))
        if norm.names != schema.raw_names or norm.out_dim != mcfg["in_dim"]:
            raise ValueError("checkpoint schema / normalizer / model dims are inconsistent")
        net = MLP(mcfg["in_dim"], mcfg["hidden"], mcfg.get("dropout", 0.0))
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu", weights_only=True))
        return cls(net, schema, norm, mcfg)
