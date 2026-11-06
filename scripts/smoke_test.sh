#!/usr/bin/env bash
# Small collect -> train -> calibrate -> evaluate run on the MOCK env, plus unit tests.
# Software validation only: results say nothing about real face perception.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-/tmp/avs_smoke}
rm -rf "$OUT"
COMMON=(collect.out_dir=$OUT/data train.datasets=[$OUT/data] train.out_dir=$OUT/run
        eval.checkpoint=$OUT/run eval.gallery_file=$OUT/data/gallery.npz eval.out_dir=$OUT/eval)
$PY -m pytest -q
$PY -m avs.collect   --config configs/mock.yaml "${COMMON[@]}" collect.n_scenes=60
$PY -m avs.train     --config configs/mock.yaml "${COMMON[@]}" train.epochs=20
$PY -m avs.calibrate --config configs/mock.yaml "${COMMON[@]}" --out $OUT/run/accept.json > /dev/null
$PY -m avs.evaluate  --config configs/mock.yaml "${COMMON[@]}" eval.accept_file=$OUT/run/accept.json \
    env.mock.render_failure_prob=0
echo "smoke test OK -> $OUT"
