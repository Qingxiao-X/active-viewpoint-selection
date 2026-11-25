#!/bin/bash
#SBATCH --export=NONE
#SBATCH -J avs_evidence
#SBATCH -p gpu
#SBATCH --exclude=t001
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:t4:1
#SBATCH -t 00:45:00
#SBATCH -o /scratch/user/chenzc2001/qx/689/hprc/logs/%x-%j.out
# Minimal evidence capture for presentation slides 1 & 4 (see scripts/evidence_capture.py).
# Does not touch runs/ or results/isaac_heads_v1/ -- independent of the main preflight/collect run.
set -uo pipefail
H=/scratch/user/chenzc2001/qx/689/hprc
R=/scratch/user/chenzc2001/qx/689/active-viewpoint-selection
USD=/scratch/user/chenzc2001/qx/689/assets_usd
SIF=$H/isaac-sim_5.1.0_avs.sif
OUT=$R/results/evidence_v1
W=$H/runtime
LOG=$H/logs
mkdir -p $W/home $W/kit_cache $W/kit_data $W/kit_logs $OUT $LOG
cd $R

stamp() { echo "=== [$(date +%T)] $*"; }
stamp "node $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

export SINGULARITYENV_ACCEPT_EULA=Y SINGULARITYENV_PRIVACY_CONSENT=Y SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=YES
SX="singularity exec --nv --home $W/home -B /scratch/user/chenzc2001/qx/689 \
  -B $W/kit_cache:/isaac-sim/kit/cache -B $W/kit_data:/isaac-sim/kit/data -B $W/kit_logs:/isaac-sim/kit/logs $SIF"
PY="$SX /isaac-sim/python.sh"

stamp "evidence_capture"
$PY scripts/evidence_capture.py \
  --usd $USD/lee_perry_smith/lee_perry_smith.usd \
  --identity lee_perry_smith \
  --manifest $USD/assets_manifest.json \
  --out $OUT \
  > $LOG/evidence_capture.log 2>&1
RC=$?
grep -v "warn rootless" $LOG/evidence_capture.log | grep -E '^\[evidence\]|Error|Traceback' | tail -n 150
stamp "evidence_capture exit=$RC"
ls -la $OUT $OUT/rgbd_sample 2>/dev/null
exit $RC
