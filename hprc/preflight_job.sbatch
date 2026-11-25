#!/bin/bash
#SBATCH --export=NONE
#SBATCH -J avs_preflight
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:t4:1
#SBATCH --exclude=t001
#SBATCH -t 01:00:00
#SBATCH -o /scratch/user/chenzc2001/qx/689/hprc/logs/%x-%j.out
# Convert the raw head scans, then run preflight only (no collect/train/evaluate).
# Stops here on purpose: the three heads' shapes need visual confirmation before
# spending GPU time on the real collection run.
set -uo pipefail
H=/scratch/user/chenzc2001/qx/689/hprc
R=/scratch/user/chenzc2001/qx/689/active-viewpoint-selection
RAW=/scratch/user/chenzc2001/qx/689/assets_raw
USD=/scratch/user/chenzc2001/qx/689/assets_usd
SIF=$H/isaac-sim_5.1.0_avs.sif
CFG=configs/isaac_heads.yaml
RUN=$R/runs/isaac_heads_v1
PRE=$RUN/preflight
W=$H/runtime
LOG=$H/logs
rm -rf $PRE  # drop any stale/git-tracked preflight output so a leftover STATUS can't be mistaken for this run's
mkdir -p $W/home $W/kit_cache $W/kit_data $W/kit_logs $PRE $LOG
cd $R

stamp() { echo "=== [$(date +%T)] $*"; }
stamp "node $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# The base image ships its own correct /etc/vulkan/icd.d/nvidia_icd.json (api_version matches
# the host driver exactly); a hand-written override here previously declared the wrong
# api_version and broke Vulkan init (ERROR_INCOMPATIBLE_DRIVER). Just use --nv's defaults.
export SINGULARITYENV_ACCEPT_EULA=Y SINGULARITYENV_PRIVACY_CONSENT=Y SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=YES
SX="singularity exec --nv --home $W/home -B /scratch/user/chenzc2001/qx/689 \
  -B $W/kit_cache:/isaac-sim/kit/cache -B $W/kit_data:/isaac-sim/kit/data -B $W/kit_logs:/isaac-sim/kit/logs $SIF"
PY="$SX /isaac-sim/python.sh"

# ------------------------------------------------------------- convert heads
stamp "convert_heads"
$PY tools/assets/convert_heads.py $RAW $USD > $LOG/convert_heads.log 2>&1
grep -v "warn rootless" $LOG/convert_heads.log | tail -n 80
[ -f $USD/assets_manifest.json ] || { stamp "convert_heads produced no manifest; stopping"; exit 1; }

# ------------------------------------------------------------------ preflight
stamp "preflight"
$PY scripts/isaac_preflight.py --config $CFG --out $PRE > $LOG/preflight.log 2>&1
grep -v "warn rootless" $LOG/preflight.log | grep -E '^\[preflight\]|^\{|^  |Error|Traceback' | tail -n 150
STATUS=$(cat $PRE/STATUS 2>/dev/null || echo missing)
stamp "preflight STATUS=$STATUS"
ls -la $PRE/grid_*.png 2>/dev/null
[ "$STATUS" = "OK" ] && exit 0 || exit 2
