#!/bin/bash
# Real Isaac Sim experiment, run by gpu_job.sbatch on one GPU node.
# preflight -> (size) -> collect -> train -> calibrate -> (size) -> evaluate
set -uo pipefail
H=/scratch/user/chenzc2001/qx/689/hprc
R=/scratch/user/chenzc2001/qx/689/active-viewpoint-selection
SIF=$H/isaac-sim_5.1.0_avs.sif
# v3: v1 is the broken run (results/isaac_heads_v1/README.md), v2 is the single-key-light
# real run (random=84.7% vs learned/greedy=100%, but learned tied with greedy -- a single
# point light makes "move away from the one glare blob" near-optimal). v3 adds a second,
# independently-randomized light (avs/scenes.py fill_light_*, avs/envs/isaac_env.py) so
# glare position alone no longer predicts the best escape direction. All three trees are
# kept side by side on purpose.
CFG=configs/isaac_heads_v3.yaml
RUN=$R/runs/isaac_heads_v3
PRE=$RUN/preflight
W=$H/runtime
LOG=$RUN/logs
mkdir -p $W/home $W/kit_cache $W/kit_data $W/kit_logs $PRE $LOG
cd $R

secs_left() {  # remaining job time in seconds (squeue %L: [d-]hh:mm:ss)
  local L; L=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
  python3 -c "
s='$L'; d=0
if '-' in s: d, s = s.split('-'); d=int(d)
p=[int(x) for x in s.split(':')]
while len(p)<3: p.insert(0,0)
print(d*86400+p[0]*3600+p[1]*60+p[2])"
}
stamp() { echo "=== [$(date +%T)] $* (time left $(secs_left)s)"; }

stamp "node $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# The base image ships its own correct /etc/vulkan/icd.d/nvidia_icd.json (api_version
# matches the host driver exactly); a hand-written override here previously declared the
# wrong api_version and broke Vulkan init (ERROR_INCOMPATIBLE_DRIVER). All RTX/OptiX driver
# libs are already listed in /etc/singularity/nvliblist.conf, so --nv binds them on its own
# -- no SINGULARITY_CONTAINLIBS or custom ICD needed (see hprc/preflight_job.sbatch).
export SINGULARITYENV_ACCEPT_EULA=Y SINGULARITYENV_PRIVACY_CONSENT=Y SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=YES
SX="singularity exec --nv --home $W/home -B /scratch/user/chenzc2001/qx/689 \
  -B $W/kit_cache:/isaac-sim/kit/cache -B $W/kit_data:/isaac-sim/kit/data -B $W/kit_logs:/isaac-sim/kit/logs $SIF"
PY="$SX /isaac-sim/python.sh"

$SX bash -c 'nvidia-smi -L'

# ---------------------------------------------------------------- preflight
stamp "preflight"
$PY scripts/isaac_preflight.py --config $CFG --out $PRE > $LOG/preflight.log 2>&1
grep -v "warn rootless" $LOG/preflight.log | grep -E '^\[preflight\]|^\{|^  |Error|Traceback' | tail -n 150
if [ "$(cat $PRE/STATUS 2>/dev/null)" != "OK" ]; then
  stamp "preflight did not pass (STATUS=$(cat $PRE/STATUS 2>/dev/null || echo missing)); stopping"
  exit 2
fi
KEY=$(python3 -c "import json;print(json.load(open('$PRE/exposure.json'))['key_light_base_intensity'])")
DOME=$(python3 -c "import json;print(json.load(open('$PRE/exposure.json'))['dome_base_intensity'])")
EXPO="env.isaac.key_light_base_intensity=$KEY env.isaac.dome_base_intensity=$DOME"
echo "exposure overrides: $EXPO"

# ---------------------------------------------------------------- collect
N=${N_SCENES:-$($SX /isaac-sim/python.sh scripts/size_run.py collect $PRE/report.json $(secs_left) 2>/dev/null | tail -1)}
stamp "collect n_scenes=$N"
$PY -m avs.collect --config $CFG $EXPO collect.n_scenes=$N > $LOG/collect.log 2>&1
grep -E '^\[collect\]|Error|Traceback' $LOG/collect.log | tail -n 60
[ -f $R/data/isaac_heads_v3/transitions.npz ] || { stamp "collection produced no dataset; stopping"; exit 3; }

# ---------------------------------------------------------------- train + calibrate (CPU, Isaac's torch)
stamp "train"
$PY -m avs.train --config $CFG > $LOG/train.log 2>&1; grep -v 'warn rootless' $LOG/train.log | tail -n 40
stamp "calibrate"
$PY -m avs.calibrate --config $CFG --out $RUN/accept.json > $LOG/calibrate.log 2>&1 \
  || echo "calibration failed; evaluator falls back to eval.accept in the config"
grep -v 'warn rootless' $LOG/calibrate.log | tail -n 30
ACC=""
[ -f $RUN/accept.json ] || ACC="eval.accept_file=null"

# ---------------------------------------------------------------- evaluate
CAP=$($SX /isaac-sim/python.sh scripts/size_run.py eval $PRE/report.json $RUN/splits.json $(secs_left) 2>/dev/null | tail -1)
stamp "evaluate max_scenes_per_split=$CAP"
$PY -m avs.evaluate --config $CFG $EXPO $ACC eval.max_scenes_per_split=$CAP > $LOG/evaluate.log 2>&1
grep -E '^\[eval\]|Error|Traceback|wrote' $LOG/evaluate.log | tail -n 60
stamp "done"
ls -la $R/results/isaac_heads_v3 2>/dev/null
