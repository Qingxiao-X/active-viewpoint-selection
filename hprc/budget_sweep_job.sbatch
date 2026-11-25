#!/bin/bash
#SBATCH --export=NONE
#SBATCH -J avs_budget_sweep
#SBATCH -p gpu
#SBATCH --exclude=t001
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:t4:1
#SBATCH -t 00:45:00
#SBATCH -o /scratch/user/chenzc2001/qx/689/hprc/logs/%x-%j.out
# Budget sweep against the already-collected v2 (single-light) data/model: does a tighter
# budget (a worse first move becomes unrecoverable) separate learned from greedy_heuristic,
# where budget=6 showed identical first-move divergence (19/72 episodes) but IDENTICAL final
# outcomes every time? Each budget writes to its own results dir.
set -uo pipefail
H=/scratch/user/chenzc2001/qx/689/hprc
R=/scratch/user/chenzc2001/qx/689/active-viewpoint-selection
SIF=$H/isaac-sim_5.1.0_avs.sif
CFG=configs/isaac_heads_v2.yaml
RUN=$R/runs/isaac_heads_v2
PRE=$RUN/preflight
W=$H/runtime
LOG=$RUN/logs
mkdir -p $W/home $W/kit_cache $W/kit_data $W/kit_logs $LOG
cd $R

secs_left() {
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
export SINGULARITYENV_ACCEPT_EULA=Y SINGULARITYENV_PRIVACY_CONSENT=Y SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=YES
SX="singularity exec --nv --home $W/home -B /scratch/user/chenzc2001/qx/689 \
  -B $W/kit_cache:/isaac-sim/kit/cache -B $W/kit_data:/isaac-sim/kit/data -B $W/kit_logs:/isaac-sim/kit/logs $SIF"
PY="$SX /isaac-sim/python.sh"

KEY=$(python3 -c "import json;print(json.load(open('$PRE/exposure.json'))['key_light_base_intensity'])")
DOME=$(python3 -c "import json;print(json.load(open('$PRE/exposure.json'))['dome_base_intensity'])")
EXPO="env.isaac.key_light_base_intensity=$KEY env.isaac.dome_base_intensity=$DOME"

for B in 1 2 3; do
  OUT=$R/results/isaac_heads_v2_budget$B
  stamp "evaluate budget=$B -> $OUT"
  CAP=$($SX /isaac-sim/python.sh scripts/size_run.py eval $PRE/report.json $RUN/splits.json $(secs_left) 2>/dev/null | tail -1)
  $PY -m avs.evaluate --config $CFG $EXPO eval.budget=$B eval.max_scenes_per_split=$CAP eval.out_dir=$OUT \
    > $LOG/evaluate_budget$B.log 2>&1
  grep -E '^\[eval\]|Error|Traceback|wrote' $LOG/evaluate_budget$B.log | tail -n 30
done
stamp "done"
