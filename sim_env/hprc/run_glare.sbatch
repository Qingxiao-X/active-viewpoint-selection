#!/bin/bash
#SBATCH -J glare_capture
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx:1
#SBATCH -t 00:40:00
#SBATCH -o hprc/logs/%x-%j.out   # relative to the folder you submit from (sim_env/)
# Face-glare capture on one Grace RTX 6000 node, Isaac Sim 5.1.0 container (headless RTX).
set -uo pipefail
P=${P:-$SLURM_SUBMIT_DIR}                 # submit from sim_env/
H=${H:-/scratch/user/qingxiao/689/hprc}   # Isaac Sim container + InsightFace models; override with H=... if you have your own copy
export H
RT=${RT:-$SCRATCH/avs_isaac_runtime}      # per-user writable caches
SIF=$H/isaac-sim_5.1.0_avs.sif
W=$RT
OUT=${OUT:-$P/outputs_new}   # outputs/ holds the reference run; keep it unchanged
CFG=${CFG:-$P/configs/glare_scene.json}
mkdir -p $W/home $W/kit_cache $W/kit_data $W/kit_logs $OUT
echo "=== $(date) node $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# RTX needs the ray-tracing/OptiX/Vulkan driver libs; bind those that --nv does not already bind
EXTRA=$(ldconfig -p | grep 'x86-64' | grep -E 'libnvidia-rtcore\.so|libnvoptix\.so|libnvidia-glvkspirv\.so|libnvidia-gpucomp\.so|libnvidia-glcore\.so|libnvidia-tls\.so|libGLX_nvidia\.so\.0|libnvidia-ngx\.so' | awk '{print $NF}' | sort -u)
NVL=$(grep -v '^#' /etc/singularity/nvliblist.conf 2>/dev/null | sed 's/\.so.*//' | grep . | sort -u)
EXTRA=$(for f in $EXTRA; do b=$(basename $f); b=${b%%.so*}; echo "$NVL" | grep -qx "$b" || echo $f; done | paste -sd, -)
export SINGULARITY_CONTAINLIBS=$EXTRA
cat > $W/nvidia_icd.json <<JSON
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}
JSON
export SINGULARITYENV_ACCEPT_EULA=Y SINGULARITYENV_PRIVACY_CONSENT=Y SINGULARITYENV_OMNI_KIT_ACCEPT_EULA=YES
export SINGULARITYENV_VK_ICD_FILENAMES=/opt/avs_nvidia_icd.json
SX="singularity exec --nv --home $W/home -B $P -B $H \
  -B $W/kit_cache:/isaac-sim/kit/cache -B $W/kit_data:/isaac-sim/kit/data -B $W/kit_logs:/isaac-sim/kit/logs \
  -B $W/nvidia_icd.json:/opt/avs_nvidia_icd.json $SIF"
cd $P
$SX cat /isaac-sim/VERSION; echo
$SX /isaac-sim/python.sh scripts/run_capture.py --config $CFG --out $OUT ${EXTRA_ARGS:-} > $OUT/capture.log 2>&1
echo "=== capture exit $? $(date)"
grep -E '^\[glare|Traceback|Error' $OUT/capture.log | grep -v 'warn rootless' | tail -n 80
$SX /isaac-sim/python.sh glare/figures.py --out $OUT 2>&1 | grep -v 'warn rootless' | tail -n 20
echo "=== done $(date)"
