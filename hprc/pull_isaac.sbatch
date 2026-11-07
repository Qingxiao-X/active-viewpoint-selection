#!/bin/bash
#SBATCH -J pull_isaac
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /scratch/user/qingxiao/689/hprc/logs/%x-%j.out
set -euo pipefail
module load WebProxy/0000
echo "proxy=$https_proxy"
LOCAL=${TMPDIR:-/tmp}/sing_$SLURM_JOB_ID
mkdir -p $LOCAL/cache $LOCAL/tmp
export SINGULARITY_CACHEDIR=$LOCAL/cache SINGULARITY_TMPDIR=$LOCAL/tmp
df -h $LOCAL
cd /scratch/user/qingxiao/689/hprc
singularity pull --force isaac-sim_5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0
ls -la isaac-sim_5.1.0.sif
rm -rf $LOCAL
