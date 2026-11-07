#!/bin/bash
#SBATCH -J avs_isaac
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:t4:1
#SBATCH -t 03:00:00
#SBATCH -o /scratch/user/qingxiao/689/hprc/logs/%x-%j.out
# Thin wrapper so the steps in gpu_main.sh can still be edited while the job waits in the queue.
exec bash /scratch/user/qingxiao/689/hprc/gpu_main.sh
