#!/bin/sh
#SBATCH --job-name=bounc
#SBATCH --partition=long
#SBATCH --cpus-per-task=16
#SBATCH --time=4320
#SBATCH --exclude=xgpd6,xgpd7,xgpd9

srun ./slurm_inner.sh
