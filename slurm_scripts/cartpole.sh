#!/bin/bash

#SBATCH --job-name=iosmc_cartpole
#SBATCH --output=/vols/bitbucket/anphilli/policy_learning/slurm_outputs/iosmc_cartpole%A.out
#SBATCH --error=/vols/bitbucket/anphilli/policy_learning/slurm_outputs/iosmc_cartpole%A.err
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodelist=swan22.cpu.stats.ox.ac.uk
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00

# Navigate to local file system
LOCAL="/data/localhost/not-backed-up/anphilli/InsideOutSMC.jl"

cd $LOCAL

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
JULIA_LOAD_PATH=@:@stdlib scripts/run_5seed_pce_pipeline.sh cartpole

echo SBATCH script done!