#!/bin/bash

#SBATCH --job-name=iosmc_lf
#SBATCH --output=/vols/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_lf%A.out
#SBATCH --error=/vols/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_lf%A.err
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodelist=swan22.cpu.stats.ox.ac.uk
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00:00

# Navigate to local file system
LOCAL="/data/localhost/not-backed-up/$USER/InsideOutSMC.jl"

cd $LOCAL

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
JULIA_LOAD_PATH=@:@stdlib scripts/run_5seed_full_pipeline.sh locationfinding

echo SBATCH script done!