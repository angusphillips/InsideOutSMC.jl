#!/bin/bash

#SBATCH --job-name=iosmc_smoke
#SBATCH --output=/vols/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_smoke%A.out
#SBATCH --error=/vols/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_smoke%A.err
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodelist=swan22.cpu.stats.ox.ac.uk
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00

# Lightweight smoke test of the full trajectory_plot pipeline (train + sPCE + NMC + trajectory)
# for one experiment, two seeds, and trivially small hyperparams. Writes to
#   experiments/<env>/data/SMOKE/
# so it does not interfere with real runs.
#
# Override defaults at submit time, e.g.:
#   sbatch --export=ALL,EXPERIMENT=double_pendulum slurm_scripts/smoke_test.sh

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-cartpole}"

LOCAL="/data/localhost/not-backed-up/$USER/InsideOutSMC.jl"
cd "$LOCAL"

export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-@:@stdlib}"

export CONFIG_TAG="SMOKE"
export SEEDS="1,2"

# Tiny training hyperparams
export NB_ITER=2
export NB_TRAJECTORIES=16
export NB_PARTICLES=8
export NB_IBIS_MOVES=3
export NB_CSMC_MOVES=1
export BATCH_SIZE=8
export NB_STEPS=10

# Tiny sPCE / NMC eval
export NB_SPCE_RUNS=2
export NB_NMC_RUNS=2
export NB_OUTER_SAMPLES=4
export NB_INNER_SAMPLES=10

echo "===================================================="
echo "SMOKE TEST"
echo "EXPERIMENT=$EXPERIMENT"
echo "SEEDS=$SEEDS"
echo "Training: NB_ITER=$NB_ITER NB_STEPS=$NB_STEPS NB_TRAJECTORIES=$NB_TRAJECTORIES NB_PARTICLES=$NB_PARTICLES"
echo "Eval:     NB_SPCE_RUNS=$NB_SPCE_RUNS NB_OUTER_SAMPLES=$NB_OUTER_SAMPLES NB_INNER_SAMPLES=$NB_INNER_SAMPLES"
echo "===================================================="

bash scripts/run_5seed_full_pipeline.sh "$EXPERIMENT"

echo SMOKE TEST done!
