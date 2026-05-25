#!/bin/bash

#SBATCH --job-name=iosmc_combo
#SBATCH --output=/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_combo%A.out
#SBATCH --error=/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs/iosmc_combo%A.err
#SBATCH --clusters=srf_cpu_01
#SBATCH --partition=standard-cpu
#SBATCH --nodelist=swan22.cpu.stats.ox.ac.uk
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-12:00:00

# Generic per-combo SLURM script. Caller must pass via --export:
#   EXPERIMENT       (nonlinear | cartpole | double_pendulum)
#   CONFIG_TAG       (e.g. T128P64; used as data/ subdir name)
#   NB_ITER, NB_TRAJECTORIES, NB_PARTICLES, NB_IBIS_MOVES, NB_CSMC_MOVES
# Override --job-name / --output / --time on the sbatch command line as needed.

set -euo pipefail

: "${EXPERIMENT:?EXPERIMENT env var required}"
: "${CONFIG_TAG:?CONFIG_TAG env var required}"

LOCAL="/bitbucket/$USER/InsideOutSMC.jl"
cd "$LOCAL"

export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-@:@stdlib}"

echo "===================================================="
echo "EXPERIMENT=$EXPERIMENT"
echo "CONFIG_TAG=$CONFIG_TAG"
echo "NB_ITER=${NB_ITER:-<default>}"
echo "NB_TRAJECTORIES=${NB_TRAJECTORIES:-<default>}"
echo "NB_PARTICLES=${NB_PARTICLES:-<default>}"
echo "NB_IBIS_MOVES=${NB_IBIS_MOVES:-<default>}"
echo "NB_CSMC_MOVES=${NB_CSMC_MOVES:-<default>}"
echo "===================================================="

bash scripts/run_5seed_full_pipeline.sh "$EXPERIMENT"

echo SBATCH script done!
