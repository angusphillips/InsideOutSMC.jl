#!/usr/bin/env bash
set -euo pipefail

# Submit one SLURM job per (experiment, combo) for the trajectory_plot pipeline.
# Defaults to 3 experiments x 5 combos = 15 jobs.
#
# Usage:
#   slurm_scripts/submit_all_combos.sh                     # all 3 experiments
#   slurm_scripts/submit_all_combos.sh cartpole            # single experiment
#   EXPERIMENTS="cartpole double_pendulum" slurm_scripts/submit_all_combos.sh
#   DRY_RUN=1 slurm_scripts/submit_all_combos.sh           # print sbatch lines without submitting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_OUT_DIR="/vols/bitbucket/anphilli/InsideOutSMC.jl/slurm_outputs"

# Each entry: NB_ITER NB_TRAJECTORIES NB_PARTICLES NB_IBIS_MOVES NB_CSMC_MOVES
COMBOS=(
  "25 128 64   3 1"
  "25 256 128  3 1"
  "25 512 256  3 1"
  "25 1024 256 3 1"
  "25 1024 512 3 1"
)

if [[ $# -ge 1 ]]; then
  EXPERIMENTS_LIST=("$@")
elif [[ -n "${EXPERIMENTS:-}" ]]; then
  read -r -a EXPERIMENTS_LIST <<< "$EXPERIMENTS"
else
  EXPERIMENTS_LIST=(nonlinear cartpole double_pendulum)
fi

DRY_RUN="${DRY_RUN:-0}"

for exp in "${EXPERIMENTS_LIST[@]}"; do
  case "$exp" in
    nonlinear|cartpole|double_pendulum|locationfinding) ;;
    *) echo "Unknown experiment: $exp"; exit 1 ;;
  esac

  for combo in "${COMBOS[@]}"; do
    read -r nb_iter nb_traj nb_part nb_ibis nb_csmc <<< "$combo"
    tag="T${nb_traj}P${nb_part}"
    job_name="iosmc_${exp}_${tag}"
    out_pat="${SLURM_OUT_DIR}/${job_name}_%A.out"
    err_pat="${SLURM_OUT_DIR}/${job_name}_%A.err"

    export_vars="ALL"
    export_vars+=",EXPERIMENT=${exp}"
    export_vars+=",CONFIG_TAG=${tag}"
    export_vars+=",NB_ITER=${nb_iter}"
    export_vars+=",NB_TRAJECTORIES=${nb_traj}"
    export_vars+=",NB_PARTICLES=${nb_part}"
    export_vars+=",NB_IBIS_MOVES=${nb_ibis}"
    export_vars+=",NB_CSMC_MOVES=${nb_csmc}"

    cmd=(sbatch
      --job-name="$job_name"
      --output="$out_pat"
      --error="$err_pat"
      --export="$export_vars"
      "${SCRIPT_DIR}/run_combo.sh"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
done

echo "Done."
