#!/usr/bin/env bash
set -euo pipefail

# Submit diagnostic runs probing why IO-SMC2 performance drops under the
# 0.1-everywhere diffusion setup. Each job = one (config, experiment) at a
# single T/P combo (T256P128) with 3 seeds. 3 configs x 3 experiments = 9 jobs.
#
# Configs (all vs. the current 0.1-everywhere, paper-penalty baseline):
#   diff001   : DIFF_ZERO_NOISE=0.001 on originally-noiseless components,
#               penalties/tempering at paper defaults.
#   slew0     : 0.1 everywhere, slew_rate_penalty=0.0, tempering default.
#   temper2x  : 0.1 everywhere, slew_rate_penalty=0.1, tempering = 2x baseline
#               (nonlinear 2.0, cartpole 0.5, double_pendulum 0.5).
#   paperorig : original paper setup - DIFF_ZERO_NOISE=0.0 (noise off on the
#               originally-noiseless components), slew_rate_penalty=1.0,
#               tempering=0.25 (intended for cartpole/double_pendulum).
#
# Output tags are prefixed so nothing collides with the existing unprefixed
# baseline results (data/T256P128/...): data/<config>_T256P128/...
#
# Usage:
#   slurm_scripts/submit_diagnostics.sh                 # all 9 jobs
#   DRY_RUN=1 slurm_scripts/submit_diagnostics.sh       # print sbatch lines only
#   CONFIGS="slew0 temper2x" EXPERIMENTS="nonlinear double_pendulum" \
#       slurm_scripts/submit_diagnostics.sh             # subset

SCRIPT_DIR="$(unset CDPATH; cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(unset CDPATH; cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_OUT_DIR="$HOME/InsideOutSMC.jl/slurm_outputs"

# Combos to run. Default: single T256P128 combo for a cheap diagnostic pass.
# Override COMBOS to promote a config to paper-grade, e.g.
#   COMBOS="25 128 64 3 1
#   25 256 128 3 1
#   25 512 256 3 1
#   25 1024 256 3 1
#   25 1024 512 3 1" SEEDS=1,2,3,4,5 CONFIGS=temper2x slurm_scripts/submit_diagnostics.sh
# Each line: NB_ITER NB_TRAJECTORIES NB_PARTICLES NB_IBIS_MOVES NB_CSMC_MOVES
COMBOS_RAW="${COMBOS:-25 256 128 3 1}"
mapfile -t COMBOS_LIST <<< "$COMBOS_RAW"

# Exported (not listed inline in --export) because its comma-separated value
# would otherwise be split by SLURM's --export parser. --export=ALL forwards it
# from the submitting environment with the commas intact.
export SEEDS="${SEEDS:-1,2,3}"

# When set to 1, training skips any seed whose policy .jld2 already exists, so
# seeds can be expanded later without retraining/overwriting earlier ones.
# Forwarded via --export=ALL. Default 0 = current behaviour (always train).
export SKIP_TRAINED_SEEDS="${SKIP_TRAINED_SEEDS:-0}"

read -r -a CONFIGS_LIST <<< "${CONFIGS:-diff001 slew0 temper2x}"
read -r -a EXPERIMENTS_LIST <<< "${EXPERIMENTS:-nonlinear cartpole double_pendulum}"
DRY_RUN="${DRY_RUN:-0}"

# 2x-baseline tempering per experiment (baselines: nl 1.0, cp 0.25, dp 0.25).
temper2x_for() {
  case "$1" in
    nonlinear)       echo "2.0" ;;
    cartpole)        echo "0.5" ;;
    double_pendulum) echo "0.5" ;;
    *) echo "ERROR: no temper2x value for $1" >&2; exit 1 ;;
  esac
}

for config in "${CONFIGS_LIST[@]}"; do
  case "$config" in
    diff001|slew0|temper2x|paperorig) ;;
    *) echo "Unknown config: $config"; exit 1 ;;
  esac

  for exp in "${EXPERIMENTS_LIST[@]}"; do
    case "$exp" in
      nonlinear|cartpole|double_pendulum) ;;
      *) echo "Unknown experiment: $exp"; exit 1 ;;
    esac

    for combo in "${COMBOS_LIST[@]}"; do
      [[ -z "${combo// }" ]] && continue
      read -r nb_iter nb_traj nb_part nb_ibis nb_csmc <<< "$combo"
      combo_tag="T${nb_traj}P${nb_part}"

      tag="${config}_${combo_tag}"
      job_name="iosmc_${config}_${exp}_${combo_tag}"
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

      # Per-config overrides (unset vars fall back to paper defaults in Julia).
      case "$config" in
        diff001)
          export_vars+=",DIFF_ZERO_NOISE=0.001"
          ;;
        slew0)
          export_vars+=",SLEW_RATE_PENALTY=0.0"
          ;;
        temper2x)
          export_vars+=",TEMPERING=$(temper2x_for "$exp")"
          ;;
        paperorig)
          # Original paper setup: noise off on the originally-noiseless
          # components, higher slew penalty, paper tempering (0.25 for
          # cartpole/double_pendulum).
          export_vars+=",DIFF_ZERO_NOISE=0.0"
          export_vars+=",SLEW_RATE_PENALTY=1.0"
          export_vars+=",TEMPERING=0.25"
          ;;
      esac

      cmd=(sbatch
        --job-name="$job_name"
        --output="$out_pat"
        --error="$err_pat"
        --export="$export_vars"
        --chdir="$PROJECT_DIR"
        "slurm_scripts/run_combo.sh"
      )

      if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${cmd[@]}"; echo
      else
        "${cmd[@]}"
      fi
    done
  done
done

echo "Done."
