#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-@:@stdlib}"

# CONFIG_TAG selects an output subdirectory under experiments/<env>/data/.
# If empty/unset, scripts fall back to the flat data/ directory (legacy mode).
export CONFIG_TAG="${CONFIG_TAG:-}"

# Forward training hyperparameters only if explicitly provided, so Julia
# falls back to per-experiment defaults otherwise.
for var in NB_ITER NB_TRAJECTORIES NB_PARTICLES NB_IBIS_MOVES NB_CSMC_MOVES BATCH_SIZE; do
  if [[ -n "${!var:-}" ]]; then export "$var"; fi
done

# Evaluation defaults (used by spce_over_training_seeds.jl).
export NB_SPCE_RUNS="${NB_SPCE_RUNS:-32}"
export NB_STEPS="${NB_STEPS:-50}"
export NB_OUTER_SAMPLES="${NB_OUTER_SAMPLES:-64}"
export NB_INNER_SAMPLES="${NB_INNER_SAMPLES:-100000}"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS_CSV"

run_one_experiment() {
  local exp_name="$1"
  local train_script
  local eval_script

  case "$exp_name" in
    nonlinear)
      train_script="experiments/pendulum/nonlinear/trajectory_plot/io_csmc_sysid.jl"
      eval_script="experiments/pendulum/nonlinear/estimators/spce_over_training_seeds.jl"
      ;;
    cartpole)
      train_script="experiments/cartpole/trajectory_plot/io_csmc_sysid.jl"
      eval_script="experiments/cartpole/estimators/spce_over_training_seeds.jl"
      ;;
    double_pendulum)
      train_script="experiments/double_pendulum/trajectory_plot/io_csmc_sysid.jl"
      eval_script="experiments/double_pendulum/estimators/spce_over_training_seeds.jl"
      ;;
    locationfinding)
      train_script="experiments/locationfinding/trajectory_plot/io_csmc_sysid.jl"
      eval_script="experiments/locationfinding/estimators/spce_over_training_seeds.jl"
      ;;
    *)
      echo "Unknown experiment: $exp_name"
      echo "Use one of: nonlinear, cartpole, double_pendulum, locationfinding, all"
      exit 1
      ;;
  esac

  echo "===================================================="
  echo "Experiment: $exp_name"
  echo "Training seeds: $SEEDS_CSV"
  echo "Config tag: ${CONFIG_TAG:-<none>}"
  echo "Load path: $JULIA_LOAD_PATH"
  echo "===================================================="

  for seed in "${SEED_ARRAY[@]}"; do
    echo "[train:$exp_name] TRAIN_SEED=$seed"
    TRAIN_SEED="$seed" julia --project=. "$train_script"
  done

  echo "[eval:$exp_name] Running sPCE over training seeds"
  TRAIN_SEEDS="$SEEDS_CSV" julia --project=. "$eval_script"
}

case "$EXPERIMENT" in
  all)
    run_one_experiment nonlinear
    run_one_experiment cartpole
    run_one_experiment double_pendulum
    run_one_experiment locationfinding
    ;;
  nonlinear|cartpole|double_pendulum|locationfinding)
    run_one_experiment "$EXPERIMENT"
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT"
    echo "Usage: scripts/run_5seed_pce_pipeline.sh [nonlinear|cartpole|double_pendulum|locationfinding|all]"
    exit 1
    ;;
esac

echo "Pipeline complete."
