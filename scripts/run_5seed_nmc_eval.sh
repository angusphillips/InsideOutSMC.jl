#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-@:@stdlib}"

export CONFIG_TAG="${CONFIG_TAG:-}"

export NB_NMC_RUNS="${NB_NMC_RUNS:-${NB_SPCE_RUNS:-32}}"
export NB_STEPS="${NB_STEPS:-50}"
export NB_OUTER_SAMPLES="${NB_OUTER_SAMPLES:-64}"
export NB_INNER_SAMPLES="${NB_INNER_SAMPLES:-100000}"

run_one_experiment() {
  local exp_name="$1"
  local eval_script

  case "$exp_name" in
    nonlinear)
      eval_script="experiments/pendulum/nonlinear/estimators/nmc_over_training_seeds.jl"
      ;;
    cartpole)
      eval_script="experiments/cartpole/estimators/nmc_over_training_seeds.jl"
      ;;
    double_pendulum)
      eval_script="experiments/double_pendulum/estimators/nmc_over_training_seeds.jl"
      ;;
    locationfinding)
      eval_script="experiments/locationfinding/estimators/nmc_over_training_seeds.jl"
      ;;
    *)
      echo "Unknown experiment: $exp_name"
      echo "Use one of: nonlinear, cartpole, double_pendulum, locationfinding, all"
      exit 1
      ;;
  esac

  echo "===================================================="
  echo "Experiment: $exp_name"
  echo "Evaluation seeds: $SEEDS_CSV"
  echo "Config tag: ${CONFIG_TAG:-<none>}"
  echo "Load path: $JULIA_LOAD_PATH"
  echo "===================================================="

  echo "[eval:$exp_name] Running NMC over training seeds"
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
    echo "Usage: scripts/run_5seed_nmc_eval.sh [nonlinear|cartpole|double_pendulum|locationfinding|all]"
    exit 1
    ;;
esac

echo "NMC evaluation complete."
