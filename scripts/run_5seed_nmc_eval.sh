#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
LOAD_PATH_VALUE="${JULIA_LOAD_PATH:-@:@stdlib}"

NB_NMC_RUNS_VALUE="${NB_NMC_RUNS:-${NB_SPCE_RUNS:-32}}"
NB_STEPS_VALUE="${NB_STEPS:-50}"
NB_OUTER_SAMPLES_VALUE="${NB_OUTER_SAMPLES:-64}"
NB_INNER_SAMPLES_VALUE="${NB_INNER_SAMPLES:-100000}"

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
    *)
      echo "Unknown experiment: $exp_name"
      echo "Use one of: nonlinear, cartpole, double_pendulum, all"
      exit 1
      ;;
  esac

  echo "===================================================="
  echo "Experiment: $exp_name"
  echo "Evaluation seeds: $SEEDS_CSV"
  echo "Load path: $LOAD_PATH_VALUE"
  echo "===================================================="

  echo "[eval:$exp_name] Running NMC over training seeds"
  TRAIN_SEEDS="$SEEDS_CSV" \
  NB_NMC_RUNS="$NB_NMC_RUNS_VALUE" \
  NB_STEPS="$NB_STEPS_VALUE" \
  NB_OUTER_SAMPLES="$NB_OUTER_SAMPLES_VALUE" \
  NB_INNER_SAMPLES="$NB_INNER_SAMPLES_VALUE" \
  JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
    julia --project=. "$eval_script"
}

case "$EXPERIMENT" in
  all)
    run_one_experiment nonlinear
    run_one_experiment cartpole
    run_one_experiment double_pendulum
    ;;
  nonlinear|cartpole|double_pendulum)
    run_one_experiment "$EXPERIMENT"
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT"
    echo "Usage: scripts/run_5seed_nmc_eval.sh [nonlinear|cartpole|double_pendulum|all]"
    exit 1
    ;;
esac

echo "NMC evaluation complete."
