#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
LOAD_PATH_VALUE="${JULIA_LOAD_PATH:-@:@stdlib}"

NB_SPCE_RUNS_VALUE="${NB_SPCE_RUNS:-32}"
NB_STEPS_VALUE="${NB_STEPS:-50}"
NB_OUTER_SAMPLES_VALUE="${NB_OUTER_SAMPLES:-64}"
NB_INNER_SAMPLES_VALUE="${NB_INNER_SAMPLES:-100000}"

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
    *)
      echo "Unknown experiment: $exp_name"
      echo "Use one of: nonlinear, cartpole, double_pendulum, all"
      exit 1
      ;;
  esac

  echo "===================================================="
  echo "Experiment: $exp_name"
  echo "Training seeds: $SEEDS_CSV"
  echo "Load path: $LOAD_PATH_VALUE"
  echo "===================================================="

  for seed in "${SEED_ARRAY[@]}"; do
    echo "[train:$exp_name] TRAIN_SEED=$seed"
    TRAIN_SEED="$seed" JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
      julia --project=. "$train_script"
  done

  echo "[eval:$exp_name] Running sPCE over training seeds"
  TRAIN_SEEDS="$SEEDS_CSV" \
  NB_SPCE_RUNS="$NB_SPCE_RUNS_VALUE" \
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
    echo "Usage: scripts/run_5seed_pce_pipeline.sh [nonlinear|cartpole|double_pendulum|all]"
    exit 1
    ;;
esac

echo "Pipeline complete."