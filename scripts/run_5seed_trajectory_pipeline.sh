#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
LOAD_PATH_VALUE="${JULIA_LOAD_PATH:-@:@stdlib}"

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS_CSV"

run_one_experiment() {
  local exp_name="$1"
  local plot_script

  case "$exp_name" in
    nonlinear)
      plot_script="experiments/pendulum/nonlinear/trajectory_plot/make_plot.jl"
      ;;
    cartpole)
      plot_script="experiments/cartpole/trajectory_plot/make_plot.jl"
      ;;
    double_pendulum)
      plot_script="experiments/double_pendulum/trajectory_plot/make_plot.jl"
      ;;
    *)
      echo "Unknown experiment: $exp_name"
      echo "Use one of: nonlinear, cartpole, double_pendulum, all"
      exit 1
      ;;
  esac

  echo "===================================================="
  echo "Experiment: $exp_name"
  echo "Trajectory seeds: $SEEDS_CSV"
  echo "Load path: $LOAD_PATH_VALUE"
  echo "===================================================="

  for seed in "${SEED_ARRAY[@]}"; do
    echo "[trajectory:$exp_name] TRAIN_SEED=$seed"
    TRAIN_SEED="$seed" JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
      julia --project=. "$plot_script"
  done
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
    echo "Usage: scripts/run_5seed_trajectory_pipeline.sh [nonlinear|cartpole|double_pendulum|all]"
    exit 1
    ;;
esac

echo "5-seed trajectory CSV generation complete."
