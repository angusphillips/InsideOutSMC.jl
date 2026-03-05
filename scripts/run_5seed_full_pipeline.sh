#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
LOAD_PATH_VALUE="${JULIA_LOAD_PATH:-@:@stdlib}"

NB_SPCE_RUNS_VALUE="${NB_SPCE_RUNS:-32}"
NB_NMC_RUNS_VALUE="${NB_NMC_RUNS:-${NB_SPCE_RUNS_VALUE}}"
NB_STEPS_VALUE="${NB_STEPS:-50}"
NB_OUTER_SAMPLES_VALUE="${NB_OUTER_SAMPLES:-64}"
NB_INNER_SAMPLES_VALUE="${NB_INNER_SAMPLES:-100000}"

echo "===================================================="
echo "Running full 5-seed pipeline"
echo "Experiment: $EXPERIMENT"
echo "Seeds: $SEEDS_CSV"
echo "Load path: $LOAD_PATH_VALUE"
echo "===================================================="

echo "[1/3] Train policies + run sPCE evaluation"
SEEDS="$SEEDS_CSV" \
NB_SPCE_RUNS="$NB_SPCE_RUNS_VALUE" \
NB_STEPS="$NB_STEPS_VALUE" \
NB_OUTER_SAMPLES="$NB_OUTER_SAMPLES_VALUE" \
NB_INNER_SAMPLES="$NB_INNER_SAMPLES_VALUE" \
JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
  bash scripts/run_5seed_pce_pipeline.sh "$EXPERIMENT"

echo "[2/3] Run NMC evaluation"
SEEDS="$SEEDS_CSV" \
NB_NMC_RUNS="$NB_NMC_RUNS_VALUE" \
NB_STEPS="$NB_STEPS_VALUE" \
NB_OUTER_SAMPLES="$NB_OUTER_SAMPLES_VALUE" \
NB_INNER_SAMPLES="$NB_INNER_SAMPLES_VALUE" \
JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
  bash scripts/run_5seed_nmc_eval.sh "$EXPERIMENT"

echo "[3/3] Export trajectories"
SEEDS="$SEEDS_CSV" \
JULIA_LOAD_PATH="$LOAD_PATH_VALUE" \
  bash scripts/run_5seed_trajectory_pipeline.sh "$EXPERIMENT"

echo "Full pipeline complete."
