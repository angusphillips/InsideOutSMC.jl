#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-nonlinear}"
SEEDS_CSV="${SEEDS:-1,2,3,4,5}"
export JULIA_LOAD_PATH="${JULIA_LOAD_PATH:-@:@stdlib}"

export CONFIG_TAG="${CONFIG_TAG:-}"

# Forward training hyperparameters only if explicitly provided.
for var in NB_ITER NB_TRAJECTORIES NB_PARTICLES NB_IBIS_MOVES NB_CSMC_MOVES BATCH_SIZE; do
  if [[ -n "${!var:-}" ]]; then export "$var"; fi
done

export NB_SPCE_RUNS="${NB_SPCE_RUNS:-32}"
export NB_NMC_RUNS="${NB_NMC_RUNS:-${NB_SPCE_RUNS}}"
export NB_STEPS="${NB_STEPS:-50}"
export NB_OUTER_SAMPLES="${NB_OUTER_SAMPLES:-64}"
export NB_INNER_SAMPLES="${NB_INNER_SAMPLES:-100000}"

echo "===================================================="
echo "Running full 5-seed pipeline"
echo "Experiment: $EXPERIMENT"
echo "Seeds: $SEEDS_CSV"
echo "Config tag: ${CONFIG_TAG:-<none>}"
echo "Load path: $JULIA_LOAD_PATH"
echo "===================================================="

echo "[1/3] Train policies + run sPCE evaluation"
SEEDS="$SEEDS_CSV" bash scripts/run_5seed_pce_pipeline.sh "$EXPERIMENT"

echo "[2/3] Run NMC evaluation"
SEEDS="$SEEDS_CSV" bash scripts/run_5seed_nmc_eval.sh "$EXPERIMENT"

echo "[3/3] Export trajectories"
SEEDS="$SEEDS_CSV" bash scripts/run_5seed_trajectory_pipeline.sh "$EXPERIMENT"

echo "Full pipeline complete."
