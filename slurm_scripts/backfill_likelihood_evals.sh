#!/usr/bin/env bash
set -euo pipefail

# Backfill the per-seed "total training likelihood evals" CSV for runs that were
# completed before that logging existed, by parsing the SLURM .out files.
#
# For each .out it reads EXPERIMENT and CONFIG_TAG (printed by run_combo.sh),
# pairs each `[train:...] TRAIN_SEED=N` with the following
# `likelihood evals (total training): V` line, and idempotently upserts rows
# into  data/<CONFIG_TAG>/<prefix>_training_likelihood_evals.csv  (same format
# and location the training script now writes live).
#
# Usage:
#   slurm_scripts/backfill_likelihood_evals.sh                       # all iosmc_*.out
#   slurm_scripts/backfill_likelihood_evals.sh slurm_outputs/iosmc_nonlinear_*.out
#   DRY_RUN=1 slurm_scripts/backfill_likelihood_evals.sh             # report only

SCRIPT_DIR="$(unset CDPATH; cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(unset CDPATH; cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_DIR"

DRY_RUN="${DRY_RUN:-0}"

if [[ $# -ge 1 ]]; then
  OUT_FILES=("$@")
else
  OUT_FILES=(slurm_outputs/iosmc_*.out)
fi

# experiment -> "data_dir|csv_prefix"
data_dir_for() {
  case "$1" in
    nonlinear)       echo "experiments/pendulum/nonlinear/data|nonlinear_pendulum" ;;
    cartpole)        echo "experiments/cartpole/data|cartpole" ;;
    double_pendulum) echo "experiments/double_pendulum/data|double_pendulum" ;;
    *)               echo "" ;;
  esac
}

updated=0
skipped=0

for out in "${OUT_FILES[@]}"; do
  [[ -f "$out" ]] || continue

  exp="$(grep -m1 -E '^EXPERIMENT=' "$out" | cut -d= -f2- || true)"
  tag="$(grep -m1 -E '^CONFIG_TAG=' "$out" | cut -d= -f2- || true)"
  if [[ -z "$exp" || -z "$tag" ]]; then
    echo "skip  $(basename "$out"): missing EXPERIMENT/CONFIG_TAG"; skipped=$((skipped+1)); continue
  fi

  mapping="$(data_dir_for "$exp")"
  if [[ -z "$mapping" ]]; then
    echo "skip  $(basename "$out"): unknown experiment '$exp'"; skipped=$((skipped+1)); continue
  fi
  data_dir="${mapping%%|*}"
  prefix="${mapping##*|}"
  target_dir="${data_dir}/${tag}"
  csv="${target_dir}/${prefix}_training_likelihood_evals.csv"

  if [[ ! -d "$target_dir" ]]; then
    echo "skip  $(basename "$out"): no data dir $target_dir"; skipped=$((skipped+1)); continue
  fi

  # Extract "seed,value" pairs from the training phase.
  pairs="$(awk '
    /\[train:.*TRAIN_SEED=/ { split($0, a, "="); cur=a[2]+0; have=1 }
    /likelihood evals \(total training\):/ { if (have) { print cur","$NF+0; have=0 } }
  ' "$out")"

  if [[ -z "$pairs" ]]; then
    echo "skip  $(basename "$out"): no training-eval lines found"; skipped=$((skipped+1)); continue
  fi

  npairs="$(printf '%s\n' "$pairs" | grep -c .)"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "would write $csv  ($npairs seeds): $(printf '%s ' $pairs)"
    updated=$((updated+1))
    continue
  fi

  # Idempotent upsert: merge existing rows with the parsed pairs, sort by seed.
  # Always feed a real header-bearing file first, so awk's FNR==NR (existing
  # vs. new) split can't misfire on an empty file and drop the first pair.
  existing_tmp="$(mktemp)"
  if [[ -f "$csv" ]]; then
    cat "$csv" > "$existing_tmp"
  else
    echo "seed,total_training_likelihood_evals" > "$existing_tmp"
  fi
  merged="$(printf '%s\n' "$pairs" | awk -F, '
    FNR==NR { if (FNR>1 && NF>=2) val[$1+0]=$2+0; next }
    NF>=2   { val[$1+0]=$2+0 }
    END {
      print "seed,total_training_likelihood_evals";
      nk=0; for (k in val) keys[nk++]=k+0;
      for (i=0;i<nk;i++) for (j=i+1;j<nk;j++) if (keys[i]>keys[j]) { t=keys[i]; keys[i]=keys[j]; keys[j]=t }
      for (i=0;i<nk;i++) print keys[i] "," val[keys[i]];
    }
  ' "$existing_tmp" -)"
  rm -f "$existing_tmp"
  printf '%s\n' "$merged" > "$csv"
  echo "wrote $csv  ($npairs seeds)"
  updated=$((updated+1))
done

echo "Done. ${updated} file(s) processed, ${skipped} skipped."
