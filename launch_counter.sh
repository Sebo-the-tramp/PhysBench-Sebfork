#!/bin/bash
set -euo pipefail

jobs=(
  # "small_counter_smaller.sh"
  # "small_counter_shift.sh"
  # "small_counter_gravity.sh"
  "big_counter_smaller.sh"
  "big_counter_shift.sh"
  "big_counter_gravity.sh"
  "big_counter_smaller_tmp.sh"
  "big_counter_shift_tmp.sh"
  "big_counter_gravity_tmp.sh"
)

RUN_NUMBER="${1:-${RUN_NUMBER:-}}"
sbatch_args=()
if [[ -n "${RUN_NUMBER}" ]]; then
  sbatch_args+=(--export=ALL,RUN_NUMBER="${RUN_NUMBER}")
fi

for job in "${jobs[@]}"; do
  sbatch "${sbatch_args[@]}" "${job}"
done
