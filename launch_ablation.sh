#!/bin/bash
set -euo pipefail

jobs=(
  # "small_roi_baseline_run_slurm.sh"
  "small_roi_no_text_run_slurm.sh"
  # "small_roi_text_run_slurm.sh"
  "small_roi_no_text_layout_position_run_slurm.sh"
  "small_roi_text_layout_position_run_slurm.sh"
  # "small_no_roi_circling_no_text_yes_layout_position_run_slurm.sh"
  # "small_no_roi_circling_yes_text_yes_layout_position_run_slurm.sh"

  "small_ablation_mass.sh",
  "small_ablation_duration.sh"
)

RUN_NUMBER="${1:-${RUN_NUMBER:-}}"
sbatch_args=()
if [[ -n "${RUN_NUMBER}" ]]; then
  sbatch_args+=(--export=ALL,RUN_NUMBER="${RUN_NUMBER}")
fi

for job in "${jobs[@]}"; do
  sbatch "${sbatch_args[@]}" "${job}"
done
