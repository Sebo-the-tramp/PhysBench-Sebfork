#!/bin/bash
set -euo pipefail

jobs=(
  "small_roi_baseline_run_slurm.sh"
  "small_roi_no_text_run_slurm.sh"
  "small_roi_text_run_slurm.sh"
  "small_roi_no_text_layout_position_run_slurm.sh"
  "small_roi_text_layout_position_run_slurm.sh"
  "small_no_roi_circling_no_text_yes_layout_position_run_slurm.sh"
  "small_no_roi_circling_yes_text_yes_layout_position_run_slurm.sh"
)

for job in "${jobs[@]}"; do
  sbatch "${job}"
done
