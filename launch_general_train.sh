#!/bin/bash
set -euo pipefail

# Config (override via env)
JOB_SCRIPT="${JOB_SCRIPT:-small_general_run_slurm.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/proj1/eu-25-92/tiny_vqa_creation/output}"

RUN_NUMBER="${1:-}"
QUANTITY_BASE="${2:-}"
MAX_SLICES="${3:-}"

if [[ -z "${RUN_NUMBER}" || -z "${QUANTITY_BASE}" || -z "${MAX_SLICES}" ]]; then
  echo "Usage: $0 RUN_NUMBER QUANTITY_BASE MAX_SLICES" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-run_${RUN_NUMBER}_general}"

for i in $(seq 1 "${MAX_SLICES}"); do
  SLICE_NUMBER="${i}"
  QUANTITY="${QUANTITY_BASE}K"
  SLICE_JSON="${OUTPUT_ROOT}/${RUN_NAME}/test_${RUN_NAME}_karo_${QUANTITY}.json"
  if [[ ! -f "${SLICE_JSON}" ]]; then
    echo "skip missing: ${SLICE_JSON}" >&2
    continue
  fi
  sbatch "${JOB_SCRIPT}" "${RUN_NUMBER}" "${QUANTITY}" "${SLICE_NUMBER}"
done
