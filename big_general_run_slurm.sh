#!/bin/bash
#SBATCH -A EU-25-92
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -t 24:00:00
#SBATCH -J interactive_gpu

source "/home/it4i-thvu/seb_dev/.telegram_bot.env"

RUN_NUMBER="${1:-}"
QUANTITY="${2:-}"
SLICE_NUMBER="${3:-}"
MODEL_SIZE="big"

if [[ -z "${RUN_NUMBER}" || -z "${QUANTITY}" ]]; then
    exit 1
fi

if [ -n "$SLICE_NUMBER" ]; then
  RUN_NAME="run_${RUN_NUMBER}_general-${SLICE_NUMBER}"
else
  RUN_NAME="run_${RUN_NUMBER}_general"
fi

SCRIPT_NAME="$(basename "$0")"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="🚀 ${SCRIPT_NAME} started on $(hostname) at $(date) | RUN_NAME=${RUN_NAME} | QUANTITY=${QUANTITY} | MODEL_SIZE=${MODEL_SIZE}" >/dev/null &
source /mnt/proj1/eu-25-92/physbench/.venv/bin/activate

python run_parallel.py \
    --model-size "${MODEL_SIZE}" \
    --run-name "${RUN_NAME}" \
    --quantity "${QUANTITY}"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="✅ ${SCRIPT_NAME} completed on $(hostname) at $(date) | RUN_NAME=${RUN_NAME} | QUANTITY=${QUANTITY} | MODEL_SIZE=${MODEL_SIZE}" >/dev/null &
