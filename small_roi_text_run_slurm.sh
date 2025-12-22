#!/bin/bash
#SBATCH -A EU-25-92
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -t 12:00:00
#SBATCH -J interactive_gpu

source ".telegram_bot.env"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="🚀 GPU session started for GENERAL_SMALL_MODELS on $(hostname) at $(date)" >/dev/null &

source /mnt/proj1/eu-25-92/physbench/.venv/bin/activate

RUN_NAME="run_11_roi_circling_text"
QUANTITY="1K"
MODEL_SIZE="small"

python run_parallel.py \
    --model-size "${MODEL_SIZE}" \
    --run-name "${RUN_NAME}" \
    --quantity "${QUANTITY}"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="✅ GPU session completed for GENERAL_SMALL_MODELS different CHMOD on $(hostname) at $(date)" >/dev/null &     