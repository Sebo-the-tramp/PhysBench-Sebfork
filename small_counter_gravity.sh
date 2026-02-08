#!/bin/bash
#SBATCH -A EU-25-92
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -t 12:00:00
#SBATCH -J interactive_gpu

source "/home/it4i-thvu/seb_dev/.telegram_bot.env"

RUN_NUMBER="${RUN_NUMBER:-25}"
RUN_NAME="run_${RUN_NUMBER}_counterfactual_gravity"
QUANTITY="10K"
MODEL_SIZE="small"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="🚀 GPU session started for ${RUN_NAME} on $(hostname) at $(date)" >/dev/null &

source /mnt/proj1/eu-25-92/physbench/.venv/bin/activate

python run_parallel.py \
    --model-size "${MODEL_SIZE}" \
    --run-name "${RUN_NAME}" \
    --quantity "${QUANTITY}"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="✅ GPU session completed for ${RUN_NAME} on $(hostname) at $(date)" >/dev/null &
