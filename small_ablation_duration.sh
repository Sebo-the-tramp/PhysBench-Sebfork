#!/bin/bash
#SBATCH -A EU-25-92
#SBATCH -p qgpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -t 4:00:00
#SBATCH -J interactive_gpu

source "/home/it4i-thvu/seb_dev/.telegram_bot.env"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="🚀 GPU session started for GENERAL_SMALL_MODELS on $(hostname) at $(date)" >/dev/null &

source /mnt/proj1/eu-25-92/physbench/.venv/bin/activate

RUN_NUMBER="${1:-${RUN_NUMBER:-}}"
if [[ -z "${RUN_NUMBER}" ]]; then
  echo "RUN_NUMBER is required (arg1 or env)" >&2
  exit 1
fi
RUN_NAME="run_${RUN_NUMBER}_ablation_physics_duration_text"
echo "Starting run ${RUN_NUMBER} (${RUN_NAME})"
QUANTITY="10K"
MODEL_SIZE="small"

python run_parallel.py \
    --model-size "${MODEL_SIZE}" \
    --run-name "${RUN_NAME}" \
    --quantity "${QUANTITY}"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" \
     --data-urlencode text="✅ GPU session completed for GENERAL_SMALL_MODELS different CHMOD on $(hostname) at $(date)" >/dev/null &     
