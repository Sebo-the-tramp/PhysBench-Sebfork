# CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH='./' python eval/test_benchmark.py --model_name "llava-1.5-7b-hf" \
    # --dataset_path /data0/sebastian.cavada/compositional-physics/tiny_vqa_creation/output --split val

# CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH='./' python eval/test_benchmark.py --model_name "vila-1.5-8b" \
#     --dataset_path /mnt/proj1/eu-25-92/tiny_vqa_creation/output --split val

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH='./' python eval/test_benchmark.py \
    --model_name instructblip-flan-t5-xxl \
    --dataset_path /data0/sebastian.cavada/compositional-physics/tiny_vqa_deterministic/output \
    --split val \
    --run_name run_06_general/test_run_06_general