MODELS_FULL=(
#instructblip-flan-t5-xl
#instructblip-flan-t5-xxl
#instructblip-vicuna-7b
#instructblip-vicuna-13b
#blip2-flant5xxl
#llava-1.5-7b-hf
#llava-1.5-13b-hf
#llava-v1.6-mistral-7b-hf
#llava-v1.6-vicuna-7b-hf
#deepseek1B
#deepseek7B
#Xinyuan-VL-2B
#Aquila-VL-2B
#Phi-3-vision-128k-instruct
#Phi-3.5V
#mPLUG-Owl3-1B-241014
#mPLUG-Owl3-2B-241014
#mPLUG-Owl3-7B-241101
#MiniCPM-V2
MiniCPM-V2.5
# MiniCPM-V2.6
# Qwen-VL-Chat
# InternVL-Chat-V1-5-quantable
#llava-interleave-qwen-7b-hf
#llava-interleave-qwen-7b-dpo-hf
#vila-1.5-3b
#vila-1.5-3b-s2
#vila-1.5-8b
#vila-1.5-13b
#cambrian-8b
#paligemma2-3b
#paligemma2-10b
#LLaVA-NeXT-Video-7B-DPO-hf
#LLaVA-NeXT-Video-7B-hf
# MolmoE-1B
# MolmoE-7B-O
# MolmoE-7B-D
#InternVL2-1B
#InternVL2-2B
#InternVL2-4B
#InternVL2-8B
#InternVL2-26B
#InternVL2-40B
#InternVL2-76B
#InternVL2_5-1B
#InternVL2_5-2B
#InternVL2_5-4B
#InternVL2_5-8B
#InternVL2_5-26B
#InternVL2_5-38B
#InternVL2_5-78B
#Mantis-8B-Idefics2
#Mantis-llava-7b
#Mantis-8B-siglip-llama3
#Mantis-8B-clip-llama3
#gpt4v
#gpt4o
#o1
#gpt4o-mini
#gemini-1.5-flash
#gemini-1.5-pro
#claude-3-5-sonnet
#claude-3-sonnet
#claude-3-opus
#claude-3-haiku
)


for MODEL in "${MODELS_FULL[@]}"; do
  echo "=== Processing with model $MODEL ==="
  start=$(date +%s)
  #CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH='./' python eval/test_benchmark.py --model_name $MODEL --dataset_path /mnt/proj1/eu-25-92/physbench --split test 
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH='./' python eval/test_benchmark.py --model_name $MODEL --dataset_path /mnt/proj1/eu-25-92/tiny_vqa_creation/output --split val
  #> /dev/null 2>&1

  end=$(date +%s)  
  runtime=$((end - start))
  echo "Model $model_name took ${runtime}s"
done