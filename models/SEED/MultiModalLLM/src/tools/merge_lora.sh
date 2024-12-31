lora_path="./checkpoints/seed_8b_lora_comm"
base_model="pretrained/seed-llama-8b-sft"

python src/tools/merge_lora_weights.py \
  --model_cfg configs/model/seed_8b_lora_sfted.yaml \
  --tokenizer_cfg configs/tokenizer/seed_llama_tokenizer.yaml \
  --base_model ${base_model} \
  --lora_model ${lora_path}/checkpoint-10000 \
  --save_path ${lora_path}/checkpoint-merged-10000 