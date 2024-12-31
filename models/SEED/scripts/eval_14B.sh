type="SEED_Llama_14B"
save_path="./outputs/${type}"


GPU_NUM=8
for i in $(seq 0 $((GPU_NUM-1))); do
    python -u scripts/eval_CoMM.py \
    --cur_gpu_id ${i} \
    --save_path ${save_path} \
    --config configs/llm/seed_llama14b_CoMM.yaml \
    >log/${type}_eval_${i}.log 2>&1 &
done