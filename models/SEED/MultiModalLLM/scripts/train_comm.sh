set -x

torchrun --nproc_per_node=8 --nnodes=1 \
    --master_port=29500 src/train/train.py \
    --model configs/model/seed_8b_lora_sft.yaml \
    --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml \
    --train_data configs/data/comm.yaml \
    --output_dir log/seed_llama_8b_comm \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --max_steps 10000 \
    --min_lr_ratio 0.1 \
    --learning_rate 5e-7 \
    --weight_decay 5e-2 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --logging_steps 1 \
    --log_level 'info' \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip \
    --deepspeed configs/deepspeed/stage2_bf16.json

