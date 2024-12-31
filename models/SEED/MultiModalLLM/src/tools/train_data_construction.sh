python3 src/tools/tot_data_construct.py \
  --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml \
  --image_transform configs/processer/blip_transform.yaml \
  --data configs/data/caption_torchdata_preprocess.yaml \
  --save_dir data/CoMM \
  --data_path "../datasets/train_data.pth" \
  --batch_size 128 --num_workers 8 --gpus 1