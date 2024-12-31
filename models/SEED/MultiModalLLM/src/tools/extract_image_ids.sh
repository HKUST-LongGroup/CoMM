data_list=$1

if [ -z "$data_list" ]; then
  data_list="train_data.pth,test_data.pth,val_data.pth"
fi

python3 src/tools/extract_image_ids_comm.py \
  --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml \
  --image_transform configs/processer/blip_transform.yaml \
  --data configs/data/caption_torchdata_preprocess.yaml \
  --save_dir ../datasets/image_ids \
  --data_list $data_list \
  --batch_size 128 --num_workers 8 --gpus 8

