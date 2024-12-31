export IS_STAGE2=False
export WEIGHTFOLDER="WEIGHT_FOLDER"
export DATAFOLDER="datasets/CoMM"

python -m torch.distributed.launch --nproc_per_node=8 \
--master_port 29500 \
--use_env train.py --is_training True \
--train_data_path train_data.pth \
--val_data_path val_data.pth \
--model_save_name MiniGPT-5_CoMM \
--store_path ./outputs/minigpt5_comm >log/minigpt5_comm.log 2>&1 &