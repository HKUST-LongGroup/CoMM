export IS_STAGE2=False
export WEIGHTFOLDER="WEIGHT_FOLDER"
export DATAFOLDER="datasets/CoMM"

eval_model=$1
ckpt=$2
test_pth=$3
if [ ${#test_pth} -eq 0 ]; then
    test_pth="test_data.pth"
fi

# bash eval.sh MiniGPT5 best_model val_data.pth
GPU_NUM=8
for i in $(seq 0 $((GPU_NUM-1)))
do
    CUDA_VISIBLE_DEVICES=$i python -m torch.distributed.run --nproc_per_node=1 --master_port 990$i inference.py --test_data_path ${test_pth} --test_weight checkpoints/$eval_model/$ckpt.ckpt --gpu_id $i --GPU_NUM $GPU_NUM > checkpoints/$eval_model/eval_${ckpt}_$i.log 2>&1 &
done