# MiniGPT-5

## Prepare

### Environment

For the environment, you can follow the instructions in the [MiniGPT-5 official repository](https://github.com/eric-ai-lab/MiniGPT-5).

### Data and checkpoints

1. Use the following commands to link CoMM dataset to the correct path:

```bash
ln -s ../../datasets/ ./datasets/CoMM
```

2. Download [Vicuna_7B](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and put it to `./checkpoints/Vicuna_7B`

3. Download [prerained_minigpt4_7b.pth](https://github.com/eric-ai-lab/MiniGPT-5/blob/main/config/prerained_minigpt4_7b.pth) and put it to `./config/prerained_minigpt4_7b.pth`

4. Download [stage1_cc3m.ckpt](https://drive.google.com/file/d/1y-VUXubIzFe0iq5_CJUaE3HKhlrdn4n2/view) and put it to `./checkpoints/stage1_cc3m.ckpt`

5. Download our fine-tuned [checkpoint](https://drive.google.com/file/d/1veGCiwDIsq0X_H-FJVW6yVH-wqsmIaRV/view) and unzip it to `./checkpoints/MiniGPT-5`


## Inference
Use the following command to run inference:

```bash
bash eval.sh MiniGPT5 best_model val_data.pth
```

## Training
bash train.sh