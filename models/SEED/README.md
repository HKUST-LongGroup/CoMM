# SEED
## Prepare

### Environment

For the environment, you can follow the instructions in the [SEED official repository](https://github.com/AILab-CVC/SEED).


### Data and checkpoints

```
# Dataset Preparation
ln -s ../../datasets ./
```

Download base model checkpoints from [SEED-Llama 8B](https://huggingface.co/AILab-CVC/seed-llama-8b-sft) and [SEED-Llama 14B](https://huggingface.co/AILab-CVC/seed-llama-14b-sft).

Put the downloaded checkpoints to `./pretrained/` and `ln -s ./pretrained ./MultiModalLLM/pretrained`.


## Evaluation

Download our finetuned lora checkpoints here [SEED-Llama 8B CoMM LoRA](https://drive.google.com/file/d/1ay9N17QvFhhVXU-ddfy6EK55YfTlem6t/view?usp=sharing) and [SEED-Llama 14B CoMM LoRA](https://drive.google.com/file/d/1NfOg17gSQgbBf8ejUyn2xudQoHveDzKL/view?usp=sharing). 
Put the downloaded checkpoints to `./MultiModalLLM/checkpoints`.

Merge LoRA checkpoints with the base model checkpoints.
```
cd MultiModalLLM
bash ./src/tools/merge_lora.sh
```

Convert images to discrete image ids.
```
bash src/tools/extract_image_ids.sh "val_data.pth,test_data.pth"
```

Inference with the finetuned model.
```
bash ./scripts/eval_8B.sh
```

## Training on CoMM

Construct training data
```
cd MultiModalLLM
bash ./src/tools/train_data_construction.sh
```

Train the model
```
bash ./scripts/train_comm.sh
```