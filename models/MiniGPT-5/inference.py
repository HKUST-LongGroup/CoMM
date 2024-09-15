import os
import math
import gc
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import transformers
from lightning.pytorch import seed_everything

from typing import Optional, Dict
from dataclasses import dataclass, field
from dataloader import SupervisedDataset, DataCollator

from constants import *

from model import MiniGPT5_Model, MiniGPT5_InputProcessor

from util import ModelArguments, DataArguments, TrainingArguments


def make_eval_data_module(data_args, training_args, data_collator, input_processor=None,
                          output_vis_processor=None, prompt_template=None, gpu_id=None, GPU_NUM=None) -> Dict:
    eval_dataset = SupervisedDataset(data_path=data_args.test_data_path, input_processor=input_processor,
                                     output_vis_processor=output_vis_processor, test=True,
                                     prompt_template=prompt_template, generation=True, gpu_id=gpu_id, GPU_NUM=GPU_NUM)
    # if local_rank == -1:
    #     eval_sampler = SequentialSampler(eval_dataset)
    # else:
    #     eval_sampler = DistributedSampler(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    val_dataloader = DataLoader(eval_dataset,
                                batch_size=1,
                                num_workers=training_args.num_workers,
                                collate_fn=data_collator,
                                sampler=eval_sampler)
    return val_dataloader


def to_device(batch, device):
    for key in batch.keys():
        if type(batch[key]) == list:
            batch[key] = batch[key]
        else:
            batch[key] = batch[key].to(device)
    return batch


def clean_cache():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def validate(model, val_dataloader, epoch, device, train_args):
    model.eval()

    pbar = tqdm(val_dataloader)
    pbar.set_description(f'Validating Epoch {epoch}/{train_args.num_train_epochs}')
    loss_list, text_loss, image_loss, caption_loss = [], [], [], []
    for batch_idx, batch in enumerate(pbar):
        batch = to_device(batch, device)
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        source_text = batch['source']
        target_text = batch['target']
        captions =batch.get('caption', None)
        input_images = batch.get('input_images', None)
        output_image = batch.get('output_image', None)
        input_images_feature = batch.get('input_images_feature', None)
        output_image_feature = batch.get('output_image_feature', None)

        bs = len(source_text)
        with torch.no_grad():
            loss_dict = model(input_ids, attention_mask, input_images, output_image, labels, captions,
                              input_images_feature,
                              output_image_feature)
        loss = loss_dict["loss"]
        loss_list.append(loss.item())
        text_loss.append(loss_dict["text_loss"].item())
        image_loss.append(loss_dict["image_loss"].item() if type(loss_dict["image_loss"]) != float else 0.0)
        caption_loss.append(loss_dict["caption_loss"].item() if "caption_loss" in loss_dict and type(
            loss_dict["caption_loss"]) != float else 0.0)
        pbar.set_postfix(loss=sum(loss_list) / len(loss_list), text_loss=sum(text_loss) / len(text_loss),
                         image_loss=sum(image_loss) / len(image_loss),
                         caption_loss=sum(caption_loss) / len(caption_loss))
    return sum(loss_list) / len(loss_dict)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def map_model_to_device(model, device, device2):
    model.model.to(device)
    model.device2 = device2
    model.sd_text_encoder.to(device2)
    model.vae.to(device2)
    model.unet.to(device2)
    model.image_pipeline.to(device2)
    model.llm_to_t2i_mapping.to(device2)
    model.fc.to(device)
    return model


if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    print("local_rank: ", local_rank, "world_size: ", world_size, "rank: ", rank)

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.distributed.barrier()
    seed_everything(42 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setup_for_distributed(rank == 0 or rank == -1)

    if isinstance(data_args.test_data_path, str):
        data_args.test_data_path = os.path.join(DATAFOLDER, data_args.test_data_path)

    batch_size = training_args.real_batch_size
    num_devices = world_size

    model = MiniGPT5_Model(encoder_model_config=model_args, device=f"cuda:{local_rank}",
                           device2=f"cuda:{local_rank}", **vars(training_args))
    checkpoint = torch.load("./checkpoints/stage1_cc3m.ckpt", map_location="cpu")["state_dict"]
    checkpoint["model.llama_model.base_model.model.lm_head.weight"] = checkpoint.pop("model.llama_model.lm_head.weight")
    checkpoint["model.llama_model.base_model.model.model.embed_tokens.weight"] = checkpoint.pop("model.llama_model.model.embed_tokens.weight")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("load stage1 model success")
    tokenizer = model.tokenizer
    sd_tokenizer = model.sd_tokenizer

    output_vis_processor = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    input_vis_processor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    input_processor = MiniGPT5_InputProcessor(tokenizer=tokenizer, image_processor=input_vis_processor)

    data_collator = DataCollator(tokenizer=tokenizer, sd_tokenizer=sd_tokenizer)
    prompt_template = None
    if model_args.model_type == "llama2":
        prompt_template = "[INST] {} [/INST] "

    val_dataloader = make_eval_data_module(data_args, training_args, data_collator, input_processor,
                                            output_vis_processor, prompt_template, data_args.gpu_id, data_args.GPU_NUM)

    assert training_args.test_weight is not None, "test weight path is None"
    ckpt_path = os.path.join("./", training_args.test_weight)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    print("unexpected_keys", unexpected_keys)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if model_args.is_load_2gpu:
        device2 = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device2 = device
    model = map_model_to_device(model, device, device2)

    model.output_folder = OUTPUT_FOLDER
    output_path = os.path.join(OUTPUT_FOLDER, *(training_args.test_weight.split("/")[1:]))
    print("output_path: ", output_path)

    model.eval()
    pbar = tqdm(val_dataloader)
    pbar.set_description(f'Test')
    loss_list, text_loss, image_loss = [], [], []
    tot = 0
    for batch_idx, batch in enumerate(pbar):
        tot += 1
        image_num = 0
        batch = to_device(batch, device)
        data_output_path = batch.get("data_output_path", None)

        if data_output_path is not None:
            assert len(data_output_path) == 1
            cur_path = os.path.join(output_path, data_output_path[0].split('/')[-1])
        else:
            cur_path = os.path.join(output_path, str(tot))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        input_text = batch["source"][0]
        input_image_list, output_gt_image_list, prediction_image_list = [], [], []
        output_dict = {
            "model_input": input_text.replace("<Img><ImageHere></Img>", "<IMAGE>"),
        }
        origin_image_path = os.path.join(cur_path, "origin")
        task_type = data_output_path[0].split("_")[-1]
        if not os.path.exists(origin_image_path):
            os.makedirs(origin_image_path)
        if "input_images" in batch and task_type != "task2" and task_type != "task4":
            for cur_image in batch["original_images"][0]:
                image_num += 1
                cur_image.save(os.path.join(origin_image_path, f"input_{image_num}.jpg"))
                input_image_list.append(f"input_{image_num}.jpg")
        cur_image_id = image_num
        if "output_origin_images" in batch and task_type != "task1":
            for cur_image in batch["output_origin_images"][0]:
                image_num += 1
                cur_image.save(os.path.join(origin_image_path, f"output_gt_{image_num}.jpg"))
                output_gt_image_list.append(f"output_gt_{image_num}.jpg")
        with torch.no_grad():
            results = model.predict_step(batch, batch_idx)
        pred_out, predict_image_list = results[1], results[3]
        pred_out = pred_out.strip("<unk>").strip("###")
        output_dict["model_output"] = pred_out.replace("[IMG0]", "<IMAGE>")

        if predict_image_list is not None:
            for predict_image in predict_image_list:
                cur_image_id += 1
                predict_image.save(os.path.join(cur_path, f"predict_{cur_image_id}.jpg"))
                prediction_image_list.append(f"predict_{cur_image_id}.jpg")

        output_dict["input_image_list"] = input_image_list
        output_dict["output_gt_image_list"] = output_gt_image_list
        output_dict["prediction_image_list"] = prediction_image_list
        with open(os.path.join(cur_path, "results.json"), "w") as f:
            json.dump(output_dict, f, indent=4)