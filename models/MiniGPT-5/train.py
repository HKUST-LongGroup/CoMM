import os
import math
import gc
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from dataloader import SupervisedDataset, DataCollator
from lightning.pytorch.loggers import WandbLogger

from constants import *

from lightning.pytorch.callbacks import BasePredictionWriter
from model import MiniGPT5_Model, MiniGPT5_InputProcessor
from metric import *

from util import ModelArguments, DataArguments, TrainingArguments


def make_supervised_data_module(data_args, training_args, data_collator, input_processor=None,
                                output_vis_processor=None, prompt_template=None, model_type=None):
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = SupervisedDataset(data_path=data_args.train_data_path, input_processor=input_processor,
                                      output_vis_processor=output_vis_processor, prompt_template=prompt_template,
                                      model_type=model_type)
    eval_dataset = SupervisedDataset(data_path=data_args.val_data_path, input_processor=input_processor,
                                     output_vis_processor=output_vis_processor, test=True,
                                     prompt_template=prompt_template,
                                     model_type=model_type)
    if local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  num_workers=training_args.num_workers,
                                  collate_fn=data_collator,
                                  prefetch_factor=4,
                                  sampler=train_sampler,
                                  pin_memory=False)
    val_dataloader = DataLoader(eval_dataset,
                                batch_size=training_args.per_device_eval_batch_size,
                                num_workers=training_args.num_workers,
                                collate_fn=data_collator,
                                prefetch_factor=4,
                                sampler=eval_sampler,
                                pin_memory=False)
    return train_dataloader, val_dataloader, train_sampler, eval_sampler


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


def train_one_epoch(model, train_dataloader, epoch, device, train_args, optimizer, scheduler):
    model.train()
    cur_idx = 0
    pbar = tqdm(train_dataloader)
    tot_len = len(pbar)
    pbar.set_description(f'Training Epoch {epoch}/{train_args.num_train_epochs}')
    loss_list, text_loss, image_loss, caption_loss = [], [], [], []
    for batch_idx, batch in enumerate(pbar):
        batch = to_device(batch, device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        source_text = batch.get('source', None)
        target_text = batch['target']
        captions =batch.get('caption', None)
        input_images = batch.get('input_images', None)
        output_image = batch.get('output_image', None)
        input_images_feature = batch.get('input_images_feature', None)
        output_image_feature = batch.get('output_image_feature', None)

        bs = len(source_text)
        # loss_dict = model(input_ids, attention_mask, input_images, output_image, labels, captions,
        #                   input_images_feature, output_image_feature, image_loss_only)
        # loss = loss_dict['loss']
        # loss_list.append(loss.item())
        # loss = loss / gradient_accumulation_steps
        # loss.backward()
        cur_idx += 1
        if cur_idx == gradient_accumulation_steps:
            loss_dict = model(input_ids, attention_mask, input_images, output_image, labels, captions,
                          input_images_feature, output_image_feature, image_loss_only)
            loss = loss_dict['loss']
            loss_list.append(loss.item())
            loss = loss / gradient_accumulation_steps
            loss.backward()
            cur_idx = 0
            model_without_ddp.model.reset_embeddings()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        else:
            with model.no_sync():
                loss_dict = model(input_ids, attention_mask, input_images, output_image, labels, captions,
                          input_images_feature, output_image_feature, image_loss_only)
                loss = loss_dict['loss']
                loss_list.append(loss.item())
                loss = loss / gradient_accumulation_steps
                loss.backward()
        text_loss.append(loss_dict["text_loss"].item())
        image_loss.append(loss_dict["image_loss"].item() if type(loss_dict["image_loss"]) != float else 0.0)
        caption_loss.append(loss_dict["caption_loss"].item() if "caption_loss" in loss_dict and type(
            loss_dict["caption_loss"]) != float else 0.0)

        pbar.set_postfix(loss=sum(loss_list) / len(loss_list), text_loss=sum(text_loss) / len(text_loss),
                         image_loss=sum(image_loss) / len(image_loss),
                         caption_loss=sum(caption_loss) / len(caption_loss))

        if epoch == 0 and batch_idx % (tot_len // 5) == 0:
            save_checkpoint(model.state_dict(), os.path.join(path, f"{batch_idx // (tot_len // 5)}.ckpt"))
            print(f"save {batch_idx // (tot_len // 5)} model")


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


def save_checkpoint(checkpoint, path):
    # remove untrainable params
    for k in list(checkpoint.keys()):
        if k not in trainable_param_names:
            del checkpoint[k]
    torch.save(checkpoint, path)


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

    if isinstance(data_args.train_data_path, str):
        data_args.train_data_path = os.path.join(DATAFOLDER, data_args.train_data_path)
    if isinstance(data_args.val_data_path, str):
        data_args.val_data_path = os.path.join(DATAFOLDER, data_args.val_data_path)
    if isinstance(data_args.test_data_path, str):
        data_args.test_data_path = os.path.join(DATAFOLDER, data_args.test_data_path)

    batch_size = training_args.real_batch_size
    num_devices = world_size
    gradient_accumulation_steps = max(1, batch_size // (training_args.per_device_train_batch_size * num_devices))
    model = MiniGPT5_Model(encoder_model_config=model_args, device=f"cuda:{local_rank}",
                           device2=f"cuda:{local_rank}", **vars(training_args))
    checkpoint = torch.load("./WEIGHT_FOLDER/stage1_cc3m.ckpt", map_location="cpu")["state_dict"]
    checkpoint["model.llama_model.base_model.model.lm_head.weight"] = checkpoint.pop("model.llama_model.lm_head.weight")
    checkpoint["model.llama_model.base_model.model.model.embed_tokens.weight"] = checkpoint.pop("model.llama_model.model.embed_tokens.weight")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print("load stage1 model success", unexpected_keys)
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    tokenizer = model.tokenizer
    sd_tokenizer = model.sd_tokenizer
    image_loss_only = False
    if training_args.resume is not None:
        ckpt_path = os.path.join("./", training_args.resume)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print("resume from: ", ckpt_path)
        print("unexpected_keys", unexpected_keys)

    output_vis_processor = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    input_vis_processor = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    input_processor = MiniGPT5_InputProcessor(tokenizer=tokenizer, image_processor=input_vis_processor)

    data_collator = DataCollator(tokenizer=tokenizer, sd_tokenizer=sd_tokenizer)
    prompt_template = None
    if model_args.model_type == "llama2":
        prompt_template = "[INST] {} [/INST] "

    train_dataloader, val_dataloader, train_sampler, eval_sampler = make_supervised_data_module(data_args,
                                                                                                training_args,
                                                                                                data_collator,
                                                                                                input_processor,
                                                                                                output_vis_processor,
                                                                                                prompt_template,
                                                                                                model_type=model_args.model_type,)
    train_param_number = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("training parameter:", train_param_number)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.model.to(device)

    device2 = f"cuda:{local_rank}"
    model.device2 = device2
    model.sd_text_encoder.to(device2)
    model.vae.to(device2)
    model.unet.to(device2)
    model.t2i_decoder_prompt.to(device2)
    model.llm_to_t2i_mapping.to(device2)
    model.fc.to(device)
    # model.consistent_encoder.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model_without_ddp = model.module

    optimizer, scheduler = model_without_ddp.configure_optimizers(is_deep_speed=False)
    optimizer = optimizer[0]
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=training_args.num_train_epochs * num_update_steps_per_epoch
    )

    min_loss = 1e9
    val_loss = validate(model, val_dataloader, -1, device, training_args)

    path = training_args.store_path
    assert path is not None, "store path is None"
    if not os.path.exists(path):
        os.makedirs(path)

    for epoch in range(training_args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_dataloader, epoch, device, training_args, optimizer, scheduler)

        val_loss = validate(model, val_dataloader, epoch, device, training_args)
        val_loss = torch.tensor(val_loss).to(device)
        torch.distributed.all_reduce(val_loss)
        latest_save_path = os.path.join(path, "latest_model.ckpt")
        if rank == 0 or rank == -1:
            print("begin save")
            save_checkpoint(model_without_ddp.state_dict(), latest_save_path)
            print("end save")

            if val_loss < min_loss:
                min_loss = val_loss
                print("save best model")
                save_checkpoint(model_without_ddp.state_dict(), os.path.join(path, "best_model.ckpt"))
                print("end save best model")
