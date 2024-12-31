import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import hydra
from omegaconf import OmegaConf
import os
import pickle
from typing import Optional
import transformers
from dataclasses import dataclass, field

from torch.multiprocessing import Process, set_start_method, Lock, Pool
import torch.multiprocessing as mp
import pyrootutils
from tqdm import tqdm
import uuid
import json
import time

import webdataset as wds
from concurrent.futures import ThreadPoolExecutor
# import multiprocessing

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


@dataclass
class ConfigPathArguments:
    image_processor: Optional[str] = field(default=None, metadata={"help": "config path of image processor"})
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    data: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})


@dataclass
class ProcessArguments:
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "save dictionary of result which will be written into a sequence of .tar"})
    data_path: Optional[str] = field(default="../datasets", metadata={"help": "data path"})
    data_list: Optional[str] = field(default="val_data.pth", metadata={"help": "data list"})
    gpus: Optional[int] = field(default=4, metadata={"help": "number of gpus to be used"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "batch size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "number of workers to load data per device"})


parser = transformers.HfArgumentParser((ConfigPathArguments, ProcessArguments))
cfg_path, args = parser.parse_args_into_dataclasses()


class ImageDataset(Dataset):
    def __init__(self, data_path, data_list, tokenizer, gpu_num, image_processor=None, image_transform=None):
        data_list = data_list.split(",")
        tot_data = []
        for cur_data_list in data_list:
            cur_data = torch.load(os.path.join(data_path, cur_data_list))
            tot_data.extend(cur_data)
        tot_image_path = self.get_image_path(tot_data)
        self.images = tot_image_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_transform = image_transform
        begin_idx, end_idx = len(self.images) * gpu_num // args.gpus, len(self.images) * (gpu_num + 1) // args.gpus
        if gpu_num == args.gpus - 1:
            end_idx = len(self.images)
        self.images = self.images[begin_idx:end_idx]
        self.data_path = data_path
        self.bak_image_path = None

    def get_image_path(self, tot_data):
        image_list = []
        for item in tqdm(tot_data):
            step_info = item["step_info"]
            dataset_type = item["dataset_type"]
            for step in step_info:
                for cur_item in step:
                    if cur_item["type"] == "image":
                        cur_image_path = os.path.join(self.data_path, "images", cur_item["image_path"])
                        cur_image_id_path = os.path.join(self.data_path, "image_ids", cur_item["image_id_path"].split(".")[0] + ".pth")
                        image_list.append((cur_image_path, cur_image_id_path))
        return image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        cur_image_path, cur_image_id_path = self.images[idx]
        try:
            image = os.path.join(cur_image_path)
            image = Image.open(image).convert('RGB')
            if self.bak_image_path is not None:
                self.bak_image_path = cur_image_path
        except Exception as e:
            image = Image.open(self.bak_image_path).convert('RGB')
            cur_image_id_path = "ERROR"
        if self.image_processor is not None:
            image = self.image_processor(image)
        elif self.image_transform is not None:
            image = self.image_transform(image)
        return image, cur_image_id_path


def main():
    print(cfg_path, args)
    os.makedirs(args.save_dir, exist_ok=True)

    children = []
    for i in range(args.gpus):
        subproc = mp.Process(target=run_worker, args=(i, ))
        children.append(subproc)
        subproc.start()

    for i in range(args.gpus):
        children[i].join()


def save(cur_image_id, cur_image_store_path):
    torch.save(cur_image_id, cur_image_store_path)


def run_worker(gpu):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6668', world_size=args.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    if cfg_path.image_processor is not None:
        processor_cfg = OmegaConf.load(cfg_path.image_processor)
        processor = hydra.utils.instantiate(processor_cfg)
    else:
        processor = None

    if cfg_path.image_transform is not None:
        transform_cfg = OmegaConf.load(cfg_path.image_transform)
        transform = hydra.utils.instantiate(transform_cfg)
    else:
        transform = None

    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device='cuda')
    tokenizer.pad_token = tokenizer.unk_token

    dataset = ImageDataset(args.data_path, args.data_list, tokenizer, gpu, image_processor=processor, image_transform=transform)

    print('Init Done')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    threadPool = ThreadPoolExecutor(max_workers=16)

    with torch.no_grad():
        time1 = time.time()
        for batch in tqdm(data_loader):
            time2 = time.time()
            if gpu == 0:
                print('time: ', time2 - time1)
            time1 = time2
            image_tensor = batch[0].cuda()
            image_store_path = batch[1]
            image_ids = tokenizer.encode_image(image_torch=image_tensor)
            for idx, cur_image_store_path in enumerate(image_store_path):
                if cur_image_store_path == "ERROR":
                    continue
                cur_image_id = image_ids[idx]
                cur_image_id = cur_image_id.view(-1).cpu().tolist()
                # save(cur_image_id, cur_image_store_path)
                threadPool.submit(save, cur_image_id, cur_image_store_path)
    threadPool.shutdown(wait=True)
                

if __name__ == '__main__':
    main()

