# 导入必要的库和模块
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
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
# import multiprocessing

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


@dataclass
class ConfigPathArguments:
    image_processor: Optional[str] = field(default=None, metadata={"help": "config path of image processor"})
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    data: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    data_path: Optional[str] = field(default="../datasets/train_data.pth", metadata={"help": "config path of tokenizer used to initialize tokenizer"})


@dataclass
class ProcessArguments:
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "save dictionary of result which will be written into a sequence of .tar"})
    gpus: Optional[int] = field(default=4, metadata={"help": "number of gpus to be used"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "batch size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "number of workers to load data per device"})


parser = transformers.HfArgumentParser((ConfigPathArguments, ProcessArguments))
cfg_path, args = parser.parse_args_into_dataclasses()


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


def run_worker(gpu):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6668', world_size=args.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    sub_save_dir = os.path.join(args.save_dir, 'part-{:04d}'.format(gpu))
    os.makedirs(sub_save_dir, exist_ok=True)

    save_pattern = sub_save_dir + "/%07d.tar"

    data = torch.load(args.data_path)

    with wds.ShardWriter(save_pattern, maxcount=5000) as sink:
        for item in tqdm(data):
            key_str = uuid.uuid4().hex
            sink.write({'__key__': key_str, 'pkl': pickle.dumps(item)})


if __name__ == '__main__':
    main()


