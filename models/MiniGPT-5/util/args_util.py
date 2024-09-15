from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

from constants import *


def default_gpus():
    return [0, 1, 2, 3]


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="multimodal_encoder") # multimodal_encoder
    snr_loss: Optional[bool] = field(default=True)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    stage1_weight: Optional[str] = field(default=None)
    is_load_2gpu: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    gpu_id: int = field(default=None, metadata={"help": "GPU id."})
    GPU_NUM: int = field(default=None, metadata={"help": "GPU number."})
    datasets_names: List[str] = field(default_factory=list, metadata={"help": "List of dataset names."})


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default=WEIGHTFOLDER)
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    real_batch_size: int = field(default=32)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    warmup_ratio: float = field(default=0.03)
    warmup_steps: int = field(default=1000)
    adam_epsilon: float = field(default=1e-8)
    store_path: str = field(default=None)


    num_workers: int = field(default=16)

    gpus: List[int] = field(default_factory=default_gpus)
    resume: Optional[str] = field(default=None)
    is_training: Optional[bool] = field(default=False)
    test_weight: Optional[str] = field(default=None)
    local_rank: Optional[int] = field(default=-1)
    zero_stage: Optional[int] = field(default=3)
