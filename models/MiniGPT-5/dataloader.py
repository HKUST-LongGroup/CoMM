import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
import transformers
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import re
import os
import numpy as np
from PIL import Image
import random
from pathlib import Path
import json
import copy
from constants import *


class CoMMDataset(Dataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False,
                prompt_template=None, model_type=None, generation=False, is_caption=None, gpu_id=None, GPU_NUM=None, max_len=2048):
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.begin_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        self.load_preprocessed_image_features = True
        self.prompt_template = prompt_template
        self.generation = generation
        if model_type == 'llama2':
            self.end_sys = "</s>"
        else:
            self.end_sys = "###"
        data = torch.load(data_path)

        self.dataset_path = data_path[:data_path.rfind('/')]
        self.sources,  self.output_image_path = [], []
        self.caption, self.task_names, self.steps = [], [], []

        data_len_list = []
        is_store = True
        data_name = data_path.split("/")[-2] + "_" + data_path.split("/")[-1][:-4]
        data_len_path = f"./data_len/{data_name}_list.pth"
        if os.path.exists(data_len_path):
            data_len_list = torch.load(data_len_path)
            is_store = False
        self.data = []

        for cur_data_idx, item in enumerate(tqdm(data)):
            cur_len = -1
            if len(data_len_list) > cur_data_idx:
                cur_len = data_len_list[cur_data_idx]
            step_info = item["step_info"]
            target_answer, image_path_list = "", []
            for cur_step_idx, cur_step in enumerate(step_info):
                target_answer += f"({cur_step_idx + 1}) "
                for sub_content in cur_step:
                    if sub_content["type"] == "text":
                        if cur_len == -1:
                            cur_text = sub_content["content"].strip(" \n")
                            target_answer += cur_text + " "
                    elif sub_content["type"] == "image":
                        target_answer += f"{ALL_IMG_TOKENS_STR} "
                        image_path = None
                        image_path_list.append(image_path)
            if cur_len == -1:
                target_ids = self.input_processor(text=target_answer, add_special_tokens=False)['input_ids']
                cur_len = target_ids.shape[1]
                data_len_list.append(cur_len)

            if len(image_path_list) <= 0 or cur_len >= max_len - 100:
                continue
            self.data.append(item)

        assert len(data) == len(data_len_list)
        if is_store:
            torch.save(data_len_list, data_len_path)
        self.generation_prompts = [
            "Here are step-by-step instructions with images about {SOURCE}: ",
        ]

        if gpu_id is not None:
            assert GPU_NUM is not None
            self.data = self.data[gpu_id::GPU_NUM]

        print("Load data done!")
        self.data_type = [
            self.task1_img_seq2text_seq,
            self.task2_text_seq2img_seq,
            self.task3_interleaved_in_and_out,
            self.task4_interleaved_generation_for_question,
        ]
        if self.test:
            cur_data = copy.deepcopy(self.data)
            self.origin_len = len(cur_data)
            self.data = []
            for _ in self.data_type:
                self.data.extend(cur_data)

    def __len__(self):
        return len(self.data)

    def get_image_path(self, image_item):
        return os.path.join(DATAFOLDER, "images", image_item["image_path"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            input_text, output_text, input_image_item_list, output_image_item_list, rest_step_num, task_type = self.get_data(i)
            if len(input_image_item_list) == 0 or len(output_image_item_list) == 0:
                raise Exception(f"No image len {len(input_image_item_list)} {len(output_image_item_list)}")
            if self.load_preprocessed_image_features:
                dataset_type = self.data[i]["dataset_type"]
                input_images = []
                for input_item in input_image_item_list:
                    in_img_path = self.get_image_path(input_item)
                    if in_img_path is not None:
                        input_image = Image.open(in_img_path).convert("RGB")
                    else:
                        input_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                    input_images.append(input_image)
                if len(input_images) == 0:
                    input_images = None
                input_dict = self.input_processor(text=input_text, images=input_images, add_special_tokens=False)
                input_dict['original_images'] = input_images
                output_image, output_origin_images = [], []
                for output_item in output_image_item_list:
                    out_img_path = self.get_image_path(output_item)
                    cur_output_image = Image.open(out_img_path).convert("RGB")
                    output_origin_images.append(cur_output_image)
                    cur_output_image = self.expand2square(cur_output_image, (255, 255, 255))
                    cur_output_image = self.output_vis_processor(cur_output_image)
                    cur_output_image = cur_output_image.unsqueeze(0)
                    output_image.append(cur_output_image)
                output_image = torch.cat(output_image, dim=0)
                input_dict["output_image"] = output_image
                input_dict["output_origin_images"] = output_origin_images
            target_ids = self.input_processor(text=output_text, add_special_tokens=False)['input_ids']
            label = torch.ones_like(input_dict["input_ids"]) * -100
            label = torch.cat((label, target_ids), dim=1)
            index = torch.nonzero(label == self.begin_img_id)
            if len(index):
                for idx, idy in index:
                    assert idx == 0
                    label[:, idy + 1:idy + IMG_TOKEN_NUM - 1] = -100
            input_dict["task_name"] = "Tot"
            input_dict["labels"] = label
            input_dict["input_ids"] = torch.cat((input_dict["input_ids"], target_ids), dim=1)
            input_dict["attention_mask"] = torch.cat((input_dict["attention_mask"], torch.ones_like(target_ids)), dim=1)
            input_dict["source"] = input_text
            input_dict["target"] = output_text
            input_dict["rest_step_num"] = rest_step_num
            input_dict["output_image_path"] = None
            input_dict["data_output_path"] = self.data[i]["data_id"] + f"_{task_type}"
            return input_dict
        except Exception as e:
            print(f"Error {self.data[i]['data_id']} \n {e}")
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def pre_caption(self, caption):
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")
        return caption

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def check_contain_image(self, idx, prev_input_step_len):
        is_contain_image = False
        for i in range(prev_input_step_len + 1, len(self.steps[idx])):
            if 'image_id' in self.steps[idx][i] and len(self.steps[idx][i]['image_id']) > 0:
                is_contain_image = True
                break
        return is_contain_image

    def get_data(self, idx):
        data_type = self.data_type
        if not self.test:
            return random.choice(data_type)(idx)
        else:
            return data_type[idx // self.origin_len](idx)

    def task1_img_seq2text_seq(self, idx, task_type="task1"):
        prompts = [
            "Here are step-by-step instructions images sequence about {SOURCE}:"
        ]
        prompts_out = [
            "According to the above images, can you generate text for each step?"
        ]
        cur_data = self.data[idx]
        data_id, dataset_type, title, step_info = \
            cur_data["data_id"], cur_data["dataset_type"], cur_data["title"], cur_data["step_info"]
        if title is None:
            title = ""
        input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
        input_image_item_list, rest_step_num = [], 0
        for cur_step_num, cur_step in enumerate(step_info):
            input_text += f"({cur_step_num + 1}) "
            target_answer += f"({cur_step_num + 1}) "
            is_contain_image = False
            for cur_content in cur_step:
                if cur_content["type"] == "text":
                    cur_text = cur_content["content"].strip(" \n")
                    target_answer += self.pre_caption(cur_text) + " "
                elif cur_content["type"] == "image":
                    input_text += f"<Img><ImageHere></Img> "
                    is_contain_image = True
                    input_image_item_list.append(cur_content)
            rest_step_num += 1
            
        input_text += random.choice(prompts_out)
        target_answer = target_answer.strip() + self.end_sys
        # ========== fix unused param error ==========
        target_answer += f"{ALL_IMG_TOKENS_STR} "
        output_image_item_list = [input_image_item_list[-1]]
        return input_text, target_answer, input_image_item_list, output_image_item_list, rest_step_num, task_type

    def task2_text_seq2img_seq(self, idx, task_type="task2"):
        prompts = [
            "Here are step-by-step instructions without images about {SOURCE}:"
        ]
        prompts_out = [
            "According to the above steps, can you generate images for each step?"
        ]
        cur_data = self.data[idx]
        data_id, dataset_type, title, step_info = \
            cur_data["data_id"], cur_data["dataset_type"], cur_data["title"], cur_data["step_info"]
        if title is None:
            title = ""
        input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
        output_image_item_list, rest_step_num = [], 0
        for cur_step_num, cur_step in enumerate(step_info):
            input_text += f"({cur_step_num + 1}) "
            target_answer += f"({cur_step_num + 1}) "
            is_contain_image = False
            for cur_content in cur_step:
                if cur_content["type"] == "text":
                    cur_text = cur_content["content"].strip(" \n")
                    input_text += self.pre_caption(cur_text) + " "
                elif cur_content["type"] == "image":
                    target_answer += f"{ALL_IMG_TOKENS_STR} "
                    is_contain_image = True
                    output_image_item_list.append(cur_content)
            rest_step_num += 1
        input_text += random.choice(prompts_out)
        target_answer = target_answer.strip() + self.end_sys
        # ========== fix unused param error ==========
        input_image_item_list = [output_image_item_list[-1]]
        target_answer += "<Img><ImageHere></Img> "
        return input_text, target_answer, input_image_item_list, output_image_item_list, rest_step_num, task_type

    def task3_interleaved_in_and_out(self, idx, task_type="task3"):
        length = len(self.data[idx]["step_info"])
        if self.test:
            prev_input_step_len = (length // 2) - 1
        else:
            prev_input_step_len = random.randint(1, length - 1) - 1

        prompts = [
            "Here are step-by-step instructions with images about {SOURCE}:"
        ]
        prompts_out = [
            "According to the above steps, can you generate the rest steps?"
        ]
        cur_data = self.data[idx]
        data_id, dataset_type, title, step_info = \
            cur_data["data_id"], cur_data["dataset_type"], cur_data["title"], cur_data["step_info"]
        if title is None:
            title = ""
        summary_info = cur_data["summary_info"]
        input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
        input_image_item_list, output_image_item_list, rest_step_num = [], [], 0
        is_contain_image = False
        for cur_step_num, cur_step in enumerate(step_info):
            if cur_step_num <= prev_input_step_len:
                input_text += f"({cur_step_num + 1}) "
                
                for cur_content in cur_step:
                    if cur_content["type"] == "text":
                        cur_text = cur_content["content"].strip(" \n")
                        input_text += self.pre_caption(cur_text) + " "
                    elif cur_content["type"] == "image":
                        input_text += f"<Img><ImageHere></Img> "
                        input_image_item_list.append(cur_content)
                if cur_step_num == prev_input_step_len:
                    input_text += random.choice(prompts_out)
            else:
                rest_step_num += 1
                target_answer += f"({cur_step_num + 1}) "
                for cur_content in cur_step:
                    if cur_content["type"] == "text":
                        cur_text = cur_content["content"].strip(" \n")
                        target_answer += self.pre_caption(cur_text) + " "
                    elif cur_content["type"] == "image":
                        target_answer += f"{ALL_IMG_TOKENS_STR} "
                        is_contain_image = True
                        output_image_item_list.append(cur_content)
        if not is_contain_image:
            raise ValueError(f"{data_id} does not contain image in prev len {prev_input_step_len}!")
        target_answer = target_answer.strip() + self.end_sys
        return input_text, target_answer, input_image_item_list, output_image_item_list, rest_step_num, task_type

    def task4_interleaved_generation_for_question(self, idx, task_type="task4"):
        prompts = [
            "Can you generate step-by-step instructions with images about {SOURCE}?"
        ]
        cur_data = self.data[idx]
        data_id, dataset_type, title, step_info = \
            cur_data["data_id"], cur_data["dataset_type"], cur_data["title"], cur_data["step_info"]
        if title is None:
            title = ""
        summary_info = cur_data["summary_info"]
        input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
        input_image_item_list, output_image_item_list, rest_step_num = [], [], 0
        is_contain_image = False
        for cur_step_num, cur_step in enumerate(step_info):
            target_answer += f"({cur_step_num + 1}) "
            for cur_content in cur_step:
                if cur_content["type"] == "text":
                    cur_text = cur_content["content"].strip(" \n")
                    target_answer += self.pre_caption(cur_text) + " "
                elif cur_content["type"] == "image":
                    target_answer += f"{ALL_IMG_TOKENS_STR} "
                    is_contain_image = True
                    output_image_item_list.append(cur_content)
        if not is_contain_image:
            raise ValueError(f"{data_id} does not contain image! ")
        target_answer = target_answer.strip() + self.end_sys
        # ========== fix unused param error ==========
        input_image_item_list = [output_image_item_list[-1]]
        target_answer += "<Img><ImageHere></Img> "
        return input_text, target_answer, input_image_item_list, output_image_item_list, rest_step_num, task_type


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sd_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        key_list = instances[0].keys()
        output_dict = {}
        for key in key_list:
            # Need to remove the batch dimension
            if key in ['input_ids', 'attention_mask', 'labels']:
                output_value = [instance[key][0] for instance in instances]
            else:
                output_value = [instance[key] for instance in instances]

            if key == "input_ids":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True,
                                                                padding_value=self.tokenizer.pad_token_id)
            elif key == "labels":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=-100)
            elif key == "attention_mask":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=0)
            elif key == 'input_images':
                output_value = [v.to(PRECISION) for v in output_value]
            elif key == 'output_image':
                output_value = torch.concat(output_value).to(PRECISION)
            elif key == 'output_image_feature':
                output_value = torch.concat(output_value)
            output_dict[key] = output_value
        return output_dict


if 'CoMM' in DATAFOLDER:
    SupervisedDataset = CoMMDataset
else:
    raise ValueError(f"Dataset {DATAFOLDER} not supported!")
