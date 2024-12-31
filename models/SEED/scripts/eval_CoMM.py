import hydra
import random
random.seed(42)
import pyrootutils
import os
import re
import copy
import torch
from tqdm import tqdm

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

# s_token = "[INST] "
# e_token = " [/INST]"
sep = "\n"
s_token = "USER:"
e_token = "ASSISTANT:"

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000


def arg_parse():
    parser = argparse.ArgumentParser(description='Seed Llama')
    parser.add_argument('--config', type=str, default='configs/llm/seed_llama14b_CoMM.yaml')
    parser.add_argument('--tokenizer', type=str, default='configs/tokenizer/seed_llama_tokenizer_hf.yaml')
    parser.add_argument('--transform', type=str, default='configs/transform/clip_transform.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='./datasets/test_data.pth')
    parser.add_argument('--image_path', type=str, default='./datasets/images')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--tot_gpu', type=int, default=8)
    parser.add_argument('--cur_gpu_id', type=int, default=-1)
    return parser.parse_args()


class EvalDataset(Dataset):
    def __init__(self, data_path, tot_gpu, cur_gpu_id, tokenizer, max_len=2048, image_path="./datasets/images"):
        data_name = data_path.split("/")[-2] + "_" + data_path.split("/")[-1][:-4]
        data = torch.load(data_path)
        is_store, data_len = True, []
        data_len_path = f"./data_len/{data_name}_list.pth"
        if os.path.exists(data_len_path):
            data_len = torch.load(data_len_path)
            is_store = False
            assert len(data) == len(data_len)
        self.tokenizer = tokenizer
        self.image_path = image_path
        self.image_ids_path = image_path.replace("images", "image_ids")

        self.data = []
        for cur_data_idx, item in enumerate(tqdm(data)):
            step_info = item["step_info"]
            image_path_list = []
            tot_text = ""
            for cur_step_idx, cur_step in enumerate(step_info):
                for sub_content in cur_step:
                    if sub_content["type"] == "image":
                        image_path = None
                        image_path_list.append(image_path)
                    elif sub_content["type"] == "text":
                        tot_text += sub_content["content"]
            if len(data_len) > cur_data_idx:
                cur_len = data_len[cur_data_idx]
            else:
                cur_len = len(tokenizer.encode(tot_text))
            if len(image_path_list) <= 0 or cur_len >= max_len - 100:
                continue
            self.data.append(item)
        assert len(data) == len(data_len)
        if is_store:
            torch.save(data_len, data_len_path)
        self.data = self.data[cur_gpu_id::tot_gpu]

        self.data_type = [
                            "qa",
                            "text_only",
                            "img_gen", 
                            "context", 
                        ]

        self.task_type_dict = {
            "text_only": "task1",
            "img_gen": "task2",
            "context": "task3",
            "qa": "task4",
        }

        cur_data = copy.deepcopy(self.data)
        random.shuffle(cur_data)
        self.origin_len = len(cur_data)
        self.data = []
        for _ in self.data_type:
            self.data.extend(cur_data)
    

    def __len__(self):
        return len(self.data)


    def load_image(self, image_list):
        PIL_image_list = []
        for image_path in image_list:
            PIL_image = Image.open(image_path).convert('RGB')
            PIL_image_list.append(PIL_image)
        return PIL_image_list


    def __getitem__(self, idx, max_length=2048):
        def get_image_id(image_file):
            image_id = image_file.split(".")[0]
            cur_image_path = os.path.join(self.image_path, image_file)
            cur_image_id_path = os.path.join(self.image_ids_path, f"{image_id}.pth")
            return torch.load(cur_image_id_path), cur_image_path
        tokenizer = self.tokenizer

        try:
            task_type_list = self.data_type
            cur_task_type = task_type_list[idx // self.origin_len]
            sample = self.data[idx]
            cur_data = sample
            data_id, dataset_type, title, step_info = \
                    cur_data["data_id"], cur_data["dataset_type"], cur_data["title"], cur_data["step_info"]
            input_ids = []
            labels = []
            if title is None:
                title = ""
            input_image_list, output_image_list = [], []
            system_message = ''
            if cur_task_type == "qa":
                prompts = [
                    "Can you generate step-by-step instructions with images about {SOURCE}?"
                ]
                input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
                is_contain_image = False
                for cur_step_num, cur_step in enumerate(step_info):
                    target_answer += f"({cur_step_num + 1}) "
                    for cur_content in cur_step:
                        if cur_content["type"] == "text":
                            cur_text = cur_content["content"].strip(" \n")
                            target_answer += cur_text + " "
                        elif cur_content["type"] == "image":
                            try:
                                image_id_list, image_path = get_image_id(cur_content["image_path"])
                            except Exception as e:
                                raise ValueError(f"Error: {e} {data_id}")
                            output_image_list.append(image_path)
                            image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id_list]) + EOI_TOKEN
                            target_answer += image_tokens
                            is_contain_image = True
                if not is_contain_image:
                    raise ValueError("No image in the data")
                question = s_token + " " + input_text + e_token
                answer = target_answer
            elif cur_task_type == "context":
                length = len(step_info)
                prev_input_step_len = (length // 2) - 1

                prompts = [
                    "Here are step-by-step instructions with images about {SOURCE}:"
                ]
                prompts_out = [
                    "According to the above steps, can you generate the rest steps?"
                ]
                input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""

                for cur_step_num, cur_step in enumerate(step_info):
                    if cur_step_num <= prev_input_step_len:
                        input_text += f"({cur_step_num + 1}) "
                        
                        for cur_content in cur_step:
                            if cur_content["type"] == "text":
                                cur_text = cur_content["content"].strip(" \n")
                                input_text += cur_text + " "
                            elif cur_content["type"] == "image":
                                try:
                                    image_id_list, image_path = get_image_id(cur_content["image_path"])
                                except Exception as e:
                                    raise ValueError(f"Error: {e} {data_id}")
                                input_image_list.append(image_path)
                                image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id_list]) + EOI_TOKEN
                                input_text += image_tokens
                        if cur_step_num == prev_input_step_len:
                            input_text += random.choice(prompts_out)
                    else:
                        target_answer += f"({cur_step_num + 1}) "
                        for cur_content in cur_step:
                            if cur_content["type"] == "text":
                                cur_text = cur_content["content"].strip(" \n")
                                target_answer += cur_text + " "
                            elif cur_content["type"] == "image":
                                try:
                                    image_id_list, image_path = get_image_id(cur_content["image_path"])
                                except Exception as e:
                                    raise ValueError(f"Error: {e} {data_id}")
                                output_image_list.append(image_path)
                                image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id_list]) + EOI_TOKEN
                                target_answer += image_tokens
                question = s_token + " " + input_text + e_token
                answer = target_answer
            elif cur_task_type == "img_gen":
                prompts = [
                    "Here are step-by-step instructions without images about {SOURCE}:"
                ]
                prompts_out = [
                    "According to the above steps, can you generate images for each step?"
                ]
                input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
                
                for cur_step_num, cur_step in enumerate(step_info):
                    input_text += f"({cur_step_num + 1}) "
                    target_answer += f"({cur_step_num + 1}) "
                    for cur_content in cur_step:
                        if cur_content["type"] == "text":
                            cur_text = cur_content["content"].strip(" \n")
                            input_text += cur_text + " "
                        elif cur_content["type"] == "image":
                            try:
                                image_id_list, image_path = get_image_id(cur_content["image_path"])
                            except Exception as e:
                                raise ValueError(f"Error: {e} {data_id}")
                            output_image_list.append(image_path)
                            image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id_list]) + EOI_TOKEN
                            target_answer += image_tokens
                question = s_token + " " + input_text + random.choice(prompts_out) + e_token
                answer = target_answer
            elif cur_task_type == "text_only":
                prompts = [
                    "Here are step-by-step instructions images sequence about {SOURCE}:"
                ]
                prompts_out = [
                    "According to the above images, can you generate text for each step?"
                ]

                input_text, target_answer = random.choice(prompts).replace("{SOURCE}", title), ""
                for cur_step_num, cur_step in enumerate(step_info):
                    input_text += f"({cur_step_num + 1}) "
                    target_answer += f"({cur_step_num + 1}) "
                    for cur_content in cur_step:
                        if cur_content["type"] == "text":
                            cur_text = cur_content["content"].strip(" \n")
                            target_answer += cur_text + " "
                        elif cur_content["type"] == "image":
                            try:
                                image_id_list, image_path = get_image_id(cur_content["image_path"])
                            except Exception as e:
                                raise ValueError(f"Error: {e} {data_id}")
                            input_image_list.append(image_path)
                            image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_id_list]) + EOI_TOKEN
                            input_text += image_tokens
                question = s_token + " " + input_text + random.choice(prompts_out) + e_token
                answer = target_answer
            else:
                raise ValueError("Error task type")
            
            q_ids = tokenizer.encode(question, add_special_tokens=False)
            a_ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(a_ids) <= 0 or len(q_ids) >= max_length - 64:
                raise ValueError("Error length")
            labels_item = [-100] * len(q_ids) + a_ids
            input_ids_item = q_ids
            input_ids.extend(input_ids_item)
            labels.extend(labels_item)
            
            input_ids = [tokenizer.bos_token_id] + input_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] + labels + [tokenizer.eos_token_id]
            
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            task_type = self.task_type_dict[cur_task_type]
            return input_ids, data_id + f"_{task_type}", input_image_list, output_image_list
        except Exception as e:
            print(f"Error: {e} {data_id}")
            return self.__getitem__(random.randint(0, len(self.data) - 1))


def generate(tokenizer, input_ids, generation_config, model, device, max_new_tokens):
    input_ids = input_ids.to(device)
    generation_config["max_new_tokens"] = max_new_tokens - input_ids.shape[1] - 32
    generate_ids = model.generate(
        input_ids=input_ids,
        **generation_config
    )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    return generate_ids


def replace_img_tags(input_text):
    img_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    img_matches = img_pattern.findall(input_text)
    
    for i, match in enumerate(img_matches):
        replacement = f'<IMAGE>'
        input_text = input_text.replace(f'<img>{match}</img>', replacement)
    
    return input_text


def save_image(store_path, image_list, previous_name, image_num=0):
    image_name_list = []
    for idx, image in enumerate(image_list):
        cur_image_num = image_num + idx + 1
        file_save_path = os.path.join(store_path, f"{previous_name}_{cur_image_num}.jpg")
        os.system(f"cp {image[0]} {file_save_path}")
        image_name_list.append(f"{previous_name}_{cur_image_num}.jpg")
    return image_name_list, image_num + len(image_list)


def decode_image_text(generate_ids, tokenizer, data_id, save_path, input_ids, input_image_list, output_image_list):
    cur_save_path = os.path.join(save_path, data_id)
    if os.path.exists(cur_save_path):
        return
    os.makedirs(cur_save_path, exist_ok=True)
    origin_file_path = os.path.join(cur_save_path, "origin")
    os.makedirs(origin_file_path, exist_ok=True)
    input_image_list, image_num = save_image(origin_file_path, input_image_list, "input")
    output_gt_image_list, _ = save_image(origin_file_path, output_image_list, "output", image_num)
    boi_list = torch.where(generate_ids == tokenizer(
        BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(
        EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    input_text = replace_img_tags(input_text)
    output_text = tokenizer.decode(generate_ids, skip_special_tokens=False)
    output_text = replace_img_tags(output_text)
    output_dict = {
        "model_input": input_text,
        "model_output": output_text,
        "input_image_list": input_image_list,
        "output_gt_image_list": output_gt_image_list
    }
    image_list, prediction_image_list = [], []
    for img_idx in range(len(eoi_list)):
        try:
            boi_index = boi_list[img_idx]
            eoi_index = eoi_list[img_idx]
            image_ids = (generate_ids[boi_index+1:eoi_index] -
                    image_id_shift).reshape(1, -1)
            images = tokenizer.decode_image(image_ids)
            image_list.append(images[0])
        except:
            continue
    for idx, image in enumerate(image_list):
        cur_image_num = image_num + idx + 1
        image.save(os.path.join(cur_save_path, f"predict_{cur_image_num}.jpg"))
        prediction_image_list.append("predict_" + str(cur_image_num) + ".jpg")
    output_dict["prediction_image_list"] = prediction_image_list
    with open(os.path.join(save_path, "results.json"), "w") as f:
        json.dump(output_dict, f, indent=4)


def main(args):
    device = f"cuda:{args.cur_gpu_id}"

    tokenizer_cfg_path = args.tokenizer
    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(
        tokenizer_cfg, device=device, load_diffusion=True)

    transform_cfg_path = args.transform
    transform_cfg = OmegaConf.load(transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    data_path = args.data_path
    dataset = EvalDataset(data_path, args.tot_gpu, args.cur_gpu_id, tokenizer, args.image_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    model_cfg = OmegaConf.load(args.config)
    model = hydra.utils.instantiate(model_cfg, torch_dtype=torch.float16)
    model = model.eval().to(device)

    generation_config = {
        'temperature': args.temperature,
        'num_beams': args.num_beams,
        'max_new_tokens': args.max_new_tokens,
        'top_p': args.top_p,
        'do_sample': args.do_sample
    }

    for input_ids, data_id, input_image_list, output_image_list in tqdm(dataloader):
        data_id = data_id[0]
        cur_save_path = os.path.join(args.save_path, data_id)
        if os.path.exists(cur_save_path):
            continue
        generate_ids = generate(tokenizer, input_ids, generation_config, model, device, args.max_new_tokens)
        decode_image_text(generate_ids, tokenizer, data_id, args.save_path, input_ids, input_image_list, output_image_list)


if __name__ == "__main__":
    _args = arg_parse()
    main(_args)

