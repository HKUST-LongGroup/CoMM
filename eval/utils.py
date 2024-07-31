import re
import os
import json


def get_each_step_text(text):
    pattern = re.compile(r'\(\d+\)\s(.*?)\s*(?=\(\d+\)|$)', re.DOTALL)
    steps = pattern.findall(text)
    return steps


def convert_data2dict(ground_truth_data):
    data_dict = {}
    for data in ground_truth_data:
        data_id = data["data_id"]
        data_dict[data_id] = data
    return data_dict


def get_steps_text(step_info):
    step_text_list = []
    for step in step_info:
        step_text = ""
        for item in step:
            if item["type"] == "text":
                step_text += item["content"]
        step_text_list.append(step_text)
    return step_text_list


def write2file(path, evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency):
    with open(os.path.join(path, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
    with open(os.path.join(path, "coherence_and_completeness.json"), "w") as f:
        json.dump(coherence_and_completeness, f, indent=4)
    with open(os.path.join(path, "texts_to_image_descriptions.json"), "w") as f:
        json.dump(texts_to_image_descriptions, f, indent=4)
    with open(os.path.join(path, "descriptions_gt_images_consistency.json"), "w") as f:
        json.dump(descriptions_gt_images_consistency, f, indent=4)


def parse_data_for_task1(input_text, output_text, cur_ground_truth_data):
    step_info = cur_ground_truth_data["step_info"]
    predict_text_steps = get_each_step_text(output_text)
    ground_truth_steps = get_steps_text(step_info)
    if len(predict_text_steps) != len(step_info):
        min_len = min(len(predict_text_steps), len(step_info))
        predict_text_steps = predict_text_steps[:min_len]
        ground_truth_steps = ground_truth_steps[:min_len]
    assert len(predict_text_steps) == len(ground_truth_steps)
    all_predict_text, all_ground_truth_text = " ".join(predict_text_steps), " ".join(ground_truth_steps)
    return predict_text_steps, ground_truth_steps, [all_predict_text], [all_ground_truth_text]


def parse_data_for_task3_and_task4(input_string):
    pattern = re.compile(r'(\[img\](.*?)\[/img\]|\[imgblock\](.*?)\[/imgblock\]|<IMAGE>|[^\[\]<]+)')
    
    matches = pattern.findall(input_string)
    result = []
    
    for match in matches:
        if match[0].startswith('/IMGB') or match[0].startswith('<>'):
            result.append({'type': 'image', 'content': match[0]})
        elif len(match[0].strip()) <= 0 or match[0].startswith('IMG1') or match[0].startswith('IMG0') \
            or  match[0].startswith('IMGB') or match[0].startswith('[imgblock]'):
            continue
        elif match[0] == '<IMAGE>':
            result.append({'type': 'image', 'content': match[0]})
        else:
            result.append({'type': 'text', 'content': match[0].strip()})
    
    return result


def data_format(input_text, input_image, sample_path, previous_img_num=0, max_img_num=10):
    input_step_info = []
    input_steps = get_each_step_text(input_text)
    input_img_idx = 0
    for step_idx, step in enumerate(input_steps):
        begin_text = f"({step_idx + 1})"
        cur_step_info = [{"type": "text", "content": begin_text}]
        results = parse_data_for_task3_and_task4(step)
        for item in results:
            if item["type"] == "text":
                cur_step_info.append(item)
            elif item["type"] == "image":
                if input_img_idx >= len(input_image):
                    break
                image_path = os.path.join(sample_path, input_image[input_img_idx])
                cur_step_info.append({"type": "image", "content": image_path})
                input_img_idx += 1
                if input_img_idx + previous_img_num >= max_img_num:
                    break
            else:
                raise NotImplementedError
        input_step_info.append(cur_step_info)
        if input_img_idx + previous_img_num >= max_img_num:
            break
    return input_step_info, input_img_idx


