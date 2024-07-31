import os
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def convert_json_to_coco_annotation_format(ground_truth_steps):
    annotations = []
    images = []
    for cur_idx, text in enumerate(ground_truth_steps):
        temp_image = {
            "license": 3,
            "url": "None",
            "file_name": "None",
            "id": cur_idx,
            "width": 640,
            "date_captured": "None",
            "height": 360
        }
        temp_image["id"] = cur_idx

        temp = {}
        temp["image_id"] = cur_idx
        temp["caption"] = text
        temp["id"] = cur_idx

        annotations.append(temp)
        images.append(temp_image)
    original_data = load_json("./tot_metric/captions_val2014.json")
    original_data["annotations"] = annotations
    original_data["images"] = images
    return original_data


def convert_json_to_coco_result_format(predict_text_steps):
    results = []
    for cur_idx, text in enumerate(predict_text_steps):
        temp = {}
        temp["image_id"] = cur_idx
        temp["caption"] = text
        results.append(temp)
    return results

def calculate_cider(predict_text_steps, ground_truth_steps):
    tmp_predict_json_path = "./tmp_predict.json"
    tmp_ground_truth_json_path = "./tmp_ground_truth.json"
    predict_json = convert_json_to_coco_result_format(predict_text_steps)
    ground_truth_json = convert_json_to_coco_annotation_format(ground_truth_steps)
    write_json(predict_json, tmp_predict_json_path)
    write_json(ground_truth_json, tmp_ground_truth_json_path)
    coco = COCO(tmp_ground_truth_json_path)
    coco_result = coco.loadRes(tmp_predict_json_path)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    os.remove(tmp_predict_json_path)
    os.remove(tmp_ground_truth_json_path)

    return coco_eval.eval