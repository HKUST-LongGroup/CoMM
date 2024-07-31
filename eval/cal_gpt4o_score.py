import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_results_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None)
    args = parser.parse_args()
    return args


def parse_scores(cur_results):
    lambda_1 = lambda x, y: {y: x["Score"] if "Score" in x else x["overall_score"]}
    lambda_2 = lambda x, y: {
        "style_consistency": x["style_consistency"], 
        "entity_consistency": x["entity_consistency"], 
        "trend_consistency": x["trend_consistency"],
        "overall_consistency": x["overall_consistency"]
    }
    key_map = {
        "completeness": lambda_1,
        "Completeness": lambda_1,
        "Image_Quality": lambda_1,
        "image quality": lambda_1,
        "text fidelity to images": lambda_1,
        "Text Fidelity to Image": lambda_1,
        "coherence_score": lambda_2,
    }
    scores = {}
    for key, value in cur_results.items():
        cur_dict = key_map[key](value, key)
        scores.update(cur_dict)
    return scores


def cal_gpt4o_scores(path, model_type, task_type):
    sub_path = "/".join(path.split("/")[-2:])
    file_path = os.path.join("./results", model_type, sub_path)
    files = os.listdir(file_path)
    score_list_dict = {}
    for file in files:
        if task_type not in file:
            continue
        cur_file_path = os.path.join(file_path, file, "evaluation_results.json")
        if not os.path.exists(cur_file_path):
            cur_file_path = os.path.join(file_path, file, "completeness_and_text_fidelity.json")
        with open(cur_file_path, "r") as f:
            cur_results = json.load(f)
        if type(cur_results) != dict:
            continue
        try:
            cur_scores = parse_scores(cur_results)
        except:
            continue
        for key, value in cur_scores.items():
            if key not in score_list_dict:
                score_list_dict[key] = []
            if value is None or value == "NA":
                value = 0
            score_list_dict[key].append(value)
    final_results = {}
    for key, value in score_list_dict.items():
        print(f"{key}: {sum(value)/len(value)}")
        final_results[key] = sum(value)/len(value)
    return final_results


if __name__ == "__main__":
    args = get_args()
    predict_results_path = args.predict_results_path
    model_type = args.model_type
    task_type = args.task_type
    cal_gpt4o_scores(predict_results_path, model_type, task_type)