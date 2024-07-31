import os
import re
import json
import torch
import random
random.seed(42)
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .utils import get_each_step_text, convert_data2dict, write2file, \
    parse_data_for_task1, parse_data_for_task3_and_task4, data_format

from tot_metric import calculate_meteor, calculate_bleu, calculate_cider, \
    calculate_IS_and_FID, calculate_ssim, calculate_psnr, gpt4o_eval_task4, \
    calculate_lpips, gpt4o_eval_task1, gpt4o_eval_task2, split_token


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_results_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--task_type', type=str)
    parser.add_argument('--ground_truth_path', type=str)
    return parser.parse_args()


def evaluation_task1(samples, ground_truth_data, task_type, path, model_type, max_img_num=10):
    max_workers = 20
    threadPool = ThreadPoolExecutor(max_workers=max_workers)
    predict_text_steps_list, ground_truth_steps_list, all_predict_text_list, all_ground_truth_text_list = [], [], [], []
    for cur_idx, sample in enumerate(tqdm(samples)):
        if task_type not in sample:
            continue
        data_id = sample[:-6]
        cur_ground_truth_data = ground_truth_data[data_id]
        sample_path = os.path.join(path, sample)
        if not os.path.isdir(sample_path):
            continue
        results_path = os.path.join(sample_path, "results.json")
        with open(results_path, "r") as f:
            datas = json.loads(f.read())

        input_text, output_text = datas["model_input"], datas["model_output"]
        input_image = [os.path.join(sample_path + "/origin", x) for x in datas["input_image_list"]]

        input_step_info, _ = data_format(input_text, input_image, sample_path, 0, max_img_num)
        output_step_info, _ = data_format(output_text, [], sample_path, 0, max_img_num)

        if (cur_idx + 1) % max_workers == 0:
            eval_and_write_task1(input_step_info, output_step_info, sample_path, model_type)
        else:
            threadPool.submit(eval_and_write_task1, input_step_info, output_step_info, sample_path, model_type)
        predict_text_steps, ground_truth_steps, all_predict_text, all_ground_truth_text = parse_data_for_task1(input_text, output_text, cur_ground_truth_data)
        predict_text_steps_list.extend(predict_text_steps)
        ground_truth_steps_list.extend(ground_truth_steps)
        all_predict_text_list.extend(all_predict_text)
        all_ground_truth_text_list.extend(all_ground_truth_text)

    threadPool.shutdown(wait=True)
    scores_dict = calculate_cider(predict_text_steps_list, ground_truth_steps_list)
    print("================= Step-level Evaluation =================")
    print(scores_dict)

    all_text_scores_dict = calculate_cider(all_predict_text_list, all_ground_truth_text_list)
    print("================= Whole Text Evaluation =================")
    print(all_text_scores_dict)


def evaluation_task2(samples, ground_truth_data, task_type, path, model_type, max_img_num=10):
    predict_image_path_list, ground_truth_image_path_list = [], []
    avg_ssim_scores_list, avg_psnr_scores_list, avg_lpips_scores_list = [], [], []
    max_workers = 20
    threadPool = ThreadPoolExecutor(max_workers=max_workers)
    for cur_idx, sample in enumerate(tqdm(samples)):
        if task_type not in sample:
            continue
        sample_path = os.path.join(path, sample)
        if not os.path.isdir(sample_path):
            continue

        results_path = os.path.join(sample_path, "results.json")
        with open(results_path, "r") as f:
            datas = json.loads(f.read())

        input_text, output_text = datas["model_input"], datas["model_output"]
        output_gt_image = [os.path.join(sample_path + "/origin", x) for x in datas["output_gt_image_list"]]
        output_image = [os.path.join(sample_path, x) for x in datas["prediction_image_list"]]

        output_step_info, _ = data_format(output_text, output_image, sample_path, 0, max_img_num)
        input_step_info, _ = data_format(input_text, [], sample_path, 0, max_img_num)
        if (cur_idx + 1) % max_workers == 0:
            eval_and_write_task2(input_step_info, output_step_info, sample_path, model_type)
        else:
            threadPool.submit(eval_and_write_task2, input_step_info, output_step_info, sample_path, model_type)
        predict_is_score, ground_truth_is_score, fid_score = calculate_IS_and_FID(output_gt_image, output_image)
        predict_image_path_list.extend(output_image)
        ground_truth_image_path_list.extend(output_gt_image)

        ssim_scores, avg_ssim_scores = calculate_ssim(output_gt_image, output_image)
        avg_ssim_scores_list.append(avg_ssim_scores)

        psnr_scores, avg_psnr_scores = calculate_psnr(output_gt_image, output_image)
        avg_psnr_scores_list.append(avg_psnr_scores)

        lpips_scores, avg_lpips_scores = calculate_lpips(output_gt_image, output_image)
        avg_lpips_scores_list.append(avg_lpips_scores)
    threadPool.shutdown(wait=True)
    tot_average_ssim = sum(avg_ssim_scores_list) / len(avg_ssim_scores_list)
    print("average ssim: ", tot_average_ssim)
    tot_average_psnr = sum(avg_psnr_scores_list) / len(avg_psnr_scores_list)
    print("average psnr: ", tot_average_psnr)
    tot_average_lpips = sum(avg_lpips_scores_list) / len(avg_lpips_scores_list)
    print("average lpips: ", tot_average_lpips)
    predict_is_score, ground_truth_is_score, fid_score = calculate_IS_and_FID(predict_image_path_list, ground_truth_image_path_list)
    print("predict_is_score: ", predict_is_score)
    print("ground_truth_is_score: ", ground_truth_is_score)
    print("fid_score: ", fid_score)


def evaluation_task3(samples, ground_truth_data, task_type, path, model_type, max_img_num=10):
    max_workers = 20
    threadPool = ThreadPoolExecutor(max_workers=max_workers)
    for cur_idx, sample in enumerate(tqdm(samples)):
        if task_type not in sample:
            continue
        data_id = sample[:-6]
        cur_ground_truth_data = ground_truth_data[data_id]
        sample_path = os.path.join(path, sample)
        if not os.path.isdir(sample_path):
            continue

        results_path = os.path.join(sample_path, "results.json")
        with open(results_path, "r") as f:
            datas = json.loads(f.read())

        output_image = [os.path.join(sample_path, x) for x in datas["prediction_image_list"]]
        input_image = [os.path.join(sample_path + "/origin", x) for x in datas["input_image_list"]]

        input_text, output_text = datas["model_input"], datas["model_output"]
        input_step_info, input_img_idx = data_format(input_text, input_image, sample_path, 0, max_img_num)
        if input_img_idx >= max_img_num:
            continue
        output_step_info, _ = data_format(output_text, output_image, sample_path, input_img_idx, max_img_num)
        if (cur_idx + 1) % max_workers == 0:
            eval_and_write_task3_and_task4(input_step_info, output_step_info, sample_path, model_type)
        else:
            threadPool.submit(eval_and_write_task3_and_task4, input_step_info, output_step_info, sample_path, model_type)


def evaluation_task4(samples, ground_truth_data, task_type, path, model_type, max_img_num=10):
    max_workers = 20
    threadPool = ThreadPoolExecutor(max_workers=max_workers)
    for cur_idx, sample in enumerate(tqdm(samples)):
        if task_type not in sample:
            continue
        data_id = sample[:-6]
        cur_ground_truth_data = ground_truth_data[data_id]
        sample_path = os.path.join(path, sample)
        if not os.path.isdir(sample_path):
            continue

        results_path = os.path.join(sample_path, "results.json")
        with open(results_path, "r") as f:
            datas = json.loads(f.read())
        
        output_image = [os.path.join(sample_path, x) for x in datas["prediction_image_list"]]
        image_idx = 0
        input_text, output_text = datas["model_input"], datas["model_output"]
        step_info = []
        output_steps = get_each_step_text(output_text)
        for step_idx, step in enumerate(output_steps):
            begin_text = f"({step_idx + 1})"
            cur_step_info = [{"type": "text", "content": begin_text}]
            
            results = parse_data_for_task3_and_task4(step)
            for item in results:
                if item["type"] == "text":
                    cur_step_info.append(item)
                elif item["type"] == "image":
                    if image_idx >= len(output_image):
                        break
                    image_path = os.path.join(sample_path, output_image[image_idx])
                    cur_step_info.append({"type": "image", "content": image_path})
                    image_idx += 1
                    if image_idx >= max_img_num:
                        break
                else:
                    raise NotImplementedError
            step_info.append(cur_step_info)
            if image_idx >= max_img_num:
                break
        if (cur_idx + 1) % max_workers == 0:
            eval_and_write_task3_and_task4(input_text, step_info, sample_path, model_type)
        else:
            threadPool.submit(eval_and_write_task3_and_task4, input_text, step_info, sample_path, model_type)
    threadPool.shutdown(wait=True)


def eval_and_write_task1(input_text, output_text, sample_path, model_type):
    max_try = 3
    for _ in range(max_try):
        try:
            sub_path = "/".join(sample_path.split("/")[-3:])
            save_path = os.path.join("./results", model_type, sub_path)
            if os.path.exists(save_path):
                return
            completeness_and_text_fidelity = gpt4o_eval_task1(input_text, output_text)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "completeness_and_text_fidelity.json"), "w") as f:
                json.dump(completeness_and_text_fidelity, f, indent=4)
            break
        except Exception as e:
            print(e)


def eval_and_write_task2(input_text, step_info, sample_path, model_type):
    max_try = 3
    for _ in range(max_try):
        try:
            sub_path = "/".join(sample_path.split("/")[-3:])
            save_path = os.path.join("./results", model_type, sub_path)
            if os.path.exists(save_path):
                return
            evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency = gpt4o_eval_task2(input_text, step_info)
            os.makedirs(save_path, exist_ok=True)
            write2file(save_path, evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency)
            break
        except Exception as e:
            print(e)


def eval_and_write_task3_and_task4(input_text, step_info, sample_path, model_type):
    max_try = 3
    for _ in range(max_try):
        try:
            sub_path = "/".join(sample_path.split("/")[-3:])
            save_path = os.path.join("./results", model_type, sub_path)
            if os.path.exists(save_path):
                return
            evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency = gpt4o_eval_task4(input_text, step_info)
            os.makedirs(save_path, exist_ok=True)
            write2file(save_path, evaluation_results, coherence_and_completeness, texts_to_image_descriptions, descriptions_gt_images_consistency)
            break
        except Exception as e:
            print(e)


def main():
    args = get_args()
    predict_results_path, model_type, task_type = args.predict_results_path, args.model_type, args.task_type
    ground_truth_data = torch.load(args.ground_truth_path)
    ground_truth_data = convert_data2dict(ground_truth_data)

    samples = os.listdir(predict_results_path)
    samples.sort()
    random.shuffle(samples)
    task_func_dict = {
        "task1": evaluation_task1,
        "task2": evaluation_task2,
        "task3": evaluation_task3,
        "task4": evaluation_task4,
    }
    task_func = task_func_dict[task_type]
    task_func(samples, ground_truth_data, task_type, predict_results_path, model_type)


if __name__ == "__main__":
    main()