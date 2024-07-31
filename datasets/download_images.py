import os
import torch
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list", nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=64)
    return parser.parse_args()


def download_image(url, save_path):
    try:
        if os.path.exists(save_path):
            return
        image_folder = os.path.dirname(save_path)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url} to {save_path}: {e}")


def read_data(data_list):
    data = []
    for data_path in data_list:
        if not os.path.exists(data_path):
            print(f"{data_path} does not exist. Please download the data first and put them to ./datasets")
            exit()
        data.extend(torch.load(data_path))
    return data


def main():
    args = get_args()
    data_list = args.data_list
    save_path = args.save_path
    num_workers = args.num_workers

    data = read_data(data_list)
    image_list = []
    for item in data:
        summary_info = item["summary_info"]
        for content in summary_info:
            if content['type'] == 'image' and content['url'] is not None:
                image_list.append((content['url'], os.path.join(save_path, content['image_path'])))
        step_info = item["step_info"]
        for step in step_info:
            for content in step:
                if content['type'] == 'image' and content['url'] is not None:
                    image_list.append((content['url'], os.path.join(save_path, content['image_path'])))
    
    thread_pool = ThreadPoolExecutor(max_workers=num_workers)
    for cur_idx, (url, save_path) in enumerate(tqdm(image_list)):
        if (cur_idx + 1) % num_workers == 0:
            download_image(url, save_path)
        else:
            thread_pool.submit(download_image, url, save_path)

    thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    main()