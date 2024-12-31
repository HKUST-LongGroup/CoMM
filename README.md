# CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation

[![arXiv](https://img.shields.io/badge/arXiv-2406.10462-b31b1b.svg)](https://arxiv.org/abs/2406.10462)
[![Static Badge](https://img.shields.io/badge/GoogleDrive-Dataset-blue)](https://drive.google.com/file/d/17AWa3wVCk4OZOdBQzDLRXsvRMnaXXP3T/view?usp=sharing)
[![Static Badge](https://img.shields.io/badge/Huggingface-Dataset-yellow)](https://huggingface.co/datasets/weisuxi/CoMM)

CoMM is a high-quality dataset designed to improve the coherence, consistency, and alignment of multimodal content. It sources raw data from diverse origins, focusing on instructional content and visual storytelling to establish a strong foundation. 
<img src="assets/data_compare.svg" width="800" alt="data comparison">

## 🔔 News 
- **[07/31/2024]** Our dataset and evaluation code are open-sourced!
- **[06/15/2024]** Our paper is released on arXiv: https://arxiv.org/abs/2406.10462.


# Dataset
- Download the dataset from [Google Drive](https://drive.google.com/file/d/17AWa3wVCk4OZOdBQzDLRXsvRMnaXXP3T/view?usp=sharing) or [Huggingface](https://huggingface.co/datasets/weisuxi/CoMM).
- Unzip the downloaded file and put three split data to `./datasets`.
- Use the following command to download the images of the dataset:
```bash scripts/download_images.sh```



# Environment Setup
```
conda create -n comm python=3.8
conda activate comm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Evaluation
The format of the prediction results is shown in [eval/example](eval/example). And we provide the evaluation scripts for the four tasks in the CoMM dataset:
```
cd eval

results_path="/path/to/predict_results"
model_type="your model_name"

# Task1  Image-to-Text Sequence Generation
python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task1 
python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task1 

# Task2  Text-to-Image Sequence Generation
python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task2 
python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task2 

# Task3  Interleaved Image-Text Content Continuation
python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task3
python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task3

# Task4   Question-based Interleaved Image-Text Generation
python -u eval_metric.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task4
python -u cal_gpt4o_score.py --predict_results_path ${results_path} --model_type ${model_type} --task_type task4
```

# Training and Inference
Please refer to [MiniGPT-5](./models/MiniGPT-5/README.md) and [SEED-Llama](./models/SEED/README.md) for the training and inference code.

# TODO
- [ ] Release the training and inference code
  - [ ] Emu2



# Citation
If you find this dataset useful, please cite our paper:
```
@article{chen2024comm,
  title={CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation},
  author={Chen, Wei and Li, Lin and Yang, Yongqi and Wen, Bin and Yang, Fan and Gao, Tingting and Wu, Yu and Chen, Long},
  journal={arXiv preprint arXiv:2406.10462},
  year={2024}
}
```