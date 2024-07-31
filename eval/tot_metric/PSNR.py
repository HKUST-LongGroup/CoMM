import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io, transform
from PIL import Image


def calculate_psnr(source_image_paths, target_image_paths, target_size=(256, 256)):
    psnr_scores = {}

    total_score = 0
    count = 0
    for i, path1 in enumerate(source_image_paths):
        for j, path2 in enumerate(target_image_paths):
            if path1 != path2:
                image1 = Image.open(path1).convert('RGB')
                image2 = Image.open(path2).convert('RGB')

                image1_resized = np.array(image1.resize(target_size))
                image2_resized = np.array(image2.resize(target_size))

                score = psnr(image2_resized, image1_resized, data_range=255)
                psnr_scores[(path1, path2)] = score
                total_score += score
                count += 1

    return psnr_scores, total_score / count
