import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io, transform
from PIL import Image


def calculate_ssim(source_image_paths, target_image_paths, target_size = (256, 256)):
    # Initialize an empty dictionary to store SSIM scores
    ssim_scores = {}

    # Load images, resize, and compute SSIM for each pair
    total_score = 0
    count = 0
    for i, path1 in enumerate(source_image_paths):
        for j, path2 in enumerate(target_image_paths):
            if path1 != path2:  # Skip comparing an image with itself
                image1 = Image.open(path1).convert('RGB')
                image2 = Image.open(path2).convert('RGB')

                # Resize images to the target size
                image1_resized = np.array(image1.resize(target_size))
                image2_resized = np.array(image2.resize(target_size))

                score = ssim(image1_resized, image2_resized, data_range=255, channel_axis=2)
                ssim_scores[(path1, path2)] = score
                total_score += score
                count += 1

    return ssim_scores, total_score / count