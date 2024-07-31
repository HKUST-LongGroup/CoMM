import numpy as np
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import torchvision
lpips = LPIPS(net_type='squeeze')
target_size = (256, 256)

def map_to_lpips_range(image):
    resize = torchvision.transforms.Resize(target_size)
    totentsor = torchvision.transforms.ToTensor()
    image = totentsor(resize(image)).unsqueeze(dim=0)
    image = image/255.0 * 2 -1

    return image


def calculate_lpips(source_image_paths, target_image_paths):
    # Initialize an empty dictionary to store lpips scores
    lpips_scores = {}

    # Load images, resize, and compute lpips for each pair
    total_score = 0
    count = 0
    for i, path1 in enumerate(source_image_paths):
        for j, path2 in enumerate(target_image_paths):
            if path1 != path2:  # Skip comparing an image with itself
                image1 = Image.open(path1).convert('RGB')
                image2 = Image.open(path2).convert('RGB')

                # Resize images to the target size
                image1_resized = map_to_lpips_range(image1)
                image2_resized = map_to_lpips_range(image2)

                score = lpips(image1_resized, image2_resized)
                lpips_scores[(path1, path2)] = score
                total_score += score
                count += 1

    return lpips_scores, total_score / count