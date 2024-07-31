import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore as IS
from PIL import Image
from tqdm import tqdm
batchsize = 1

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, image_path_list, transform=None):
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def calculate_IS_and_FID(predict_image_path_list, ground_truth_path_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_is_metric = IS(normalize=True).to(device)
    ground_truth_is_metric = IS(normalize=True).to(device)
    fid_metric = FID(normalize=True).to(device)

    # Load the datasets
    predict_dataset = ImageFolder(predict_image_path_list, transform=transform)
    ground_truth_dataset = ImageFolder(ground_truth_path_list, transform=transform)

    # Create data loaders
    predict_dataloader = torch.utils.data.DataLoader(predict_dataset, batch_size=batchsize)
    ground_truth_dataloader = torch.utils.data.DataLoader(ground_truth_dataset, batch_size=batchsize)

    # Calculate IS and FID for the real images
    for images in tqdm(predict_dataloader):
        images = images.to(device)
        fid_metric.update(images, real=True)
        predict_is_metric.update(images)

    for images in tqdm(ground_truth_dataloader):
        images = images.to(device)
        fid_metric.update(images, real=False)
        ground_truth_is_metric.update(images)

    # Compute the final scores
    predict_is_score = predict_is_metric.compute()[0].cpu().item()
    ground_truth_is_score = ground_truth_is_metric.compute()[0].cpu().item()
    fid_score = fid_metric.compute().cpu().item()

    return predict_is_score, ground_truth_is_score, fid_score

