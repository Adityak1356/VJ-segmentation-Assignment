import os
from torch.utils.data import Dataset
import torchvision.transforms as TF
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.image_transform = TF.Compose([
            TF.Resize((128, 128)),
            TF.ToTensor()
        ])
        self.mask_transform = TF.Compose([
            TF.Resize((128, 128)),
            TF.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask