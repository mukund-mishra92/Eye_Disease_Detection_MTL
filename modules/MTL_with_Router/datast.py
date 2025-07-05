import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

LESION_TYPES = ["EX", "HE", "MA", "SE", "OD"]

class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, label_json=None, use_segmentation_only=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_segmentation_only = use_segmentation_only

        # Load classification labels from JSON
        self.labels = {}
        if label_json and os.path.exists(label_json):
            with open(label_json, "r") as f:
                self.labels = json.load(f)

        # List of image filenames
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not use_segmentation_only:
            self.image_filenames = [f for f in self.image_filenames if f in self.labels]

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # Classification label
        label = torch.tensor(self.labels.get(fname, 0))

        # Multi-channel segmentation mask (if available)
        if self.mask_dir:
            mask_tensor = torch.zeros((len(LESION_TYPES), 128, 128))
            for i, lesion in enumerate(LESION_TYPES):
                lesion_folder = os.path.join(self.mask_dir, lesion)
                mask_path = os.path.join(lesion_folder, fname.replace('.jpg', f"_{lesion}.tif"))
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask_tensor[i] = self.mask_transform(mask)[0]
        else:
            mask_tensor = torch.zeros((len(LESION_TYPES), 128, 128))

        # If segmentation only, we return dummy label
        if self.use_segmentation_only:
            return image, torch.tensor(0), torch.zeros(1), mask_tensor, label  # force_class=label
        else:
            return image, label, torch.zeros(1), mask_tensor, torch.tensor(0)  # dummy mask class

