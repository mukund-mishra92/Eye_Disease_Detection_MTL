import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, label_json=None, use_segmentation_only=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_segmentation_only = use_segmentation_only

        # Load labels from JSON
        self.labels = {}
        if label_json and os.path.exists(label_json):
            with open(label_json, "r") as f:
                self.labels = json.load(f)

        # List of image filenames that are in the directory
        self.image_filenames = sorted([
            fname for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        # Keep only files that have labels if classification is required
        if not use_segmentation_only:
            self.image_filenames = [f for f in self.image_filenames if f in self.labels]

        # Define image and mask transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Make sure it matches model output
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        label = torch.tensor(self.labels.get(img_name, 0))  # Default to class 0 if not found

        # Handle segmentation mask
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.tif'))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, 128, 128))  # fallback dummy mask
        else:
            mask = torch.zeros((1, 128, 128))  # no mask available

        if self.use_segmentation_only:
            return image, torch.tensor(0), mask  # dummy label
        elif self.mask_dir:
            return image, label, mask
        else:
            return image, label, torch.zeros((1, 128, 128))  # dummy mask


class EyeMTLDataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels

        self.transform_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((128, 128)),  # Match model output
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        label = self.labels[idx]

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        return img, mask, label
