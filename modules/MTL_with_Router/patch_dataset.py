# patch_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

PATCH_SIZE = 512
LESION_TYPE_MAP = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
    "5. Optic Disc": "OD"
}

class PatchifiedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        self.patch_data = []  # List of (image_patch, mask_patch)

        print(f"\n🔍 Scanning mask directory: {mask_dir}")

        for folder in os.listdir(mask_dir):
            full_folder_path = os.path.join(mask_dir, folder)
            if not os.path.isdir(full_folder_path):
                continue

            lesion_suffix = LESION_TYPE_MAP.get(folder)
            if not lesion_suffix:
                print(f"⚠️ Skipping unknown folder: {folder}")
                continue

            tif_files = [f for f in os.listdir(full_folder_path) if f.endswith(".tif")]
            for tif_file in tif_files:
                image_id = tif_file.replace(f"_{lesion_suffix}.tif", ".jpg")
                image_path = os.path.join(image_dir, image_id)
                mask_path = os.path.join(full_folder_path, tif_file)

                if not os.path.exists(image_path):
                    continue

                img = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                img = np.array(img)
                mask = np.array(mask)

                for i in range(0, img.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
                    for j in range(0, img.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
                        img_patch = img[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                        mask_patch = mask[i:i + PATCH_SIZE, j:j + PATCH_SIZE]

                        if mask_patch.sum() > 0:  # Keep only non-empty masks
                            self.patch_data.append((img_patch, mask_patch))

        print(f"✅ Total valid patch regions: {len(self.patch_data)}")

    def __len__(self):
        return len(self.patch_data)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patch_data[idx]
        img = Image.fromarray(img_patch)
        mask = Image.fromarray(mask_patch)
        return self.transform(img), self.transform(mask)
