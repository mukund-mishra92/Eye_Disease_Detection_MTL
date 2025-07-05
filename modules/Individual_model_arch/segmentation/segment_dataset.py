# -----------------------------
# Dataset Class
# -----------------------------
import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Lesion suffix mapping
LESION_SUFFIXES = {
    "1. Microaneurysms": "_MA.tif",
    "2. Haemorrhages": "_HE.tif",
    "3. Hard Exudates": "_EX.tif",
    "4. Soft Exudates": "_ES.tif",
    "5. Optic Disc": "_OD.tif"
}

class EyeSegDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Read and normalize image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256)) / 255.0

        mask = []
        for folder, suffix in LESION_SUFFIXES.items():
            gt_path = os.path.join(self.gt_dir, folder, img_name + suffix)
            try:
                if os.path.exists(gt_path):
                    m = imread(gt_path)
                    m = (m > 0).astype(np.uint8)
                    if m.ndim > 2:
                        m = m[..., 0]
                    m = cv2.resize(m, (256, 256)).astype(np.float32)
                else:
                    m = np.zeros((256, 256), dtype=np.float32)

                if m.shape != (256, 256):
                    print(f"Shape issue in {gt_path}, got {m.shape}. Resizing.")
                    m = cv2.resize(m, (256, 256)).astype(np.float32)

            except Exception as e:
                print(f"[ERROR] Failed on mask {gt_path}: {e}")
                m = np.zeros((256, 256), dtype=np.float32)

            mask.append(m)

        try:
            mask = np.stack(mask, axis=0)
        except Exception as e:
            print(f"\n[CRITICAL] Mask stacking failed for {img_name} - Mask Shapes: {[m.shape for m in mask]}")
            raise e

        return torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
