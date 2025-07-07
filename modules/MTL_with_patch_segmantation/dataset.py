import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np

LESION_FOLDER_MAP = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
    "5. Optic Disc": "OD"
}
LESION_TYPES = list(LESION_FOLDER_MAP.keys())

class HybridMultiTaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_json, patch_size=256, debug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.debug = debug

        with open(label_json, 'r') as f:
            self.labels = json.load(f)

        self.image_filenames = sorted([
            f for f in os.listdir(image_dir) if f.lower().endswith(".jpg") and f in self.labels
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        self.patch_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        full_path = os.path.join(self.image_dir, fname)

        image = Image.open(full_path).convert("RGB")
        full_image = self.img_transform(image)
        label = torch.tensor(self.labels.get(fname, 0))

        patches = []
        mask_patches = []

        for i, lesion in enumerate(LESION_TYPES):
            lesion_code = LESION_FOLDER_MAP[lesion]
            lesion_path = os.path.join(self.mask_dir, lesion, fname.replace(".jpg", f"_{lesion_code}.tif"))

            if os.path.exists(lesion_path):
                mask = Image.open(lesion_path).convert("L")
                mask_np = np.array(mask)

                for y in range(0, mask_np.shape[0], self.patch_size):
                    for x in range(0, mask_np.shape[1], self.patch_size):
                        patch = mask_np[y:y+self.patch_size, x:x+self.patch_size]
                        if patch.sum() > 0:
                            patch_img = image.crop((x, y, x+self.patch_size, y+self.patch_size))
                            patch_tensor = self.patch_transform(patch_img)

                            patch_mask = Image.fromarray(patch)
                            patch_mask_tensor = self.mask_transform(patch_mask)[0]
                            patch_mask_tensor = (patch_mask_tensor > 0.1).float()

                            patches.append(patch_tensor)
                            mask_patches.append(patch_mask_tensor.unsqueeze(0))  # (1, H, W)

        if len(patches) == 0:
            patches = [torch.zeros(3, self.patch_size, self.patch_size)]
            mask_patches = [torch.zeros(1, self.patch_size, self.patch_size)]

        patches = torch.stack(patches)
        masks = torch.stack(mask_patches)

        return full_image, label, patches, masks
