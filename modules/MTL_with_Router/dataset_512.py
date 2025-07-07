import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Mapping lesion folders to code
LESION_FOLDER_MAP = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages": "HE",
    "3. Hard Exudates": "EX",
    "4. Soft Exudates": "SE",
    "5. Optic Disc": "OD"
}

LESION_TYPES = list(LESION_FOLDER_MAP.keys())  # Folder names


class MultiTaskDataset(Dataset):
    def __init__(
        self, image_dir, mask_dir=None, label_json=None,
        use_segmentation_only=False, filter_empty_masks=True, debug=False
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_segmentation_only = use_segmentation_only
        self.filter_empty_masks = filter_empty_masks
        self.debug = debug

        # Load classification labels
        self.labels = {}
        if label_json and os.path.exists(label_json):
            with open(label_json, "r") as f:
                self.labels = json.load(f)

        # Get image filenames
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Filter images without classification labels
        if not use_segmentation_only:
            self.image_filenames = [f for f in self.image_filenames if f in self.labels]

        # Filter out images without any mask
        if self.mask_dir and filter_empty_masks:
            filtered = []
            for fname in self.image_filenames:
                has_mask = any(os.path.exists(
                    os.path.join(self.mask_dir, lesion, fname.replace(".jpg", f"_{LESION_FOLDER_MAP[lesion]}.tif"))
                ) for lesion in LESION_TYPES)
                if has_mask:
                    filtered.append(fname)
            self.image_filenames = filtered

        # Define transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
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

        # Multi-channel segmentation mask
        mask_tensor = torch.zeros((len(LESION_TYPES), 512, 512))
        if self.mask_dir:
            for i, lesion in enumerate(LESION_TYPES):
                lesion_code = LESION_FOLDER_MAP[lesion]
                lesion_mask_name = fname.replace(".jpg", f"_{lesion_code}.tif")
                lesion_path = os.path.join(self.mask_dir, lesion, lesion_mask_name)

                if os.path.exists(lesion_path):
                    mask = Image.open(lesion_path).convert("L")
                    mask_resized = self.mask_transform(mask)[0]  # shape: (256, 256)

                    # Robust binarization (works even for low [0-255] scaled masks)
                    binarized_mask = (mask_resized > 0.01).float()
                    mask_tensor[i] = binarized_mask

                    if self.debug:
                        print(f"[DEBUG] {lesion_path} | max={mask_resized.max().item():.4f} | nonzeros={binarized_mask.sum().item()}")

        # Output: image, label, dummy, mask_tensor, force_class
        if self.use_segmentation_only:
            return image, torch.tensor(0), torch.zeros(1), mask_tensor, label
        else:
            return image, label, torch.zeros(1), mask_tensor, torch.tensor(0)
