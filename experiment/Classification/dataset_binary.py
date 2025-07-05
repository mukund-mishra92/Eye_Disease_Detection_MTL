# dataset.py
import os
import json
from PIL import Image
from torch.utils.data import Dataset

class EyeDiseaseDataset(Dataset):
    def __init__(self, image_dir, labels_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load full JSON
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)

        # Get image list
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label_data = self.labels[image_name]

        # ✅ Get only the "is_edema" label
        label = int(label_data["is_edema"])

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
