import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class IDRiDDataset(Dataset):
    def __init__(self, image_dir, label_dict, mask_dir, transform=None):
        self.image_dir = image_dir
        self.label_dict = label_dict  # {image_name: class_label}
        self.mask_dir = mask_dir
        self.image_list = list(label_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        label = self.label_dict[image_name]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, label, mask
