# inference.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.training.segmentation.model import UNet  # 👈 if UNet is in a separate model.py file; otherwise, define it here
from modules.training.segmentation.segment_dataset import EyeSegDataset  # 👈 if dataset is in separate file
from modules.training.segmentation.utils import dice_score  # 👈 if you move dice_score into a utils.py
from modules.training.segmentation.config import load_config  # 👈 if you have a config.py for loading configurations



def main():
    # Load config
    config = load_config()

    test_img_dir = config["test_img_dir"]
    test_gt_dir = config["test_gt_dir"]
    model_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/unet_eye_segmentation.pth"

    # Dataset and DataLoader
    test_dataset = EyeSegDataset(test_img_dir, test_gt_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inference and Evaluation
    total_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            preds = model(images)
            total_dice += dice_score(preds, masks)

    avg_dice = total_dice / len(test_loader)
    print(f"✅ Test Dice Score: {avg_dice:.4f}")

if __name__ == "__main__":
    main()
