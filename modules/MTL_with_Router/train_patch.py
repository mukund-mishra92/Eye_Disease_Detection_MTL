# train_patch.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import models
from modules.MTL_with_Router.patch_dataset import PatchifiedSegmentationDataset


def train_patch_segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}\n")

    # Config (can replace with config.json loading if needed)
    config = {
        "batch_size": 4,
        "epochs": 10,
        "lr": 1e-4,
        "image_dir": "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/train/train_images",
        "mask_dir": "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/train/groundtruth"
    }

    # Dataset and Dataloader
    train_dataset = PatchifiedSegmentationDataset(
        image_dir=config["image_dir"],
        mask_dir=config["mask_dir"]
    )

    if len(train_dataset) == 0:
        raise ValueError("🚨 No valid data in the dataset! Check mask/image paths.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Model (simple UNet-style conv net or pretrained backbone)
    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📅 Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train_patch_segmentation()
