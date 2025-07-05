import os
import cv2
import yaml
import numpy as np
from glob import glob
from tifffile import imread
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import load_config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lesion suffix mapping
LESION_SUFFIXES = {
    "1. Microaneurysms": "_MA.tif",
    "2. Haemorrhages": "_HE.tif",
    "3. Hard Exudates": "_EX.tif",
    "4. Soft Exudates": "_ES.tif",
    "5. Optic Disc": "_OD.tif"
}


# -----------------------------
# Dataset Class
# -----------------------------
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

# -----------------------------
# UNet Model
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.output(d1))

# -----------------------------
# Dice Score Metric
# -----------------------------
def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=2, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                val_dice += dice_score(pred, mask)
        print(f"Validation Dice: {val_dice / len(val_loader):.4f}")

    return model

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load paths from config
    config = load_config()

    train_img_dir = config["train_img_dir"]
    train_gt_dir = config["train_gt_dir"]
    test_img_dir = config["test_img_dir"]
    test_gt_dir = config["test_gt_dir"]

    # Load datasets
    train_dataset = EyeSegDataset(train_img_dir, train_gt_dir)
    test_dataset = EyeSegDataset(test_img_dir, test_gt_dir)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Train
    model = UNet()
    model = train_model(model, train_loader, test_loader, epochs=2)

    # Save model
    torch.save(model.state_dict(), "unet_eye_segmentation_test.pth")
    print("✅ Model saved as unet_eye_segmentation.pth")

    # Final test evaluation
    model.eval()
    test_dice = 0
    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            test_dice += dice_score(pred, mask)
    print(f"✅ Final Test Dice Score: {test_dice / len(test_loader):.4f}")
