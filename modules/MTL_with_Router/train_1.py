## this is for hyperparameter tuning with optuna

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from itertools import cycle
import logging
import json

from modules.MTL_with_Router.model import MultiTaskModel
from modules.MTL_with_Router.datast import MultiTaskDataset
from modules.MTL.utils import EarlyStopping
from modules.MTL_with_Router.losses import FocalLoss  # Custom focal loss

# ---------- Loss Function (Add BCEDiceLoss) ----------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

        return bce_loss + dice_loss

# ---------- Load Config ----------
with open("config.json") as f:
    config = json.load(f)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device):
    model.eval()
    total_val_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, labels_disease, _, _, _ in val_cls_loader:
            images = images.to(device)
            labels_disease = labels_disease.to(device)
            out_disease, _, _ = model(images)
            loss_d = criterion_cls(out_disease, labels_disease)
            total_val_loss += loss_d.item()
            total_batches += 1

        for images, _, _, masks, force_classes in val_seg_loader:
            images = images.to(device)
            masks = masks.to(device)
            force_classes = force_classes.to(device)
            _, _, pred_masks = model(images, force_class=force_classes)
            loss_seg = criterion_seg(pred_masks, masks)
            total_val_loss += loss_seg.item()
            total_batches += 1

    return total_val_loss / total_batches

def train_dual_datasets(
    model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
    optimizer, criterion_cls, criterion_seg, device, epochs=30
):
    early_stopping = EarlyStopping(patience=5, save_path="models/mtl_model.pth")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)
        steps_per_epoch = max(len(train_cls_loader), len(train_seg_loader))

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # --- Classification Step ---
            images_cls, labels_disease, _, _, _ = next(cls_iter)
            images_cls = images_cls.to(device)
            labels_disease = labels_disease.to(device)

            optimizer.zero_grad()
            out_disease, _, _ = model(images_cls)
            loss_cls = criterion_cls(out_disease, labels_disease)
            loss_cls.backward()
            optimizer.step()
            total_train_loss += loss_cls.item()

            # --- Segmentation Step ---
            images_seg, _, _, masks_seg, force_classes = next(seg_iter)
            images_seg = images_seg.to(device)
            masks_seg = masks_seg.to(device)
            force_classes = force_classes.to(device)

            optimizer.zero_grad()
            _, _, pred_masks = model(images_seg, force_class=force_classes)
            loss_seg = criterion_seg(pred_masks, masks_seg)
            loss_seg.backward()
            optimizer.step()
            total_train_loss += loss_seg.item()

        avg_train_loss = total_train_loss / (2 * steps_per_epoch)
        avg_val_loss = validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device)

        logging.info(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logging.info("🛑 Early stopping triggered.")
            break

def main():
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Datasets ---
    train_cls_dataset = MultiTaskDataset(
        image_dir=config["train_cls_image_dir"],
        label_json=config["train_cls_label_json"]
    )
    val_cls_dataset = MultiTaskDataset(
        image_dir=config["val_cls_image_dir"],
        label_json=config["val_cls_label_json"]
    )

    train_seg_dataset = MultiTaskDataset(
        image_dir=config["train_seg_image_dir"],
        mask_dir=config["train_seg_mask_dir"],
        use_segmentation_only=True
    )
    val_seg_dataset = MultiTaskDataset(
        image_dir=config["val_seg_image_dir"],
        mask_dir=config["val_seg_mask_dir"],
        use_segmentation_only=True
    )

    # --- Dataloaders ---
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=batch_size, shuffle=True)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=batch_size, shuffle=False)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskModel(num_disease_classes=5).to(device)

    criterion_cls = FocalLoss(gamma=2)
    criterion_seg = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_dual_datasets(
        model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
        optimizer, criterion_cls, criterion_seg, device, epochs
    )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mtl_model_final.pth")
    logging.info("✅ Final model saved to models/mtl_model_final.pth")

if __name__ == "__main__":
    main()
