import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import logging

from modules.MTL_with_Router.model import MultiTaskModel
from modules.MTL_with_Router.datast import MultiTaskDataset
from modules.MTL_with_Router.train import BCEDiceLoss

# ---------- Load Config ----------
with open("config.json") as f:
    config = json.load(f)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = target.bool()
    intersection = (pred & target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = ((2 * intersection + smooth) / (union + smooth)).mean()
    return dice.item()

def evaluate(model, test_cls_loader, test_seg_loader, device):
    model.eval()

    # Classification metrics
    all_preds = []
    all_labels = []

    # Segmentation metrics
    total_dice = 0.0
    total_bce = 0.0
    bce_loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        # --- Classification Evaluation ---
        for images, labels, _, _, _ in test_cls_loader:
            images, labels = images.to(device), labels.to(device)
            out_cls, _, _ = model(images)
            preds = torch.argmax(out_cls, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        logging.info(f"✅ Classification Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

        # --- Segmentation Evaluation ---
        for images, _, _, masks, force_classes in test_seg_loader:
            images = images.to(device)
            masks = masks.to(device)
            force_classes = force_classes.to(device)
            _, _, pred_masks = model(images, force_class=force_classes)

            dice = dice_coefficient(pred_masks, masks)
            bce = bce_loss_fn(pred_masks, masks).item()

            total_dice += dice
            total_bce += bce

        avg_dice = total_dice / len(test_seg_loader)
        avg_bce = total_bce / len(test_seg_loader)
        logging.info(f"✅ Segmentation Dice: {avg_dice:.4f} | BCE Loss: {avg_bce:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load test datasets ---
    test_cls_dataset = MultiTaskDataset(
        image_dir=config["val_cls_image_dir"],
        label_json=config["val_cls_label_json"]
    )
    test_seg_dataset = MultiTaskDataset(
        image_dir=config["val_seg_image_dir"],
        mask_dir=config["val_seg_mask_dir"],
        use_segmentation_only=True
    )

    test_cls_loader = DataLoader(test_cls_dataset, batch_size=config["batch_size"], shuffle=False)
    test_seg_loader = DataLoader(test_seg_dataset, batch_size=config["batch_size"], shuffle=False)

    # --- Load model ---
    model = MultiTaskModel(num_disease_classes=5).to(device)
    model.load_state_dict(torch.load("models/mtl_model_final.pth"))
    model.eval()

    evaluate(model, test_cls_loader, test_seg_loader, device)

if __name__ == "__main__":
    main()
