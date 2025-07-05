import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from itertools import cycle
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from modules.MTL_with_Router.model import MultiTaskModel
from modules.MTL_with_Router.datast import MultiTaskDataset
from modules.MTL.utils import EarlyStopping
from modules.MTL_with_Router.losses import FocalLoss

# ---------- Loss Function ----------
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
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

# ---------- Metrics ----------
def compute_classification_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

def compute_dice_score(pred, target):
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = ((2. * intersection + smooth) / (union + smooth)).mean().item()
    return dice

# ---------- Validation ----------
def validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device):
    model.eval()
    cls_loss_total = 0.0
    seg_loss_total = 0.0
    acc_sum = 0.0
    f1_sum = 0.0
    dice_sum = 0.0

    with torch.no_grad():
        for images, labels_disease, _, _, _ in val_cls_loader:
            images, labels_disease = images.to(device), labels_disease.to(device)
            out_disease, _, _ = model(images)
            cls_loss = criterion_cls(out_disease, labels_disease)
            acc, f1 = compute_classification_metrics(out_disease, labels_disease)
            cls_loss_total += cls_loss.item()
            acc_sum += acc
            f1_sum += f1

        for images, _, _, masks, force_classes in val_seg_loader:
            images, masks, force_classes = images.to(device), masks.to(device), force_classes.to(device)
            _, _, pred_masks = model(images, force_class=force_classes)
            seg_loss = criterion_seg(pred_masks, masks)
            dice = compute_dice_score(pred_masks, masks)
            seg_loss_total += seg_loss.item()
            dice_sum += dice

    avg_cls_loss = cls_loss_total / len(val_cls_loader)
    avg_seg_loss = seg_loss_total / len(val_seg_loader)
    avg_acc = acc_sum / len(val_cls_loader)
    avg_f1 = f1_sum / len(val_cls_loader)
    avg_dice = dice_sum / len(val_seg_loader)
    return avg_cls_loss, avg_seg_loss, avg_acc, avg_f1, avg_dice

# ---------- Training ----------
def train_dual_datasets(
    model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
    optimizer, criterion_cls, criterion_seg, device, epochs=30, task_weights=(1.0, 1.0)
):
    early_stopping = EarlyStopping(patience=5, save_path="models/mtl_model.pth")
    metrics_log = []

    for epoch in range(epochs):
        model.train()
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        acc_sum = 0.0
        f1_sum = 0.0
        dice_sum = 0.0

        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)
        steps_per_epoch = max(len(train_cls_loader), len(train_seg_loader))

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            images_cls, labels_disease, _, _, _ = next(cls_iter)
            images_cls, labels_disease = images_cls.to(device), labels_disease.to(device)
            optimizer.zero_grad()
            out_disease, _, _ = model(images_cls)
            cls_loss = criterion_cls(out_disease, labels_disease)
            acc, f1 = compute_classification_metrics(out_disease, labels_disease)
            acc_sum += acc
            f1_sum += f1
            cls_loss_total += cls_loss.item()

            images_seg, _, _, masks_seg, force_classes = next(seg_iter)
            images_seg, masks_seg, force_classes = images_seg.to(device), masks_seg.to(device), force_classes.to(device)
            _, _, pred_masks = model(images_seg, force_class=force_classes)
            seg_loss = criterion_seg(pred_masks, masks_seg)
            dice = compute_dice_score(pred_masks, masks_seg)
            seg_loss_total += seg_loss.item()
            dice_sum += dice

            loss = task_weights[0] * cls_loss + task_weights[1] * seg_loss
            loss.backward()
            optimizer.step()

        avg_cls_loss = cls_loss_total / steps_per_epoch
        avg_seg_loss = seg_loss_total / steps_per_epoch
        avg_acc = acc_sum / steps_per_epoch
        avg_f1 = f1_sum / steps_per_epoch
        avg_dice = dice_sum / steps_per_epoch

        val_cls_loss, val_seg_loss, val_acc, val_f1, val_dice = validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device)

        logging.info(f"✅ Epoch {epoch+1} | Train CLS Loss: {avg_cls_loss:.4f} | Train SEG Loss: {avg_seg_loss:.4f} | Val CLS Loss: {val_cls_loss:.4f} | Val SEG Loss: {val_seg_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Dice: {val_dice:.4f}")

        metrics_log.append({
            "epoch": epoch + 1,
            "train_cls_loss": avg_cls_loss,
            "train_seg_loss": avg_seg_loss,
            "train_acc": avg_acc,
            "train_f1": avg_f1,
            "train_dice": avg_dice,
            "val_cls_loss": val_cls_loss,
            "val_seg_loss": val_seg_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_dice": val_dice
        })

        early_stopping(val_cls_loss + val_seg_loss, model)
        if early_stopping.early_stop:
            logging.info("🛑 Early stopping triggered.")
            break

    with open("logs/metrics_log.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

# ---------- Main ----------
def main():
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_cls_loader = DataLoader(train_cls_dataset, batch_size=batch_size, shuffle=True)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=batch_size, shuffle=False)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskModel(num_disease_classes=5).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    train_dual_datasets(
        model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
        optimizer, criterion_cls, criterion_seg, device, epochs,
        task_weights=(config["cls_loss_weight"], config["seg_loss_weight"])
    )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mtl_model_final.pth")
    logging.info("✅ Final model saved to models/mtl_model_final.pth")

if __name__ == "__main__":
    main()
