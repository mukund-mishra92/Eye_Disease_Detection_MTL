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
import optuna

from modules.MTL_with_Router.model import MultiTaskModel
from modules.MTL_with_Router.dataset_512 import MultiTaskDataset
from modules.MTL_with_Router.losses import FocalLoss
from modules.MTL_with_Router.utils import CompositeEarlyStopping  # ✅ using your old function


# ---------- Loss Functions ----------
class BCESoftDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.2, dice_weight=0.8):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = ((2. * intersection + smooth) / (union + smooth)).mean()
        dice_loss = 1 - dice
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


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
    pred = (pred > 0.01).float()
    intersection = (pred * target).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    valid = union > 0
    dice = ((2. * intersection + smooth) / (union + smooth))[valid]
    return dice.mean().item() if dice.numel() > 0 else 0.0


# ---------- Validation ----------
def validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device):
    model.eval()
    cls_loss_total, seg_loss_total, acc_sum, f1_sum, dice_sum = 0.0, 0.0, 0.0, 0.0, 0.0

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


# ---------- Training Core ----------
def run_training(lr, weight_decay, gamma, bce_weight, dice_weight, cls_loss_weight, seg_loss_weight):
    with open("config/config.json") as f:
        config = json.load(f)

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ]
    )

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
    criterion_cls = FocalLoss(gamma=gamma)
    criterion_seg = BCESoftDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = CompositeEarlyStopping(patience=5, save_path="models/mtl_model_best.pth", alpha=0.5, beta=0.5)
    metrics_log = []

    for epoch in range(epochs):
        model.train()
        cls_loss_total, seg_loss_total, acc_sum, f1_sum, dice_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)
        steps = max(len(train_cls_loader), len(train_seg_loader))

        for _ in tqdm(range(steps), desc=f"Epoch {epoch+1}/{epochs}"):
            images_cls, labels_disease, _, _, _ = next(cls_iter)
            images_cls, labels_disease = images_cls.to(device), labels_disease.to(device)
            optimizer.zero_grad()
            out_disease, _, _ = model(images_cls)
            cls_loss = criterion_cls(out_disease, labels_disease)
            acc, f1 = compute_classification_metrics(out_disease, labels_disease)
            cls_loss_total += cls_loss.item()
            acc_sum += acc
            f1_sum += f1

            images_seg, _, _, masks_seg, force_classes = next(seg_iter)
            images_seg, masks_seg, force_classes = images_seg.to(device), masks_seg.to(device), force_classes.to(device)
            _, _, pred_masks = model(images_seg, force_class=force_classes)
            seg_loss = criterion_seg(pred_masks, masks_seg)
            dice = compute_dice_score(pred_masks, masks_seg)
            seg_loss_total += seg_loss.item()
            dice_sum += dice

            loss = cls_loss_weight * cls_loss + seg_loss_weight * seg_loss
            loss.backward()
            optimizer.step()

        avg_cls_loss = cls_loss_total / steps
        avg_seg_loss = seg_loss_total / steps
        avg_acc = acc_sum / steps
        avg_f1 = f1_sum / steps
        avg_dice = dice_sum / steps

        val_cls_loss, val_seg_loss, val_acc, val_f1, val_dice = validate(
            model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device)

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

        early_stopping(val_cls_loss, val_dice, model)
        if early_stopping.early_stop:
            break

    with open("logs/metrics_log.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    return val_acc, val_f1, val_dice


# ---------- Optuna Main ----------
if __name__ == "__main__":
    

    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
        gamma = trial.suggest_uniform("gamma", 1.0, 5.0)
        bce_weight = trial.suggest_uniform("bce_weight", 0.1, 0.9)
        dice_weight = 1.0 - bce_weight
        cls_loss_weight = trial.suggest_uniform("cls_loss_weight", 0.5, 2.0)
        seg_loss_weight = trial.suggest_uniform("seg_loss_weight", 0.5, 2.0)

        val_acc, val_f1, val_dice = run_training(
            lr, weight_decay, gamma,
            bce_weight, dice_weight,
            cls_loss_weight, seg_loss_weight
        )
        return 0.4 * val_acc + 0.3 * val_f1 + 0.3 * val_dice

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best Trial:")
    print(study.best_trial)