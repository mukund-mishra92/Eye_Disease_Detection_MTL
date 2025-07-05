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
import optuna
from optuna.trial import Trial

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

def objective(trial: Trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
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
    criterion_seg = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')

    for epoch in range(10):
        model.train()
        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)
        steps = min(len(train_cls_loader), len(train_seg_loader))

        for _ in range(steps):
            images_cls, labels_disease, *_ = next(cls_iter)
            images_cls, labels_disease = images_cls.to(device), labels_disease.to(device)
            optimizer.zero_grad()
            out_disease, _, _ = model(images_cls)
            loss_cls = criterion_cls(out_disease, labels_disease)
            loss_cls.backward()
            optimizer.step()

            images_seg, _, _, masks_seg, force_classes = next(seg_iter)
            images_seg, masks_seg, force_classes = images_seg.to(device), masks_seg.to(device), force_classes.to(device)
            optimizer.zero_grad()
            _, _, pred_masks = model(images_seg, force_class=force_classes)
            loss_seg = criterion_seg(pred_masks, masks_seg)
            loss_seg.backward()
            optimizer.step()

        val_loss = validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device)
        best_val_loss = min(best_val_loss, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best hyperparameters:", study.best_params)
