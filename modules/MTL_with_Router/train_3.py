import os
import json
import torch
import logging
from torch.utils.data import DataLoader
from modules.MTL_with_Router.model import MultiTaskModel
from modules.MTL_with_Router.patch_dataset import PatchifiedSegmentationDataset
from modules.MTL_with_Router.datast import MultiTaskDataset

from modules.MTL_with_Router.utils import compute_classification_metrics, compute_dice_score, CompositeEarlyStopping
from modules.MTL_with_Router.losses import BCEDiceLoss
from tqdm import tqdm
from itertools import cycle
import torch.nn as nn
import torch.optim as optim


def validate(model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device):
    model.eval()
    val_cls_loss = 0.0
    val_seg_loss = 0.0
    acc_sum = 0.0
    f1_sum = 0.0
    dice_sum = 0.0

    with torch.no_grad():
        for images, labels, _, _, _ in val_cls_loader:
            images, labels = images.to(device), labels.to(device)
            out_disease, _, _ = model(images)
            val_cls_loss += criterion_cls(out_disease, labels).item()
            acc, f1 = compute_classification_metrics(out_disease, labels)
            acc_sum += acc
            f1_sum += f1

        for images, _, _, masks, force_classes in val_seg_loader:
            images, masks, force_classes = images.to(device), masks.to(device), force_classes.to(device)
            _, _, pred_masks = model(images, force_class=force_classes)
            val_seg_loss += criterion_seg(pred_masks, masks).item()
            dice = compute_dice_score(pred_masks, masks)
            dice_sum += dice

    n_cls = len(val_cls_loader)
    n_seg = len(val_seg_loader)
    return (val_cls_loss / n_cls, val_seg_loss / n_seg,
            acc_sum / n_cls, f1_sum / n_cls, dice_sum / n_seg)


def train():
    # Load config
    with open("config/config.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and loaders
    train_cls_dataset = MultiTaskDataset(
        image_dir=config["train_cls_image_dir"],
        label_json=config["train_cls_label_json"]
    )
    val_cls_dataset = MultiTaskDataset(
        image_dir=config["val_cls_image_dir"],
        label_json=config["val_cls_label_json"]
    )

    train_seg_dataset = PatchifiedSegmentationDataset(
        image_dir=config["train_seg_image_dir"],
        mask_dir=config["train_seg_mask_dir"]
    )
    val_seg_dataset = PatchifiedSegmentationDataset(
        image_dir=config["val_seg_image_dir"],
        mask_dir=config["val_seg_mask_dir"]
    )

    batch_size = config["batch_size"]

    train_cls_loader = DataLoader(train_cls_dataset, batch_size=batch_size, shuffle=True)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=batch_size, shuffle=False)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False)

    # Model and training setup
    model = MultiTaskModel().to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    early_stopping = CompositeEarlyStopping(
        patience=5,
        save_path="models/mtl_model_best.pth",
        alpha=0.5,
        beta=0.5
    )

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        cls_loss_total = 0.0
        seg_loss_total = 0.0
        acc_sum = 0.0
        f1_sum = 0.0
        dice_sum = 0.0

        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)
        steps_per_epoch = max(len(train_cls_loader), len(train_seg_loader))

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            images_cls, labels, _, _, _ = next(cls_iter)
            images_cls, labels = images_cls.to(device), labels.to(device)

            optimizer.zero_grad()
            out_disease, _, _ = model(images_cls)
            cls_loss = criterion_cls(out_disease, labels)
            acc, f1 = compute_classification_metrics(out_disease, labels)
            acc_sum += acc
            f1_sum += f1
            cls_loss_total += cls_loss.item()

            images_seg, _, _, masks_seg, force_classes = next(seg_iter)
            images_seg, masks_seg, force_classes = images_seg.to(device), masks_seg.to(device), force_classes.to(device)

            if masks_seg.sum() == 0:
                print("⚠️ All-zero masks encountered in this batch")

            _, _, pred_masks = model(images_seg, force_class=force_classes)
            seg_loss = criterion_seg(pred_masks, masks_seg)
            dice = compute_dice_score(pred_masks, masks_seg)
            seg_loss_total += seg_loss.item()
            dice_sum += dice

            loss = config["cls_loss_weight"] * cls_loss + config["seg_loss_weight"] * seg_loss
            loss.backward()
            optimizer.step()

        avg_cls_loss = cls_loss_total / steps_per_epoch
        avg_seg_loss = seg_loss_total / steps_per_epoch
        avg_acc = acc_sum / steps_per_epoch
        avg_f1 = f1_sum / steps_per_epoch
        avg_dice = dice_sum / steps_per_epoch

        val_cls_loss, val_seg_loss, val_acc, val_f1, val_dice = validate(
            model, val_cls_loader, val_seg_loader, criterion_cls, criterion_seg, device
        )

        logging.info(f"Epoch {epoch+1} | Train CLS Loss: {avg_cls_loss:.4f} | Train SEG Loss: {avg_seg_loss:.4f} | Val CLS Loss: {val_cls_loss:.4f} | Val SEG Loss: {val_seg_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Dice: {val_dice:.4f}")

        early_stopping(val_cls_loss, val_dice, model)
        if early_stopping.early_stop:
            print("⛔ Early stopping triggered.")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
