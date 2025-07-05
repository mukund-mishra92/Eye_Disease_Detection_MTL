import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle
from tqdm import tqdm
import logging

from modules.MTL.model import MultiTaskModel
from modules.MTL.dataset import MultiTaskDataset
from modules.MTL.utils import EarlyStopping

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
        for images, labels, _ in val_cls_loader:
            images, labels = images.to(device), labels.to(device)
            class_logits, _ = model(images)
            loss_cls = criterion_cls(class_logits, labels)
            total_val_loss += loss_cls.item()
            total_batches += 1

        for images, _, masks in val_seg_loader:
            images, masks = images.to(device), masks.to(device)
            _, seg_logits = model(images)
            loss_seg = criterion_seg(seg_logits, masks)
            total_val_loss += loss_seg.item()
            total_batches += 1

    return total_val_loss / total_batches


def train_dual_datasets(
    model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
    optimizer, criterion_cls, criterion_seg, device, epochs=20
):
    early_stopping = EarlyStopping(patience=5, save_path="models/mtl_model.pth")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        cls_iter = cycle(train_cls_loader)
        seg_iter = cycle(train_seg_loader)

        steps_per_epoch = max(len(train_cls_loader), len(train_seg_loader))

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # --- Classification step ---
            images_cls, labels_cls, _ = next(cls_iter)
            images_cls, labels_cls = images_cls.to(device), labels_cls.to(device)
            optimizer.zero_grad()
            class_logits, _ = model(images_cls)
            loss_cls = criterion_cls(class_logits, labels_cls)
            loss_cls.backward()
            optimizer.step()
            total_train_loss += loss_cls.item()

            # --- Segmentation step ---
            images_seg, _, masks_seg = next(seg_iter)
            images_seg, masks_seg = images_seg.to(device), masks_seg.to(device)
            optimizer.zero_grad()
            _, seg_logits = model(images_seg)
            loss_seg = criterion_seg(seg_logits, masks_seg)
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
    batch_size = 8
    epochs = 30
    num_classes = 5
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Datasets ---
    train_cls_dataset = MultiTaskDataset(
        image_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_images",
        label_json="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_iamge_lable_mapping.json",
        use_segmentation_only=False
    )
    val_cls_dataset = MultiTaskDataset(
        image_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images",
        label_json="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_image_label_mapping.json",
        use_segmentation_only=False
    )
    train_seg_dataset = MultiTaskDataset(
        image_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/train/train_images",
        mask_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/train/groundtruth",
        use_segmentation_only=True
    )
    val_seg_dataset = MultiTaskDataset(
        image_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/test/test_images",
        mask_dir="/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/test/groundtruth",
        use_segmentation_only=True
    )

    # --- Dataloaders ---
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=batch_size, shuffle=True)
    train_seg_loader = DataLoader(train_seg_dataset, batch_size=batch_size, shuffle=True)
    val_cls_loader = DataLoader(val_cls_dataset, batch_size=batch_size, shuffle=False)
    val_seg_loader = DataLoader(val_seg_dataset, batch_size=batch_size, shuffle=False)

    model = MultiTaskModel(num_classes=num_classes).to(device)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dual_datasets(
        model, train_cls_loader, train_seg_loader, val_cls_loader, val_seg_loader,
        optimizer, criterion_cls, criterion_seg, device, epochs
    )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mtl_model_final.pth")
    logging.info("✅ Final model saved to models/mtl_model_final.pth")


if __name__ == "__main__":
    main()
