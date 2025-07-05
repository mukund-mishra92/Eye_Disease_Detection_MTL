# File: scripts/prepare_idrid_dataset.py

import os
import shutil
import pandas as pd
from PIL import Image
import json
import torch
import torchvision.transforms as T
from pathlib import Path

def normalize_idrid_id(image_id):
    """Ensure IDRiD IDs are always 3-digit padded (e.g., IDRiD_01 → IDRiD_001)"""
    if image_id.startswith("IDRiD_"):
        parts = image_id.split("_")
        return f"IDRiD_{int(parts[1]):03d}"
    return image_id

def prepare_disease_grading_dataset():
    base = Path(__file__).resolve().parents[1] / "Data" / "B. Disease Grading"
    out_dir = Path(__file__).resolve().parents[1] / "Preprocessed_Data" / "classification"

    image_dir = base / "Original Images" / "Training Set"
    csv_path = base / "Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"

    print("Looking for CSV at:", csv_path)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    label_dict = {}
    for _, row in df.iterrows():
        base_name = normalize_idrid_id(row["Image name"].strip())
        label = int(row["Retinopathy grade"])
        file_name = base_name + ".jpg"
        src = image_dir / file_name
        dst = out_dir / file_name

        if os.path.exists(src):
            shutil.copy(src, dst)
            label_dict[file_name] = label

    with open(out_dir / "labels.json", 'w') as f:
        json.dump(label_dict, f, indent=2)

    print(f"✅ Classification data prepared: {len(label_dict)} samples")

def prepare_segmentation_dataset():
    base = Path(__file__).resolve().parents[1] / "Data" / "A. Segmentation"
    out_base = Path(__file__).resolve().parents[1] / "Preprocessed_Data" / "segmentation"

    seg_img_dir = base / "Original Images" / "Training Set"
    seg_mask_root = base / "All Segmentation Groundtruths" / "Training Set"
    out_img_dir = out_base / "images"
    out_mask_dir = out_base / "masks"

    lesion_info = [
        ("1. Microaneurysms", "_MA.tif"),
        ("2. Haemorrhages", "_HE.tif"),
        ("3. Hard Exudates", "_EX.tif"),
        ("4. Soft Exudates", "_ES.tif"),
        ("5. Optic Disc", "_OD.tif")
    ]

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    total_processed = 0
    for file in os.listdir(seg_img_dir):
        if not file.lower().endswith(".jpg"): continue
        raw_name = file.replace(".jpg", "").replace(".JPG", "")
        base_name = normalize_idrid_id(raw_name)

        src = seg_img_dir / file
        dst = out_img_dir / f"{base_name}.jpg"

        # Check all 5 masks exist
        mask_stack = []
        missing = False
        for lesion_folder, suffix in lesion_info:
            norm_name = normalize_idrid_id(raw_name)
            mask_path = seg_mask_root / lesion_folder / f"{norm_name}{suffix}"
            if not mask_path.exists():
                print(f"❌ Missing mask: {mask_path}")
                missing = True
                break
            try:
                mask = Image.open(mask_path).convert("L")
                mask_stack.append(T.ToTensor()(mask))
            except Exception as e:
                print(f"❌ Failed to read mask {mask_path}: {e}")
                missing = True
                break

        if not missing:
            shutil.copy(src, dst)
            multi_mask = torch.cat(mask_stack, dim=0)  # Shape: [5, H, W]
            torch.save(out_mask_dir / f"{base_name}.pt", multi_mask)
            total_processed += 1

    print(f"✅ Segmentation data prepared: {total_processed} samples")

if __name__ == '__main__':
    prepare_disease_grading_dataset()
    prepare_segmentation_dataset()
