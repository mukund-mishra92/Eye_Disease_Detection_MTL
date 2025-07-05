import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from modules.training.binary_is_edema.dataset_binary import EyeDiseaseDataset  # Update import if needed

def main():
    # --- Paths ---
    test_data_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images'
    labels_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_is_edema_binary_image_label_mapping.json'
    model_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/binary_model/best_edema_model.pth'  # Or full path to saved model

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset and Loader ---
    test_dataset = EyeDiseaseDataset(test_data_path, labels_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model Setup ---
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    # --- Inference ---
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).int()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # --- Metrics ---
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')  # AUC requires at least one positive and one negative sample

    # --- Report ---
    print(f"✅ Binary Classification Metrics for Edema Detection:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC       : {auc:.4f}")

if __name__ == '__main__':
    main()
