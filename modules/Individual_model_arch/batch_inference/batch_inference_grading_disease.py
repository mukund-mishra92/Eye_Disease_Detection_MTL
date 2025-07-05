# inference.py
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from modules.training.multi_class_grading.dataset import EyeDiseaseDataset

def main():
    # Paths
    test_data_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images'
    labels_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_image_label_mapping.json"
    model_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/resnet50_eye_disease.pth'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset and Loader
    test_dataset = EyeDiseaseDataset(test_data_path, labels_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model setup
    num_classes = max(test_dataset.labels.values()) + 1
    model = resnet50(weights=None)  # Updated for warning-free use
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()
