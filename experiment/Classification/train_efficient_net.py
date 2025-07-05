# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7
)
from torch.utils.data import DataLoader
from dataset import EyeDiseaseDataset  # 👈 Import from separate file

def get_efficientnet(model_name, num_classes):
    # Load EfficientNet model by name
    model_fns = {
        "efficientnet_b0": efficientnet_b0,
        "efficientnet_b1": efficientnet_b1,
        "efficientnet_b2": efficientnet_b2,
        "efficientnet_b3": efficientnet_b3,
        "efficientnet_b4": efficientnet_b4,
        "efficientnet_b5": efficientnet_b5,
        "efficientnet_b6": efficientnet_b6,
        "efficientnet_b7": efficientnet_b7,
    }
    if model_name not in model_fns:
        raise ValueError(f"Unsupported EfficientNet model: {model_name}")

    model = model_fns[model_name](pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

if __name__ == '__main__':  # 👈 Required for multiprocessing
    dataset_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_images'
    labels_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_iamge_lable_mapping.json"

    # Select EfficientNet version
    model_name = "efficientnet_b2"  # 👈 Change to b0, b1, b2, ..., b7 as needed
    image_size_map = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,
    }

    image_size = image_size_map[model_name]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = EyeDiseaseDataset(dataset_path, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = max(dataset.labels.values()) + 1
    model = get_efficientnet(model_name, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    model_path = f'/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/experiment/model/{model_name}_eye_disease.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
