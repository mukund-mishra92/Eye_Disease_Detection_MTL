# train_binary_edema_with_val.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from dataset import EyeDiseaseDataset

def train_model(model, train_loader, val_loader, device, patience=3, epochs=20, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/binary_model/best_edema_model.pth")
        else:
            early_stop_counter += 1
            print(f"⚠️  No improvement for {early_stop_counter} epoch(s).")

        if early_stop_counter >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

    return model

if __name__ == '__main__':
    # Paths — change to your actual validation dataset paths
    train_img_dir = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_images'
    train_labels_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_is_edema_binary_image_label_mapping.json'

    val_img_dir = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images'
    val_labels_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_is_edema_binary_image_label_mapping.json'

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = EyeDiseaseDataset(train_img_dir, train_labels_path, transform=transform)
    val_dataset = EyeDiseaseDataset(val_img_dir, val_labels_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model setup
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    # Train with early stopping
    model = train_model(model, train_loader, val_loader, device, patience=3, epochs=20)

    print("✅ Training completed. Best model saved as 'models/binary_model/best_edema_model.pth'")



