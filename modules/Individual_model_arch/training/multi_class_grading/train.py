# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101
from torch.utils.data import DataLoader
from dataset import EyeDiseaseDataset  # 👈 Import from separate file

if __name__ == '__main__':  # 👈 Required for multiprocessing
    dataset_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_images'
    labels_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_iamge_lable_mapping.json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = EyeDiseaseDataset(dataset_path, labels_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = resnet18(pretrained=True)
    model = resnet50(pretrained=True)  # You can also use resnet101 or resnet18
    num_classes = max(dataset.labels.values()) + 1
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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


    model_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/experiment/model/resnet50_eye_disease.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
