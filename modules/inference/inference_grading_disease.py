import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
# from inference_utils import predict_image_class

def load_model(model_path, num_classes, device):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_image_class(image_path, model_path, label_map_path=None, device=None):
    """
    Predict class index for a single image.

    Args:
        image_path (str): Path to image file
        model_path (str): Path to saved .pth model
        label_map_path (str): Path to JSON file with label mapping (used to determine num_classes)
        device (str): 'cuda' or 'cpu'. If None, auto-detect

    Returns:
        int: Predicted class index
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine number of classes
    if label_map_path:
        with open(label_map_path, 'r') as f:
            label_dict = json.load(f)
        num_classes = max(label_dict.values()) + 1
    else:
        raise ValueError("label_map_path must be provided to determine num_classes.")

    # Load model and preprocess image
    model = load_model(model_path, num_classes, device)
    image_tensor = preprocess_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()


def main():
# --- Configuration ---
    model_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/resnet50_eye_disease.pth'
    label_map_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_image_label_mapping.json"
    image_path = '/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images/IDRiD_006.jpg'

    predicted_class = predict_image_class(image_path, model_path, label_map_path)
    print(f"Predicted class: {predicted_class}")


if __name__ == '__main__':
    main()
