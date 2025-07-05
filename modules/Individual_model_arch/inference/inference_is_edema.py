import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

def load_edema_model(model_path, device):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classifier
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
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_is_edema(image_path, model_path, device=None):
    """
    Predict whether edema is present in a single image.

    Args:
        image_path (str): Path to input image
        model_path (str): Path to trained binary classifier model
        device (str or torch.device): 'cuda', 'cpu', or None (auto)

    Returns:
        Tuple[int, float]: predicted label (0 or 1), probability of edema
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_edema_model(model_path, device)
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)

    return pred, prob

# Example usage
if __name__ == "__main__":
    image_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/test/test_images/IDRiD_014.jpg"
    model_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/binary_model/best_edema_model.pth"

    pred, prob = predict_is_edema(image_path, model_path)
    print(f"✅ Predicted Edema: {pred} (Probability: {prob:.4f})")
