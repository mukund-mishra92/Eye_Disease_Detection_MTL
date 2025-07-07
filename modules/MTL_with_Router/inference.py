import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

from modules.MTL_with_Router.model import MultiTaskModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and load weights
model = MultiTaskModel(num_disease_classes=5, num_segmentation_channels=5)
model.load_state_dict(torch.load("/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/mtl_model_final.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, predicted_class, mask_output = model(input_tensor)

        class_idx = torch.argmax(class_logits, dim=1).item()
        mask_output = torch.sigmoid(mask_output)  # apply sigmoid to get [0,1]

    return class_idx, mask_output.squeeze(0).cpu()

# Visualization utility
def visualize(image_path, masks, class_idx):
    labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    lesion_names = ['Microaneurysms', 'Hemorrhages', 'Hard Exudates', 'Soft Exudates', 'Optic Disc']

    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title(f"Class: {labels[class_idx]}")
    plt.axis("off")

    for i in range(5):
        plt.subplot(2, 4, i + 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title(lesion_names[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_image_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/test/test_images/IDRiD_59.jpg"  # change as needed

    if not os.path.exists(test_image_path):
        print(f"Image not found at: {test_image_path}")
    else:
        predicted_class, predicted_masks = infer(test_image_path)

        print(f"Predicted Class Index: {predicted_class}")
        print("Predicted Masks Shape:", predicted_masks.shape)
        
        visualize(test_image_path, predicted_masks, predicted_class)
