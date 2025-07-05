import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os

from modules.MTL.model import MultiTaskModel  # Your model definition

def load_model(model_path, device, num_classes=5):
    model = MultiTaskModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image.resize((256, 256))  # Tensor, PIL resized

def postprocess_mask(mask_tensor, threshold=0.5):
    mask = torch.sigmoid(mask_tensor)
    mask = (mask > threshold).cpu().numpy().astype(np.uint8)
    return np.squeeze(mask)  # [H, W]

def overlay_mask_on_image(original_img, mask_array, color=(255, 0, 0), alpha=0.5):
    mask = Image.fromarray(mask_array * 255).convert("L").resize(original_img.size)
    color_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=color).convert("RGBA")
    original_rgba = original_img.convert("RGBA")
    blended = Image.blend(original_rgba, color_mask, alpha=alpha)
    return blended

def inference(image_path, model_path, output_dir="result", class_labels=None):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)

    # Preprocess
    image_tensor, original_img = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        class_logits, seg_logits = model(image_tensor)

    # Classification output
    pred_class = class_logits.argmax(dim=1).item()
    severity = class_labels[pred_class] if class_labels else str(pred_class)

    # Segmentation output
    mask = postprocess_mask(seg_logits)

    # Save original
    original_save_path = os.path.join(output_dir, "original.png")
    original_img.save(original_save_path)

    # Save mask
    mask_img = Image.fromarray(mask * 255).convert("L")
    mask_path = os.path.join(output_dir, "mask.png")
    mask_img.save(mask_path)

    # Overlay
    overlay = overlay_mask_on_image(original_img, mask)
    overlay_path = os.path.join(output_dir, "overlay.png")
    overlay.save(overlay_path)

    print(f"✅ Saved: original, mask, overlay in `{output_dir}`")
    print(f"🧠 Predicted Disease Grade: {pred_class} ({severity})")

    return pred_class, severity, original_save_path, mask_path, overlay_path

if __name__ == "__main__":
    # Example usage
    image_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/test/test_images/IDRiD_70.jpg"
    model_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/mtl_model_final.pth"

    # Optional: label mapping
    class_labels = {
        0: "Healthy",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }

    inference(image_path, model_path, class_labels=class_labels)
