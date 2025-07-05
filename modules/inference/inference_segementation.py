import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from modules.training.segmentation.model import UNet

from PIL import Image, ImageOps

def overlay_mask_on_image(original_path, mask_array, output_path="result/overlay.png", color=(255, 0, 0), alpha=0.5):
    # Load original image
    original = Image.open(original_path).convert("RGB").resize((256, 256))

    # Convert mask to image with same size and RGB color
    mask = Image.fromarray((mask_array * 255).astype(np.uint8)).convert("L")
    color_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=color).convert("RGBA")
    
    # Convert original image to RGBA
    original_rgba = original.convert("RGBA")
    
    # Blend the images
    blended = Image.blend(original_rgba, color_mask, alpha=alpha)

    #blended.save(output_path)
    #print(f"✅ Overlay saved to {output_path}")
    return blended


def load_unet_model(model_path, device):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def postprocess_mask(pred_mask, threshold=0.5):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > threshold).cpu().numpy().astype(np.uint8)  # shape: [1, 1, H, W]
    return pred_mask

def predict_single_image(image_path, model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_unet_model(model_path, device)
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    predicted_mask = postprocess_mask(output)
    return predicted_mask  # shape: [1, 1, H, W]

if __name__ == "__main__":
    image_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/segmentation/test/test_images/IDRiD_63.jpg"
    model_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/unet_eye_segmentation.pth"
    output_path = "result/predicted_mask.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mask = predict_single_image(image_path, model_path)

    # ✅ Ensure shape is [H, W]
    #mask_2d = np.squeeze(mask)  # from shape [1, 1, H, W] to [H, W]
    mask_2d = mask[0, 0, :, :]

    # ✅ Save the mask as grayscale image
    #Image.fromarray(mask_2d * 255).convert("L").save(output_path)

    original_image = Image.open(image_path).convert("RGB").resize((256, 256))

    original_output_path = "result/original_image.png"
    original_image.save(original_output_path)
    print(f"✅ Saved original image to {original_output_path}")
    Image.fromarray(mask_2d * 255).convert("L").save(output_path)

    overlay_mask_on_image(image_path, mask_2d, output_path="result/overlay.png")




    print(f"✅ Saved predicted mask to {output_path}")
