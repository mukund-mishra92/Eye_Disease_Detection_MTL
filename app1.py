import os
import torch
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from modules.MTL_with_Router.model import MultiTaskModel

# ------------------------
# Configuration
# ------------------------
MODEL_PATH = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/final_models/mtl_model_fina_6julyl.pth"
CLASS_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
LESION_LABELS = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
LESION_COLORS = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 255), (0, 255, 0)]  # red, orange, yellow, cyan, green

# ------------------------
# Load Model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(num_disease_classes=5, num_segmentation_channels=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ------------------------
# Transforms
# ------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# ------------------------
# Utilities
# ------------------------
def predict_and_segment(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, predicted_class, seg_masks = model(input_tensor)
        class_index = torch.argmax(class_logits, dim=1).item()
        seg_masks = torch.sigmoid(seg_masks).squeeze(0).cpu().numpy()
        return class_index, seg_masks, image

def overlay_mask_on_image(original_image, mask_array, color=(255, 0, 0), alpha=0.5):
    original = original_image.convert("RGB")
    mask_array = np.array(mask_array)

    if mask_array.max() <= 1.0:
        mask_array = (mask_array * 255).astype(np.uint8)

    mask = Image.fromarray(mask_array).convert("L")
    color_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=color).convert("RGBA")
    original_rgba = original.convert("RGBA")
    blended = Image.blend(original_rgba, color_mask, alpha=alpha)
    return blended

# ------------------------
# Streamlit UI
# ------------------------
st.title("🧠 Multi-task Eye Disease Detection")

uploaded = st.file_uploader("Upload a retinal image", type=["jpg", "png", "jpeg"])
if uploaded:
    st.image(uploaded, caption="Input Image", use_column_width=True)
    st.write("⏳ Analyzing...")

    img_path = "temp_input.jpg"
    img = Image.open(uploaded).convert("RGB")
    img.save(img_path)

    class_index, masks, resized_image = predict_and_segment(img_path)

    st.write(f"**Predicted Class**: `{CLASS_LABELS[class_index]}`")
    
    if class_index == 0:
        st.success("✅ No signs of disease detected.")
    else:
        st.warning("⚠️ Signs of retinopathy detected. Visualizing lesion segmentation...")

        option = st.radio("Choose segmentation display mode:", ("Combined Overlay", "Per-Lesion Overlays"))
        resized_for_display = img.resize((512, 512))

        if option == "Combined Overlay":
            combined_mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
            overlay = overlay_mask_on_image(resized_for_display, combined_mask, color=(255, 0, 0))
            st.image(overlay, caption="🩸 Combined Lesion Overlay", use_column_width=True)

        elif option == "Per-Lesion Overlays":
            for i in range(5):
                lesion_mask = masks[i]
                overlay = overlay_mask_on_image(resized_for_display, lesion_mask, color=LESION_COLORS[i])
                st.image(overlay, caption=f"🩸 {LESION_LABELS[i]}", use_column_width=True)
