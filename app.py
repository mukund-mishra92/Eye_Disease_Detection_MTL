import streamlit as st
from PIL import Image, ImageOps
import numpy as np

from modules.Individual_model_arch.inference.inference_grading_disease import predict_image_class
from modules.Individual_model_arch.inference.inference_segementation import predict_single_image

# --- Overlay function ---
def overlay_mask_on_image(original_image, mask_array, color=(255, 0, 0), alpha=0.5):
    original = original_image.convert("RGB")
    mask_array = np.array(mask_array)

    if mask_array.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask_array.shape}")

    if mask_array.max() <= 1.0:
        mask_array = mask_array * 255
    mask_array = mask_array.astype(np.uint8)

    mask = Image.fromarray(mask_array).convert("L")
    color_mask = ImageOps.colorize(mask, black=(0, 0, 0), white=color).convert("RGBA")
    original_rgba = original.convert("RGBA")

    blended = Image.blend(original_rgba, color_mask, alpha=alpha)
    return blended

# --- Constants ---
label_map_path = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/Preprocessed_Data/classification/train/train_iamge_lable_mapping.json"
CLASSIFIER_MODEL = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/resnet50_eye_disease.pth"
SEGMENTATION_MODEL = "/Users/balmukundmishra/Desktop/2025-Learning/Eye_Disease_Detection_MTL/models/4-july/unet_eye_segmentation.pth"

lesion_labels = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
lesion_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 255), (0, 255, 0)]  # red, orange, yellow, cyan, green

# --- Streamlit UI ---
st.title("🔍 Eye Health Analyzer")

uploaded = st.file_uploader("Upload a retinal image", type=["jpg", "png"], accept_multiple_files=False)
if uploaded:
    st.image(uploaded, caption="Input Image", use_column_width=True)
    st.write("Analyzing...")

    # Save uploaded file
    img = Image.open(uploaded)
    img_path = "temp_input.jpg"
    img.save(img_path)

    # Classification
    severity = predict_image_class(img_path, CLASSIFIER_MODEL, label_map_path)
    st.write(f"**Predicted Severity Class**: {severity}")

    if severity == 0:
        st.success("✅ Eye Health Looks Good — No signs of disease detected.")
    else:
        st.warning(f"⚠️ Detected severity class {severity}. Running segmentation...")

        # Segmentation
        mask = predict_single_image(img_path, SEGMENTATION_MODEL)
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.array(mask)
        st.write(f"Raw mask shape: {mask.shape}")

        # Ask user how to view the output
        option = st.radio("Choose mask overlay mode:", ("Combined Overlay", "Per-Lesion Overlays"))

        resized_img = img.resize((256, 256))

        if option == "Combined Overlay":
            # Combine all lesion channels
            if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 5:
                combined_mask = np.any(mask[0], axis=0).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected mask shape for combined mode: {mask.shape}")

            st.write("Displaying combined lesion overlay")
            overlay = overlay_mask_on_image(resized_img, combined_mask, color=(255, 0, 0))
            st.image(overlay, caption="⚠️ Combined Segmentation Overlay", use_column_width=True)

        elif option == "Per-Lesion Overlays":
            if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 5:
                st.write("Displaying each lesion in different color:")
                for i in range(5):
                    lesion_mask = mask[0, i]
                    if lesion_mask.max() > 1:
                        lesion_mask = (lesion_mask > 127).astype(np.uint8)
                    overlay = overlay_mask_on_image(resized_img, lesion_mask, color=lesion_colors[i])
                    st.image(overlay, caption=f"🩸 {lesion_labels[i]}", use_column_width=True)
            else:
                raise ValueError(f"Unexpected mask shape for per-lesion mode: {mask.shape}")
