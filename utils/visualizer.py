import matplotlib.pyplot as plt
import numpy as np
import torch

def overlay_mask(image, mask):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = torch.sigmoid(mask).squeeze().cpu().numpy()
    mask = np.stack([mask]*3, axis=-1)
    overlay = (0.6 * image + 0.4 * mask)
    return np.clip(overlay, 0, 1)


def show_result(image, mask, pred_mask):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0).cpu())
    axs[0].set_title("Input Image")
    axs[1].imshow(mask.squeeze().cpu(), cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[2].imshow(overlay_mask(image, pred_mask))
    axs[2].set_title("Predicted Mask Overlay")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()