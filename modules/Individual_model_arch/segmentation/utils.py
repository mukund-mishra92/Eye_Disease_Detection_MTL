# -----------------------------
# Dice Score Metric
# -----------------------------
def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()