import torch

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def dice_score(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


def iou_score(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)
