# modules/MTL/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = pt.gather(1, targets.view(-1, 1)).squeeze(1)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets.view(-1))
            logpt *= at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        pred: raw logits from the model (before sigmoid)
        target: binary mask ground truth
        """
        bce_loss = self.bce(pred, target)

        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth)).mean()

        return bce_loss + dice_loss
