import torch.nn as nn

def multitask_loss(class_out, class_target, seg_out, seg_target, alpha=1.0, beta=1.0):
    ce_loss = nn.CrossEntropyLoss()(class_out, class_target)
    bce_loss = nn.BCEWithLogitsLoss()(seg_out, seg_target)
    return alpha * ce_loss + beta * bce_loss
