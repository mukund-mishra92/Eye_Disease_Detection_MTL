import torch.nn.functional as F
import torch


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path='models/mtl_model.pth'):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss, model):
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"✅ Model improved. Saving to {self.save_path}")
        else:
            self.counter += 1
            print(f"⏳ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def multitask_loss(class_logits, true_labels, seg_logits, true_masks):
    classification_loss = F.cross_entropy(class_logits, true_labels)
    segmentation_loss = F.binary_cross_entropy_with_logits(seg_logits, true_masks)
    return classification_loss + segmentation_loss, classification_loss.item(), segmentation_loss.item()
