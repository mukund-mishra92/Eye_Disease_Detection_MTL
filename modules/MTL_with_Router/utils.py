import torch
import copy
import torch.nn.functional as F

class MultiMetricEarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path='models/mtl_model.pth', 
                 monitor_metrics=None):
        """
        monitor_metrics: dict where key = metric name, value = 'min' or 'max'
        Example: {'val_loss_task1': 'min', 'val_acc_task2': 'max'}
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.monitor_metrics = monitor_metrics or {}
        
        self.best_metrics = {metric: float('inf') if mode == 'min' else float('-inf')
                             for metric, mode in self.monitor_metrics.items()}
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_metrics: dict, model):
        """
        current_metrics: dict with current values for each metric
        """
        improved = False
        for metric, mode in self.monitor_metrics.items():
            current = current_metrics.get(metric)
            best = self.best_metrics[metric]

            if current is None:
                continue

            if (mode == 'min' and current < best - self.delta) or \
               (mode == 'max' and current > best + self.delta):
                self.best_metrics[metric] = current
                improved = True
                print(f"✅ {metric} improved to {current:.4f}")

        if improved:
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"📦 Model saved at {self.save_path}")
        else:
            self.counter += 1
            print(f"⏳ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("⛔ Early stopping triggered.")


class CompositeEarlyStopping:
    def __init__(self, patience=5, delta=1e-4, save_path='models/mtl_model.pth',
                 alpha=0.5, beta=0.5, verbose=True):
        """
        Early stops based on a composite score:
        composite_score = alpha * (1 - val_cls_loss) + beta * val_dice

        Args:
            patience: Number of epochs with no improvement after which to stop
            delta: Minimum change to qualify as improvement
            save_path: File path to save the best model
            alpha: Weight for classification loss (converted to reward by 1 - val_cls_loss)
            beta: Weight for segmentation Dice score (already a reward)
            verbose: Whether to print detailed updates
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

        self.best_score = float('-inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_cls_loss, val_dice, model):
        # Handle cases where dice = 0 or not computed
        composite_score = self.alpha * (1 - val_cls_loss) + self.beta * val_dice

        if self.verbose:
            print(f"ℹ️ Composite Score: {composite_score:.4f} (Val Loss: {val_cls_loss:.4f}, Dice: {val_dice:.4f})")

        if composite_score >= self.best_score + self.delta:
            self.best_score = composite_score
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"✅ Improved composite score. Model saved at {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("⛔ Early stopping triggered.")

def compute_classification_metrics(logits, labels):
    """
    Computes accuracy and F1 score for classification task.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # Compute F1 Score (macro)
    num_classes = logits.size(1)
    f1 = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = tp + 0.5 * (fp + fn)
        f1 += tp / denom if denom > 0 else 0.0
    f1 = f1 / num_classes if num_classes > 0 else 0.0

    return accuracy, 

# def compute_dice_score(pred_mask, true_mask, threshold=0.01, eps=1e-6):
#     """
#     Computes average Dice score across all channels in the batch.
#     Assumes `pred_mask` is raw logits (before sigmoid).
#     """
#     pred_mask = torch.sigmoid(pred_mask)
#     pred_mask = (pred_mask > threshold).float()

#     intersection = (pred_mask * true_mask).sum(dim=(2, 3))
#     union = pred_mask.sum(dim=(2, 3)) + true_mask.sum(dim=(2, 3))
#     dice = (2 * intersection + eps) / (union + eps)

#     return dice.mean().item()

def compute_dice_score(pred_mask, true_mask, eps=1e-6):
    """
    Computes average Soft Dice score across all channels and batches.
    Works directly with sigmoid outputs (no thresholding).
    """
    pred_mask = torch.sigmoid(pred_mask)  # Keep soft values
    intersection = (pred_mask * true_mask).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + true_mask.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()
