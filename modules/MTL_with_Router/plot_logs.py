import re
import matplotlib.pyplot as plt

log_path = "logs/training1.log"

epoch_data = []
with open(log_path, "r") as f:
    for line in f:
        match = re.search(
            r"Epoch (\d+) \| Train CLS Loss: ([\d\.]+) \| Train SEG Loss: ([\d\.]+) \| Val CLS Loss: ([\d\.]+) \| Val SEG Loss: ([\d\.]+) \| Acc: ([\d\.]+) \| F1: ([\d\.]+) \| Dice: ([\d\.]+)", 
            line
        )
        if match:
            epoch_data.append({
                "epoch": int(match.group(1)),
                "train_cls_loss": float(match.group(2)),
                "train_seg_loss": float(match.group(3)),
                "val_cls_loss": float(match.group(4)),
                "val_seg_loss": float(match.group(5)),
                "val_acc": float(match.group(6)),
                "val_f1": float(match.group(7)),
                "val_dice": float(match.group(8))
            })

# ---------- Step 2: Extract lists ----------
epochs = [d["epoch"] for d in epoch_data]
train_cls_loss = [d["train_cls_loss"] for d in epoch_data]
train_seg_loss = [d["train_seg_loss"] for d in epoch_data]
val_cls_loss = [d["val_cls_loss"] for d in epoch_data]
val_seg_loss = [d["val_seg_loss"] for d in epoch_data]
val_acc = [d["val_acc"] for d in epoch_data]
val_f1 = [d["val_f1"] for d in epoch_data]
val_dice = [d["val_dice"] for d in epoch_data]

# ---------- Step 3: Plot ----------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(epochs, train_cls_loss, label="Train CLS Loss", marker='o')
plt.plot(epochs, train_seg_loss, label="Train SEG Loss", marker='o')
plt.plot(epochs, val_cls_loss, label="Val CLS Loss", linestyle='--', marker='x')
plt.plot(epochs, val_seg_loss, label="Val SEG Loss", linestyle='--', marker='s')
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
plt.plot(epochs, val_f1, label="Validation F1", marker='x')
plt.plot(epochs, val_dice, label="Validation Dice", marker='s')
plt.title("Validation Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("logs/mtl_loss_and_metrics_plot3.png")
plt.show()
