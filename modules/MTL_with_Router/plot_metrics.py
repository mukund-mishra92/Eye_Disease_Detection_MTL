import json
import matplotlib.pyplot as plt

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_metric(metric_name, train_key, val_key, metrics):
    epochs = [entry["epoch"] for entry in metrics]
    train_values = [entry[train_key] for entry in metrics]
    val_values = [entry[val_key] for entry in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label=f"Train {metric_name}", marker='o')
    plt.plot(epochs, val_values, label=f"Val {metric_name}", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{metric_name.replace(' ', '_').lower()}.png")
    plt.close()

def main():
    metrics = load_metrics("logs/metrics_log.json")

    # Create output folder if not exists
    import os
    os.makedirs("plots", exist_ok=True)

    plot_metric("Classification Loss", "train_cls_loss", "val_cls_loss", metrics)
    plot_metric("Segmentation Loss", "train_seg_loss", "val_seg_loss", metrics)
    plot_metric("Accuracy", "train_acc", "val_acc", metrics)
    plot_metric("F1 Score", "train_f1", "val_f1", metrics)
    plot_metric("Dice Score", "train_dice", "val_dice", metrics)

    print("✅ Plots saved in the `plots/` directory")

if __name__ == "__main__":
    main()
