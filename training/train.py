import torch
from torch.utils.data import DataLoader
from utils.metrics import accuracy, dice_score
from training.losses import multitask_loss
from tqdm import tqdm


def train_model(model, train_dataset, val_dataset, optimizer, num_epochs=10, device='cuda'):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels, masks in tqdm(train_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            class_out, seg_out = model(images)
            loss = multitask_loss(class_out, labels, seg_out, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        accs, dices = [], []
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                class_out, seg_out = model(images)
                accs.append(accuracy(class_out, labels))
                dices.append(dice_score(seg_out, masks))

        print(f"Validation Accuracy: {sum(accs)/len(accs):.4f}, Dice Score: {sum(dices)/len(dices):.4f}")
