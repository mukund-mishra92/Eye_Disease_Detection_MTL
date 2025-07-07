import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_hybrid(model, dataloader, optimizer, device):
    model.train()
    total_cls_loss, total_seg_loss = 0, 0

    for full_img, cls_label, patch_imgs, patch_masks in tqdm(dataloader):
        full_img = full_img.to(device)
        cls_label = cls_label.to(device)
        patch_imgs = patch_imgs.to(device)
        patch_masks = patch_masks.to(device)

        optimizer.zero_grad()
        cls_logits, seg_outputs = model(full_img, patch_imgs)

        cls_loss = F.cross_entropy(cls_logits, cls_label)

        seg_outputs = torch.sigmoid(seg_outputs.squeeze(2))  # (B*N, H, W)
        seg_loss = F.binary_cross_entropy(seg_outputs, patch_masks.squeeze(2))

        loss = cls_loss + seg_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_seg_loss += seg_loss.item()

    return total_cls_loss / len(dataloader), total_seg_loss / len(dataloader)
