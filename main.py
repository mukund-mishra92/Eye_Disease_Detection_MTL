import json
import torch
import torchvision.transforms as T
from modules.multitask_model import MultiTaskModel
from utils.dataloader import IDRiDDataset
from training.train import train_model
from torch.optim import Adam

import yaml

# Load config
with open("config/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

#print(cfg['paths']['label_file'])

# Load label mapping
with open(cfg['paths']['label_file'], 'r') as f:
    label_dict = json.load(f)

# Transforms
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

# Datasets
train_dataset = IDRiDDataset(cfg['paths']['image_dir'], label_dict, cfg['paths']['mask_dir'], transform)
val_dataset = IDRiDDataset(cfg['paths']['image_dir'], label_dict, cfg['paths']['mask_dir'], transform)  # Use separate val split in practice

# Model
model = MultiTaskModel(
    num_classes=cfg['model']['num_classes'],
    seg_channels=cfg['model']['seg_channels']
)

# Optimizer
optimizer = Adam(model.parameters(), lr=cfg['training']['learning_rate'])

# Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    num_epochs=cfg['training']['num_epochs'],
    device=cfg['training']['device']
)

