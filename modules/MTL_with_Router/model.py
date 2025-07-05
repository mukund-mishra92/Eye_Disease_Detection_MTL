from torchvision import models
import torch.nn as nn
import torch
from modules.MTL_with_Router.unet import UNet  # your UNet file

class MultiTaskModel(nn.Module):
    def __init__(self, num_disease_classes=5, num_segmentation_channels=5):
        super(MultiTaskModel, self).__init__()

        # Shared encoder
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (B, 512, 8, 8)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_disease_classes)
        )

        # Router-controlled segmentation experts (each a small UNet)
        self.segmentation_experts = nn.ModuleList([
            UNet(in_channels=512, out_channels=num_segmentation_channels)
            for _ in range(num_disease_classes)
        ])

    def forward(self, x, force_class=None):
        features = self.backbone(x)

        logits = self.classifier(self.avgpool(features))
        selected_classes = force_class if force_class is not None else torch.argmax(logits, dim=1)

        batch_size = x.size(0)
        seg_outputs = torch.zeros(batch_size, 5, 128, 128).to(x.device)

        for i in range(batch_size):
            class_id = selected_classes[i].item()
            expert = self.segmentation_experts[class_id]
            seg_outputs[i] = expert(features[i].unsqueeze(0)).squeeze(0)

        return logits, selected_classes, seg_outputs
