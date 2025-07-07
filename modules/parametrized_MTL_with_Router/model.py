# === model.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from modules.MTL_with_Router.unet import UNet

class MultiTaskModel(nn.Module):
    def __init__(self, num_disease_classes=5, num_segmentation_channels=5, image_size=512):
        super(MultiTaskModel, self).__init__()
        self.image_size = image_size

        resnet = models.resnet18(pretrained=True)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_disease_classes)
        )

        self.segmentation_experts = nn.ModuleList([
            UNet(in_channels=512, out_channels=num_segmentation_channels,
                 skip_channels=[256, 128, 64, 64])
            for _ in range(num_disease_classes)
        ])

    def forward(self, x, force_class=None):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        logits = self.classifier(self.avgpool(e5))
        selected_classes = force_class if force_class is not None else torch.argmax(logits, dim=1)

        batch_size = x.size(0)
        seg_outputs = torch.zeros(batch_size, 5, self.image_size, self.image_size).to(x.device)

        for i in range(batch_size):
            class_id = selected_classes[i].item()
            expert = self.segmentation_experts[class_id]
            seg_output = expert(
                e5[i].unsqueeze(0), skips=[
                    e4[i].unsqueeze(0), e3[i].unsqueeze(0),
                    e2[i].unsqueeze(0), e1[i].unsqueeze(0)
                ]
            )
            seg_outputs[i] = seg_output.squeeze(0)

        return logits, selected_classes, seg_outputs
