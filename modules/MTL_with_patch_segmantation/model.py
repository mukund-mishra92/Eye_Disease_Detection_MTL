import torch
import torch.nn as nn
from torchvision import models
from modules.MTL_with_Router.unet import UNet

class HybridMultiTaskModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()

        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(512, num_classes)
        self.segmentor = UNet(in_channels=3, out_channels=1)

    def forward(self, image_full, patches):
        cls_features = self.feature_extractor(image_full)
        cls_logits = self.classifier(cls_features.view(cls_features.size(0), -1))

        b, n, c, h, w = patches.shape
        patches_flat = patches.view(-1, c, h, w)
        seg_outputs = self.segmentor(patches_flat)
        seg_outputs = seg_outputs.view(b, -1, 1, h, w)

        return cls_logits, seg_outputs
