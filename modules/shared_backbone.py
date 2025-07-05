import torch
import torch.nn as nn
import torchvision.models as models

class SharedBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(SharedBackbone, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # up to conv5_x

    def forward(self, x):
        return self.features(x)
