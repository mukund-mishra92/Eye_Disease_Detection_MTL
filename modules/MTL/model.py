import torch
import torch.nn as nn
import torchvision.models as models

class SharedResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # output: [B, 2048, 8, 8]

    def forward(self, x):
        return self.encoder(x)

class ClassificationHead(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.avgpool(x)
        return self.fc(x)

class SegmentationDecoder(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.up1 = self._upsample(in_channels, 512)
        self.up2 = self._upsample(512, 256)
        self.up3 = self._upsample(256, 128)
        self.up4 = self._upsample(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _upsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = SharedResNetBackbone()
        self.class_head = ClassificationHead(num_classes=num_classes)
        self.seg_head = SegmentationDecoder()

    def forward(self, x):
        features = self.backbone(x)          # [B, 2048, 8, 8]
        cls_logits = self.class_head(features)   # [B, num_classes]
        seg_logits = self.seg_head(features)     # [B, 1, 256, 256] after upsampling
        return cls_logits, seg_logits
