import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=512, out_channels=5):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 256)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(256, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(512, 256)

        self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)       # [B, 256, 8, 8]
        x2 = self.enc2(self.pool1(x1))  # [B, 128, 4, 4]
        x3 = self.bottleneck(self.pool2(x2))  # [B, 64, 2, 2]

        x = self.up2(x3)        # [B, 128, 4, 4]
        x = self.dec2(torch.cat([x, x2], dim=1))  # [B, 128, 4, 4]
        x = self.up1(x)         # [B, 256, 8, 8]
        x = self.dec1(torch.cat([x, x1], dim=1))  # [B, 256, 8, 8]

        return F.interpolate(self.final_conv(x), scale_factor=16, mode='bilinear', align_corners=False)
