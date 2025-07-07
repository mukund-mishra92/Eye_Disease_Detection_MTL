import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNet, self).__init__()

        self.up1 = self.upsample_block(in_channels + skip_channels[0], 256)
        self.up2 = self.upsample_block(256 + skip_channels[1], 128)
        self.up3 = self.upsample_block(128 + skip_channels[2], 64)
        self.up4 = self.upsample_block(64 + skip_channels[3], 64)

        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # (256 → 512)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skips):
        # skips: [e4, e3, e2, e1] = [256, 128, 64, 64]
        # Input x: [B, 512, 8, 8]
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # → 16x16
        x = torch.cat([x, self._match_size(skips[0], x)], dim=1)
        x = self.up1(x)  # → [B, 256, 16, 16]

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # → 32x32
        x = torch.cat([x, self._match_size(skips[1], x)], dim=1)
        x = self.up2(x)  # → [B, 128, 32, 32]

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # → 64x64
        x = torch.cat([x, self._match_size(skips[2], x)], dim=1)
        x = self.up3(x)  # → [B, 64, 64, 64]

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # → 128x128
        x = torch.cat([x, self._match_size(skips[3], x)], dim=1)
        x = self.up4(x)  # → [B, 64, 128, 128]

        x = self.up5(x)  # → [B, 32, 256, 256]
        return self.final_conv(x)  # → [B, num_classes, 512, 512]

    def _match_size(self, skip, target):
        return F.interpolate(skip, size=target.shape[2:], mode='bilinear', align_corners=True)
