import torch.nn as nn

class SegmenterHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmenterHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)
