import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(features, features*2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(features*2, features*4)

        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = conv_block(features*4, features*2)
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = conv_block(features*2, features)

        self.final = nn.Conv2d(features, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.final(d1)
