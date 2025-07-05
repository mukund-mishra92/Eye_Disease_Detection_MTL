import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassifierHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)