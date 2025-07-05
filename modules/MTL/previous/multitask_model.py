from modules.MTL.previous.shared_backbone import SharedBackbone
from modules.MTL.previous.classifier_head import ClassifierHead
from modules.MTL.previous.segmenter_head import SegmenterHead

import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes, seg_channels):
        super(MultiTaskModel, self).__init__()
        self.backbone = SharedBackbone()
        self.classifier = ClassifierHead(512, num_classes)
        self.segmenter = SegmenterHead(512, seg_channels)

    def forward(self, x):
        shared_features = self.backbone(x)
        class_out = self.classifier(shared_features)
        seg_out = self.segmenter(shared_features)
        return class_out, seg_out
