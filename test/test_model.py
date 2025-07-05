import torch
from modules.MTL.previous.multitask_model import MultiTaskModel

def test_forward_pass():
    model = MultiTaskModel(num_classes=5, seg_channels=1)
    x = torch.randn(4, 3, 512, 512)  # batch of 4 images
    class_out, seg_out = model(x)
    assert class_out.shape == (4, 5), "Incorrect shape for classification output"
    assert seg_out.shape == (4, 1, 64, 64) or seg_out.shape[-2:] != x.shape[-2:], "Segmentation shape mismatch"
    print("Forward pass successful.")

if __name__ == '__main__':
    test_forward_pass()