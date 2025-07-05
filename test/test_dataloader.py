from utils.dataloader import IDRiDDataset
import torchvision.transforms as T
import json

if __name__ == '__main__':
    with open("./data/labels.json", 'r') as f:
        labels = json.load(f)

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    dataset = IDRiDDataset("./data/images", labels, "./data/masks", transform)
    img, label, mask = dataset[0]
    print("Image shape:", img.shape)
    print("Label:", label)
    print("Mask shape:", mask.shape)
