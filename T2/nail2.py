import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_names = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_names[idx])
        label_name = os.path.join(self.label_folder, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

dataset = SegmentationDataset(r'D:\pythonProject\T2\archive\images', r'D:\pythonProject\T2\archive\labels', transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
