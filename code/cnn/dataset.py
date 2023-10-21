import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class CXRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.file_paths = []
        self.labels = []

        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.file_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(class_id)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=-1)  # Convert to 3D array (H, W, C)

        if self.transform:
            img = self.transform(img)

        return img, label

def get_dataloaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalization values for ImageNet
    ])

    train_dataset = CXRDataset(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = CXRDataset(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
