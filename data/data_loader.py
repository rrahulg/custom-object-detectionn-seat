
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SeatDetectionDataset(Dataset):
    def __init__(self, img_folder, label_folder, img_size=640, transform=None):
        """
        Custom PyTorch Dataset for loading seat detection images and labels.

        Args:
        - img_folder (str): Path to the image directory.
        - label_folder (str): Path to the YOLO label directory.
        - img_size (int): Target image size (default 640x640 for YOLO).
        - transform (callable, optional): Optional transform to apply to images.
        """
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.img_size = img_size
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size)) / 255.0  # Normalize to [0,1]
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32)

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    labels.append(values)  # [class_id, x_center, y_center, width, height]
        
        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

        return image, labels

# Example usage
if __name__ == "__main__":
    dataset = SeatDetectionDataset(img_folder="../train/images", label_folder="../train/labels")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
