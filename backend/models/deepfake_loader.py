import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, image_size=224, mode="train"):
        """
        root_dir structure:

        dataset/
            real/
                img1.jpg
                img2.jpg
            fake/
                img1.jpg
                img2.jpg
        """

        self.root_dir = root_dir
        self.mode = mode
        self.image_paths = []
        self.labels = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        real_images = glob.glob(os.path.join(real_dir, "*"))
        fake_images = glob.glob(os.path.join(fake_dir, "*"))

        for img in real_images:
            self.image_paths.append(img)
            self.labels.append(0)

        for img in fake_images:
            self.image_paths.append(img)
            self.labels.append(1)

        if mode == "train":

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)