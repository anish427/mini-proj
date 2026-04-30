import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):

    def __init__(self, root_dir):

        self.image_paths = []
        self.labels = []
        

        real_path = os.path.join(root_dir, "real")
        fake_path = os.path.join(root_dir, "fake")

        for img in os.listdir(real_path):
            self.image_paths.append(os.path.join(real_path, img))
            self.labels.append(0)

        for img in os.listdir(fake_path):
            self.image_paths.append(os.path.join(fake_path, img))
            self.labels.append(1)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label