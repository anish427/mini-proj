import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DeepfakeDataset(Dataset):

    def __init__(self, csv_file, root_dir):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.root_dir + "/" + self.data.iloc[idx,0]
        label = self.data.iloc[idx,1]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label