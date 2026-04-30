import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.model import DeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _extract_video_frames(video_dir: Path, out_dir: Path, stride: int = 10) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for vf in video_dir.rglob("*"):
        if vf.suffix.lower() not in VIDEO_EXTS or not vf.is_file():
            continue
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % stride == 0:
                rel = vf.relative_to(video_dir).as_posix().replace("/", "_")
                out_name = f"{Path(rel).stem}_{frame_id}.jpg"
                cv2.imwrite(str(out_dir / out_name), frame)
                extracted += 1
            frame_id += 1
        cap.release()
    return extracted


def prepare_video_dataset(dataset_root: Path) -> None:
    """
    Add video-derived frames into dataset/frames/{real,fake} automatically.
    Supports both:
      - dataset/real_videos + dataset/fake_videos
      - dataset/videos/real + dataset/videos/fake
    """
    frames_real = dataset_root / "frames" / "real"
    frames_fake = dataset_root / "frames" / "fake"

    candidates = [
        (dataset_root / "real", dataset_root / "fake"),
        (dataset_root / "real_videos", dataset_root / "fake_videos"),
        (dataset_root / "videos" / "real", dataset_root / "videos" / "fake"),
    ]

    for real_videos, fake_videos in candidates:
        if real_videos.exists() and fake_videos.exists():
            # Refresh generated frame cache so training reflects latest videos.
            for old in frames_real.glob("*.jpg"):
                old.unlink(missing_ok=True)
            for old in frames_fake.glob("*.jpg"):
                old.unlink(missing_ok=True)
            real_n = _extract_video_frames(real_videos, frames_real, stride=12)
            fake_n = _extract_video_frames(fake_videos, frames_fake, stride=12)
            print(f"Video dataset added: real_frames={real_n}, fake_frames={fake_n}")
            return
    print("Video dataset folders not found, skipping video extraction.")


class DeepfakeDataset(Dataset):
    def __init__(self, root_dirs: list[Path]):
        self.images = []
        self.labels = []
        for root in root_dirs:
            real_path = root / "real"
            fake_path = root / "fake"
            if not real_path.exists() or not fake_path.exists():
                continue
            for img in real_path.glob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    self.images.append(str(img))
                    self.labels.append(0)
            for img in fake_path.glob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    self.images.append(str(img))
                    self.labels.append(1)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def main():
    dataset_root = Path("dataset")
    prepare_video_dataset(dataset_root)

    dataset = DeepfakeDataset([dataset_root / "images", dataset_root / "frames"])
    if len(dataset) == 0:
        raise RuntimeError(
            "No training images found. Expected dataset/images/{real,fake} "
            "and/or video frames in dataset/frames/{real,fake}."
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = DeepfakeDetector().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch: {epoch + 1}/{epochs} Loss: {running_loss / len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/deepfake_model.pth")
    print("Model saved successfully")


if __name__ == "__main__":
    main()