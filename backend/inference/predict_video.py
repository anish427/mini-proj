import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.model import DeepfakeDetector


model = DeepfakeDetector()
model.load_state_dict(torch.load("models/deepfake_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def detect_video(video_path):

    cap=cv2.VideoCapture(video_path)

    fake=0
    total=0

    while True:

        ret,frame=cap.read()

        if not ret:
            break

        if total%15==0:

            img=Image.fromarray(frame)

            img=transform(img).unsqueeze(0)

            output=model(img)

            pred=torch.argmax(output)

            if pred==1:
                fake+=1

        total+=1

    cap.release()

    ratio=fake/max(1,total)

    if ratio>0.3:
        return "DEEPFAKE VIDEO"
    else:
        return "REAL VIDEO"


print(detect_video("test_video.mp4"))