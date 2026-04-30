import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

from models.model import DeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load trained model
model = DeepfakeDetector()
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -------- IMAGE DETECTION --------
def detect_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)
        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs,1)

    label = "FAKE IMAGE" if pred.item() == 1 else "REAL IMAGE"

    print("\n===== IMAGE RESULT =====")
    print("Prediction:", label)
    
    print("========================")


# -------- VIDEO DETECTION --------
def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    fake_frames = 0
    checked_frames = 0
    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # analyze every 15th frame
        if frame_id % 15 == 0:

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():

                output = model(image)
                pred = torch.argmax(output)

            if pred.item() == 1:
                fake_frames += 1

            checked_frames += 1

        frame_id += 1

    cap.release()

    fake_ratio = fake_frames / max(1,checked_frames)

    if fake_ratio > 0.3:
        result = "DEEPFAKE VIDEO"
    else:
        result = "REAL VIDEO"

    print("\n===== VIDEO RESULT =====")
    print("Prediction:", result)
    print("Fake frame ratio:", round(fake_ratio*100,2), "%")
    print("========================")


# -------- MAIN --------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict.py <file_path>")
        sys.exit()

    file_path = sys.argv[1]

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".jpg",".jpeg",".png"]:
        detect_image(file_path)

    elif ext in [".mp4",".avi",".mov"]:
        detect_video(file_path)

    else:
        print("Unsupported file type")