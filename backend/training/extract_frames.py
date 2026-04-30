import cv2
import os

def extract_frames(video_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for video in os.listdir(video_folder):

        path = os.path.join(video_folder, video)

        cap = cv2.VideoCapture(path)

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % 10 == 0:

                name = f"{video}_{frame_id}.jpg"

                save_path = os.path.join(output_folder, name)

                cv2.imwrite(save_path, frame)

            frame_id += 1

        cap.release()


extract_frames("dataset/real_videos","frames/real")
extract_frames("dataset/fake_videos","frames/fake")