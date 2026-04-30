from fastapi import FastAPI, UploadFile
import shutil

from inference.predict_image import predict
from inference.video_detector import detect_video

app = FastAPI()

@app.post("/detect-image")

async def detect_image(file:UploadFile):

    path="temp.jpg"

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    result=predict(path)

    return {"result":result}

@app.post("/detect-video")

async def detect_video_api(file:UploadFile):

    path="temp.mp4"

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    result=detect_video(path)

    return {"result":result}