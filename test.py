import torch
import cv2

from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="6th_video.mp4",show=True,save=True,conf=0.5)

# source - along with the file extension ex. "image.jpg", "video.mp4"
