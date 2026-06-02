from ultralytics import YOLO
import os

os.makedirs(
    "outputs/videos",
    exist_ok=True
)

model = YOLO(
    "models/yolov8m.pt"
)

model.predict(
    source=0,
    show=True,
    save=True,
    conf=0.25,
    project="outputs",
    name="videos",
    exist_ok=True
)

print(
    "\n视频已保存至 outputs/videos/"
)