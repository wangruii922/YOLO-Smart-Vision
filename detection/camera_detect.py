from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")

model.predict(
    source=0,
    show=True,
    conf=0.25
)