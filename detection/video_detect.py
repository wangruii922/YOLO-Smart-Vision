from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")

model.predict(
    source="data/videos/test.mp4",
    save=True,
    conf=0.25,
    project="outputs",
    name="videos",
    exist_ok=True
)

print("视频检测完成")
print("结果保存在 outputs/videos/")