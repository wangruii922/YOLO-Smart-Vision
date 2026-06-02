from ultralytics import YOLO
import os

os.makedirs(
    "outputs/videos",
    exist_ok=True
)

model = YOLO(
    "models/yolov8m.pt"
)

track_ids = set()

results = model.track(
    source="data/videos/test.mp4",
    tracker="bytetrack.yaml",
    save=True,
    persist=True,
    stream=True,
    project="outputs",
    name="videos",
    exist_ok=True
)

for result in results:

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy()

        for track_id in ids:

            track_ids.add(
                int(track_id)
            )

print("\n===================")
print("统计结果")
print("===================")

print(
    "独立目标数量:",
    len(track_ids)
)

print("目标ID列表:")

print(
    sorted(track_ids)
)

print("\n跟踪视频保存在:")
print("outputs/videos/")