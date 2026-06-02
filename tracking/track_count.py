from ultralytics import YOLO

model = YOLO(
    "models/yolov8m.pt"
)

class_tracks = {}

results = model.track(
    source="data/videos/test.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

for result in results:

    if result.boxes.id is None:
        continue

    ids = result.boxes.id.cpu().numpy()

    classes = result.boxes.cls.cpu().numpy()

    for track_id, cls in zip(
        ids,
        classes
    ):

        name = result.names[
            int(cls)
        ]

        if name not in class_tracks:

            class_tracks[name] = set()

        class_tracks[name].add(
            int(track_id)
        )

print("\n===================")
print("分类统计结果")
print("===================\n")

for name, ids in sorted(
    class_tracks.items()
):

    print(
        f"{name}: "
        f"{len(ids)}个独立目标"
    )