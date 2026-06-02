from ultralytics import YOLO
import csv
import os

os.makedirs("outputs/csv", exist_ok=True)

model = YOLO("models/yolov8m.pt")

results = model.track(
    source="data/videos/test.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

frame_id = 0

csv_path = "outputs/csv/tracking_xy.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    # =========================
    # ⭐ 增加 confidence
    # =========================
    writer.writerow([
        "frame",
        "track_id",
        "class",
        "conf",
        "x",
        "y"
    ])

    for result in results:

        frame_id += 1

        if result.boxes.id is None:
            continue

        ids = result.boxes.id.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xywh.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for track_id, cls, box, conf in zip(
            ids, classes, boxes, confs
        ):

            x = float(box[0])
            y = float(box[1])

            # ⭐ 可选：过滤低置信度噪声（推荐）
            if conf < 0.3:
                continue

            writer.writerow([
                frame_id,
                int(track_id),
                result.names[int(cls)],
                float(conf),
                x,
                y
            ])

print("\n保存完成")
print("CSV文件位置:")
print(csv_path)