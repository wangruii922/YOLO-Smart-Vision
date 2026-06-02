from ultralytics import YOLO
import cv2
import os

# 创建输出目录
os.makedirs("outputs/videos", exist_ok=True)

model = YOLO("models/yolov8m.pt")

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle"
]

counted_ids = set()

vehicle_count = 0

line_y = 400

video_path = "data/videos/test.mp4"

results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    stream=True,
    persist=True
)

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

save_path = "outputs/videos/vehicle_count_result.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    save_path,
    fourcc,
    fps,
    (width, height)
)

for result in results:

    frame = result.orig_img

    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (0, 255, 0),
        3
    )

    if result.boxes.id is not None:

        ids = result.boxes.id.cpu().numpy()

        classes = result.boxes.cls.cpu().numpy()

        boxes = result.boxes.xywh.cpu().numpy()

        for track_id, cls, box in zip(
            ids,
            classes,
            boxes
        ):

            class_name = result.names[int(cls)]

            if class_name not in vehicle_classes:
                continue

            center_x = int(box[0])
            center_y = int(box[1])

            if abs(center_y - line_y) < 10:

                if track_id not in counted_ids:

                    counted_ids.add(track_id)

                    vehicle_count += 1

                    print(
                        f"车辆通过: ID={int(track_id)}"
                    )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    out.write(frame)

    cv2.imshow(
        "Vehicle Counting",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

out.release()

cv2.destroyAllWindows()

print("\n统计完成")
print("车辆总数:", vehicle_count)
print("结果保存至:")
print(save_path)