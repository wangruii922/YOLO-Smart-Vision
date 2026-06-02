from ultralytics import YOLO

model = YOLO(
    "models/yolov8m.pt"
)

results = model.predict(
    source="data/images/test.jpg",
    conf=0.1
)

result = results[0]

print("\nResult对象类型:")

print(type(result))

print("\nBoxes对象:")

print(result.boxes)

print("\n检测目标数量:")

print(len(result.boxes))

if len(result.boxes) > 0:

    first_box = result.boxes[0]

    print("\n第一个目标信息")

    print("类别ID:")
    print(int(first_box.cls))

    print("置信度:")
    print(float(first_box.conf))

    print("坐标:")
    print(first_box.xyxy)