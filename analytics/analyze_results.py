from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")

results = model.predict(
    source="data/images",
    conf=0.25
)

result = results[0]

print("图片路径:")
print(result.path)

print("\n图片尺寸:")
print(result.orig_shape)

print("\n检测到的目标数量:")
print(len(result.boxes))

print("\n详细信息:\n")

for i, box in enumerate(result.boxes):

    cls = int(box.cls)
    conf = float(box.conf)

    x1, y1, x2, y2 = box.xyxy[0]

    print(f"目标{i+1}")

    print("类别:", result.names[cls])

    print("置信度:", round(conf, 3))

    print(
        f"位置: ({int(x1)},{int(y1)}) -> ({int(x2)},{int(y2)})"
    )

    print("-"*40)
    