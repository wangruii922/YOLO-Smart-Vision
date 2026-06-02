from ultralytics import YOLO

model = YOLO("models/yolov8m.pt")

results = model.predict(
    source="data/images/test.jpg",
    save=True,
    conf=0.25,
    project="outputs",
    name="images",
    exist_ok=True
)

result = results[0]

print("\n检测结果：")

for box in result.boxes:

    cls = int(box.cls)
    conf = float(box.conf)

    name = result.names[cls]

    print(f"{name}: {conf:.2f}")

print("\n检测完成")
print("图片保存在 outputs/images/")