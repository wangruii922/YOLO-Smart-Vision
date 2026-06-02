from ultralytics import YOLO
import os
import shutil

output_dir = "outputs/images"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

model = YOLO("models/yolov8m.pt")

results = model.predict(
    source="data/images",
    save=True,
    conf=0.25,
    project="outputs",
    name="images",
    exist_ok=True
)

print("\n检测结果：")

for result in results:

    filename = os.path.basename(result.path)

    print(f"\n图片: {filename}")

    for box in result.boxes:

        cls = int(box.cls)
        conf = float(box.conf)

        print(
            f"{result.names[cls]}  {conf:.2f}"
        )

print("\n全部完成")
print("结果保存至:")
print(output_dir)