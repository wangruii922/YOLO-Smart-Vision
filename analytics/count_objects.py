from ultralytics import YOLO
import shutil
import os

# 删除旧结果
if os.path.exists("outputs/images"):
    shutil.rmtree("outputs/images")

# 创建目录
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

model = YOLO("models/yolov8m.pt")

results = model.predict(
    source="data/images",
    conf=0.25,
    save=True,
    project="outputs",
    name="images",
    exist_ok=True
)

counter = {}

for result in results:

    for box in result.boxes:

        cls = int(box.cls)

        name = result.names[cls]

        counter[name] = counter.get(name, 0) + 1

result_path = "outputs/reports/result.txt"

with open(result_path, "w", encoding="utf-8") as f:

    f.write("目标统计结果\n\n")

    for k, v in sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        line = f"{k}: {v}\n"

        print(line, end="")

        f.write(line)

print("\n结果已保存到:")
print(result_path)

print("预测图片已保存到:")
print("outputs/images")