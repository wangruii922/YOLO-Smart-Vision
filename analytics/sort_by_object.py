from ultralytics import YOLO
import shutil
import os

model = YOLO("models/yolov8m.pt")

results = model.predict(
    source="data/images",
    conf=0.25
)

root_dir = "outputs/classified_images"

os.makedirs(root_dir, exist_ok=True)

for result in results:

    filename = os.path.basename(result.path)

    classes_in_image = set()

    for box in result.boxes:

        cls = int(box.cls)

        name = result.names[cls]

        classes_in_image.add(name)

    for class_name in classes_in_image:

        save_dir = os.path.join(
            root_dir,
            class_name
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        shutil.copy(
            result.path,
            os.path.join(
                save_dir,
                filename
            )
        )

print("\n分类完成")

print("保存位置:")
print(root_dir)