from ultralytics import YOLO
import os

model = YOLO("models/yolov8m.pt")

img_path = input("请输入图片路径：")

if not os.path.exists(img_path):

    print("文件不存在")

else:

    model.predict(
        source=img_path,
        save=True,
        conf=0.25,
        project="outputs",
        name="images",
        exist_ok=True
    )

    print("检测完成")
    print("结果保存在 outputs/images/")