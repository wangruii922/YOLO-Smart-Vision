import cv2
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QFileDialog, QTextEdit,
    QHBoxLayout, QVBoxLayout,
    QSizePolicy
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

model = YOLO("models/yolov8m.pt")


class ImagePage(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        layout = QHBoxLayout(self)

        left = QVBoxLayout()

        self.btn = QPushButton("选择图片检测")
        self.btn.clicked.connect(self.detect_image)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # ⭐ 关键修复
        self.image_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored
        )
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border:1px solid black;")

        left.addWidget(self.btn)
        left.addWidget(self.image_label)

        self.result = QTextEdit()
        self.result.setReadOnly(True)

        layout.addLayout(left, 2)
        layout.addWidget(self.result, 1)

    def detect_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.jpg *.png *.jpeg)"
        )

        if not file_path:
            return

        results = model.predict(file_path)
        result = results[0]

        img = result.plot()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # ⭐ 关键修复：限定最大缩放
        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        text = f"检测目标: {len(result.boxes)}\n\n"

        for i, box in enumerate(result.boxes):
            cls = int(box.cls)
            conf = float(box.conf)
            text += f"{i+1}. {result.names[cls]} {conf:.2f}\n"

        self.result.setText(text)