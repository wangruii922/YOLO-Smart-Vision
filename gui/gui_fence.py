import cv2
import time

from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer


model = YOLO("models/yolov8m.pt")


class FencePage(QWidget):

    def __init__(self):

        super().__init__()

        self.cap = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.running = False
        self.paused = False

        self.total_intrusions = 0

        self.prev_time = time.time()
        self.fps = 0

        self.roi = (200, 150, 900, 650)

        self.init_ui()

    def init_ui(self):

        layout = QHBoxLayout(self)

        left = QVBoxLayout()

        # =========================
        # 控制按钮
        # =========================
        self.btn_open = QPushButton("选择视频")
        self.btn_start = QPushButton("开始")
        self.btn_pause = QPushButton("暂停")
        self.btn_reset = QPushButton("重置")

        self.btn_open.clicked.connect(self.open_video)
        self.btn_start.clicked.connect(self.start_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_reset.clicked.connect(self.reset_video)

        # =========================
        # ⭐ 核心修复：固定视频窗口
        # =========================
        self.video_label = QLabel("未加载视频")

        self.video_label.setAlignment(Qt.AlignCenter)

        self.video_label.setFixedSize(900, 650)

        self.video_label.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed
        )

        self.video_label.setStyleSheet(
            "border:2px solid black;"
            "background-color: #111;"
        )

        left.addWidget(self.btn_open)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_pause)
        left.addWidget(self.btn_reset)
        left.addWidget(self.video_label)

        # =========================
        # 信息面板
        # =========================
        self.info = QTextEdit()
        self.info.setReadOnly(True)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(left, 3)
        layout.addWidget(self.info, 1)
        layout.addWidget(self.log, 1)

    def write_log(self, msg):

        now = time.strftime("%H:%M:%S")
        self.log.append(f"[{now}] {msg}")

    def open_video(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video (*.mp4 *.avi)"
        )

        if not path:
            return

        self.cap = cv2.VideoCapture(path)

        self.total_intrusions = 0

        self.write_log("视频加载成功")

    def start_video(self):

        if self.cap is None:
            return

        self.running = True
        self.paused = False

        self.timer.start(30)

        self.write_log("电子围栏启动")

    def pause_video(self):

        self.paused = True
        self.write_log("电子围栏暂停")

    def reset_video(self):

        self.running = False
        self.paused = False

        self.timer.stop()

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.total_intrusions = 0

        self.video_label.setText("已重置")

        self.write_log("系统重置")

    def update_frame(self):

        if not self.running or self.paused:
            return

        current_time = time.time()

        self.fps = 1 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time

        ret, frame = self.cap.read()

        if not ret:
            self.timer.stop()
            self.write_log("视频结束")
            return

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        result = results[0]

        x1, y1, x2, y2 = self.roi

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        intrusion_now = 0

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:

                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                if x1 < cx < x2 and y1 < cy < y2:

                    intrusion_now += 1
                    self.total_intrusions += 1

                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"FPS:{self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # ⭐ 关键：只适配 label，不反向影响 layout
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.info.setText(
            f"FPS: {self.fps:.1f}\n\n"
            f"当前入侵: {intrusion_now}\n\n"
            f"累计入侵: {self.total_intrusions}"
        )