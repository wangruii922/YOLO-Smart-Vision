import cv2
import time
import pandas as pd
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QFileDialog, QTextEdit,
    QHBoxLayout, QVBoxLayout,
    QSizePolicy
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer


model = YOLO("models/yolov8m.pt")


class VehiclePage(QWidget):

    def __init__(self):

        super().__init__()

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.running = False
        self.paused = False

        self.total_ids = set()

        # =========================
        # trajectory cache (核心数据层)
        # =========================
        self.tracking_data = []
        self.frame_id = 0

        self.prev_time = time.time()
        self.fps = 0

        self.init_ui()

    # =========================
    # UI (Dashboard Architecture)
    # =========================
    def init_ui(self):

        layout = QHBoxLayout(self)

        left = QVBoxLayout()

        # ================= Controls =================
        self.btn_open = QPushButton("选择视频")
        self.btn_start = QPushButton("开始检测")
        self.btn_pause = QPushButton("暂停")
        self.btn_reset = QPushButton("重置")

        # ===== Dashboard extension =====
        self.btn_export_csv = QPushButton("导出Tracking CSV")
        self.btn_cluster = QPushButton("Trajectory Clustering")

        self.btn_open.clicked.connect(self.open_video)
        self.btn_start.clicked.connect(self.start_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_reset.clicked.connect(self.reset_video)

        self.btn_export_csv.clicked.connect(self.save_tracking_csv)
        self.btn_cluster.clicked.connect(self.open_cluster_module)

        # ================= Video Display =================
        self.video_label = QLabel("未加载视频")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.video_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored
        )
        self.video_label.setMinimumSize(720, 420)
        self.video_label.setStyleSheet("border:1px solid black;")

        # ================= Layout =================
        left.addWidget(self.btn_open)
        left.addWidget(self.btn_start)
        left.addWidget(self.btn_pause)
        left.addWidget(self.btn_reset)
        left.addWidget(self.btn_export_csv)
        left.addWidget(self.btn_cluster)
        left.addWidget(self.video_label)

        # ================= Right Panels =================
        self.info = QTextEdit()
        self.info.setReadOnly(True)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(left, 3)
        layout.addWidget(self.info, 1)
        layout.addWidget(self.log, 1)

    # =========================
    # open video
    # =========================
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

        self.total_ids.clear()
        self.tracking_data.clear()
        self.frame_id = 0

        self.log.append("[OK] Video loaded")

    # =========================
    # start
    # =========================
    def start_video(self):

        if self.cap is None:
            return

        self.running = True
        self.paused = False
        self.timer.start(30)

    # =========================
    # pause
    # =========================
    def pause_video(self):
        self.paused = True

    # =========================
    # reset
    # =========================
    def reset_video(self):

        self.running = False
        self.paused = False
        self.timer.stop()

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.total_ids.clear()
        self.tracking_data.clear()
        self.frame_id = 0

        self.video_label.setText("RESET DONE")

        self.log.append("[RESET] System reset complete")

    # =========================
    # main loop
    # =========================
    def update_frame(self):

        if not self.running or self.paused:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.timer.stop()
            self.save_tracking_csv()
            self.log.append("[END] Video finished & CSV saved")
            return

        self.frame_id += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        result = results[0]
        img = result.plot()

        # ================= FPS =================
        now = time.time()
        self.fps = 1 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        cv2.putText(
            img,
            f"FPS:{self.fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # ================= tracking data =================
        if result.boxes.id is not None:

            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            for i in range(len(ids)):

                x, y, w, h = boxes[i]
                tid = int(ids[i])
                c = int(cls[i])

                self.total_ids.add(tid)

                # center point (更适合 trajectory clustering)
                self.tracking_data.append([
                    self.frame_id,
                    tid,
                    c,
                    float(x),
                    float(y)
                ])

        # ================= display =================
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        qimg = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(qimg)

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.info.setText(
            f"FPS: {self.fps:.1f}\n"
            f"Objects: {len(result.boxes)}\n"
            f"Track IDs: {len(self.total_ids)}\n"
            f"Frame: {self.frame_id}"
        )

    # =========================
    # save CSV
    # =========================
    def save_tracking_csv(self):

        if not self.tracking_data:
            self.log.append("[WARN] No tracking data")
            return

        df = pd.DataFrame(
            self.tracking_data,
            columns=["frame", "track_id", "class", "x", "y"]
        )

        save_path = "outputs/csv/tracking_xy.csv"
        df.to_csv(save_path, index=False)

        self.log.append(f"[OK] CSV saved:\n{save_path}")

    # =========================
    # Dashboard hook
    # =========================
    def open_cluster_module(self):

        self.log.append("[INFO] Opening Trajectory Clustering module...")
        self.log.append("[NEXT] Connect to TrajectoryClusteringGUI")