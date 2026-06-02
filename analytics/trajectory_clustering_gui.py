import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from collections import defaultdict

from PyQt5.QtWidgets import (
    QWidget, QPushButton, QFileDialog,
    QTextEdit, QLabel, QHBoxLayout, QVBoxLayout
)

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class TrajectoryClusteringGUI(QWidget):

    def __init__(self):
        super().__init__()

        self.df = None
        self.cluster_map = {}
        self.trajectories = {}

        self.init_ui()

    # =========================
    # UI
    # =========================
    def init_ui(self):

        layout = QHBoxLayout(self)

        left = QVBoxLayout()

        self.btn_load = QPushButton("加载CSV")
        self.btn_cluster = QPushButton("开始聚类")
        self.btn_export = QPushButton("导出视频")

        self.btn_load.clicked.connect(self.load_csv)
        self.btn_cluster.clicked.connect(self.run_cluster)
        self.btn_export.clicked.connect(self.export_video)

        self.image_label = QLabel("结果显示")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border:1px solid black;")

        left.addWidget(self.btn_load)
        left.addWidget(self.btn_cluster)
        left.addWidget(self.btn_export)
        left.addWidget(self.image_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(left, 3)
        layout.addWidget(self.log, 1)

    # =========================
    # load CSV
    # =========================
    def load_csv(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "CSV",
            "",
            "CSV (*.csv)"
        )

        if not path:
            return

        self.df = pd.read_csv(path)

        self.log.append(f"[OK] CSV loaded: {path}")

    # =========================
    # build trajectories
    # =========================
    def build_trajectories(self):

        traj = defaultdict(list)

        for _, row in self.df.iterrows():

            traj[row["track_id"]].append(
                (row["frame"], row["x"], row["y"])
            )

        for k in traj:
            traj[k] = sorted(traj[k], key=lambda x: x[0])

        self.trajectories = traj

    # =========================
    # clustering (DBSCAN)
    # =========================
    def run_cluster(self):

        if self.df is None:
            self.log.append("[ERROR] No CSV loaded")
            return

        self.build_trajectories()

        features = []
        ids = []

        for tid, traj in self.trajectories.items():

            coords = np.array([(x, y) for _, x, y in traj])

            if len(coords) < 5:
                continue

            feat = np.diff(coords, axis=0).flatten()

            features.append(feat)
            ids.append(tid)

        if len(features) == 0:
            self.log.append("[WARN] No valid trajectories")
            return

        model = DBSCAN(eps=12, min_samples=3)
        labels = model.fit_predict(np.array(features))

        self.cluster_map = {
            tid: label for tid, label in zip(ids, labels)
        }

        self.log.append(f"[OK] clustering done: {len(set(labels))} clusters")

        self.show_result()

    # =========================
    # show simple visualization
    # =========================
    def show_result(self):

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        for tid, traj in self.trajectories.items():

            if tid not in self.cluster_map:
                continue

            label = self.cluster_map[tid]

            color = self.get_color(label)

            for i in range(1, len(traj)):

                _, x1, y1 = traj[i - 1]
                _, x2, y2 = traj[i]

                cv2.line(
                    canvas,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )

        self.show_image(canvas)

    # =========================
    # color mapping
    # =========================
    def get_color(self, label):

        if label == -1:
            return (0, 0, 255)

        np.random.seed(label * 99 + 7)

        return tuple(np.random.randint(0, 255, 3).tolist())

    # =========================
    # export video
    # =========================
    def export_video(self):

        if self.df is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存视频",
            "",
            "MP4 (*.mp4)"
        )

        if not save_path:
            return

        width, height = 1280, 720
        fps = 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_max = int(self.df["frame"].max())

        for f in range(frame_max):

            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            for tid, traj in self.trajectories.items():

                if tid not in self.cluster_map:
                    continue

                color = self.get_color(self.cluster_map[tid])

                pts = [(x, y) for frame, x, y in traj if frame <= f]

                for i in range(1, len(pts)):

                    cv2.line(
                        canvas,
                        (int(pts[i-1][0]), int(pts[i-1][1])),
                        (int(pts[i][0]), int(pts[i][1])),
                        color,
                        2
                    )

            out.write(canvas)

        out.release()
        self.log.append(f"[OK] video saved: {save_path}")

    # =========================
    # show image
    # =========================
    def show_image(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape

        qimg = QImage(img.data, w, h, ch*w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio)
        )