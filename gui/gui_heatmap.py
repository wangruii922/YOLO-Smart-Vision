import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout
)

from PyQt5.QtGui import (
    QPixmap,
    QImage
)

from PyQt5.QtCore import Qt

from analytics.behavior_heatmap import BehaviorAnalysisSystem
from analytics.overlay_heatmap import HeatmapOverlayGenerator
from analytics.roi_selector import ROISelector


class HeatmapPage(QWidget):

    def __init__(self):

        super().__init__()

        self.b = None
        self.rois = []

        self.init_ui()

    def init_ui(self):

        layout = QHBoxLayout(self)

        left = QVBoxLayout()

        # =====================
        # Heatmap Module (FINAL)
        # =====================
        self.btn_load = QPushButton("加载CSV")
        self.btn_roi = QPushButton("ROI框选")
        self.btn_car = QPushButton("Car热力图")
        self.btn_person = QPushButton("Person热力图")
        self.btn_overlay = QPushButton("Overlay Heatmap")

        self.btn_load.clicked.connect(self.load_csv)
        self.btn_roi.clicked.connect(self.select_roi)
        self.btn_car.clicked.connect(
            lambda: self.class_heatmap("car")
        )
        self.btn_person.clicked.connect(
            lambda: self.class_heatmap("person")
        )
        self.btn_overlay.clicked.connect(self.overlay_heatmap)

        # =====================
        # Display
        # =====================
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border:1px solid black;")

        left.addWidget(self.btn_load)
        left.addWidget(self.btn_roi)
        left.addWidget(self.btn_car)
        left.addWidget(self.btn_person)
        left.addWidget(self.btn_overlay)
        left.addWidget(self.image_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addLayout(left, 3)
        layout.addWidget(self.log, 1)

    # =========================
    # Load CSV
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

        self.b = BehaviorAnalysisSystem(path)

        self.log.append("[OK] CSV Loaded")
        self.log.append(path)

    # =========================
    # ROI selection (KEEP for later stages)
    # =========================
    def select_roi(self):

        if self.b is None:
            self.log.append("[ERROR] 请先加载CSV")
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景图",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not image_path:
            return

        selector = ROISelector(image_path)

        self.rois = selector.select()

        self.log.append(f"[OK] ROI数量: {len(self.rois)}")

        for i, roi in enumerate(self.rois):
            self.log.append(f"ROI {i+1}: {roi}")

    # =========================
    # Class heatmap (FINAL USE ONLY)
    # =========================
    def class_heatmap(self, cls):

        if self.b is None:
            self.log.append("[ERROR] 请先加载CSV")
            return

        heat = self.b.class_heatmap(cls)

        if heat is None:
            self.log.append(f"[WARN] {cls} Heatmap为空")
            return

        self.show_heatmap_array(heat)

        self.log.append(f"[OK] {cls} Heatmap完成")

    # =========================
    # Global overlay (FINAL USE ONLY)
    # =========================
    def overlay_heatmap(self):

        if self.b is None:
            self.log.append("[ERROR] 请先加载CSV")
            return

        bg_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择背景图",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not bg_path:
            return

        heatmap = self.b.global_heat()

        if heatmap is None:
            self.log.append("[WARN] Heatmap为空")
            return

        generator = HeatmapOverlayGenerator(heatmap)

        save_path = "outputs/heatmap/overlay_heatmap.png"

        overlay = generator.generate_overlay(
            bg_path,
            save_path
        )

        self.show_cv_image(overlay)

        self.log.append("[OK] Overlay Heatmap完成")

    # =========================
    # heatmap -> image
    # =========================
    def show_heatmap_array(self, heat):

        heat = np.uint8(heat * 255)

        heat = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        self.show_cv_image(heat)

    # =========================
    # OpenCV -> Qt
    # =========================
    def show_cv_image(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape

        qimg = QImage(
            img.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio
            )
        )