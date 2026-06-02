import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QSplitter,
    QSizePolicy
)

from PyQt5.QtCore import Qt

from gui.gui_images import ImagePage
from gui.gui_vehicle import VehiclePage
from gui.gui_fence import FencePage
from gui.gui_heatmap import HeatmapPage


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Smart Vision System")

        # =========================
        # ✅ 固定窗口策略（防无限放大）
        # =========================
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        self.setMaximumSize(1600, 900)

        self.init_ui()

    def init_ui(self):

        main = QWidget()
        self.setCentralWidget(main)

        layout = QHBoxLayout(main)

        # =========================
        # 左侧菜单（固定宽度）
        # =========================
        self.menu = QListWidget()
        self.menu.setFixedWidth(200)

        self.menu.addItem("图片检测")
        self.menu.addItem("视频跟踪")
        self.menu.addItem("电子围栏")
        self.menu.addItem("热力图分析")   # ⭐ 新增

        self.menu.currentRowChanged.connect(self.switch_page)

        # =========================
        # 页面栈
        # =========================
        self.stack = QStackedWidget()

        self.stack.addWidget(ImagePage())
        self.stack.addWidget(VehiclePage())
        self.stack.addWidget(FencePage())
        self.stack.addWidget(HeatmapPage())   # ⭐ 新增

        # =========================
        # ⭐ 关键修复：禁止 stack 被无限拉伸
        # =========================
        self.stack.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        # =========================
        # splitter（稳定布局）
        # =========================
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.menu)
        splitter.addWidget(self.stack)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)

        # ⭐ 防止左侧被拖没 / 右侧无限扩张
        splitter.setSizes([200, 1000])

        layout.addWidget(splitter)

        self.menu.setCurrentRow(0)

    def switch_page(self, i):
        self.stack.setCurrentIndex(i)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())