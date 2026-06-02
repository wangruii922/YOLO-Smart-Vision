import cv2
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal

import os

os.makedirs("outputs/videos", exist_ok=True)


class FenceDetectionThread(QThread):

    change_pixmap = pyqtSignal(object)
    update_info = pyqtSignal(str)

    def __init__(self, video_path):

        super().__init__()

        self.video_path = video_path

        self.model = YOLO("models/yolov8m.pt")

        # ROI区域（工业默认矩形）
        self.roi = (200, 150, 900, 650)

        self.running = True

    def run(self):

        cap = cv2.VideoCapture(self.video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        fps = cap.get(cv2.CAP_PROP_FPS)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            "outputs/videos/fence_output.mp4",
            fourcc,
            fps,
            (w, h)
        )

        total_intrusions = 0

        while self.running:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml"
            )

            result = results[0]

            x1, y1, x2, y2 = self.roi

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            intrusion_count = 0

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()

                for box in boxes:

                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)

                    if x1 < cx < x2 and y1 < cy < y2:

                        intrusion_count += 1
                        total_intrusions += 1

                        cv2.circle(
                            frame,
                            (cx, cy),
                            6,
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            frame,
                            "INTRUSION",
                            (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )

            # 信息叠加
            cv2.putText(
                frame,
                f"Current: {intrusion_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Total: {total_intrusions}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # 保存视频
            out.write(frame)

            # GUI显示
            self.change_pixmap.emit(frame)

            self.update_info.emit(
                f"当前入侵: {intrusion_count}\n"
                f"累计入侵: {total_intrusions}"
            )

        cap.release()
        out.release()

        self.update_info.emit(
            "\n处理完成\n已保存视频"
        )

    def stop(self):
        self.running = False