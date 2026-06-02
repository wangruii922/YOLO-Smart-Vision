import pandas as pd


class LoiteringDetector:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.df["frame"] = self.df["frame"].astype(int)
        self.df["track_id"] = self.df["track_id"].astype(int)

    # =========================
    # ROI Loitering Detection
    # =========================
    def detect(self, roi, threshold_frames=50):

        x1, y1, x2, y2 = roi

        alerts = {}

        for tid in self.df["track_id"].unique():

            obj = self.df[self.df["track_id"] == tid]

            # 在ROI内的点
            in_roi = obj[
                (obj["x"] >= x1) &
                (obj["x"] <= x2) &
                (obj["y"] >= y1) &
                (obj["y"] <= y2)
            ]

            if len(in_roi) == 0:
                continue

            stay_frames = len(in_roi)

            if stay_frames >= threshold_frames:

                alerts[tid] = {
                    "stay_frames": stay_frames,
                    "start_frame": int(in_roi["frame"].min()),
                    "end_frame": int(in_roi["frame"].max())
                }

        return alerts