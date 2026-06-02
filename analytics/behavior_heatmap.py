import pandas as pd
import numpy as np
import math
from scipy.ndimage import gaussian_filter


class BehaviorAnalysisSystem:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        # =====================
        # 类型统一
        # =====================
        self.df["x"] = self.df["x"].astype(float)
        self.df["y"] = self.df["y"].astype(float)
        self.df["frame"] = self.df["frame"].astype(int)
        self.df["track_id"] = self.df["track_id"].astype(int)
        self.df["class"] = self.df["class"].astype(str)

        # =====================
        # 固定画布尺寸（核心修复）
        # 防止ROI shape mismatch
        # =====================
        self.W = 1280
        self.H = 720

        self.df = self._compute_speed()

    # ==================================================
    # speed
    # ==================================================
    def _compute_speed(self):

        df = self.df.copy()
        df = df.sort_values(["track_id", "frame"])

        df["speed"] = 0.0

        for tid in df["track_id"].unique():

            obj = df[df["track_id"] == tid]

            if len(obj) < 2:
                continue

            prev = None

            for idx in obj.index:

                x = df.loc[idx, "x"]
                y = df.loc[idx, "y"]

                if prev is not None:
                    dx = x - prev[0]
                    dy = y - prev[1]
                    df.loc[idx, "speed"] = math.sqrt(dx * dx + dy * dy)

                prev = (x, y)

        return df

    # ==================================================
    # ROI heatmap（稳定版）
    # ==================================================
    def roi_heatmap(self, rois):

        heat = self._generate_heat(self.df)

        if heat is None:
            print("[ROI] no data")
            return None

        overlay = np.zeros((self.H, self.W), dtype=np.float32)

        for (x1, y1, x2, y2) in rois:

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(self.W, int(x2))
            y2 = min(self.H, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            overlay[y1:y2, x1:x2] = heat[y1:y2, x1:x2]

        return overlay

    # ==================================================
    # class heatmap
    # ==================================================
    def class_heatmap(self, class_name):

        df = self.df[self.df["class"].str.lower() == class_name.lower()]

        if len(df) == 0:
            print(f"[WARN] class not found: {class_name}")
            return None

        return self._generate_heat(df)

    # ==================================================
    # global heat
    # ==================================================
    def global_heat(self):

        return self._generate_heat(self.df)

    # ==================================================
    # core heat
    # ==================================================
    def _generate_heat(self, df):

        if df is None or len(df) == 0:
            return None

        heat = np.zeros((self.H, self.W), dtype=np.float32)

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        for xi, yi in zip(x, y):

            xi = int(xi)
            yi = int(yi)

            if 0 <= xi < self.W and 0 <= yi < self.H:
                heat[yi, xi] += 1

        if heat.max() == 0:
            return None

        heat = gaussian_filter(
            heat,
            sigma=35
        )
        heat = heat / (heat.max() + 1e-6)

        return heat

    # ==================================================
    # dwell
    # ==================================================
    def dwell_time_analysis(self, threshold=30):

        result = {}

        for tid in self.df["track_id"].unique():

            obj = self.df[self.df["track_id"] == tid]

            if len(obj) >= threshold:
                result[tid] = len(obj)

        return result

    # ==================================================
    # speed
    # ==================================================
    def speed_analysis(self):

        return self.df.groupby("track_id")["speed"].mean()