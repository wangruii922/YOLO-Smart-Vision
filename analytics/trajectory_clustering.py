import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


class TrajectoryClustering:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.df["frame"] = self.df["frame"].astype(int)
        self.df["track_id"] = self.df["track_id"].astype(int)

        self.trajectories = self._build_trajectories()

    # =========================
    # 构建轨迹
    # =========================
    def _build_trajectories(self):

        trajectories = {}

        for tid in self.df["track_id"].unique():

            obj = self.df[self.df["track_id"] == tid]

            obj = obj.sort_values("frame")

            traj = list(zip(obj["x"], obj["y"]))

            # 过滤太短轨迹
            if len(traj) < 10:
                continue

            trajectories[tid] = traj

        return trajectories

    # =========================
    # 轨迹向量化（核心）
    # =========================
    def _trajectory_to_feature(self, traj):

        traj = np.array(traj)

        # 计算位移向量（比绝对坐标更稳定）
        diff = np.diff(traj, axis=0)

        feature = diff.flatten()

        return feature

    # =========================
    # 聚类
    # =========================
    def cluster(self, eps=10, min_samples=3):

        features = []
        ids = []

        for tid, traj in self.trajectories.items():

            feat = self._trajectory_to_feature(traj)

            features.append(feat)
            ids.append(tid)

        features = np.array(features)

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(features)

        result = {}

        for tid, label in zip(ids, labels):

            result[tid] = label

        return result