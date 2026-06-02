import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict


class TrajectoryClusteringViz:

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.df["frame"] = self.df["frame"].astype(int)
        self.df["track_id"] = self.df["track_id"].astype(int)

        self.trajectories = self._build_trajectories()

    # =========================
    # build trajectories
    # =========================
    def _build_trajectories(self):

        traj = defaultdict(list)

        for _, row in self.df.iterrows():

            traj[row["track_id"]].append(
                (row["frame"], row["x"], row["y"])
            )

        for k in traj:
            traj[k] = sorted(traj[k], key=lambda x: x[0])

        return traj

    # =========================
    # feature extraction
    # =========================
    def _extract_feature(self, trajectory):

        coords = np.array([(x, y) for _, x, y in trajectory])

        if len(coords) < 5:
            return None

        diff = np.diff(coords, axis=0)

        return diff.flatten()

    # =========================
    # clustering
    # =========================
    def cluster(self, eps=10, min_samples=3):

        features = []
        ids = []

        for tid, traj in self.trajectories.items():

            feat = self._extract_feature(traj)

            if feat is None:
                continue

            features.append(feat)
            ids.append(tid)

        features = np.array(features)

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(features)

        self.cluster_map = {
            tid: label for tid, label in zip(ids, labels)
        }

        return self.cluster_map

    # =========================
    # trajectory color map
    # =========================
    def get_color(self, cluster_id):

        if cluster_id == -1:
            return (0, 0, 255)  # red noise

        np.random.seed(cluster_id * 10 + 7)

        return tuple(
            np.random.randint(0, 255, 3).tolist()
        )

    # =========================
    # draw + export video
    # =========================
    def export_video(self, video_path, output_path):

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        frame_tracks = defaultdict(list)

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            # collect points up to current frame
            for tid, traj in self.trajectories.items():

                for f, x, y in traj:

                    if f == frame_id:

                        frame_tracks[tid].append((x, y))

            # draw trajectories
            for tid, points in frame_tracks.items():

                if tid not in self.cluster_map:
                    continue

                color = self.get_color(
                    self.cluster_map[tid]
                )

                for i in range(1, len(points)):

                    x1, y1 = points[i - 1]
                    x2, y2 = points[i]

                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2
                    )

            cv2.putText(
                frame,
                "Trajectory Clustering",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            out.write(frame)

        cap.release()
        out.release()