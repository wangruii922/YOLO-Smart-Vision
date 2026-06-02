import cv2
import numpy as np


class HeatmapOverlayGenerator:

    def __init__(self, heatmap):

        self.heatmap = heatmap

    def generate_overlay(
            self,
            background_path,
            save_path="outputs/heatmap/overlay_heatmap.png"
    ):

        frame = cv2.imread(background_path)

        if frame is None:
            raise ValueError(
                f"Cannot load image: {background_path}"
            )

        h, w = frame.shape[:2]

        heat = cv2.resize(
            self.heatmap,
            (w, h)
        )

        heat = np.uint8(255 * heat)

        colored_heatmap = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            frame,
            0.65,
            colored_heatmap,
            0.35,
            0
        )

        cv2.imwrite(
            save_path,
            overlay
        )

        return overlay