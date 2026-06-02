import cv2


class ROISelector:

    def __init__(self, image_path):

        self.image = cv2.imread(image_path)

        self.rois = []

        self.start_x = 0
        self.start_y = 0

        self.drawing = False

    def mouse_callback(
            self,
            event,
            x,
            y,
            flags,
            param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drawing = True

            self.start_x = x
            self.start_y = y

        elif event == cv2.EVENT_LBUTTONUP:

            self.drawing = False

            self.rois.append(
                (
                    self.start_x,
                    self.start_y,
                    x,
                    y
                )
            )

            cv2.rectangle(
                self.image,
                (self.start_x, self.start_y),
                (x, y),
                (0, 255, 0),
                2
            )

    def select(self):

        cv2.namedWindow("ROI Selector")

        cv2.setMouseCallback(
            "ROI Selector",
            self.mouse_callback
        )

        while True:

            cv2.imshow(
                "ROI Selector",
                self.image
            )

            key = cv2.waitKey(1)

            if key == 27:
                break

        cv2.destroyAllWindows()

        return self.rois