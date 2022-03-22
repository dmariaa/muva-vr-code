import cv2.cv2 as cv2
import numpy as np

from F1.vector import Vector

bgr_color_blue = [255, 0, 0]
bgr_color_green = [0, 255, 0]
bgr_color_red = [0, 0, 255]


class EyeController:
    line_color_pattern = np.array([
        [[0, 100, 20], [10, 255, 255]],
        [[160, 100, 20], [179, 255, 255]]
    ])

    erosion_kernel = np.ones((3, 3), np.uint8)

    def __init__(self, dead_zone: float = 0.05, test_image=None):
        self.dead_zone = dead_zone
        self.centroid2_y = None
        self.centroid1_y = None
        self.current_image = None
        self.normalized_centroid2 = None
        self.normalized_centroid1 = None
        self.view_center = None
        self.centroid1 = None
        self.centroid2 = None
        self.view_width = None
        self.test_image = test_image

    def __get_image__(self):
        return self.test_image if self.test_image is not None else HAL.getImage()

    def _get_line__(self):
        self.current_image = self.__get_image__()
        hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_image, self.line_color_pattern[0, 0], self.line_color_pattern[0, 1])
        mask2 = cv2.inRange(hsv_image, self.line_color_pattern[1, 0], self.line_color_pattern[1, 1])
        mask = mask1 + mask2

        mask = cv2.erode(mask, self.erosion_kernel, iterations=2)
        mask = cv2.dilate(mask, self.erosion_kernel, iterations=2)

        return mask

    def __get_centroids__(self):
        mask = self._get_line__()

        d = np.argwhere(mask > 0)

        if d.shape[0] == 0:
            return self.centroid1, self.centroid2, 0, 0

        y1 = d[0, 0]
        b1 = np.nonzero(mask[y1:y1 + 1, :])[1]

        y2 = d[0, 0] + ((d[-1, 0] - d[0, 0]) // 2)
        b2 = np.nonzero(mask[y2:y2 + 1, :])[1]

        try:
            c1 = b1[b1.shape[0] // 2]
            c2 = b2[b2.shape[0] // 2]
        except Exception as ex:
            print(f"c1={y1}, y2={y2}, b1 shape: {b1.shape}, b2 shape {b2.shape}")
            c2 = c1

        return c1, c2, y1, y2

    def initialize(self):
        _, self.view_width, _ = self.__get_image__().shape
        self.view_center = self.view_width // 2
        self.centroid1 = self.view_center
        self.centroid2 = self.view_center
        self.normalized_centroid1 = 0
        self.normalized_centroid2 = 0

    def update(self):
        self.centroid1, self.centroid2, self.centroid1_y, self.centroid2_y = self.__get_centroids__()
        self.normalized_centroid1 = (self.centroid1 - self.view_center) / self.view_center
        self.normalized_centroid2 = (self.centroid2 - self.view_center) / self.view_center

        if abs(self.normalized_centroid1) < self.dead_zone:
            self.normalized_centroid1 = 0.0

        if abs(self.normalized_centroid2) < self.dead_zone:
            self.normalized_centroid2 = 0.0

    def draw_centroids(self):
        self.current_image = cv2.drawMarker(self.current_image, (self.view_center, self.centroid1_y),
                                            color=bgr_color_blue,
                                            thickness=2)

        self.current_image = cv2.line(self.current_image, (self.view_center, self.centroid1_y),
                                      (self.centroid1, self.centroid1_y),
                                      color=bgr_color_blue,
                                      thickness=2)

        self.current_image = cv2.drawMarker(self.current_image, (self.view_center, self.centroid2_y),
                                            color=bgr_color_green,
                                            thickness=2)

        self.current_image = cv2.line(self.current_image, (self.view_center, self.centroid2_y),
                                      (self.centroid2, self.centroid2_y),
                                      color=bgr_color_green,
                                      thickness=2)



if __name__ == "__main__":
    import cv2.cv2 as cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    image = cv2.imread("../data/image1.jpeg")
    eye_controller = EyeController(test_image=image)
    eye_controller.initialize()

    start = time.time()
    eye_controller.update()
    print(f"Updated in {time.time()-start:.4f}")

    start = time.time()
    eye_controller.draw_centroids()
    print(f"Drawn in {time.time()-start:.4f}")

    final_image = cv2.cvtColor(eye_controller.current_image, cv2.COLOR_BGR2RGB)
    plt.imshow(final_image)
    plt.show()