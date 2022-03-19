import os
import time
import math

from GUI import GUI
from HAL import HAL

import cv2.cv2 as cv2
import numpy as np

bgr_color_blue = [255, 0, 0]
bgr_color_green = [0, 255, 0]
bgr_color_red = [0, 0, 255]


class EyeController:
    line_color_pattern = np.array([
        [[0, 100, 20], [10, 255, 255]],
        [[160, 100, 20], [179, 255, 255]]
    ])

    erosion_kernel = np.ones((3, 3), np.uint8)

    def __init__(self, test_image = None):
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
            return self.view_center, self.view_center, 0, 0

        y1 = d[0, 0]
        b1 = np.nonzero(mask[y1:y1 + 1, :])[1]
        c1 = b1[b1.shape[0] // 2]

        y2 = d[0, 0] + ((d[-1, 0] - d[0, 0]) // 2)
        b2 = np.nonzero(mask[y2:y2 + 1, :])[1]
        c2 = b2[b2.shape[0] // 2]

        return c1, c2, y1, y2

    def initialize(self):
        _, self.view_width, _ = self.__get_image__().shape
        self.view_center = self.view_width // 2

    def update(self):
        self.centroid1, self.centroid2, self.centroid1_y, self.centroid2_y = self.__get_centroids__()
        self.normalized_centroid1 = (self.centroid1 - self.view_center) / self.view_center
        self.normalized_centroid2 = (self.centroid2 - self.view_center) / self.view_center

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


class PDIController:
    def __init__(self, kp=0.0, kd=0.0, ki=0.0):
        # s = kp + kd + ki
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki

        self.error = 0
        self.prev_error = 0.
        self.integral = 0

        self.P = 0.
        self.D = 0.
        self.I = 0.
        self.PDI = 0.

    def update(self, error, delta_time: float = 1.0):
        self.P = error
        self.D = (error - self.prev_error) / delta_time
        self.I = (self.I + error) * delta_time

        self.PDI = (self.P * self.Kp) + (self.I * self.Ki) + (self.D * self.Kd)

        self.prev_error = error


def printDebugLine(img, line, text):
    img = cv2.putText(img, f"{text}", (10, 20 * line),
                      cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img


initialized = False
acc_time = 0.
eye_controller = EyeController()
pdi_controller = PDIController(kp=1.0, kd=0.45, ki=0.3)
# pdi_controller = PDIController(kp=1.1, kd = 0.65, ki = 0.4)
# pdi_controller = PDIController(kp=1.0, kd = 0.6, ki = 0.45)

print(f"Starting loop")

last_time = time.time()

while True:
    delta_time = time.time() - last_time
    acc_time = acc_time + delta_time

    if not initialized:
        eye_controller.initialize()
        HAL.setW(0.)
        HAL.setV(0.)

        if acc_time >= 3.0:
            initialized = True
            print(f"Pilot initialized")

    eye_controller.update()
    eye_controller.draw_centroids()

    final_image = np.copy(eye_controller.current_image)
    final_image = printDebugLine(final_image, 1, f"Delta time: {delta_time:.4f}, Acc time: {acc_time:.4f}")

    if initialized:
        pdi_controller.update(eye_controller.normalized_centroid1, delta_time)
        HAL.setW(-pdi_controller.PDI)
        # v = 2 + 5 * (1 - abs(eye_controller.centroid.x))
        # v = 4 + 3 * (1 - abs(eye_controller.centroid.x ** 2))
        # v = 2 + 5 * math.exp(-((0.5 * eye_controller.centroid.x) ** 2))
        v = 4
        HAL.setV(v)
        final_image = printDebugLine(final_image, 2,
                                     f"P: {pdi_controller.P:.4f}, D: {pdi_controller.D:.4f}, I: {pdi_controller.I:.4f}")
        final_image = printDebugLine(final_image, 3, f"W: {-pdi_controller.PDI:.4f}, V: {v:.4f}")

    GUI.showImage(final_image)

    last_time = time.time()

