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


class PDIController:
    def __init__(self, kp=0.0, kd=0.0, ki=0.0, pdi_clamp=3.0, i_clamp=2.5):
        # s = kp + kd + ki
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki

        self.pdi_clamp = pdi_clamp
        self.i_clamp = i_clamp

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

        self.I = self.I + error * delta_time
        self.I = max(min(self.I, self.i_clamp), -self.i_clamp)

        self.PDI = (self.P * self.Kp) + (self.I * self.Ki) + (self.D * self.Kd)
        self.PDI = max(min(self.PDI, self.pdi_clamp), -self.pdi_clamp)

        self.prev_error = error


def printDebugLine(img, line, text):
    img = cv2.putText(img, f"{text}", (10, 20 * line),
                      cv2.FONT_HERSHEY_PLAIN, 1, bgr_color_red, 1)
    return img


initialized = False
acc_time = 0.
eye_controller = EyeController(dead_zone=0.0)
# pdi_controller = PDIController(kp=4.5, kd=5.5, ki=0.0)
pdi_controller = PDIController(kp=4.25, kd=5.75, ki=0.0, pdi_clamp=4.0, i_clamp=2.5)

print(f"Starting loop")

last_time = time.monotonic()

while True:
    now = time.monotonic()
    delta_time = now - last_time
    acc_time = acc_time + delta_time
    last_time = now

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
        error = eye_controller.normalized_centroid1
        pdi_controller.update(error, 1.0)
        w = -pdi_controller.PDI
        # w = round(w, 1)
        HAL.setW(w)

        v_e = pdi_controller.PDI * 1.5
        # v_e = min(eye_controller.normalized_centroid1, eye_controller.normalized_centroid2) * 2
        v = max(8.0, 15 * math.exp(-abs(v_e) ** 2))
        # v = 9
        # v = 8
        v = round(v, 2)
        HAL.setV(v)
        final_image = printDebugLine(final_image, 2,
                                     f"P: {pdi_controller.P:.4f}, D: {pdi_controller.D:.4f}, I: {pdi_controller.I:.4f}")
        final_image = printDebugLine(final_image, 3,
                                     f"Kp: {pdi_controller.Kp}, Kd: {pdi_controller.Kd}, Ki: {pdi_controller.Ki}")
        final_image = printDebugLine(final_image, 4, f"W: {w:.4f}, V: {v:.4f}")
        final_image = printDebugLine(final_image, 5,
                                     f"NC1: {eye_controller.normalized_centroid1:.4f}, NC2:{eye_controller.normalized_centroid2}")

    GUI.showImage(final_image)
