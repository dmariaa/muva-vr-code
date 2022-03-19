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


if __name__ == "__main__":
    from eye_controller import EyeController
    import cv2.cv2 as cv2
    import time

    image = cv2.imread("../data/image1.jpeg")
    eye_controller = EyeController(test_image=image)
    eye_controller.initialize()

    start = time.time()
    eye_controller.update()
    print(f"Updated in {time.time() - start:.4f}")

    start = time.time()
    eye_controller.draw_centroids()
    print(f"Drawn in {time.time() - start:.4f}")

    pdi_controller = PDIController(kp=1, kd=1, ki=1)

    start = time.time()
    pdi_controller.update(eye_controller.normalized_centroid1)
    print(f"PDI calculated in {time.time() - start:.4f}")

    print(f"PDI: {pdi_controller.PDI} [{pdi_controller.P}, {pdi_controller.D}, {pdi_controller.I}]")

    for i in range(10):
        pdi_controller.update(eye_controller.normalized_centroid1)
        print(f"PDI: {pdi_controller.PDI} [{pdi_controller.P}, {pdi_controller.D}, {pdi_controller.I}]")

