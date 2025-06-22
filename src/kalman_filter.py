import cv2
import numpy as np

class KalmanFilterBox:
    def __init__(self, initial_bbox: tuple):
        # Kalman filter with 8 state variables: [x, y, w, h, vw, vy, vw, vh]
        # x, y, w, h are position and size
        # vw, vy, vw, vh are velocities
        self.kf = cv2.KalmanFilter(8, 4)

        # how states are changing from t to t+1
        # constant velocity model:
        # x' = x + vw * dt, y = v + vy * dt
        # dt = 1 (assuming 1 frame interval)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ], np.float32)

        # relates state vector to measurement vector
        # we only measure x, y, w, h
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        # how much uncertainty in state transition
        # this should be tuned based on expected object motion
        self.kf.processNoiseCov = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0, 0, 0],     # low noise for velocities
            [0, 0, 0, 0, 0, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.0005, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.0005]
        ], np.float32)                      # process noise for position

        # how much uncertainty in measurements
        # tune this based on detector/tracker accuracy
        self.kf.measurementNoiseCov = np.array([
            [10, 0, 0, 0],    # x
            [0, 10, 0, 0],    # y
            [0, 0, 100, 0],   # w
            [0, 0, 0, 100]    # h
        ], np.float32)

        # initial uncertainty in state
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1

        # initial state
        x, y, w, h = initial_bbox
        self.kf.statePost = np.array([[x], [y], [w], [h], [0.], [0.], [0.], [0.]], np.float32)

    def predict(self):
        """
        Predicts the next position of the bbox, 
        and returns the x, y, height, width for it.
        """
        predicted = self.kf.predict()
        return tuple(map(int, predicted[:4].flatten()))
    
    def update(self, bbox: tuple):
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], np.float32)
        corrected = self.kf.correct(measurement)

        return tuple(map(int, corrected[:4].flatten()))
    