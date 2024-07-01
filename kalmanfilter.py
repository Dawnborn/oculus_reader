import numpy as np

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x  # State vector dimension
        self.dim_z = dim_z  # Measurement vector dimension
        self.x = np.zeros(dim_x)  # State vector
        self.P = np.eye(dim_x)  # State covariance matrix
        self.Q = np.eye(dim_x)  # Process noise covariance matrix
        self.F = np.eye(dim_x)  # State transition matrix
        self.H = np.zeros((dim_z, dim_x))  # Measurement function
        self.R = np.eye(dim_z)  # Measurement noise covariance matrix
        self.z = np.array([None] * dim_z)  # Measurement vector

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        self.z = z
        y = self.z - np.dot(self.H, self.x)  # Innovation or residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

    def set_state_transition_matrix(self, F):
        self.F = F

    def set_measurement_matrix(self, H):
        self.H = H

    def set_process_noise_covariance(self, Q):
        self.Q = Q

    def set_measurement_noise_covariance(self, R):
        self.R = R

    def set_initial_state(self, x, P=None):
        self.x = x
        if not (P is None):
            self.P = P