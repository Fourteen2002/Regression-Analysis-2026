import numpy as np

class CustomOLS:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # 增加截距项
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        # 正规方程求解（因为data_prep做了drop_first，不会奇异矩阵）
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.beta