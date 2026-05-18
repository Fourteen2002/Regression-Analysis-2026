# -*- coding: utf-8 -*-
import numpy as np

class AnalyticalOLS:
    """解析解线性回归"""
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class GradientDescentOLS:
    """梯度下降线性回归"""
    def __init__(
        self,
        learning_rate=0.01,
        tol=1e-6,
        max_iter=1000,
        gd_type="full_batch",
        batch_fraction=0.2
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X, y, seed=42):
        np.random.seed(seed)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        for i in range(self.max_iter):
            # 批量选择
            if self.gd_type == "mini_batch":
                batch_size = int(n_samples * self.batch_fraction)
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            else:
                X_batch = X
                y_batch = y

            # 梯度计算
            y_pred = X_batch @ self.coef_
            error = y_pred - y_batch
            grad = (2 / len(X_batch)) * (X_batch.T @ error)

            # 更新参数
            self.coef_ -= self.learning_rate * grad

            # 损失记录
            mse = np.mean((X @ self.coef_ - y) ** 2)
            self.loss_history_.append(mse)

            # 收敛停止
            if i > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)