import numpy as np
from scipy.stats import f

class MyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.beta_hat_ = None
        self.n_ = 0
        self.p_ = 0
        self.X_b_ = None
        self.y_ = None

    def fit(self, X, y):
        self.n_, self.p_ = X.shape
        self.y_ = y.copy()
        # 加入截距项
        self.X_b_ = np.c_[np.ones((self.n_, 1)), X]
        
        # ✅ 关键修改：使用 lstsq 代替 inv，彻底解决奇异矩阵问题
        self.beta_hat_ = np.linalg.lstsq(self.X_b_, y, rcond=None)[0]
        
        self.intercept_ = self.beta_hat_[0]
        self.coef_ = self.beta_hat_[1:]
        return self

    def predict(self, X):
        if self.beta_hat_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit 方法")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.beta_hat_

    def score(self, X, y):
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        if ss_total < 1e-10:
            return 1.0 if np.allclose(y, y_pred) else 0.0
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - ss_res / ss_total

    def f_test(self, C, d):
        """
        F检验：检验 H0: C @ beta = d
        """
        if self.beta_hat_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit 方法")
        
        # 无约束模型的残差平方和
        y_pred_unrestricted = self.X_b_ @ self.beta_hat_
        rss_unrestricted = np.sum((self.y_ - y_pred_unrestricted) ** 2)
        
        # 计算受限估计（使用伪逆防止奇异）
        XTX = self.X_b_.T @ self.X_b_
        XTX_inv = np.linalg.pinv(XTX)
        
        # 求解受限参数
        try:
            C_XTX_inv = C @ XTX_inv
            temp = C_XTX_inv @ C.T
            temp_inv = np.linalg.inv(temp)
            lambda_ = temp_inv @ (C @ self.beta_hat_ - d)
            beta_restricted = self.beta_hat_ - XTX_inv @ C.T @ lambda_
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        
        # 受限模型的残差平方和
        y_pred_restricted = self.X_b_ @ beta_restricted
        rss_restricted = np.sum((self.y_ - y_pred_restricted) ** 2)
        
        # F统计量
        q = C.shape[0]
        df_residual = self.n_ - (self.p_ + 1)
        F_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / df_residual)
        p_value = 1 - f.cdf(F_stat, q, df_residual)
        
        return F_stat, p_value