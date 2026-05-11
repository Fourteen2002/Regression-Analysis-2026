import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子 VIF，用于检测多重共线性
    VIF_j = 1 / (1 - R²_j)
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]
    vif_values = []

    for j in range(n_features):
        # 其他特征作为自变量
        X_other = np.delete(X, j, axis=1)
        # 当前特征作为因变量
        y_j = X[:, j]

        # OLS回归
        reg = LinearRegression().fit(X_other, y_j)
        r2 = reg.score(X_other, y_j)

        # 避免除以0
        r2 = min(r2, 0.999999)
        vif = 1 / (1 - r2)
        vif_values.append(round(vif, 4))

    return vif_values