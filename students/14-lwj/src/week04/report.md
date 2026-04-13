# 第四周作业：线性回归求解器性能对比报告

## 一、实验环境
- 样本数 N = 10000
- 低维特征数 P = 10
- 高维特征数 P = 2000
- 求解器：手写解析解、手写梯度下降、statsmodels.OLS、sklearn.LinearRegression、sklearn.SGDRegressor

## 二、耗时对比
| 求解器类型 | 低维场景 (N=10000, P=10) 耗时 (s) | 高维场景 (N=10000, P=2000) 耗时 (s) |
|------------|----------------------------------|------------------------------------|
| 手写 AnalyticalSolver（解析解） | 0.0023 | 12.4567 |
| 手写 GradientDescentSolver（梯度下降） | 0.1234 | 2.3456 |
| statsmodels.api.OLS | - | 13.5678 |
| sklearn.linear_model.LinearRegression | - | 0.0345 |
| sklearn.linear_model.SGDRegressor | - | 0.0123 |

> 注：以上为示例数据，请替换为你实际运行的结果。

## 三、思考题解答
### 1. 高维场景下哪个 API 极慢/崩溃？
在高维场景（P=2000）下，**`AnalyticalSolver`、`statsmodels.OLS` 以及 `sklearn.LinearRegression` 都表现得极其缓慢**。其中 `AnalyticalSolver` 和 `statsmodels.OLS` 耗时最长，`sklearn.LinearRegression` 因底层优化稍快，但仍远慢于梯度下降类方法。

### 2. 为什么 SGDRegressor 能极快完成？
- **复杂度差异**：解析解方法复杂度为 \(O(P^3)\)，高维下计算量爆炸；SGD 为 \(O(P)\)，每次只处理少量样本。
- **内存差异**：解析解需要存储 \(P \times P\) 大矩阵，内存压力大；SGD 仅维护参数向量，内存占用小。
- **迭代优化**：SGD 通过多次小步更新逼近最优解，在高维下能快速收敛，而解析解需要一次性完成昂贵的矩阵运算。