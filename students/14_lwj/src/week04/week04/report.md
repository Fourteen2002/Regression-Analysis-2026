# 第四周作业：线性回归求解器性能对比报告

## 一、实验环境
- 样本数 N = 10000
- 低维特征数 P = 10
- 高维特征数 P = 2000
- 求解器：手写解析解、手写梯度下降、statsmodels.OLS、sklearn.LinearRegression、sklearn.SGDRegressor

## 二、耗时对比
| 求解器类型 | 低维场景 (N=10000, P=10) 耗时 (s) | 高维场景 (N=10000, P=2000) 耗时 (s) |
|------------|----------------------------------|------------------------------------|
| 手写 AnalyticalSolver（解析解） | 0.0204 | 1.3478 |
| 手写 GradientDescentSolver（梯度下降） | 0.6274 | 99.5827 |
| statsmodels.api.OLS | - | 44.9730 |
| sklearn.linear_model.LinearRegression | - | 16.6182 |
| sklearn.linear_model.SGDRegressor | - | 54.7901 |

> 说明：低维场景仅测试手写求解器，高维场景测试全部 5 种方法。

## 三、思考题解答

### 1. 高维场景下哪个 API 极慢/接近崩溃？
在高维场景（P=2000）下，**手写 GradientDescentSolver 耗时最长（99.5827s）**，其次是 `statsmodels.OLS`（44.9730s）和 `sklearn.SGDRegressor`（54.7901s），`sklearn.LinearRegression` 相对最快（16.6182s）。手写解析解 `AnalyticalSolver` 在高维下反而耗时仅 1.3478s，这是因为底层使用了高效的线性代数库。

### 2. 为什么 SGDRegressor 能在极短时间内完成任务？
- **计算复杂度差异**：
  解析解类方法需要计算 \(X^T X\) 并求解线性方程，时间复杂度为 \(O(P^3)\)；而 `SGDRegressor` 每次仅对小批量样本计算梯度，复杂度为 \(O(P)\)，在高维下依然高效。
- **内存占用差异**：
  解析解需要存储 \(P \times P\) 的大矩阵，内存压力极大；SGD 仅维护参数向量和小批量数据，内存占用极低。
- **迭代 vs 直接求解**：
  解析解是“一步到位”求精确解，高维下计算代价极高；SGD 是迭代优化，通过多次小步更新逼近最优解，在高维下能快速完成迭代（即使未完全收敛，也能在设定迭代次数内结束）。
> 补充：本次实验中 `sklearn.SGDRegressor` 出现了收敛警告，说明在 1000 次迭代内未完全收敛，若增加 `max_iter` 可进一步降低 MSE，但耗时会相应增加。