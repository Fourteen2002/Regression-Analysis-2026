import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

# ====================== 1. 全局参数 ======================
TRUE_BETA = np.array([5.0, 3.0])
SIGMA = 2.0
N_SIMULATIONS = 1000
N_SAMPLES = 100
np.random.seed(42)

# ====================== 2. 生成数据 ======================
def generate_design_matrix(n_samples, rho):
    x1 = np.random.normal(0, 1, n_samples)
    x2 = rho * x1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_samples)
    return np.column_stack((x1, x2))

# ====================== 3. 蒙特卡洛模拟 ======================
def run_monte_carlo(X, true_beta, sigma, n_simulations):
    n_samples = X.shape[0]
    beta_hat_list = []
    xtx_inv = np.linalg.inv(X.T @ X)
    proj = xtx_inv @ X.T

    for _ in range(n_simulations):
        eps = np.random.normal(0, sigma, n_samples)
        y = X @ true_beta + eps
        beta_hat = proj @ y
        beta_hat_list.append(beta_hat)
    return np.array(beta_hat_list)

# ====================== 4. 执行实验 ======================
print("=== 实验A：正交特征 rho=0.0 ===")
X_A = generate_design_matrix(N_SAMPLES, 0.0)
beta_hat_A = run_monte_carlo(X_A, TRUE_BETA, SIGMA, N_SIMULATIONS)

print("\n=== 实验B：高度共线性 rho=0.99 ===")
X_B = generate_design_matrix(N_SAMPLES, 0.99)
beta_hat_B = run_monte_carlo(X_B, TRUE_BETA, SIGMA, N_SIMULATIONS)

# ====================== 5. 协方差矩阵 ======================
print("\n===== 协方差矩阵对比 =====")
empirical_cov = np.cov(beta_hat_B.T)
theoretical_cov = (SIGMA**2) * np.linalg.inv(X_B.T @ X_B)
print("经验协方差矩阵：")
print(np.round(empirical_cov, 4))
print("\n理论协方差矩阵：")
print(np.round(theoretical_cov, 4))

# ====================== 6. 绘图 & 保存到 week05 ======================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(beta_hat_A[:,0], beta_hat_A[:,1], alpha=0.5, s=20, color="#1f77b4", label="实验A (正交)")
ax.scatter(beta_hat_B[:,0], beta_hat_B[:,1], alpha=0.5, s=20, color="#ff7f0e", label="实验B (共线)")
ax.scatter(TRUE_BETA[0], TRUE_BETA[1], color="red", s=100, marker="*", label="真实参数")

def plot_cov_ellipse(cov, center, ax):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 4 * np.sqrt(vals)
    ellip = Ellipse(center, w, h, angle=theta, edgecolor='k', facecolor='none', lw=2)
    ax.add_artist(ellip)

plot_cov_ellipse(np.cov(beta_hat_A.T), TRUE_BETA, ax)
plot_cov_ellipse(np.cov(beta_hat_B.T), TRUE_BETA, ax)

ax.set_xlabel(r"$\hat{\beta}_1$")
ax.set_ylabel(r"$\hat{\beta}_2$")
ax.set_title("正交 vs 共线性 OLS估计分布")
ax.legend()
plt.tight_layout()


current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "beta_scatter.png")


plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()


print(f"\n✅ 图片已保存！")
print(f"📂 完整路径：{save_path}")
print(f"🗂️  所在文件夹：{current_dir}")

# 强制刷新 VSCode 文件夹（Linux 系统命令）
os.system(f"cd {current_dir} && ls -l")