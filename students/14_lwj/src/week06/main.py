import numpy as np
import pandas as pd
import time
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR
from models import MyLinearRegression

# ====================== Task2 通用模型评价函数 ======================
def evaluate_model(model, X, y, model_name):
    # 计时拟合全流程
    start_time = time.time()
    model.fit(X, y)
    fit_time = time.time() - start_time

    # 预测+计算R²精度
    r2_score = model.score(X, y)

    # 输出性能对比
    print(f"\n【{model_name} 性能报告】")
    print(f"模型拟合耗时：{fit_time:.6f} 秒")
    print(f"拟合优度 R²：{r2_score:.6f}")
    return {
        "模型名称": model_name,
        "拟合耗时(s)": round(fit_time, 6),
        "R²拟合优度": round(r2_score, 6)
    }

# ====================== Task4 初始化结果文件夹 ======================
def init_result_dir():
    """程序启动自动创建/清空重置results文件夹"""
    result_path = "./results"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    print("✅ results文件夹已重置创建完成")

# ====================== Task3 场景A：合成数据白盒验证 ======================
def scene_A():
    print("\n===== 场景A：合成数据DGP白盒测试 =====")
    # 1. 生成1000条已知真实参数的线性数据
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 3)
    true_beta = np.array([2, 5, -3, 1.2])  # 截距+3个特征真实系数
    y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n) * 0.5

    # 2. 自定义模型拟合
    my_model = MyLinearRegression()
    my_res = evaluate_model(my_model, X, y, "自研线性回归引擎")

    # 3. 白盒断言校验R²合理性
    assert my_res["R²拟合优度"] > 0.95, "合成数据R²不符合预期，模型计算错误！"
    print("✅ 场景A断言校验通过，模型精度符合真实DGP设定")
    return my_res

# ============ Task3 场景B：真实市场数据分组F检验 ============
def scene_B():
    print("\n===== 场景B: 营销真实数据 双市场独立建模&联合F检验 =====")
    # 1. 读取csv数据
    df = pd.read_csv("q3_marketing.csv")
    # 拆分北美NA、欧洲EU两个独立市场
    df_A = df[df["Region"] == "NA"]
    df_B = df[df["Region"] == "EU"]

    # 广告特征X，销量因变量y
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    X_A, y_A = df_A[feature_cols].values, df_A["Sales"].values
    X_B, y_B = df_B[feature_cols].values, df_B["Sales"].values

    # 2. 分别建立独立模型
    model_A = MyLinearRegression()
    model_A.fit(X_A, y_A)
    
    model_B = MyLinearRegression()
    model_B.fit(X_B, y_B)

    # 3. 计算R²和F检验
    r2_A = model_A.score(X_A, y_A)
    r2_B = model_B.score(X_B, y_B)
    
    # 4. 联合F检验：H0 所有广告系数全=0
    k = len(feature_cols)
    C = np.hstack([np.zeros((k, 1)), np.eye(k)])  # 约束矩阵
    d = np.zeros(k)  # 约束值
    
    # 注意：这里需要models.py中有f_test方法
    try:
        F_A, p_A = model_A.f_test(C, d)
        F_B, p_B = model_B.f_test(C, d)
    except AttributeError:
        # 如果f_test方法不存在，使用替代方案
        print("⚠️ f_test方法未实现，使用基于R²的F统计量计算")
        # 计算F统计量的替代方法
        F_A = (r2_A / k) / ((1 - r2_A) / (len(y_A) - k - 1))
        p_A = 1 - f.cdf(F_A, k, len(y_A) - k - 1)
        F_B = (r2_B / k) / ((1 - r2_B) / (len(y_B) - k - 1))
        p_B = 1 - f.cdf(F_B, k, len(y_B) - k - 1)

    # 输出检验结论
    print(f"\n北美市场NA:")
    print(f"  R² = {r2_A:.4f}")
    print(f"  F = {F_A:.4f}, p值 = {p_A:.6f}")
    print(f"  结论: {'广告策略显著有效 ✅' if p_A < 0.05 else '广告策略无显著效果 ❌'}")
    
    print(f"\n欧洲市场EU:")
    print(f"  R² = {r2_B:.4f}")
    print(f"  F = {F_B:.4f}, p值 = {p_B:.6f}")
    print(f"  结论: {'广告策略显著有效 ✅' if p_B < 0.05 else '广告策略无显著效果 ❌'}")
    
    # 返回包含所有结果的字典
    return {
        '北美R²': r2_A,
        '北美F值': F_A,
        '北美p值': p_A,
        '欧洲R²': r2_B,
        '欧洲F值': F_B,
        '欧洲p值': p_B,
        'model_A': model_A,
        'model_B': model_B
    }

# ====================== Task4 自动生成markdown报告 ======================
def save_report(res_A, res_B, compare_res):
    """
    保存分析报告
    res_A: scene_A的返回结果（包含R²）
    res_B: scene_B的返回结果（包含F检验结果）
    compare_res: 模型对比结果列表
    """
    report_content = f"""# 线性回归建模全流程分析报告

## 1. 自研引擎 vs Sklearn 工业模型性能对比

| 模型名称 | 拟合耗时(s) | R²拟合优度 |
|---------|------------|------------|
| {compare_res[0]['模型名称']} | {compare_res[0]['拟合耗时(s)']} | {compare_res[0]['R²拟合优度']} |
| {compare_res[1]['模型名称']} | {compare_res[1]['拟合耗时(s)']} | {compare_res[1]['R²拟合优度']} |

## 2. 合成数据白盒测试结果

- **样本量**：1000条
- **特征数**：3个
- **真实系数**：截距=2, β₁=5, β₂=-3, β₃=1.2
- **模型R²**：{res_A['R²拟合优度']}
- **校验状态**：✅ 断言通过，模型数理逻辑正确

## 3. 双市场广告策略F检验结果

### 北美市场 (Region NA)
- 样本量：{res_B.get('北美样本量', 'N/A')}
- 模型拟合R²：{res_B['北美R²']:.4f}
- 联合F统计量：{res_B['北美F值']:.4f}
- 检验p值：{res_B['北美p值']:.6f}
- **结论**：{'✅ 广告整体投放效果显著' if res_B['北美p值'] < 0.05 else '❌ 广告投放无统计显著性'}

### 欧洲市场 (Region EU)
- 样本量：{res_B.get('欧洲样本量', 'N/A')}
- 模型拟合R²：{res_B['欧洲R²']:.4f}
- 联合F统计量：{res_B['欧洲F值']:.4f}
- 检验p值：{res_B['欧洲p值']:.6f}
- **结论**：{'✅ 广告整体投放效果显著' if res_B['欧洲p值'] < 0.05 else '❌ 广告投放无统计显著性'}

## 4. 业务洞察与建议

### 市场对比分析
- 北美市场R² = {res_B['北美R²']:.4f}，模型解释力{ '较强' if res_B['北美R²'] > 0.7 else '中等' if res_B['北美R²'] > 0.5 else '较弱'}
- 欧洲市场R² = {res_B['欧洲R²']:.4f}，模型解释力{ '较强' if res_B['欧洲R²'] > 0.7 else '中等' if res_B['欧洲R²'] > 0.5 else '较弱'}

### 广告策略建议
{'北美和欧洲市场的广告投放都具有显著效果，建议继续优化预算分配' if res_B['北美p值'] < 0.05 and res_B['欧洲p值'] < 0.05 
 else '北美市场广告效果显著，欧洲市场需要重新评估广告策略' if res_B['北美p值'] < 0.05 
 else '欧洲市场广告效果显著，北美市场需要重新评估广告策略' if res_B['欧洲p值'] < 0.05 
 else '两个市场的广告策略都需要重新审视和优化'}

---
*报告生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    # 写入报告文件
    with open("./results/summary_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\n📄 分析报告已自动保存至 results/summary_report.md")

# ====================== 程序唯一入口 ======================
if __name__ == "__main__":
    # 导入scipy的f分布（用于p值计算）
    from scipy.stats import f
    
    # Task4 初始化结果目录
    init_result_dir()

    # Task2 模型精度&速度横向对比
    print("\n" + "="*50)
    print("Task2: 模型性能对比测试")
    print("="*50)
    X_test = np.random.rand(500, 3)
    y_test = 2 + 3*X_test[:,0] + 1.5*X_test[:,1] - 2*X_test[:,2] + np.random.randn(500) * 0.1
    
    my_model = MyLinearRegression()
    sk_model = SklearnLR()
    
    my_res = evaluate_model(my_model, X_test, y_test, "自研手写线性回归")
    sk_res = evaluate_model(sk_model, X_test, y_test, "Sklearn工业级线性回归")

    # Task3 双场景全流程验证
    print("\n" + "="*50)
    print("Task3: 双场景验证")
    print("="*50)
    sceneA_res = scene_A()
    sceneB_res = scene_B()

    # Task4 自动导出全部结果报告
    print("\n" + "="*50)
    print("Task4: 生成分析报告")
    print("="*50)
    save_report(sceneA_res, sceneB_res, [my_res, sk_res])
    
    print("\n✅ 所有任务执行完毕！")