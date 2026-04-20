import numpy as np
import pandas as pd
import time
import os
import shutil
from scipy.stats import f
from sklearn.linear_model import LinearRegression as SklearnLR
from models import MyLinearRegression

# ====================== Task2 通用模型评价函数 ======================
def evaluate_model(model, X, y, model_name):
    """评估模型性能"""
    start_time = time.time()
    model.fit(X, y)
    fit_time = time.time() - start_time
    
    r2_score = model.score(X, y)
    
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
    """初始化结果文件夹"""
    result_path = "./results"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)
    print("✅ results文件夹已重置创建完成")

# ====================== Task3 场景A：合成数据白盒验证 ======================
def scene_A():
    """场景A：合成数据测试"""
    print("\n===== 场景A：合成数据DGP白盒测试 =====")
    
    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 3)
    true_beta = np.array([2, 5, -3, 1.2])
    y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n) * 0.5
    
    my_model = MyLinearRegression()
    res = evaluate_model(my_model, X, y, "自研线性回归引擎")
    
    assert res["R²拟合优度"] > 0.95, "R²不符合预期"
    print("✅ 场景A断言校验通过，模型精度符合真实DGP设定")
    
    return res

# ====================== Task3 场景B：真实市场数据 ======================
def scene_B():
    """场景B：真实营销数据"""
    print("\n===== 场景B：营销真实数据 双市场独立建模&联合F检验 =====")
    
    # 读取数据
    df = pd.read_csv("q3_marketing.csv")
    
    print(f"\n📊 数据概览：")
    print(f"总样本量: {len(df)}")
    print(f"列名: {list(df.columns)}")
    
    # 处理Region列：将空值填充为'NA'
    if 'Region' in df.columns:
        print(f"原始Region列唯一值: {df['Region'].unique()}")
        # 将空值填充为'NA'
        df['Region'] = df['Region'].fillna('NA')
        print(f"处理后Region列唯一值: {df['Region'].unique()}")
    
    # 拆分数据
    df_na = df[df["Region"] == "NA"]
    df_eu = df[df["Region"] == "EU"]
    
    print(f"\n拆分结果：")
    print(f"北美市场(NA): {len(df_na)} 条")
    print(f"欧洲市场(EU): {len(df_eu)} 条")
    
    # 如果还是没有NA数据，使用索引拆分（前一半EU，后一半NA）
    if len(df_na) == 0:
        print("\n⚠️ 未找到'NA'标签，使用索引拆分（前500条EU，后500条NA）")
        n_half = len(df) // 2
        df_eu = df.iloc[:n_half].copy()
        df_na = df.iloc[n_half:].copy()
        print(f"北美市场(NA): {len(df_na)} 条")
        print(f"欧洲市场(EU): {len(df_eu)} 条")
    
    # 特征列
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget"]
    
    # 检查特征列是否存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：缺少特征列 {missing_cols}")
        return {}
    
    X_na = df_na[feature_cols].values
    y_na = df_na["Sales"].values
    X_eu = df_eu[feature_cols].values
    y_eu = df_eu["Sales"].values
    
    n_na = len(y_na)
    n_eu = len(y_eu)
    p = len(feature_cols)
    
    print(f"\n数据维度：")
    print(f"北美: X shape={X_na.shape}, y shape={y_na.shape}")
    print(f"欧洲: X shape={X_eu.shape}, y shape={y_eu.shape}")
    
    # 检查是否有数据
    if n_na == 0:
        print("❌ 错误：没有北美数据！")
        return {}
    
    if n_eu == 0:
        print("❌ 错误：没有欧洲数据！")
        return {}
    
    # 建立模型
    model_na = MyLinearRegression()
    model_eu = MyLinearRegression()
    
    model_na.fit(X_na, y_na)
    model_eu.fit(X_eu, y_eu)
    
    # 计算R²
    r2_na = model_na.score(X_na, y_na)
    r2_eu = model_eu.score(X_eu, y_eu)
    
    print(f"\n📈 R²计算结果：")
    print(f"北美市场 R² = {r2_na:.6f}")
    print(f"欧洲市场 R² = {r2_eu:.6f}")
    
    # 输出模型系数
    print(f"\n北美市场模型系数：")
    print(f"  截距 (Intercept): {model_na.intercept_:.4f}")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {model_na.coef_[i]:.4f}")
    
    print(f"\n欧洲市场模型系数：")
    print(f"  截距 (Intercept): {model_eu.intercept_:.4f}")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: {model_eu.coef_[i]:.4f}")
    
    # F检验：所有系数是否为0
    k = p
    
    # 安全计算F统计量
    F_na, p_na = np.nan, np.nan
    F_eu, p_eu = np.nan, np.nan
    
    # 北美市场F检验
    if n_na > k + 1 and r2_na < 0.9999:
        F_na = (r2_na / k) / ((1 - r2_na) / (n_na - k - 1))
        p_na = 1 - f.cdf(F_na, k, n_na - k - 1)
    else:
        if n_na <= k + 1:
            print(f"\n⚠️ 北美市场样本量不足 (n={n_na} ≤ p+1={k+1})，无法进行F检验")
        if r2_na >= 0.9999:
            print(f"\n⚠️ 北美市场R²={r2_na}过高，可能样本量太小")
    
    # 欧洲市场F检验
    if n_eu > k + 1 and r2_eu < 0.9999:
        F_eu = (r2_eu / k) / ((1 - r2_eu) / (n_eu - k - 1))
        p_eu = 1 - f.cdf(F_eu, k, n_eu - k - 1)
    else:
        if n_eu <= k + 1:
            print(f"\n⚠️ 欧洲市场样本量不足 (n={n_eu} ≤ p+1={k+1})，无法进行F检验")
        if r2_eu >= 0.9999:
            print(f"\n⚠️ 欧洲市场R²={r2_eu}过高，可能样本量太小")
    
    print(f"\n{'='*60}")
    print(f"北美市场 (NA) 分析结果：")
    print(f"  样本量: {n_na}")
    print(f"  R² = {r2_na:.6f}")
    if not np.isnan(F_na):
        print(f"  F统计量 = {F_na:.4f}")
        print(f"  p值 = {p_na:.6e}")
        print(f"  结论: {'✅ 广告策略显著有效' if p_na < 0.05 else '❌ 广告策略无显著效果'}")
    else:
        print(f"  ⚠️ 无法进行F检验")
    
    print(f"\n{'='*60}")
    print(f"欧洲市场 (EU) 分析结果：")
    print(f"  样本量: {n_eu}")
    print(f"  R² = {r2_eu:.6f}")
    if not np.isnan(F_eu):
        print(f"  F统计量 = {F_eu:.4f}")
        print(f"  p值 = {p_eu:.6e}")
        print(f"  结论: {'✅ 广告策略显著有效' if p_eu < 0.05 else '❌ 广告策略无显著效果'}")
    else:
        print(f"  ⚠️ 无法进行F检验")
    
    return {
        '北美样本量': n_na,
        '北美R²': r2_na,
        '北美F值': F_na if not np.isnan(F_na) else 0,
        '北美p值': p_na if not np.isnan(p_na) else 1,
        '欧洲样本量': n_eu,
        '欧洲R²': r2_eu,
        '欧洲F值': F_eu if not np.isnan(F_eu) else 0,
        '欧洲p值': p_eu if not np.isnan(p_eu) else 1,
        'model_na': model_na,
        'model_eu': model_eu
    }

# ====================== Task4 生成报告 ======================
def save_report(res_A, res_B, compare_res):
    """生成Markdown报告"""
    
    # 处理可能的nan值
    na_f = res_B['北美F值'] if not np.isnan(res_B['北美F值']) else 0
    na_p = res_B['北美p值'] if not np.isnan(res_B['北美p值']) else 1
    eu_f = res_B['欧洲F值'] if not np.isnan(res_B['欧洲F值']) else 0
    eu_p = res_B['欧洲p值'] if not np.isnan(res_B['欧洲p值']) else 1
    
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
- 样本量：{res_B['北美样本量']}
- 模型拟合R²：{res_B['北美R²']:.6f}
- 联合F统计量：{na_f:.4f}
- 检验p值：{na_p:.6e}
- **结论**：{'✅ 广告整体投放效果显著' if na_p < 0.05 else '❌ 广告投放无统计显著性'}

### 欧洲市场 (Region EU)
- 样本量：{res_B['欧洲样本量']}
- 模型拟合R²：{res_B['欧洲R²']:.6f}
- 联合F统计量：{eu_f:.4f}
- 检验p值：{eu_p:.6e}
- **结论**：{'✅ 广告整体投放效果显著' if eu_p < 0.05 else '❌ 广告投放无统计显著性'}

## 4. 业务洞察与建议

### 市场对比分析
- 北美市场R² = {res_B['北美R²']:.4f}，模型解释力{'较强' if res_B['北美R²'] > 0.7 else '中等' if res_B['北美R²'] > 0.5 else '较弱'}
- 欧洲市场R² = {res_B['欧洲R²']:.4f}，模型解释力{'较强' if res_B['欧洲R²'] > 0.7 else '中等' if res_B['欧洲R²'] > 0.5 else '较弱'}

### 广告策略建议
{get_advice(na_p, eu_p)}

---
*报告生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open("./results/summary_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\n📄 分析报告已自动保存至 results/summary_report.md")

def get_advice(p_na, p_eu):
    """根据p值生成建议"""
    na_sig = p_na < 0.05
    eu_sig = p_eu < 0.05
    
    if na_sig and eu_sig:
        return "北美和欧洲市场的广告投放都具有显著效果，建议继续优化预算分配，重点关注高ROI的广告渠道。"
    elif na_sig and not eu_sig:
        return "北美市场广告效果显著，欧洲市场需要重新评估广告策略。建议：1) 检查欧洲市场的广告投放渠道 2) 考虑地区文化差异 3) 增加样本量重新测试。"
    elif not na_sig and eu_sig:
        return "欧洲市场广告效果显著，北美市场需要重新评估广告策略。建议：1) 优化北美市场的广告投放组合 2) 考虑季节性因素 3) 增加样本量重新测试。"
    else:
        return "两个市场的广告策略都需要重新审视和优化。建议：1) 重新评估广告预算分配 2) 考虑其他影响因素（价格、竞品等） 3) 收集更多数据进行分析。"

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*60)
    print("线性回归建模全流程分析")
    print("="*60)
    
    # 初始化
    init_result_dir()
    
    # Task2: 模型对比
    print("\n" + "="*60)
    print("Task2: 模型性能对比测试")
    print("="*60)
    
    np.random.seed(42)
    X_test = np.random.rand(500, 3)
    y_test = 2 + 3*X_test[:,0] + 1.5*X_test[:,1] - 2*X_test[:,2] + np.random.randn(500) * 0.1
    
    my_res = evaluate_model(MyLinearRegression(), X_test, y_test, "自研手写线性回归")
    sk_res = evaluate_model(SklearnLR(), X_test, y_test, "Sklearn工业级线性回归")
    
    # Task3: 双场景验证
    print("\n" + "="*60)
    print("Task3: 双场景验证")
    print("="*60)
    
    sceneA_res = scene_A()
    sceneB_res = scene_B()
    
    # Task4: 生成报告
    print("\n" + "="*60)
    print("Task4: 生成分析报告")
    print("="*60)
    
    save_report(sceneA_res, sceneB_res, [my_res, sk_res])
    
    print("\n" + "="*60)
    print("✅ 所有任务执行完毕！")
    print("="*60)