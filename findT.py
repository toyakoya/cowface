import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# 配置
VALIDATION_FEATURE_FILE = "features_validation18.npz" # 验证集特征文件
VALIDATION_PAIRS_CSV = "val_pairs_from_train_with_paths.csv" # 验证集对及其真实标签

# 阈值搜索范围
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.005

# === 更改: 定义新的分隔符 ===
PAIR_SEPARATOR = "__"
# ===========================

# --- 加载特征 ---
data = np.load(VALIDATION_FEATURE_FILE, allow_pickle=True)
names = data["names"].astype(str) # 这里的 names 已经是相对路径了
feats = data["feats"]

# 构建 name->index 映射
name_to_idx = {name: i for i, name in enumerate(names)}


# --- 读取验证集对 ---
df_val_pairs = pd.read_csv(VALIDATION_PAIRS_CSV)
y_true = df_val_pairs['TARGET'].tolist()

# --- 计算所有验证对的相似度 (一次性计算，提高效率) ---
print("计算所有验证对的余弦相似度...")
all_sims = []
for idx, row in tqdm(df_val_pairs.iterrows(), total=len(df_val_pairs), desc="Calculating Similarities"):
    pid = row['ID__ID'] # 例如 'cow_A/img1.jpg__cow_B/img2.jpg'
    
    # *** 修复下划线分割逻辑 ***
    # 使用 rsplit(PAIR_SEPARATOR, 1) 从右侧分割一次，确保得到的是两个完整的路径
    try:
        a, b = pid.rsplit(PAIR_SEPARATOR, 1) 
    except ValueError:
        print(f"错误: 无法正确分割 ID_ID '{pid}'。它可能不包含预期的分隔符 '{PAIR_SEPARATOR}'。")
        sim = 0.0 # 默认处理，或者跳过这个对
        all_sims.append(sim) # 确保即使错误也添加一个相似度，保持列表长度一致
        continue 

    idx_a = name_to_idx.get(a)
    idx_b = name_to_idx.get(b)

    if idx_a is None or idx_b is None:
        sim = 0.0
        # 警告：找不到图像的特征。这可能意味着生成对和提取特征的图像列表不一致。
        if idx_a is None: print(f"警告: 找不到图像 '{a}' 的特征。")
        if idx_b is None: print(f"警告: 找不到图像 '{b}' 的特征。")
    else:
        fa = feats[idx_a]
        fb = feats[idx_b]
        sim = float(np.dot(fa, fb))
    all_sims.append(sim)
    
# --- 遍历阈值，计算指标 ---
threshold_results = []
best_f1 = -1
best_threshold = -1

print("开始调优阈值...")
thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP/2, THRESHOLD_STEP) # 这里改正了一个笔误 THRESHID_STEP -> THRESHOLD_STEP

for thres in tqdm(thresholds, desc="Testing Thresholds"):
    y_pred = [1 if s >= thres else 0 for s in all_sims]
    
    if not y_true:
        print("警告: 验证集没有真实标签，无法计算指标。")
        break

    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    threshold_results.append({
        'threshold': thres,
        'f1_score': f1,
        'accuracy': acc,
        'precision': prec,
        'recall': rec
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thres

# --- 打印和可视化结果 ---
if threshold_results: # 确保有结果才打印和绘图
    results_df = pd.DataFrame(threshold_results)
    print("\n--- 阈值调优结果 ---")
    print(results_df.round(4))

    print(f"\n最佳 F1-Score: {best_f1:.4f} @ 阈值: {best_threshold:.2f}")

    # 可视化 F1-Score
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['f1_score'], marker='o', linestyle='-')
    plt.scatter(best_threshold, best_f1, color='red', s=100, zorder=5, label=f'Best F1: {best_f1:.4f} @ {best_threshold:.2f}')
    plt.title('F1-Score vs. Cosine Similarity Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('threshold_f1_curve.png')
    plt.show()

    print(f"推荐的阈值是: {best_threshold:.2f}")
else:
    print("没有生成阈值调优结果，请检查数据和流程。")
