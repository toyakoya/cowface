import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 配置
FEATURE_FILE = "cow_model.npz"  # 由 extract_features.py 生成
TEST_CSV = "test-1118.csv"
OUT_SUB = "./submission/submission_new.csv"
# COMP_SUB="submission.csv"
TEST_DIR = "./test-new"  # 用于容错（如果需要检查文件是否存在）
THRESHOLD = 0.72  # 余弦相似度阈值，建议用验证集调优

# 加载特征
data = np.load(FEATURE_FILE, allow_pickle=True)
names = data["names"].astype(str)  # array of filenames like '0001.jpg'
feats = data["feats"]  # shape (n_images, D)

# 构建 name->index 映射，支持带或不带扩展名的键
name_to_idx = {}
for i, n in enumerate(names):
    name_to_idx[n] = i
    base = os.path.splitext(n)[0]
    name_to_idx[base] = i

# 读取测试对 CSV
df = pd.read_csv(TEST_CSV)
pairs = df['ID_ID'].tolist()

results = []
for pid in tqdm(pairs, desc="Predicting"):
    a, b = pid.split('_')
    idx_a = name_to_idx.get(a)
    idx_b = name_to_idx.get(b)
    if idx_a is None or idx_b is None:
        # 若找不到任意图，默认预测为0
        pred = 0
    else:
        fa = feats[idx_a]
        fb = feats[idx_b]
        sim = float(np.dot(fa, fb))  # feats 已归一化，余弦相似度 = 点积
        pred = 1 if sim >= THRESHOLD else 0
    results.append((pid, pred))
# comp=readcsv
out_df = pd.DataFrame(results, columns=['ID_ID', 'TARGET'])
out_df.to_csv(OUT_SUB, index=False)
print(f"Saved predictions to {OUT_SUB}")
