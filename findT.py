import pandas as pd

# 定义两个提交文件的路径
submission_075_path = "./submission/submission_075.csv"
submission_080_path = "./submission/submission_076.csv"

# 加载两个提交文件
df_075 = pd.read_csv(submission_075_path)
df_080 = pd.read_csv(submission_080_path)

# 为了方便对比，确保两个DataFrame的顺序一致 (通常'ID_ID'列是唯一的且有序的)
# 如果不确定，可以基于'ID_ID'进行排序
df_075 = df_075.sort_values(by='ID_ID').reset_index(drop=True)
df_080 = df_080.sort_values(by='ID_ID').reset_index(drop=True)

# 将两个DataFrame合并，以便进行逐行对比
# 使用suffixes来区分来自不同阈值的结果
merged_df = pd.merge(df_075, df_080, on='ID_ID', suffixes=('_075', '_080'))

# 找出'TARGET'列发生变化的图像对
changed_pairs_df = merged_df[merged_df['TARGET_075'] != merged_df['TARGET_080']]

# 打印发生变化的图像对数量
print(f"在阈值 0.75 和 0.80 之间，共有 {len(changed_pairs_df)} 对图像的预测结果发生了变化。")

# 打印发生变化的图像对及其在新旧阈值下的预测结果
print("\n发生变化的图像对及其预测结果：")
print(changed_pairs_df[['ID_ID', 'TARGET_075', 'TARGET_080']])

# 您可以将这些变化的图像对保存到一个新的CSV文件，以便后续分析
changed_pairs_df[['ID_ID', 'TARGET_075', 'TARGET_080']].to_csv("changed_predictions7576.csv", index=False)
print("\n变化的预测结果已保存到 changed_predictions7580.csv")

# 进一步分析：
# 可以加载原始的特征文件或测试CSV，结合这些变化的图像对进行分析。
# 例如，如果您想知道这些图像对的实际相似度，可以重新计算或从原始特征中提取。