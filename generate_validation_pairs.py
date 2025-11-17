import pandas as pd
import os
import itertools
import random
from tqdm import tqdm # 导入tqdm用于显示进度

# 配置
TRAIN_IMAGES_DIR = "./train" # 您的原始训练集图像根目录
# OUTPUT_PAIRS_CSV = "validation_pairs_with_gt.csv" # 输出的验证对CSV文件
# 建议为不同场景设置不同的输出文件名，例如:
OUTPUT_PAIRS_CSV = "val_pairs_from_train_with_paths.csv"

# 从训练集中选取多少比例的牛类别来生成验证对
VALIDATION_COW_RATIO = 0.2  # 选取20%的牛类别用于生成验证对
MIN_COWS_FOR_VALIDATION = 2 # 至少需要多少头牛才能生成验证对

NUM_POSITIVE_PAIRS_PER_COW = 5 # 每头牛尝试生成多少个正样本对（如果图片足够）
NUM_NEGATIVE_PAIRS_RATIO = 2   # 负样本对数量相对于正样本对总数的倍数

def generate_pairs_from_subset(image_root_dir, val_cow_ratio, min_cows_for_val, num_pos_per_cow=5, neg_ratio=2):
    """
    从训练集中选取部分牛类别，并生成它们的图像对和真实标签。
    图像ID将使用相对于image_root_dir的完整路径。
    Args:
        image_root_dir (str): 原始训练集图像的根目录。
        val_cow_ratio (float): 用于生成验证对的牛类别占总牛类别的比例。
        min_cows_for_val (int): 至少需要多少头牛才能组成验证集。
        num_pos_per_cow (int): 每头牛生成多少个正样本对。
        neg_ratio (int): 负样本对数量是正样本对总数的多少倍。
    Returns:
        pd.DataFrame: 包含 'ID_ID' 和 'TARGET' 列的DataFrame。
        list: 选定用于验证的牛类别列表。
    """
    all_cows_in_train = [d for d in os.listdir(image_root_dir) if os.path.isdir(os.path.join(image_root_dir, d))]
    
    if len(all_cows_in_train) < min_cows_for_val:
        print(f"错误: 训练集中的牛类别太少 ({len(all_cows_in_train)}), 至少需要 {min_cows_for_val} 头牛才能生成验证对。")
        return pd.DataFrame(), []

    # 随机选择部分牛类别作为验证集
    num_val_cows = max(min_cows_for_val, int(len(all_cows_in_train) * val_cow_ratio))
    validation_cow_ids = random.sample(all_cows_in_train, num_val_cows)
    print(f"从 {len(all_cows_in_train)} 头牛中，选择了 {num_val_cows} 头牛 ({validation_cow_ids}) 用于生成验证对。")

    all_images_for_validation = {} # {cow_id: [full_img_path1, full_img_path2, ...]}

    # 收集选定牛的所有图像，存储完整路径
    print("收集验证牛的图像...")
    for cow_id in tqdm(validation_cow_ids, desc="Collecting images"):
        cow_path = os.path.join(image_root_dir, cow_id)
        # 存储相对于 TRAIN_IMAGES_DIR 的路径，例如 'cow_A/img1.jpg'
        images_in_folder = [os.path.join(cow_id, f) for f in os.listdir(cow_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if images_in_folder:
            all_images_for_validation[cow_id] = images_in_folder
        else:
            print(f"警告: 牛 '{cow_id}' 文件夹中没有找到图片。")
            validation_cow_ids.remove(cow_id) # 从列表中移除没有图片的牛

    # 确保至少有两头牛有图片，才能生成负样本对
    if len(validation_cow_ids) < 2:
        print("错误: 选定的验证牛类别中，至少需要两头牛有图片才能生成负样本对。")
        return pd.DataFrame(), []

    positive_pairs = []
    negative_pairs = []

    # 生成正样本对 (同一头牛的图片)
    print("生成正样本对...")
    for cow_id, images in all_images_for_validation.items():
        if len(images) >= 2:
            all_cow_pairs = list(itertools.combinations(images, 2))
            positive_pairs.extend(random.sample(all_cow_pairs, min(len(all_cow_pairs), num_pos_per_cow)))
        elif len(images) == 1:
            print(f"警告: 牛 '{cow_id}' 只有一张图片，无法生成正样本对。")

    print(f"生成的正样本对数量: {len(positive_pairs)}")

    # 生成负样本对 (不同头牛的图片)
    num_negative_to_generate = len(positive_pairs) * neg_ratio
    
    # 随机选择负样本对，直到达到目标数量或遍历完所有组合
    cow_id_combinations = list(itertools.combinations(validation_cow_ids, 2))
    random.shuffle(cow_id_combinations)

    print("生成负样本对...")
    for cow_id1, cow_id2 in tqdm(cow_id_combinations, desc="Generating neg pairs"):
        if len(negative_pairs) >= num_negative_to_generate:
            break
        
        images1 = all_images_for_validation.get(cow_id1)
        images2 = all_images_for_validation.get(cow_id2)

        if images1 and images2:
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            negative_pairs.append((img1, img2))
        
    # 如果通过组合生成的负样本对不够，继续随机抽取
    while len(negative_pairs) < num_negative_to_generate:
        if len(validation_cow_ids) < 2: # 再次检查是否有足够的牛
            break
        
        cow1, cow2 = random.sample(validation_cow_ids, 2)
        images1 = all_images_for_validation.get(cow1)
        images2 = all_images_for_validation.get(cow2)
        
        if images1 and images2:
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            # 确保不添加重复的对 (虽然概率低)
            if (img1, img2) not in negative_pairs and (img2, img1) not in negative_pairs:
                negative_pairs.append((img1, img2))
        
        # 避免无限循环，如果长时间无法生成新的唯一对则退出
        if len(negative_pairs) >= len(cow_id_combinations) * min(len(images1), len(images2)): # 粗略估计最大负样本数
             break


    negative_pairs = random.sample(negative_pairs, min(len(negative_pairs), num_negative_to_generate))
    print(f"生成的负样本对数量: {len(negative_pairs)}")

    # 组合所有对
    all_pairs_data = []
    for img1_path, img2_path in positive_pairs:
        # ID_ID 使用相对于 TRAIN_IMAGES_DIR 的完整路径作为 ID
        all_pairs_data.append([f"{img1_path}__{img2_path}", 1])

    for img1_path, img2_path in negative_pairs:
        all_pairs_data.append([f"{img1_path}__{img2_path}", 0])

    df = pd.DataFrame(all_pairs_data, columns=['ID__ID', 'TARGET'])
    return df, validation_cow_ids

if __name__ == "__main__":
    if not os.path.exists(TRAIN_IMAGES_DIR):
        print(f"错误: 训练集图像目录 '{TRAIN_IMAGES_DIR}' 不存在。请确保路径正确。")
    else:
        df_val_pairs, selected_val_cows = generate_pairs_from_subset(
            TRAIN_IMAGES_DIR,
            VALIDATION_COW_RATIO,
            MIN_COWS_FOR_VALIDATION,
            NUM_POSITIVE_PAIRS_PER_COW,
            NUM_NEGATIVE_PAIRS_RATIO
        )
        if not df_val_pairs.empty:
            df_val_pairs.to_csv(OUTPUT_PAIRS_CSV, index=False)
            print(f"验证集图像对及其真实标签已保存到 {OUTPUT_PAIRS_CSV}")
            print(f"总共生成了 {len(df_val_pairs)} 对验证样本。")
            print(f"用于生成验证对的牛类别有: {selected_val_cows}")
            
            # --- 额外步骤: 将这些验证牛的图像路径保存到一个文件，以便特征提取脚本使用 ---
            validation_image_paths = []
            for cow_id in selected_val_cows:
                cow_path = os.path.join(TRAIN_IMAGES_DIR, cow_id)
                images_in_folder = [os.path.join(cow_id, f) for f in os.listdir(cow_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                validation_image_paths.extend(images_in_folder)
            
            # 保存所有验证图像的相对路径到一个文本文件
            with open("validation_image_list.txt", "w") as f:
                for path in validation_image_paths:
                    f.write(f"{path}\n")
            print(f"所有用于验证的图像相对路径已保存到 validation_image_list.txt")
