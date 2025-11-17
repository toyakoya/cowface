# extract_features.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms

# 配置
TEST_DIR = "test/test"
OUT_FEATURE_FILE = "cow_model_final_trained50.npz"
BATCH_SIZE = 64  # 提取时可用较大 batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = "cow_model_final_trained50.pth"  # 你的已保存模型（包含 fc）

# 图像预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载模型并替换 fc 为 Identity（只做特征提取）
model = models.resnet50(pretrained=False)
num_classes = 7807 # 你需要知道你训练时的类别数
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Identity()
# 使用 strict=False 以兼容保存了 fc 参数的权重文件
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state, strict=False)
model = model.to(DEVICE)
model.eval()

# 收集图片列表并按批次处理
img_names = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])
n = len(img_names)
features = []
names = []

with torch.no_grad():
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Extracting"):
        batch_names = img_names[i:i+BATCH_SIZE]
        imgs = []
        for name in batch_names:
            path = os.path.join(TEST_DIR, name)
            img = Image.open(path).convert("RGB")
            img = transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0).to(DEVICE)
        feats = model(imgs)  # (B, D)
        feats = feats.cpu().numpy()
        # L2 归一化每个向量

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        features.append(feats)
        names.extend(batch_names)

# 合并并保存为 npz
features = np.vstack(features)  # (n_images, D)
np.savez_compressed(OUT_FEATURE_FILE, names=np.array(names), feats=features)
print(f"Saved {features.shape} to {OUT_FEATURE_FILE}")
# 修改后的 extract_features.py (伪代码 - 您需要根据实际情况实现)
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import os
# import numpy as np
# from torchvision import models
# from tqdm import tqdm

# # 配置
# MODEL_PATH = 'cow_model_final_trained50.pth' # 您的特征提取模型
# # INPUT_IMAGES_DIR = './validation_images' # 不再直接扫描目录，而是读取列表
# IMAGE_ROOT_DIR = './test/test' # 图像的根目录，因为 validation_image_list.txt 存的是相对路径
# IMAGE_LIST_FILE = 'validation_image_list.txt' # 包含要提取特征的图像列表
# OUTPUT_FEATURE_FILE = 'features_validation.npz'
# BATCH_SIZE = 32

# # 设备配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 1. 加载模型 (与之前相同)
# model = models.resnet50(pretrained=False) # 不加载预训练权重
# # 如果模型训练时fc层被替换，这里也要替换
# num_classes = 7807 # 你需要知道你训练时的类别数
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# model.fc = torch.nn.Identity() # 将FC层替换为恒等层
# model = model.to(device)
# model.eval() # 设置为评估模式

# # 2. 定义图像预处理 (与之前相同)
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 3. 提取特征
# image_full_paths = []
# image_names_for_feature_file = [] # 这些是存储在.npz文件中的'names'，应与ID_ID中的图像ID匹配

# # 读取图像列表文件
# if not os.path.exists(IMAGE_LIST_FILE):
#     print(f"错误: 图像列表文件 '{IMAGE_LIST_FILE}' 不存在。请先运行 generate_validation_pairs.py。")
#     exit()

# with open(IMAGE_LIST_FILE, 'r') as f:
#     for line in f:
#         relative_path = line.strip() # 例如: cow_A/img1.jpg
#         full_path = os.path.join(IMAGE_ROOT_DIR, relative_path)
#         if os.path.exists(full_path):
#             image_full_paths.append(full_path)
#             image_names_for_feature_file.append(relative_path) # 将相对路径作为ID存储
#         else:
#             print(f"警告: 图像文件 '{full_path}' 不存在，将跳过。")

# all_features = []
# print(f"正在从 {len(image_full_paths)} 张图片中提取特征...")

# # 批处理提取特征
# for i in tqdm(range(0, len(image_full_paths), BATCH_SIZE), desc="提取特征"):
#     batch_paths = image_full_paths[i:i + BATCH_SIZE]
#     batch_images = []
#     for p in batch_paths:
#         try:
#             img = Image.open(p).convert('RGB')
#             batch_images.append(transform(img))
#         except Exception as e:
#             print(f"加载或处理图像 {p} 失败: {e}")
#             # 处理失败的图像，例如跳过或用全零特征填充
#             # 这里的处理方式取决于您对错误处理的需求
#             continue
    
#     if not batch_images: # 如果批次中所有图像都失败了
#         continue

#     inputs = torch.stack(batch_images).to(device)
    
#     with torch.no_grad():
#         features = model(inputs)
    
#     # 归一化特征向量
#     features = features / features.norm(dim=1, keepdim=True)
#     all_features.append(features.cpu().numpy())

# if not all_features:
#     print("没有成功提取任何特征。请检查图像路径和处理过程。")
#     exit()

# all_features = np.concatenate(all_features, axis=0)

# # 4. 保存特征
# np.savez_compressed(OUTPUT_FEATURE_FILE, names=np.array(image_names_for_feature_file), feats=all_features)
# print(f"特征已保存到 {OUTPUT_FEATURE_FILE}")
