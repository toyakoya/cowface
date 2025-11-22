import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torchvision import models
from tqdm import tqdm

# 配置
MODEL_PATH = 'cow_model3.pth' # 您的特征提取模型
# INPUT_IMAGES_DIR = './validation_images' # 不再直接扫描目录，而是读取列表
IMAGE_ROOT_DIR = './train' # 图像的根目录，因为 validation_image_list.txt 存的是相对路径
IMAGE_LIST_FILE = 'validation_image_list.txt' # 包含要提取特征的图像列表
OUTPUT_FEATURE_FILE = 'features_validation18.npz'
BATCH_SIZE = 32

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载模型 (与之前相同)
model = models.resnet18(pretrained=False) # 不加载预训练权重
# 如果模型训练时fc层被替换，这里也要替换
num_classes = 7807 # 你需要知道你训练时的类别数
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.fc = torch.nn.Identity() # 将FC层替换为恒等层
model = model.to(device)
model.eval() # 设置为评估模式

# 2. 定义图像预处理 (与之前相同)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 提取特征
image_full_paths = []
image_names_for_feature_file = [] # 这些是存储在.npz文件中的'names'，应与ID_ID中的图像ID匹配

# 读取图像列表文件
if not os.path.exists(IMAGE_LIST_FILE):
    print(f"错误: 图像列表文件 '{IMAGE_LIST_FILE}' 不存在。请先运行 generate_validation_pairs.py。")
    exit()

with open(IMAGE_LIST_FILE, 'r') as f:
    for line in f:
        relative_path = line.strip() # 例如: cow_A/img1.jpg
        full_path = os.path.join(IMAGE_ROOT_DIR, relative_path)
        if os.path.exists(full_path):
            image_full_paths.append(full_path)
            image_names_for_feature_file.append(relative_path) # 将相对路径作为ID存储
        else:
            print(f"警告: 图像文件 '{full_path}' 不存在，将跳过。")

all_features = []
print(f"正在从 {len(image_full_paths)} 张图片中提取特征...")

# 批处理提取特征
for i in tqdm(range(0, len(image_full_paths), BATCH_SIZE), desc="提取特征"):
    batch_paths = image_full_paths[i:i + BATCH_SIZE]
    batch_images = []
    for p in batch_paths:
        try:
            img = Image.open(p).convert('RGB')
            batch_images.append(transform(img))
        except Exception as e:
            print(f"加载或处理图像 {p} 失败: {e}")
            # 处理失败的图像，例如跳过或用全零特征填充
            # 这里的处理方式取决于您对错误处理的需求
            continue
    
    if not batch_images: # 如果批次中所有图像都失败了
        continue

    inputs = torch.stack(batch_images).to(device)
    
    with torch.no_grad():
        features = model(inputs)
    
    # 归一化特征向量
    features = features / features.norm(dim=1, keepdim=True)
    all_features.append(features.cpu().numpy())

if not all_features:
    print("没有成功提取任何特征。请检查图像路径和处理过程。")
    exit()

all_features = np.concatenate(all_features, axis=0)

# 4. 保存特征
np.savez_compressed(OUTPUT_FEATURE_FILE, names=np.array(image_names_for_feature_file), feats=all_features)
print(f"特征已保存到 {OUTPUT_FEATURE_FILE}")
