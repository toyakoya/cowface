import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义模型（加载保存的模型，提取特征用）
# 加载模型（只用截断到特征层）

resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Identity()  # 用 Identity 替换
resnet.load_state_dict(torch.load('cow_model.pth'), strict=False)  # 使用 strict=False
resnet = resnet.to(device)
resnet.eval()

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. 采集所有测试图片路径（假设在 'archive/test/test/'）
test_img_dir = 'archive/test/test/'
test_img_names = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 构造字典：图片名 -> 特征向量
img_features = {}

def extract_feature(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img)  # 输出为特征（B,512）
    feat = feat.squeeze(0).cpu().numpy()
    feat /= np.linalg.norm(feat)  # 归一化
    return feat

print("开始提取所有测试图片特征...")
for fname in tqdm(test_img_names):
    path = os.path.join(test_img_dir, fname)
    feat = extract_feature(path)
    img_features[fname] = feat

# 4. 读取测试文件
test_csv = 'archive/test/test-0930.csv'
df = pd.read_csv(test_csv)

# 设定阈值（根据验证集调优，暂取0.8作为示例）
threshold = 0.77

# 5. 计算每对图片相似度
predictions = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pair_id = row['ID_ID']
    img1_name, img2_name = pair_id.split('_')
    feat1 = img_features.get(img1_name + '.jpg', None)
    feat2 = img_features.get(img2_name + '.jpg', None)
    if feat1 is None or feat2 is None:
        # 若缺失，默认预测为不相同
        pred = 0
    else:
        # 计算余弦相似度
        sim = np.dot(feat1, feat2)
        pred = 1 if sim >= threshold else 0
    predictions.append((pair_id, pred))

# 6. 保存输出
output_df = pd.DataFrame(predictions, columns=['ID_ID', 'TARGET'])
output_df.to_csv('submission.csv', index=False)
print("预测完成，已保存到 submission.csv")
