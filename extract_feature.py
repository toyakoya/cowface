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
OUT_FEATURE_FILE = "features_test_detec3.npz"
BATCH_SIZE = 64  # 提取时可用较大 batch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
MODEL_PATH = "cow_model3.pth"  # 你的已保存模型（包含 fc）

# 图像预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载模型并替换 fc 为 Identity（只做特征提取）
model = models.resnet18(pretrained=False)
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
