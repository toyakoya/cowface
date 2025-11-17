import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
import random
from pathlib import Path
from dataloader import get_dataloader 
import os
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
train_loader, val_loader, meta=get_dataloader()


# --- 新增: 配置要加载的权重文件路径 ---
# 如果您想加载之前保存的 cow_model3.pth，就保持这个路径
# 如果您想从另一个文件加载，请修改这里
LOAD_WEIGHTS_PATH = 'cow_model_final_trained50.pth'
# ------------------------------------


# 1. 定义模型
num_classes = meta['train_num_classes']
print(num_classes)
model = models.resnet50(pretrained=True) # pretrained=True 会加载ImageNet预训练权重

# 替换最后的全连接层，使其适应您的数据集类别数量
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 2. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
Explr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# 3. 训练与验证
num_epochs = 50
best_model_wts = copy.deepcopy(model.state_dict()) # 初始化为当前模型状态
best_acc = 0.0
best_loss = float('inf') # 使用正无穷大作为初始最佳损失，确保任何有效损失都会被视为更优

# --- 新增: 加载保存的权重 (如果存在) ---
if os.path.exists(LOAD_WEIGHTS_PATH):
    print(f"正在加载模型权重: {LOAD_WEIGHTS_PATH}")
    # map_location 参数确保模型加载到正确的设备上，即使保存时是在不同设备上
    model.load_state_dict(torch.load(LOAD_WEIGHTS_PATH, map_location=device))
    print("模型权重加载成功。")

    # --- 评估加载模型的初始性能，以正确初始化 best_acc 和 best_loss ---
    # 这样做可以确保如果加载的模型本身就是最好的，后续训练也不会轻易覆盖它
    print("评估加载模型的初始性能...")
    model.eval() # 切换到评估模式
    initial_val_loss = 0.0
    initial_val_corrects = 0
    with torch.no_grad():
        for inputs, labels_idx, labels_name, paths in val_loader:
            inputs = inputs.to(device)
            labels = labels_idx.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            initial_val_loss += loss.item() * inputs.size(0)
            initial_val_corrects += torch.sum(preds == labels.data)

    initial_epoch_val_loss = initial_val_loss / len(val_loader.dataset)
    initial_epoch_val_acc = initial_val_corrects.double() / len(val_loader.dataset)
    
    best_loss = initial_epoch_val_loss
    best_acc = initial_epoch_val_acc
    best_model_wts = copy.deepcopy(model.state_dict()) # 更新 best_model_wts 为加载模型的权重
    
    print(f"加载模型初始验证 loss: {initial_epoch_val_loss:.4f}  acc: {initial_epoch_val_acc:.4f}")
    # ------------------------------------------------------------------
else:
    print(f"未找到预训练权重文件: {LOAD_WEIGHTS_PATH}，将从头开始训练 (使用ImageNet预训练权重作为起点)。")

# ------------------------------------------

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # 训练阶段
    model.train() # 切换到训练模式
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels_idx, labels_name, paths in train_loader:
        inputs = inputs.to(device)
        labels = labels_idx.to(device)

        optimizer.zero_grad() # 梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f"训练 loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

    # 验证阶段
    model.eval() # 切换到评估模式
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad(): # 在验证阶段不计算梯度
        for inputs, labels_idx, labels_name, paths in val_loader:
            inputs = inputs.to(device)
            labels = labels_idx.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
    print(f"验证 loss: {epoch_val_loss:.4f}  acc: {epoch_val_acc:.4f}")
    Explr.step()
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        best_acc = epoch_val_acc # 在更新最佳损失时，同时记录对应的最佳准确率
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"--- 发现更优模型 (验证 loss: {best_loss:.4f}, acc: {best_acc:.4f})，已更新最佳权重。---")

# 训练结束后，加载在验证集上表现最佳的模型权重
model.load_state_dict(best_model_wts)
print(f"\n训练完成。在验证集上观察到的最佳 loss: {best_loss:.4f}, 对应的最佳准确率: {best_acc:.4f}")

# 保存最终（即整个训练过程中表现最佳的）模型
# 建议使用一个不同的文件名，以防您想保留以前的训练结果
FINAL_SAVE_PATH = 'cow_model_final_trained50.pth' 
torch.save(model.state_dict(), FINAL_SAVE_PATH)
print(f"最佳模型已保存到 {FINAL_SAVE_PATH}")