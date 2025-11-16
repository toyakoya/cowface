import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy
import random
from pathlib import Path
from dataloader import get_dataloader
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

train_loader, val_loader, meta=get_dataloader()


# 1. 定义模型
num_classes = meta['train_num_classes']
model = models.resnet18(pretrained=True)

# 替换最后的全连接层
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 2. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# 3. 训练与验证
num_epochs = 100
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
best_loss = 100000
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # 训练阶段
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels_idx, labels_name, paths in train_loader:
    # for inputs, labels_name, paths in train_loader:
        inputs = inputs.to(device)
        # labels_idx=label_to_idx(labels_name)
        labels = labels_idx.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f"训练 loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

    # model.load_state_dict(best_model_wts)
    # print(f"训练完成，验证集最高准确率: {best_acc:.4f}")

    # 你可以保存模型
    # torch.save(model.state_dict(), 'cow_model.pth')

    # if epoch_acc > best_acc:
    #     best_acc = epoch_acc
    #     best_model_wts = copy.deepcopy(model.state_dict())

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
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

    # 保存最佳模型
    # if epoch_val_acc > best_acc:
    #     best_acc = epoch_val_acc
    #     best_model_wts = copy.deepcopy(model.state_dict())
    if epoch_val_loss < best_loss:
        best_loss=epoch_val_loss
        est_model_wts = copy.deepcopy(model.state_dict())
# 加载最佳模型
model.load_state_dict(best_model_wts)
print(f"训练完成，验证集最高准确率: {best_acc:.4f}")

# 你可以保存模型
torch.save(model.state_dict(), 'cow_model3.pth')

