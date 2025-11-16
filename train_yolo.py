"""
train_yolo.py

YOLOv8 微调训练示例（基于 ultralytics）

使用说明：
- 安装依赖: pip install ultralytics==8.* torch torchvision pillow
- 确保 yolo_dataset 已生成（使用 convert_labelme_to_yolo.py）
- 修改 DATA_YAML、MODEL_NAME、EPOCHS、BATCH、IMGSZ 如需调整
- 运行: python train_yolo.py
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

# 配置
YOLO_WEIGHTS = "weights/yolov8n.pt"
OUT_DIR = "yolov8_runs"
DATA_DIR = Path("yolo_dataset")
DATA_YAML = "data.yaml"
EPOCHS = 30
BATCH = 8   # 受显存限制，8GB 推荐 4~8
IMGSZ = 640
DEVICE = 0  # GPU id or 'cpu'

# 生成 data.yaml
data_cfg = {
    "train": str(DATA_DIR / "images" / "train"),
    "val": str(DATA_DIR / "images" / "val"),
    "nc": 1,
    "names": ["cow"]
}
with open(DATA_YAML, "w") as f:
    yaml.dump(data_cfg, f)
print(f"Written data yaml to {DATA_YAML}")

# 训练
model = YOLO(YOLO_WEIGHTS)
# model.info()  # 可选，查看模型信息
model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, save=True, patience=10)
# 训练后权重保存在 ./runs/detect/train/weights/best.pt 或类似路径
print("Training finished. Check runs/ directory for results.")
