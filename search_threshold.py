"""
search_threshold.py

在验证集上搜索 detection confidence threshold，使得 precision 达到目标值或查看 precision/recall 曲线。

假设你已经训练好模型并有 best.pt 路径（修改 MODEL_PT）。
会在 yolo_dataset/images/val 读取图片并用模型预测，然后与 yolo_dataset/labels/val 的标签比较计算 TP/FP/FN。

注意：此脚本为简单实现，适用于单对象场景（每图一个 ground-truth box）。
"""

import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import glob

MODEL_PT = "runs/detect/train/weights/best.pt" # 修改为实际路径
VAL_IMG_DIR = Path("yolo_dataset/images/val")
VAL_LABEL_DIR = Path("yolo_dataset/labels/val")
DEVICE = 0
IOU_THRESH = 0.5

def load_gt_for_image(img_path):
    stem = Path(img_path).stem
    label_path = VAL_LABEL_DIR / (stem + ".txt")
    if not label_path.exists():
        return None
    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        # format: class xc yc w h (normalized)
        xc, yc, w, h = map(float, parts[1:5])
    img = Image.open(img_path)
    W, H = img.size
    x1 = (xc - w/2.0) * W
    y1 = (yc - h/2.0) * H
    x2 = (xc + w/2.0) * W
    y2 = (yc + h/2.0) * H
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    if boxAArea == 0 or boxBArea == 0:
        return 0.0
    return inter / (boxAArea + boxBArea - inter)

def main():
    model = YOLO(MODEL_PT)
    img_paths = sorted(list(VAL_IMG_DIR.glob("*.*")))
    results = {}
    for p in img_paths:
        res = model.predict(source=str(p), conf=0.01, device=DEVICE, verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                dets.append((list(map(float, b)), float(c)))
        results[str(p)] = dets
    # evaluate precision at various thresholds
    thresholds = np.linspace(0.1, 0.99, 19)
    metrics = []
    for th in thresholds:
        TP = FP = FN = 0
        for p in img_paths:
            gt = load_gt_for_image(p)
            dets = results[str(p)]
            # choose best det >= th
            dets_f = [d for d in dets if d[1] >= th]
            if len(dets_f) == 0:
                # no detection
                if gt is not None:
                    FN += 1
                continue
            # take highest score
            best_box, score = max(dets_f, key=lambda x: x[1])
            if gt is None:
                FP += 1
            else:
                if iou(best_box, gt) >= IOU_THRESH:
                    TP += 1
                else:
                    FP += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        metrics.append((th, precision, recall, TP, FP, FN))
    # print table
    print("th, precision, recall, TP, FP, FN")
    for row in metrics:
        print(f"{row[0]:.2f}, {row[1]:.3f}, {row[2]:.3f}, {row[3]}, {row[4]}, {row[5]}")
    # suggest threshold where precision >= 0.95 if exists
    cand = [r for r in metrics if r[1] >= 0.95]
    if cand:
        best = cand[0]
        print(f"Suggested threshold for precision>=0.95: {best[0]:.2f}")
    else:
        print("No threshold found with precision>=0.95; inspect metrics.")
if __name__ == "__main__":
    main()
