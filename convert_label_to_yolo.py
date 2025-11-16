"""
convert_labelme_to_yolo.py

功能：
- 将 LabeledImages/LabeledImages 下的 labelme JSON 标注转换为 YOLO 格式 labels (.txt)
- 将图片与 labels 拷贝到 yolo_dataset/images/{train,val} 和 yolo_dataset/labels/{train,val}
- 按比例划分 train/val（按图片随机划分）

使用说明：
- 修改 BASE_DIR 指向你的 LabeledImages 根目录（包含 .jpg 和 .json）
- 修改 OUT_DIR 为输出 yolo 数据集目录
- 运行：python convert_labelme_to_yolo.py
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import List

# 配置
BASE_DIR = Path("LabeledImages/")  # 含 .jpg 和 .json
OUT_DIR = Path("yolo_dataset")
TRAIN_RATIO = 0.8
SEED = 42
CLASS_NAME = "cow"  # 假设只有一个类
CLASS_ID = 0

random.seed(SEED)

def find_jsons(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.iterdir() if p.suffix.lower() == ".json"])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def convert_one(json_path: Path, out_images_train: Path, out_images_val: Path,
                out_labels_train: Path, out_labels_val: Path, is_train: bool):
    with json_path.open('r', encoding='utf-8') as f:
        j = json.load(f)
    img_name = j.get("imagePath") or json_path.with_suffix(".jpg").name
    img_w = j.get("imageWidth")
    img_h = j.get("imageHeight")
    shapes = j.get("shapes", [])
    if len(shapes) == 0:
        return False
    pts = shapes[0].get("points", [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    # Convert to YOLO format (x_center, y_center, w, h) normalized
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    txt_line = f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
    # Copy image and write label
    src_img = json_path.with_suffix(".jpg")
    if not src_img.exists():
        # try png
        src_img = json_path.with_suffix(".png")
        if not src_img.exists():
            print(f"Image not found for {json_path}")
            return False
    if is_train:
        dst_img = out_images_train / src_img.name
        dst_label = out_labels_train / (src_img.stem + ".txt")
    else:
        dst_img = out_images_val / src_img.name
        dst_label = out_labels_val / (src_img.stem + ".txt")
    shutil.copy2(src_img, dst_img)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write(txt_line)
    return True

def main():
    jsons = find_jsons(BASE_DIR)
    print(f"Found {len(jsons)} json files")
    # prepare output dirs
    img_train = OUT_DIR / "images" / "train"
    img_val = OUT_DIR / "images" / "val"
    lbl_train = OUT_DIR / "labels" / "train"
    lbl_val = OUT_DIR / "labels" / "val"
    for p in [img_train, img_val, lbl_train, lbl_val]:
        ensure_dir(p)
    # shuffle and split
    idxs = list(range(len(jsons)))
    random.shuffle(idxs)
    split = int(len(idxs) * TRAIN_RATIO)
    train_idx = set(idxs[:split])
    cnt = 0
    for i, js in enumerate(jsons):
        is_train = i in train_idx
        ok = convert_one(js, img_train, img_val, lbl_train, lbl_val, is_train)
        if ok:
            cnt += 1
    print(f"Converted {cnt} samples to YOLO format in {OUT_DIR}")

if __name__ == "__main__":
    main()
