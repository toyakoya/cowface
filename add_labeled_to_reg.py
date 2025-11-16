# add_labeled_to_recog.py
"""
将 LabeledImages 中的带标注图片裁切并加入识别训练集 ./archive/tain/tain/<cow_id>/
假设每个 json 为 labelme 格式，且每个 json 只含一个 rectangle 标注。
"""

import os
import json
from pathlib import Path
from PIL import Image
import csv

# 配置
LABELED_DIR = Path("LabeledImages")  # 含 .jpg 与 .json
RECOG_TRAIN_ROOT = Path("train")      # 目标识别训练集根目录
LOG_CSV = "labeled_integration_log.csv"
MARGIN_RATIO = 0.05   # bbox 扩展比例
SAVE_FORMAT = "jpg"
IMG_QUALITY = 95

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_bbox_from_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        j = json.load(f)
    shapes = j.get("shapes", [])
    if len(shapes) == 0:
        return None, None, None
    pts = shapes[0].get("points", [])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    img_w = j.get("imageWidth")
    img_h = j.get("imageHeight")
    return [float(x1), float(y1), float(x2), float(y2)], img_w, img_h

def apply_margin(box, img_w, img_h, margin_ratio=0.05):
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    mx = w * margin_ratio
    my = h * margin_ratio
    nx1 = max(0, int(round(x1 - mx)))
    ny1 = max(0, int(round(y1 - my)))
    nx2 = min(img_w, int(round(x2 + mx)))
    ny2 = min(img_h, int(round(y2 + my)))
    return [nx1, ny1, nx2, ny2]

def main():
    if not LABELED_DIR.exists():
        raise FileNotFoundError(f"Labeled dir not found: {LABELED_DIR}")
    ensure_dir(RECOG_TRAIN_ROOT)
    json_files = sorted([p for p in LABELED_DIR.iterdir() if p.suffix.lower() == ".json"])
    log_rows = []
    for jp in json_files:
        try:
            bbox, img_w, img_h = read_bbox_from_json(jp)
            if bbox is None:
                continue
            img_path = jp.with_suffix(".jpg")
            if not img_path.exists():
                img_path = jp.with_suffix(".png")
                if not img_path.exists():
                    print(f"Image not found for {jp.name}, skip")
                    continue
            img = Image.open(img_path).convert("RGB")
            if img_w is None or img_h is None:
                img_w, img_h = img.size
            bboxm = apply_margin(bbox, img_w, img_h, MARGIN_RATIO)
            crop = img.crop((bboxm[0], bboxm[1], bboxm[2], bboxm[3]))
            # cow_id from filename prefix before underscore
            stem = img_path.stem  # e.g., KQ25003806_2
            cow_id = stem.split("_")[0]
            target_dir = RECOG_TRAIN_ROOT / cow_id
            ensure_dir(target_dir)
            out_name = f"{stem}_label.{SAVE_FORMAT}"
            out_path = target_dir / out_name
            crop.save(out_path, quality=IMG_QUALITY)
            log_rows.append({
                "source_image": str(img_path),
                "cow_id": cow_id,
                "bbox_x1": bboxm[0], "bbox_y1": bboxm[1], "bbox_x2": bboxm[2], "bbox_y2": bboxm[3],
                "saved_path": str(out_path)
            })
        except Exception as e:
            print(f"Error processing {jp}: {e}")
    # write log
    keys = ["source_image","cow_id","bbox_x1","bbox_y1","bbox_x2","bbox_y2","saved_path"]
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in log_rows:
            writer.writerow(r)
    print(f"Done. Processed {len(log_rows)} labeled images. Log: {LOG_CSV}")

if __name__ == "__main__":
    main()