# integrate_detection_to_recognition.py
"""
Integrate detection results into recognition training set (detection-only mode)

- Uses a trained YOLOv8 detection model to detect the most prominent cow face in each image
  under UnlabeledImages/UnlabeledImages/<cow_id>/.
- Selects the top-scoring bbox if score >= SELECT_THRESH (default 0.74).
- Applies a small margin, crops the image, saves to archive/tain/tain/<cow_id>/.
- Logs all operations into LOG_CSV.

Usage:
  1) Ensure ultralytics is installed: pip install ultralytics torch torchvision pillow tqdm
  2) Set MODEL_PT to your trained YOLOv8 best checkpoint path.
  3) Run: python integrate_detection_to_recognition.py
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from ultralytics import YOLO

# ----------------- CONFIG -----------------
MODEL_PT = "runs/detect/train/weights/best.pt"  # <-- 修改为你的 best.pt 路径
SELECT_THRESH = 0.74    # detection score 阈值（根据 validation 建议）
BATCH = 8               # 推理批量大小（显存允许下可调）
DEVICE = 0              # GPU id (ultralytics accepts device=0) or 'cpu'
UNLABELED_ROOT = Path("UnlabeldImages/")
RECOG_TRAIN_ROOT = Path("./train")
LOG_CSV = "detection_integration_log.csv"
MARGIN_RATIO = 0.05     # 裁切外扩比例
SAVE_FORMAT = "jpg"
IMG_QUALITY = 95
# ------------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def apply_margin(box, img_w, img_h, margin_ratio=0.05):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    mx = w * margin_ratio
    my = h * margin_ratio
    nx1 = max(0, int(round(x1 - mx)))
    ny1 = max(0, int(round(y1 - my)))
    nx2 = min(img_w, int(round(x2 + mx)))
    ny2 = min(img_h, int(round(y2 + my)))
    return [nx1, ny1, nx2, ny2]

def save_crop(crop_img, target_dir: Path, base_name: str, score: float):
    ensure_dir(target_dir)
    out_name = f"{base_name}_det_{int(score*100):02d}.{SAVE_FORMAT}"
    out_path = target_dir / out_name
    crop_img.save(out_path, quality=IMG_QUALITY)
    return str(out_path)

def main():
    # validate paths
    if not Path(MODEL_PT).exists():
        raise FileNotFoundError(f"Detection model not found: {MODEL_PT}")
    if not UNLABELED_ROOT.exists():
        raise FileNotFoundError(f"Unlabeled root not found: {UNLABELED_ROOT}")
    ensure_dir(RECOG_TRAIN_ROOT)

    # load detection model
    det_model = YOLO(MODEL_PT)

    log_rows = []
    cow_dirs = sorted([p for p in UNLABELED_ROOT.iterdir() if p.is_dir()])
    if len(cow_dirs) == 0:
        print(f"No cow folders found under {UNLABELED_ROOT}")
        return

    for cow_dir in tqdm(cow_dirs, desc="Cow folders"):
        cow_id = cow_dir.name
        img_files = sorted([p for p in cow_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if len(img_files) == 0:
            continue

        # batch infer per folder
        paths = [str(p) for p in img_files]
        results = det_model.predict(source=paths, conf=0.01, device=DEVICE, batch=BATCH, verbose=False)

        for img_path, res in zip(paths, results):
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy, confs):
                    dets.append({"bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "score": float(c)})
            if not dets:
                # no detection
                log_rows.append({
                    "source_image": img_path,
                    "target_cow_id": cow_id,
                    "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                    "score": "", "saved_path": ""
                })
                continue

            # select highest score bbox
            best = max(dets, key=lambda x: x["score"])
            score = best["score"]
            if score < SELECT_THRESH:
                # below threshold: skip
                log_rows.append({
                    "source_image": img_path,
                    "target_cow_id": cow_id,
                    "bbox_x1": best["bbox"][0], "bbox_y1": best["bbox"][1],
                    "bbox_x2": best["bbox"][2], "bbox_y2": best["bbox"][3],
                    "score": score, "saved_path": ""
                })
                continue

            bbox = best["bbox"]
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            bboxm = apply_margin(bbox, W, H, MARGIN_RATIO)
            crop = img.crop((bboxm[0], bboxm[1], bboxm[2], bboxm[3]))

            target_dir = RECOG_TRAIN_ROOT / cow_id
            base = Path(img_path).stem
            out_path = save_crop(crop, target_dir, base, score)

            log_rows.append({
                "source_image": img_path,
                "target_cow_id": cow_id,
                "bbox_x1": bboxm[0], "bbox_y1": bboxm[1], "bbox_x2": bboxm[2], "bbox_y2": bboxm[3],
                "score": score, "saved_path": out_path
            })

    # write CSV log
    keys = ["source_image", "target_cow_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "score", "saved_path"]
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in log_rows:
            writer.writerow(r)

    print(f"Done. Saved crops logged to {LOG_CSV}")

if __name__ == "__main__":
    main()
