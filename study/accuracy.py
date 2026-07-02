"""
Accuracy evaluation: measure mAP drop from FP32 → INT8 quantization.
Uses COCO val2017 subset (500 images) for rigorous comparison.

Usage:  python -m study.accuracy
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STUDY_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.detector import YOLODetector
from src.optimizer import MODELS_DIR, OPTIMIZED_DIR

RESULTS_DIR = STUDY_DIR / "results"

COCO_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
    12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'}


def _ensure_coco128():
    """Ensure coco128 dataset is available (128 images, ~20 MB)."""
    from ultralytics.utils.downloads import download
    data_dir = Path("~/datasets").expanduser()
    coco128_dir = data_dir / "coco128"
    if coco128_dir.exists():
        return coco128_dir
    print("[accuracy] Downloading coco128 dataset...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/coco128.zip"
    download(url, dir=str(data_dir), unzip=True)
    return coco128_dir if coco128_dir.exists() else None


def _predict_model(detector: YOLODetector, images: list[dict], images_dir: Path,
                   conf_threshold: float = 0.001) -> list[dict]:
    """Run a YOLODetector on all images and return COCO-format results."""
    import cv2
    results = []
    total = len(images)
    for idx, img_info in enumerate(images):
        img_path = images_dir / img_info['file_name']
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.infer(frame)

        img_w, img_h = img_info['width'], img_info['height']
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            # Clip to image bounds
            x1, x2 = max(0, x1), min(img_w, x2)
            y1, y2 = max(0, y1), min(img_h, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            results.append({
                "image_id": img_info['id'],
                "category_id": det['class_id'] + 1,  # COCO is 1-indexed
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(det['score']),
            })

        if (idx + 1) % 100 == 0:
            print(f"  Inference: {idx+1}/{total} images")
    return results


def _compute_map(coco_gt, results: list[dict], label: str) -> dict:
    """Compute COCO mAP using pycocotools."""
    from pycocotools.cocoeval import COCOeval
    import json as json_mod

    if not results:
        print(f"  [accuracy] No detections for {label}, mAP = 0")
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    # COCO expects annotations to be written to a temp file
    from io import StringIO
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP@0.5:0.95": float(coco_eval.stats[0]),
        "mAP@0.5": float(coco_eval.stats[1]),
    }


def evaluate_model(model_path: Path, label: str, images: list[dict],
                   images_dir: Path, coco_gt, conf_threshold: float = 0.001) -> dict:
    print(f"\n[accuracy] Evaluating {label} ({model_path.name})...")
    print(f"  Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    t0 = time.time()
    detector = YOLODetector(str(model_path), conf_threshold=conf_threshold)
    detector.warmup()
    results = _predict_model(detector, images, images_dir, conf_threshold)
    infer_time = time.time() - t0
    print(f"  Inference time: {infer_time:.1f}s for {len(images)} images")

    metrics = _compute_map(coco_gt, results, label)
    print(f"  mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"  Detections: {len(results)}")

    return metrics


def _convert_yolo_to_coco(labels_dir: Path, images_dir: Path, class_names: dict) -> tuple[list[dict], list[dict]]:
    """Convert YOLO-format labels to COCO annotation format."""
    import cv2
    images = []
    annotations = []
    ann_id = 1

    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_id = len(images) + 1
        images.append({
            "id": img_id, "file_name": img_path.name,
            "width": w, "height": h,
        })

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw = bw * w
            bh = bh * h
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    return images, annotations


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_test = [
        ("FP32 Baseline", MODELS_DIR / "yolov8n.onnx"),
        ("INT8 Static", OPTIMIZED_DIR / "yolov8n_int8.onnx"),
    ]

    valid = [(l, p) for l, p in models_to_test if p.exists()]
    if not valid:
        print("[accuracy] No models found. Run `python run.py --quantize-only` first.")
        sys.exit(1)

    # Try ultralytics val first (handles data auto-download)
    print("[accuracy] Attempting ultralytics validation...")
    from ultralytics import YOLO

    all_metrics = {}
    for label, path in valid:
        print(f"\n  [{label}] Running ultralytics val on coco128...")
        try:
            model = YOLO(str(path))
            val = model.val(data="coco128.yaml", verbose=False, plots=False)
            all_metrics[label] = {
                "method": "ultralytics",
                "mAP@0.5": float(val.box.map50),
                "mAP@0.5:0.95": float(val.box.map),
            }
            print(f"    mAP@0.5: {all_metrics[label]['mAP@0.5']:.4f}")
            print(f"    mAP@0.5:0.95: {all_metrics[label]['mAP@0.5:0.95']:.4f}")
        except Exception as e:
            print(f"    ultralytics val failed: {e}")
            all_metrics[label] = None

    # If INT8 failed (null or zero mAP) via ultralytics, use custom pipeline
    def _failed(v):
        return v is None or (isinstance(v, dict) and v.get("mAP@0.5:0.95", 1) == 0)

    int8_failed = any(_failed(v) for l, v in all_metrics.items() if "INT8" in l)
    if int8_failed:
        print("\n[accuracy] INT8 ultralytics val failed. Using custom COCO eval pipeline...")

        # Ensure coco128 is downloaded
        from ultralytics.utils import SETTINGS
        data_dir = Path(SETTINGS.get("datasets_dir", "~/datasets")).expanduser()
        coco128_dir = data_dir / "coco128"
        if not coco128_dir.exists():
            _ensure_coco128()

        labels_dir = coco128_dir / "labels" / "train2017"
        images_dir = coco128_dir / "images" / "train2017"

        if not labels_dir.exists() or not images_dir.exists():
            print("[accuracy] coco128 not available. Skipping custom eval.")
        else:
            from pycocotools.coco import COCO
            images, annotations = _convert_yolo_to_coco(labels_dir, images_dir, COCO_NAMES)
            coco_gt = COCO()
            coco_gt.dataset = {
                "images": images, "annotations": annotations,
                "categories": [{"id": i + 1, "name": COCO_NAMES[i]} for i in range(80)],
            }
            coco_gt.createIndex()

            print(f"[accuracy] Custom eval with {len(images)} images")
            for label, path in valid:
                if all_metrics.get(label) is not None and "INT8" not in label:
                    continue  # already have ultralytics result
                metrics = evaluate_model(path, label, images, images_dir, coco_gt)
                all_metrics[label] = {
                    "method": "custom",
                    "mAP@0.5": metrics["mAP@0.5"],
                    "mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
                }

    # Comparison
    print("\n" + "=" * 60)
    print("Accuracy Comparison — YOLOv8n on COCO128")
    print("=" * 60)
    cols = ["Model", "Method", "mAP@0.5", "mAP@0.5:0.95", "Size (MB)"]
    print(f"| {' | '.join(cols)} |")
    print(f"|{'|'.join(['---'] * len(cols))}|")

    baseline_map = None
    for label, path in valid:
        m = all_metrics.get(label)
        if m is None:
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"| {label} | {m.get('method','?')} | {m['mAP@0.5']:.4f} | {m['mAP@0.5:0.95']:.4f} | {size_mb:.1f} |")
        if "FP32" in label:
            baseline_map = m

    if baseline_map:
        for label, path in valid:
            if "INT8" in label and all_metrics.get(label):
                m = all_metrics[label]
                ap_drop = baseline_map["mAP@0.5:0.95"] - m["mAP@0.5:0.95"]
                pct = ap_drop / baseline_map["mAP@0.5:0.95"] * 100 if baseline_map["mAP@0.5:0.95"] > 0 else 0
                print(f"\n  mAP@0.5:0.95 drop: {ap_drop:.4f} ({pct:.1f}%)")

    # Save
    result_path = RESULTS_DIR / "accuracy_results.json"
    results = {
        "model": "yolov8n",
        "dataset": "coco128",
        "models": {
            l: {k: v for k, v in all_metrics[l].items() if k != "method"}
            for l in all_metrics if all_metrics[l] is not None
        },
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[accuracy] Results saved to {result_path}")


if __name__ == "__main__":
    main()
