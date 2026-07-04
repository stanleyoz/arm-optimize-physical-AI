import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.benchmark import run_benchmarks, generate_chart
from src.camera import Camera
from src.detector import YOLODetector, draw_detections
from src.optimizer import (
    export_onnx,
    quantize_onnx_static,
    print_comparison,
    MODELS_DIR,
    OPTIMIZED_DIR,
)
from src.tracker import PanTiltTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Arm-Optimized Real-Time Vision Assistant")
    parser.add_argument("--model", default="yolov8n", help="YOLO model name (default: yolov8n)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    parser.add_argument("--demo", action="store_true", help="Run real-time camera demo")
    parser.add_argument("--track", action="store_true", help="Run Physical AI pan-tilt tracking demo")
    parser.add_argument("--serial", default="/dev/ttyACM0", help="Serial port for servo controller")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--quantize-only", action="store_true", help="Only export + quantize, no benchmark or demo")
    parser.add_argument("--runs", type=int, default=200, help="Inference runs for benchmark (default: 200)")
    return parser.parse_args()


def run_demo(model_path: Path, camera_id: int, conf_threshold: float):
    print("[main] Loading detector...")
    detector = YOLODetector(str(model_path), conf_threshold=conf_threshold)
    detector.warmup()

    print(f"[main] Opening camera #{camera_id}...")
    cam = Camera(source=camera_id, width=640, height=480)
    cam.start()

    cv2.namedWindow("Arm Vision Assistant", cv2.WINDOW_NORMAL)
    fps_buffer = []

    print("[main] Demo running — press 'q' to quit, 's' to save frame")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            t0 = time.perf_counter()
            detections = detector.infer(frame)
            t1 = time.perf_counter()

            fps_buffer.append(1.0 / (t1 - t0))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            current_fps = np.mean(fps_buffer)

            display = draw_detections(frame.copy(), detections)
            cv2.putText(
                display,
                f"FPS: {current_fps:.1f} | Detections: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            model_label = Path(detector.model_path).stem
            cv2.putText(
                display,
                f"Model: {model_label}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

            cv2.imshow("Arm Vision Assistant", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = int(time.time())
                cv2.imwrite(f"capture_{timestamp}.jpg", display)
                print(f"[main] Saved capture_{timestamp}.jpg")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def run_tracker(model_path: Path, camera_id: int, conf_threshold: float, serial_port: str):
    print("[main] Loading detector...")
    detector = YOLODetector(str(model_path), conf_threshold=conf_threshold)
    detector.warmup()

    print("[main] Initializing pan-tilt tracker...")
    tracker = PanTiltTracker(serial_port=serial_port)
    tracker.servo.center()

    print(f"[main] Opening camera #{camera_id}...")
    cam = Camera(source=camera_id, width=640, height=480)
    cam.start()

    cv2.namedWindow("Arm Physical AI Tracker", cv2.WINDOW_NORMAL)
    fps_buffer = []

    print("[main] Physical AI Tracking — press 'q' to quit, 'c' to re-center")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            t0 = time.perf_counter()
            detections = detector.infer(frame)
            t1 = time.perf_counter()

            fps_buffer.append(1.0 / (t1 - t0))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            current_fps = np.mean(fps_buffer)

            pan, tilt = tracker.update(detections, frame.shape[1], frame.shape[0])

            display = draw_detections(frame.copy(), detections)
            cv2.putText(
                display,
                f"FPS: {current_fps:.1f} | Detections: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                display,
                f"Pan: {pan:.0f}*  Tilt: {tilt:.0f}*  |  Model: {Path(detector.model_path).stem}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

            cv2.imshow("Arm Physical AI Tracker", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                tracker.servo.center()
    finally:
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Arm AI Optimization Challenge — Real-Time Vision Assistant")
    print("=" * 50)

    fp32_path = export_onnx(args.model)
    print_comparison(fp32_path, fp32_path)

    int8_path = quantize_onnx_static(fp32_path)
    print_comparison(fp32_path, int8_path)

    if args.quantize_only:
        return

    if args.benchmark:
        print("\n--- Running Benchmarks ---")
        results = run_benchmarks(fp32_path, int8_path, results_dir, runs=args.runs)
        generate_chart(results, results_dir)
        return

    if args.demo:
        run_demo(int8_path, args.camera, args.conf)
        return

    if args.track:
        run_tracker(int8_path, args.camera, args.conf, args.serial)
        return

    print("\nNo action specified. Use --demo for camera, --track for Physical AI, or --benchmark for tests.")
    print("Examples:")
    print("  python run.py --demo              # Real-time detection")
    print("  python run.py --track             # Physical AI pan-tilt tracking")
    print("  python run.py --benchmark          # Benchmark FP32 vs INT8")
    print("  python run.py --quantize-only      # Just export + quantize")


if __name__ == "__main__":
    main()
