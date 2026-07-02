"""
Orchestrates the full optimization study across model sizes × configs.
Usage:  python -m study.runner
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STUDY_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.detector import YOLODetector
from src.benchmark import BenchmarkResult, generate_test_frames, benchmark_model
from src.optimizer import MODELS_DIR, OPTIMIZED_DIR, export_onnx, quantize_onnx_static, quantize_onnx_dynamic


MODEL_NAMES = ["yolov8n", "yolov8s", "yolov8m"]
RESULTS_DIR = STUDY_DIR / "results"


def load_configs() -> list[dict]:
    configs_dir = STUDY_DIR / "configs"
    configs = []
    for f in sorted(configs_dir.glob("*.json")):
        with open(f) as fh:
            cfg = json.load(fh)
            configs.append(cfg)
    return configs


def get_model_path(model_name: str, config: dict) -> Path:
    base = MODELS_DIR / f"{model_name}.onnx"
    q = config.get("quantization")
    if q == "static":
        return OPTIMIZED_DIR / f"{model_name}_int8.onnx"
    if q == "dynamic":
        return OPTIMIZED_DIR / f"{model_name}_int8_dynamic.onnx"
    return base


def ensure_model(model_name: str, config: dict):
    fp32 = export_onnx(model_name)
    q = config.get("quantization")
    if q == "static":
        int8_path = OPTIMIZED_DIR / f"{model_name}_int8.onnx"
        if not int8_path.exists():
            quantize_onnx_static(fp32, int8_path)
    elif q == "dynamic":
        int8_path = OPTIMIZED_DIR / f"{model_name}_int8_dynamic.onnx"
        if not int8_path.exists():
            quantize_onnx_dynamic(fp32, int8_path)


def graph_opt_level(name: str):
    import onnxruntime as ort
    levels = {
        "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return levels.get(name, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)


def exec_mode(name: str):
    import onnxruntime as ort
    modes = {
        "ORT_SEQUENTIAL": ort.ExecutionMode.ORT_SEQUENTIAL,
        "ORT_PARALLEL": ort.ExecutionMode.ORT_PARALLEL,
    }
    return modes.get(name, ort.ExecutionMode.ORT_PARALLEL)


def benchmark_with_config(
    model_path: Path,
    config: dict,
    test_frames: list[np.ndarray],
    runs: int = 200,
) -> BenchmarkResult:
    sopts = config.get("session_opts", {})
    extra = sopts.get("extra", {})

    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = graph_opt_level(sopts.get("graph_optimization_level", "ORT_ENABLE_ALL"))
    sess_opts.intra_op_num_threads = sopts.get("intra_op_num_threads", 4)
    sess_opts.inter_op_num_threads = sopts.get("inter_op_num_threads", 2)
    sess_opts.execution_mode = exec_mode(sopts.get("execution_mode", "ORT_PARALLEL"))
    sess_opts.enable_mem_pattern = sopts.get("enable_mem_pattern", True)
    if "enable_cpu_mem_arena" in sopts:
        sess_opts.enable_cpu_mem_arena = sopts["enable_cpu_mem_arena"]

    for k, v in extra.items():
        sess_opts.add_session_config_entry(k, str(v))

    detector = YOLODetector(str(model_path), session_opts=sess_opts)
    detector.warmup(20)

    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    latencies = []
    for frame in test_frames * (runs // len(test_frames) + 1):
        blob, scale, x_off, y_off = detector.preprocess(frame)
        t0 = time.perf_counter()
        outputs = detector.session.run(None, {detector.input_name: blob})
        detector.postprocess(outputs, scale, x_off, y_off, frame.shape[1], frame.shape[0])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if len(latencies) >= runs:
            break

    mem_after = process.memory_info().rss / (1024 * 1024)
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    return BenchmarkResult(
        model_name=f"{model_path.stem}_{config['name'].replace(' ', '_')}",
        model_size_mb=model_path.stat().st_size / (1024 * 1024),
        mean_latency_ms=float(np.mean(latencies_sorted)),
        median_latency_ms=float(latencies_sorted[n // 2]) if n else 0,
        p95_latency_ms=float(latencies_sorted[int(n * 0.95)]) if n else 0,
        p99_latency_ms=float(latencies_sorted[int(n * 0.99)]) if n else 0,
        fps=1000.0 / float(np.mean(latencies_sorted)) if latencies_sorted else 0,
        peak_memory_mb=mem_after - mem_before,
        latencies_ms=latencies_sorted,
    )


def run_study(runs: int = 200):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    test_frames = generate_test_frames()

    csv_path = RESULTS_DIR / "study_results.csv"
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([
            "model", "config", "size_mb", "mean_latency_ms", "median_latency_ms",
            "p95_latency_ms", "p99_latency_ms", "fps", "peak_memory_mb",
            "quantization", "graph_opt", "threads",
        ])

        for model_name in MODEL_NAMES:
            for config in configs:
                label = f"{model_name} / {config['name']}"
                print(f"\n{'='*60}")
                print(f"[study] {label}")
                print(f"  {config['description']}")

                ensure_model(model_name, config)
                model_path = get_model_path(model_name, config)
                if not model_path.exists():
                    print(f"  SKIP — model not found: {model_path}")
                    continue

                result = benchmark_with_config(model_path, config, test_frames, runs=runs)
                print(
                    f"  FPS: {result.fps:.1f}  |  "
                    f"Lat: {result.mean_latency_ms:.1f} ms  |  "
                    f"Size: {result.model_size_mb:.1f} MB  |  "
                    f"Mem: {result.peak_memory_mb:.1f} MB"
                )

                sopts = config.get("session_opts", {})
                w.writerow([
                    model_name,
                    config["name"],
                    f"{result.model_size_mb:.2f}",
                    f"{result.mean_latency_ms:.2f}",
                    f"{result.median_latency_ms:.2f}",
                    f"{result.p95_latency_ms:.2f}",
                    f"{result.p99_latency_ms:.2f}",
                    f"{result.fps:.1f}",
                    f"{result.peak_memory_mb:.2f}",
                    config.get("quantization", "none"),
                    sopts.get("graph_optimization_level", "none"),
                    sopts.get("intra_op_num_threads", 1),
                ])

    print(f"\n[study] Full results saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    run_study(runs=args.runs)
