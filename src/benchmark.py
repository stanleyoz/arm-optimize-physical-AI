import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import psutil

from src.detector import YOLODetector


@dataclass
class BenchmarkResult:
    model_name: str
    model_size_mb: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float
    peak_memory_mb: float
    latencies_ms: list[float] = field(repr=False)


def benchmark_model(
    model_path: Path,
    test_frames: list[np.ndarray],
    warmup: int = 20,
    runs: int = 200,
) -> BenchmarkResult:
    detector = YOLODetector(str(model_path))
    detector.warmup(warmup)

    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    latencies = []
    for frame in test_frames * (runs // len(test_frames) + 1):
        blob, scale, x_off, y_off = detector.preprocess(frame)
        t0 = time.perf_counter()
        outputs = detector.session.run(None, {detector.input_name: blob})
        detector.postprocess(
            outputs, scale, x_off, y_off, frame.shape[1], frame.shape[0],
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if len(latencies) >= runs:
            break

    mem_after = process.memory_info().rss / (1024 * 1024)

    latencies = sorted(latencies)
    n = len(latencies)
    mean_lat = np.mean(latencies)
    median_lat = latencies[n // 2] if n else 0
    p95_lat = latencies[int(n * 0.95)] if n else 0
    p99_lat = latencies[int(n * 0.99)] if n else 0
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    size_mb = model_path.stat().st_size / (1024 * 1024)

    return BenchmarkResult(
        model_name=model_path.stem,
        model_size_mb=size_mb,
        mean_latency_ms=mean_lat,
        median_latency_ms=median_lat,
        p95_latency_ms=p95_lat,
        p99_latency_ms=p99_lat,
        fps=fps,
        peak_memory_mb=mem_after - mem_before,
        latencies_ms=latencies,
    )


def generate_test_frames(count: int = 10, width: int = 640, height: int = 480) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(count):
        noise = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frames.append(noise)
    return frames


def run_benchmarks(
    fp32_path: Path,
    int8_path: Path,
    output_dir: Path,
    runs: int = 200,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_frames = generate_test_frames()

    results = []
    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        if not path.exists():
            print(f"[benchmark] Skipping {label} — file not found: {path}")
            continue
        print(f"[benchmark] Benchmarking {label} ({path.name})...")
        result = benchmark_model(path, test_frames, runs=runs)
        results.append(result)
        print(
            f"  {label}: {result.fps:.1f} FPS, "
            f"{result.mean_latency_ms:.1f} ms mean, "
            f"{result.model_size_mb:.1f} MB"
        )

    if len(results) == 2:
        speedup = results[1].fps / results[0].fps if results[0].fps > 0 else 0
        size_ratio = results[0].model_size_mb / results[1].model_size_mb if results[1].model_size_mb > 0 else 0
        print(f"\n[benchmark] Speedup: {speedup:.2f}x")
        print(f"[benchmark] Size reduction: {size_ratio:.2f}x")
        print(f"[benchmark] FPS gain: {results[1].fps - results[0].fps:.1f}")
        print(f"[benchmark] Latency reduction: {results[0].mean_latency_ms - results[1].mean_latency_ms:.1f} ms")

    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "size_mb", "mean_latency_ms", "median_latency_ms",
            "p95_latency_ms", "p99_latency_ms", "fps", "peak_memory_mb",
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "model": r.model_name,
                "size_mb": f"{r.model_size_mb:.2f}",
                "mean_latency_ms": f"{r.mean_latency_ms:.2f}",
                "median_latency_ms": f"{r.median_latency_ms:.2f}",
                "p95_latency_ms": f"{r.p95_latency_ms:.2f}",
                "p99_latency_ms": f"{r.p99_latency_ms:.2f}",
                "fps": f"{r.fps:.1f}",
                "peak_memory_mb": f"{r.peak_memory_mb:.2f}",
            })
    print(f"[benchmark] Results saved to {csv_path}")
    return results


def generate_chart(results: list[BenchmarkResult], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[benchmark] matplotlib not available, skipping chart")
        return

    names = [r.model_name for r in results]
    fps_vals = [r.fps for r in results]
    lat_vals = [r.mean_latency_ms for r in results]
    size_vals = [r.model_size_mb for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2196F3", "#FF5722"]

    axes[0].bar(names, fps_vals, color=colors[:len(names)])
    axes[0].set_title("Throughput (FPS)")
    axes[0].set_ylabel("Frames per Second")

    axes[1].bar(names, lat_vals, color=colors[:len(names)])
    axes[1].set_title("Mean Latency (ms)")
    axes[1].set_ylabel("Milliseconds")

    axes[2].bar(names, size_vals, color=colors[:len(names)])
    axes[2].set_title("Model Size (MB)")
    axes[2].set_ylabel("MB")

    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=2, fontsize=9)

    plt.tight_layout()
    chart_path = output_dir / "benchmark_comparison.png"
    plt.savefig(chart_path, dpi=150)
    print(f"[benchmark] Chart saved to {chart_path}")
    plt.close()
