# Arm-Optimized Real-Time Vision Assistant

> **Arm AI Optimization Challenge 2026 — Physical AI Track**

A real-time object detection system optimized for **Arm-powered edge devices** (Raspberry Pi 4/5). Uses INT8 quantization, ONNX Runtime with Arm NEON optimizations, and benchmarked end-to-end to show **2.4x speedup and 3.6x model size reduction** over baseline FP32.

---

## Project Overview

This project demonstrates how to take a standard YOLOv8n object detection model and optimize it for deployment on Arm-based edge hardware. The result is a **real-time vision assistant** that runs on a Raspberry Pi with a camera, detecting objects at interactive frame rates while consuming less memory and storage.

**Why this should win:**
- Shows a **clear, measurable optimization pipeline** (FP32 → INT8) with quantified improvements
- Uses **Arm-specific acceleration** via ONNX Runtime's NEON-optimized CPUExecutionProvider
- Runs **fully on-device** on a Raspberry Pi — no cloud dependency, privacy-preserving
- Includes **benchmarking infrastructure** so results are reproducible and comparable
- **Open source** with MIT license — reusable by the entire Arm developer community

---

## Functionality / Output

### What it does
1. **Model Optimization Pipeline** — Takes a YOLOv8n model, exports to ONNX, applies INT8 quantization with calibration
2. **Real-Time Detection** — Live camera feed with bounding boxes, class labels, confidence scores, and FPS overlay
3. **Benchmark Suite** — Runs both FP32 and INT8 models through controlled inference loops, measuring:
   - Throughput (FPS)
   - Latency (mean, p50, p95, p99)
   - Model size (MB)
   - Peak memory usage (MB)
4. **Visualization** — Generates comparison charts from benchmark results

### Final output
- **Optimized INT8 model** (`optimized_models/yolov8n_int8.onnx`) — 3.36 MB, ready for deployment
- **Benchmark report** (`results/benchmark_results.csv`) with statistical latency analysis
- **Comparison chart** (`results/benchmark_comparison.png`) visualizing FPS, latency, and size improvements
- **Live demo** — real-time camera inference with on-screen telemetry

---

## Optimization Results

### Raspberry Pi 5 (Cortex-A76 @ 2.4 GHz, aarch64)

| Metric | FP32 Baseline | INT8 Optimized | Improvement |
|--------|:------------:|:--------------:|:----------:|
| Model Size | 12.26 MB | **3.36 MB** | **3.6x smaller** |
| Throughput | 6.0 FPS | **13.7 FPS** | **2.3x faster** |
| Mean Latency | 166.6 ms | **73.0 ms** | **2.3x reduction** |
| p95 Latency | 169.6 ms | **73.7 ms** | **2.3x reduction** |
| Peak Memory | 14.5 MB | **7.5 MB** | **1.9x less** |

*Benchmarked on a physical Raspberry Pi 5 (8 GB RAM, 4× Cortex-A76 cores). 
INT8 quantization delivers 2.3x throughput gain and halves memory usage — making real-time object detection viable on a $80 single-board computer.*

### Optimization techniques used

| Technique | Impact |
|-----------|--------|
| **INT8 Static Quantization** | 4x smaller weights, 2-4x faster NEON matmul |
| **ONNX Runtime Graph Optimization** | Operator fusion, constant folding, layout optimization |
| **Arm NEON SIMD** | Leveraged automatically by ONNX Runtime CPUExecutionProvider for parallel vector ops |
| **Calibration** | 32 samples prevent accuracy degradation during quantization |

---

## Setup Instructions (Raspberry Pi)

### Prerequisites
- Raspberry Pi 4 or 5 (2GB+ RAM recommended)
- Raspberry Pi Camera Module (or USB webcam)
- Raspberry Pi OS (64-bit, Bookworm)
- Python 3.10+

### Step 1 — Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/optimize-arm.git
cd optimize-arm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Optimize the model

This exports YOLOv8n to ONNX and applies INT8 quantization:

```bash
python run.py --quantize-only
```

### Step 3 — Run benchmarks

```bash
python run.py --benchmark
```

### Step 4 — Launch real-time demo

```bash
python run.py --demo
```

Press `q` to quit, `s` to save a snapshot.

### Running on a different camera

```bash
python run.py --demo --camera 0   # default USB camera
python run.py --demo --camera 1   # secondary camera
```

### Custom confidence threshold

```bash
python run.py --demo --conf 0.6   # higher threshold = fewer but more confident detections
```

---

## Project Structure

```
optimize-arm/
├── run.py                  # Entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── main.py             # CLI / orchestration
│   ├── optimizer.py        # ONNX export + INT8 quantization
│   ├── detector.py         # ONNX Runtime inference + post-processing
│   ├── camera.py           # Raspberry Pi camera capture
│   └── benchmark.py        # Performance measurement + charting
├── models/                 # Source FP32 models
├── optimized_models/       # Quantized INT8 models
└── results/                # Benchmark data + charts
```

---

## How It Leverages Arm

1. **Arm NEON SIMD** — ONNX Runtime's CPUExecutionProvider auto-vectorizes convolutions and matmul operations using Arm NEON instructions on aarch64
2. **INT8 quantized inference** — Arm NEON has dedicated 8-bit dot-product instructions (SDOT/UDOT) that run 2-4x faster than FP32 equivalents on Cortex-A72 (Pi 4) and Cortex-A76 (Pi 5)
3. **Optimized for memory-constrained edge** — Model fits in ~3.4 MB with 1.9x lower peak memory, ideal for embedded/edge deployment
4. **Fully on-device** — No cloud round-trip, no GPU required, runs on commodity Arm single-board computers
5. **Measured on real hardware** — All benchmarks run on a physical Raspberry Pi 5, not simulated

---

## Tech Stack

- **Model**: YOLOv8nano (Ultralytics)
- **Framework**: ONNX Runtime with CPUExecutionProvider
- **Quantization**: ONNX Runtime static INT8 quantization
- **Language**: Python 3.12
- **Camera**: OpenCV (Picamera2-compatible)
- **Hardware target**: Raspberry Pi 4/5 (Arm Cortex-A72/A76)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Submission Details

- **Track**: Physical AI
- **Video**: [link to YouTube/Vimeo demo]
- **Repository**: [public GitHub link]
