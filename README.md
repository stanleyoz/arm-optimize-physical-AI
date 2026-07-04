# Arm-Optimized Physical AI Tracker

> **Arm AI Optimization Challenge 2026 — Physical AI Track**

A closed-loop person tracking system running on a Raspberry Pi 5: YOLOv8n/s/m optimized via INT8 quantization and ONNX Runtime pan-tilt servos that follow people in real time. Includes a **reproducible 18-point optimization study framework** with COCO128 accuracy validation and Pareto analysis.

---

## Project Overview

This project demonstrates the complete pipeline from model optimization to Physical AI deployment on Arm hardware:

1. **Optimization Study** — Sweep 3 model sizes × 6 configs = 18 data points, validate accuracy, find Pareto-optimal operating points
2. **Real-Time Tracking** — Camera → YOLO (INT8 Dynamic, 13.7 FPS, 73 ms latency) → PID controller → pan-tilt servos (SR431 via RP2040)
3. **Key Discovery** — Static INT8 quantization silently collapses YOLO confidences to zero (mAP=0.0). Documented with proof so the community doesn't waste time debugging it.

### Key results

| Metric | FP32 Baseline | INT8 Dynamic | Improvement |
|--------|:------------:|:------------:|:-----------:|
| Throughput | 6.0 FPS | **13.7 FPS** | **2.3x** |
| Latency | 166.6 ms | **73.0 ms** | **2.3x** |
| Model Size | 12.3 MB | **3.4 MB** | **3.6x** |
| Memory | 14.5 MB | **7.5 MB** | **1.9x** |
| Accuracy (mAP) | 0.451 | **0.435** | **−3.5%** |

*All measurements: physical Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz, 8 GB RAM)*

---

## Quick Start

```bash
git clone https://github.com/stanleyoz/optimize-arm.git
cd optimize-arm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Export + quantize models
python run.py --quantize-only

# Live camera demo
python run.py --demo

# Physical AI tracking (requires hardware, see hardware/WIRING.md)
python run.py --track

# Run benchmarks
python run.py --benchmark
```

---

## Running the Full Optimization Study

```bash
# On a Pi 5 with SSH access:
python deploy_study.py

# Or locally:
python -m study.runner --runs 200
python -m study.analyze          # Pareto playbook
python -m study.accuracy         # COCO128 mAP validation
```

Results land in `study_results/`:
- `optimization_playbook.md` — Full 18-point report
- `pareto_frontier.png` — Size vs throughput scatter plot
- `accuracy_results.json` — mAP comparison

---

## Hardware

Components: Pi 5 + Camera Module 3 + 2× SR431 servos + RP2040 + pan-tilt bracket.

See `hardware/BOM.md` for the bill of materials (~$30) and `hardware/WIRING.md` for assembly instructions. RP2040 firmware is in `hardware/pan_tilt_firmware/`.

```bash
# One-time Pi setup:
chmod +x scripts/setup_pan_tilt.sh && ./scripts/setup_pan_tilt.sh
```

---

## Project Structure

```
├── run.py                   # Entry point
├── src/
│   ├── main.py              # CLI: --demo, --track, --benchmark, --quantize-only
│   ├── detector.py          # YOLO ONNX Runtime inference
│   ├── tracker.py           # Pan-tilt PID + serial servo control
│   ├── optimizer.py         # ONNX export + INT8 quantization
│   ├── benchmark.py         # FPS/latency/memory measurement
│   └── camera.py            # Pi Camera / USB capture
├── study/
│   ├── runner.py            # 3-model × 6-config ablation
│   ├── accuracy.py          # COCO128 mAP validation
│   ├── analyze.py           # Pareto frontier + playbook
│   └── configs/             # 6 JSON config files
├── hardware/
│   ├── BOM.md               # Bill of materials
│   ├── WIRING.md            # Step-by-step assembly guide
│   └── pan_tilt_firmware/   # RP2040 Arduino sketch
├── scripts/
│   └── setup_pan_tilt.sh    # Pi camera + serial setup
├── models/                  # FP32 ONNX models
├── optimized_models/        # INT8 quantized models
├── results/                 # Local benchmarks
├── results_pi/              # Pi 5 benchmarks
├── study_results/           # Study report + charts + accuracy
├── deploy_pi.py             # SSH auto-deploy + benchmark
├── deploy_study.py          # SSH auto-deploy + full study
├── DEVPOST_SUBMISSION.md    # Hackathon submission write-up
└── LICENSE                  # MIT
```

---

## Tech Stack

- **Models:** YOLOv8n/s/m (Ultralytics)
- **Runtime:** ONNX Runtime with Arm NEON CPUExecutionProvider
- **Quantization:** ONNX Runtime INT8 static + dynamic
- **Control:** Custom PID → serial protocol → RP2040
- **Firmware:** Arduino sketch (Servo library) on RP2040
- **Platform:** Raspberry Pi 5 (Cortex-A76) + Pi Pico (Cortex-M0+)
- **Servos:** 2× SR431 analog

---

## License

MIT — see [LICENSE](LICENSE).