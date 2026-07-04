# Submission Checklist — Arm AI Optimization Challenge

Use this to verify everything is in place before submitting.

## Track Selection

- [x] **Track 1 — Physical AI** (selected)
- [ ] Track 2 — Migration/Adoption
- [ ] Track 3 — Scale + Learning

## Repository Requirements

- [x] Repository is **public** on GitHub
- [x] **Open source license** file included → `LICENSE` (MIT)
- [x] License detectable in GitHub "About" section

## Source Code

- [x] `run.py` — Entry point with `--demo`, `--track`, `--benchmark`, `--quantize-only`
- [x] `src/detector.py` — YOLO inference via ONNX Runtime
- [x] `src/tracker.py` — Pan-tilt PID control + serial servo protocol
- [x] `src/optimizer.py` — ONNX export + INT8 static/dynamic quantization
- [x] `src/benchmark.py` — FPS, latency, memory measurement
- [x] `src/camera.py` — Pi Camera / USB webcam capture
- [x] `src/main.py` — CLI orchestration

## Study Framework (bonus — differentiator)

- [x] `study/runner.py` — 18-point ablation study (3 models × 6 configs)
- [x] `study/accuracy.py` — COCO128 mAP validation
- [x] `study/analyze.py` — Pareto analysis + playbook generation
- [x] `study/configs/` — 6 JSON config files
- [x] `study/results/accuracy_results.json` — Accuracy validation data

## Hardware

- [x] `hardware/BOM.md` — Bill of materials
- [x] `hardware/WIRING.md` — Step-by-step assembly guide
- [x] `hardware/pan_tilt_firmware/pan_tilt.ino` — RP2040 servo PWM firmware

## Results / Artifacts

- [x] `study_results/optimization_playbook.md` — Full report with tables
- [x] `study_results/pareto_frontier.png` — Size vs throughput chart
- [x] `study_results/study_results.csv` — Full 18-row data matrix
- [x] `results_pi/benchmark_results.csv` — Pi 5 benchmark (FP32 vs INT8)
- [x] `results_pi/benchmark_comparison.png` — Benchmark bar chart
- [x] `optimized_models/` — Quantized models (9 ONNX files)

## Project Write-Up

- [x] `DEVPOST_SUBMISSION.md` — Complete with:
  - [x] Project Overview + "Why it should win"
  - [x] Functionality / Output description
  - [x] Setup Instructions (step-by-step)
  - [x] Optimization study results tables
  - [x] Update log (significant changes during submission period)
  - [x] How it leverages Arm (5 points)

## Submission Form (in browser)

- [ ] **URL to public GitHub repo** → `https://github.com/stanleyoz/optimize-arm`
- [ ] **Demo video uploaded** to YouTube/Vimeo (< 3 min)
  - [ ] Shows the system functioning on Pi 5
  - [ ] Pan-tilt tracking in action
  - [ ] On-screen telemetry (FPS, model name, servo angles)
- [ ] **Video URL** pasted in submission form
- [ ] **Text description** pasted (or link to DEVPOST_SUBMISSION.md)

## Pre-Submission Verification

- [ ] `python run.py --benchmark` completes on Pi 5
- [ ] `python run.py --demo` shows detections with FPS overlay
- [ ] `python run.py --track` moves servos in response to person detection
- [ ] Servos center on `c` keypress
- [ ] Clean exit on `q` keypress
- [ ] Camera stream is stable (no dropped frames after warmup)

## Video Script Outline (< 3 min)

Suggested structure:
1. **0:00–0:30** — Project intro + what makes it different (study framework, static INT8 discovery)
2. **0:30–1:00** — Optimization results: FP32 (6 FPS) → INT8 (13.7 FPS), accuracy preserved
3. **1:00–1:30** — Full study demo (show the 18-point data, Pareto chart)
4. **1:30–2:30** — Physical AI tracking demo (camera follows a person walking)
5. **2:30–3:00** — Why Arm + why this matters (cost, efficiency, reproducibility)

## Files to Check Before Pushing

- [ ] No hardcoded passwords or secrets in deploy scripts
- [ ] `.gitignore` excludes `__pycache__/`, `.venv/`, large model files
- [ ] License is detectable on GitHub repo page
- [ ] Repo description is set on GitHub