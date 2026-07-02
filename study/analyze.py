"""
Generates Pareto frontier analysis + markdown report from study results.
Usage:  python -m study.analyze
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STUDY_DIR / "results"


def load_results() -> list[dict]:
    csv_path = RESULTS_DIR / "study_results.csv"
    if not csv_path.exists():
        print(f"[analyze] No results found at {csv_path}. Run study/runner.py first.")
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["fps"] = float(r["fps"])
            r["mean_latency_ms"] = float(r["mean_latency_ms"])
            r["size_mb"] = float(r["size_mb"])
            r["peak_memory_mb"] = float(r["peak_memory_mb"])
            r["threads"] = int(r["threads"])
            rows.append(r)
    return rows


def find_pareto_frontier(points: list[dict]) -> list[dict]:
    """Find Pareto-optimal points maximizing FPS and minimizing size."""
    sorted_pts = sorted(points, key=lambda p: (-p["fps"], p["size_mb"]))
    frontier = []
    min_size = float("inf")
    for p in sorted_pts:
        if p["size_mb"] < min_size:
            min_size = p["size_mb"]
            frontier.append(p)
    return frontier


def generate_report(rows: list[dict]) -> str:
    if not rows:
        return "No data."

    models = sorted(set(r["model"] for r in rows))
    configs = sorted(set(r["config"] for r in rows))

    lines = []
    def L(s=""): lines.append(s)

    L("# Arm Cortex-A76 Optimization Study — Raspberry Pi 5")
    L()
    L(f"**Platform:** Raspberry Pi 5 (4× Cortex-A76 @ 2.4 GHz, 8 GB RAM)")
    L(f"**OS:** Raspberry Pi OS 64-bit, aarch64")
    L(f"**Runtime:** ONNX Runtime {__import__('onnxruntime').__version__}")
    L()
    L(f"## Overview")
    L()
    L(f"Tested **{len(models)} model sizes** × **{len(configs)} optimization configs** = **{len(rows)} data points**.")
    L(f"Each cell: 200 inference runs, synthetic 640×480 frames, mean latency reported.")
    L()

    # Summary table — best config per model
    L("## Best Config Per Model (by FPS)")
    L()
    L("| Model | Best Config | FPS | Latency (ms) | Size (MB) | Memory (MB) | Speedup vs Baseline |")
    L("|-------|-------------|:---:|:------------:|:---------:|:-----------:|:-------------------:|")
    for model in models:
        entries = [r for r in rows if r["model"] == model]
        baseline = next((r for r in entries if "Baseline" in r["config"]), None)
        best = max(entries, key=lambda r: r["fps"])
        b_fps = baseline["fps"] if baseline else 0
        speedup = best["fps"] / b_fps if b_fps > 0 else 0
        L(
            f"| {model} | {best['config']} | {best['fps']:.1f} | "
            f"{best['mean_latency_ms']:.1f} | {best['size_mb']:.1f} | "
            f"{best['peak_memory_mb']:.1f} | {speedup:.2f}x |"
        )
    L()

    # Full matrix — FPS
    L("## Full Comparison Matrix — Throughput (FPS)")
    L()
    header = "| Config | " + " | ".join(m.replace("yolov8", "YOLOv8-") for m in models) + " |"
    sep = "|" + "|".join("---" for _ in range(len(models) + 1)) + "|"
    L(header)
    L(sep)
    for config in configs:
        row = [config]
        for model in models:
            match = next((r for r in rows if r["model"] == model and r["config"] == config), None)
            row.append(f"{match['fps']:.1f}" if match else "-")
        L("| " + " | ".join(row) + " |")
    L()

    # Full matrix — Latency
    L("## Full Comparison Matrix — Mean Latency (ms)")
    L()
    L(header)
    L(sep)
    for config in configs:
        row = [config]
        for model in models:
            match = next((r for r in rows if r["model"] == model and r["config"] == config), None)
            row.append(f"{match['mean_latency_ms']:.1f}" if match else "-")
        L("| " + " | ".join(row) + " |")
    L()

    # Full matrix — Size
    L("## Full Comparison Matrix — Model Size (MB)")
    L()
    L(header)
    L(sep)
    for config in configs:
        row = [config]
        for model in models:
            match = next((r for r in rows if r["model"] == model and r["config"] == config), None)
            row.append(f"{match['size_mb']:.1f}" if match else "-")
        L("| " + " | ".join(row) + " |")
    L()

    # Pareto analysis
    L("## Pareto Frontier — Size vs Throughput")
    L()
    all_points = []
    for r in rows:
        all_points.append({"fps": r["fps"], "size_mb": r["size_mb"], "label": f"{r['model']} / {r['config']}", "model": r["model"]})
    frontier = find_pareto_frontier(all_points)
    L("Points on the Pareto frontier (no other config is both faster AND smaller):")
    L()
    for p in frontier:
        L(f"  - **{p['label']}** — {p['fps']:.1f} FPS, {p['size_mb']:.1f} MB")
    L()

    # Key insights
    L("## Key Insights")
    L()
    for model in models:
        entries = [r for r in rows if r["model"] == model]
        baseline = next((r for r in entries if "Baseline" in r["config"]), None)
        if not baseline:
            continue
        b_fps = baseline["fps"]
        b_size = baseline["size_mb"]
        best = max(entries, key=lambda r: r["fps"])
        smallest = min(entries, key=lambda r: r["size_mb"])

        int8s = next((r for r in entries if r["config"] == "INT8 Static" and r["model"] == model), None)
        int8_1t = next((r for r in entries if r["config"] == "INT8 Static 1T" and r["model"] == model), None)
        fp32_opt = next((r for r in entries if r["config"] == "FP32 Optimized" and r["model"] == model), None)

        L(f"### {model}")
        L(f"- **Total speedup:** {b_fps:.1f} → {best['fps']:.1f} FPS ({best['fps']/b_fps:.1f}x)")
        L(f"- **Size reduction:** {b_size:.1f} → {smallest['size_mb']:.1f} MB ({b_size/smallest['size_mb']:.1f}x)")

        if int8s and int8_1t:
            thread_gain = int8s["fps"] / int8_1t["fps"] if int8_1t["fps"] > 0 else 0
            L(f"- **Thread scaling (INT8):** {int8_1t['fps']:.1f} FPS (1T) → {int8s['fps']:.1f} FPS (4T) = {thread_gain:.2f}x from parallelism")

        if fp32_opt and int8s:
            quant_gain = int8s["fps"] / fp32_opt["fps"] if fp32_opt["fps"] > 0 else 0
            L(f"- **Quantization gain (FP32 opt → INT8 static):** {fp32_opt['fps']:.1f} → {int8s['fps']:.1f} FPS ({quant_gain:.2f}x)")

        L()

    # Recommendations
    L("## Recommendations for Arm Cortex-A76")
    L()
    L("Based on the Pareto analysis, here are the recommended configs for different use cases:")
    L()
    L("| Use Case | Recommended Config | Expected FPS | Model Size |")
    L("|----------|-------------------|:------------:|:----------:|")
    for model in models:
        best_int8 = max((r for r in rows if r["model"] == model and "INT8 Static" in r["config"] and r["threads"] > 1),
                        key=lambda r: r["fps"], default=None)
        if best_int8:
            speed = "real-time" if best_int8["fps"] >= 30 else "interactive" if best_int8["fps"] >= 15 else "near-real-time"
            L(f"| {speed} detection (YOLO) | {model} + INT8 Static + {best_int8['threads']}T | {best_int8['fps']:.1f} | {best_int8['size_mb']:.1f} MB |")

    L()
    L("---")
    L(f"*Generated by optimize-arm-yolo study framework. {len(rows)} data points from {len(models)} models × {len(configs)} configs on Raspberry Pi 5.*")

    return "\n".join(lines)


def main():
    rows = load_results()
    if not rows:
        return

    report = generate_report(rows)
    report_path = RESULTS_DIR / "optimization_playbook.md"
    report_path.write_text(report)
    print(f"[analyze] Report saved to {report_path}")

    # Generate Pareto chart
    try:
        import matplotlib.pyplot as plt
        models = sorted(set(r["model"] for r in rows))
        colors = {"yolov8n": "#2196F3", "yolov8s": "#FF9800", "yolov8m": "#F44336"}
        markers = {"yolov8n": "o", "yolov8s": "s", "yolov8m": "^"}

        fig, ax = plt.subplots(figsize=(10, 6))
        for model in models:
            pts = [r for r in rows if r["model"] == model]
            xs = [r["size_mb"] for r in pts]
            ys = [r["fps"] for r in pts]
            labels = [r["config"] for r in pts]
            ax.scatter(xs, ys, c=colors.get(model, "#333"), marker=markers.get(model, "o"),
                       s=100, label=model.replace("yolov8", "YOLOv8-"), zorder=5)
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl.replace(" ", "\n"), (x, y), fontsize=6, ha="center", va="bottom",
                           alpha=0.7)

        # Pareto frontier
        all_pts = [{"fps": r["fps"], "size_mb": r["size_mb"]} for r in rows]
        frontier = find_pareto_frontier(all_pts)
        frontier = sorted(frontier, key=lambda p: p["size_mb"])
        if len(frontier) > 1:
            fx = [p["size_mb"] for p in frontier]
            fy = [p["fps"] for p in frontier]
            ax.plot(fx, fy, "--", color="#4CAF50", linewidth=2, alpha=0.7, label="Pareto frontier", zorder=3)

        ax.set_xlabel("Model Size (MB)")
        ax.set_ylabel("Throughput (FPS)")
        ax.set_title("Optimization Study — Raspberry Pi 5 (Cortex-A76)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        chart_path = RESULTS_DIR / "pareto_frontier.png"
        plt.savefig(chart_path, dpi=150)
        print(f"[analyze] Pareto chart saved to {chart_path}")
        plt.close()
    except ImportError:
        print("[analyze] matplotlib not available, skipping chart")

    # Print summary to stdout
    print(f"\n{'='*60}")
    print("Optimization Playbook Preview")
    print(f"{'='*60}")
    # Print first ~40 lines
    for line in report.split("\n")[:45]:
        print(line)


if __name__ == "__main__":
    main()
