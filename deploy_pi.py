#!/usr/bin/env python3
"""
Deploy optimize-arm to Raspberry Pi 5, run benchmarks, restore original app.
"""

import io
import sys
import tarfile
import time
from pathlib import Path

import paramiko

HOST = "smartshelf.taila277ca.ts.net"
USER = "icp"
PASSWORD = "icp12345"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/home/icp/optimize-arm"


def ssh(client, cmd, timeout=120, echo=True):
    if echo:
        print(f"  $ {cmd}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if echo and out:
        for l in out.split("\n"):
            print(f"    {l}")
    if echo and err:
        for l in err.split("\n"):
            print(f"    [err] {l}")
    return status, out


def main():
    print("=" * 60)
    print("Arm Optimization Challenge — Pi 5 Deployment")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=22, username=USER, password=PASSWORD, timeout=15)
    print("[ssh] Connected to Raspberry Pi 5\n")

    # Step 0: System info
    print("=== System Info ===")
    ssh(client, "cat /proc/cpuinfo | grep -E 'Model|Hardware|cpu cores' | head -5")
    ssh(client, "free -h | head -2")
    ssh(client, "uname -m")
    print()

    # Step 1: Stop traffic-monitor service
    print("=== Stopping traffic-monitor service ===")
    ssh(client, "sudo systemctl stop traffic-monitor.service")
    ssh(client, "systemctl is-active traffic-monitor.service")
    print()

    # Step 2: Copy project (exclude heavy/irrelevant dirs)
    print("=== Copying project to Pi ===")
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for f in PROJECT_DIR.rglob("*"):
            rel = f.relative_to(PROJECT_DIR)
            parts = rel.parts
            if any(p.startswith(".") or p in ("__pycache__", ".venv", "models", "optimized_models", "results", "venv") for p in parts):
                continue
            if f.is_file() and not f.name.startswith("."):
                tar.add(f, arcname=str(rel))
    tar_buf.seek(0)

    sftp = client.open_sftp()
    with sftp.open("/tmp/optimize-arm.tar.gz", "wb") as f:
        f.write(tar_buf.read())
    sftp.close()
    ssh(client, f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}")
    ssh(client, f"tar xzf /tmp/optimize-arm.tar.gz -C {REMOTE_DIR} && rm /tmp/optimize-arm.tar.gz")
    print("[copy] Done\n")

    # Step 3: Setup venv + install deps on Pi
    print("=== Setting up Python environment ===")
    ssh(client, f"cd {REMOTE_DIR} && python3 -m venv .venv", timeout=30)
    ssh(client, f"cd {REMOTE_DIR} && .venv/bin/pip install --upgrade pip setuptools wheel", timeout=120)
    ssh(client, f"cd {REMOTE_DIR} && .venv/bin/pip install -r requirements.txt", timeout=600)
    print()

    # Step 4: Export + Quantize on Pi
    print("=== Exporting YOLOv8n to ONNX & Quantizing ===")
    rc, out = ssh(client, f"cd {REMOTE_DIR} && .venv/bin/python run.py --quantize-only", timeout=300)
    if rc != 0:
        print(f"[ERROR] Quantize step failed (rc={rc})")
        sys.exit(1)
    print()

    # Step 5: Run benchmarks on Pi
    print("=== Running Benchmarks on Raspberry Pi 5 ===")
    rc, out = ssh(client, f"cd {REMOTE_DIR} && .venv/bin/python run.py --benchmark --runs 200", timeout=600)
    print()

    # Step 6: Fetch results back
    print("=== Fetching results ===")
    local_results = PROJECT_DIR / "results_pi"
    local_results.mkdir(exist_ok=True)
    (local_results / "models").mkdir(exist_ok=True)
    (local_results / "optimized_models").mkdir(exist_ok=True)

    sftp = client.open_sftp()
    for remote_subdir, local_subdir in [
        ("results", "."),
        ("models", "models"),
        ("optimized_models", "optimized_models"),
    ]:
        remote_path = f"{REMOTE_DIR}/{remote_subdir}"
        local_path = local_results / local_subdir
        local_path.mkdir(exist_ok=True)
        try:
            for entry in sftp.listdir_attr(remote_path):
                if entry.filename.startswith(".") or entry.st_mode & 0o40000:
                    continue
                sftp.get(f"{remote_path}/{entry.filename}", str(local_path / entry.filename))
                print(f"  {local_subdir}/{entry.filename} ({entry.st_size // 1024} KB)")
        except FileNotFoundError:
            print(f"  {remote_subdir}: not found")
    sftp.close()
    print()

    # Print benchmark results
    csv_path = local_results / "benchmark_results.csv"
    if csv_path.exists():
        print("=== Benchmark Results ===")
        print(csv_path.read_text())

    chart_path = local_results / "benchmark_comparison.png"
    if chart_path.exists():
        print(f"[chart] saved: {chart_path} ({chart_path.stat().st_size // 1024} KB)")

    # Step 7: Restart traffic-monitor service
    print("=== Restarting traffic-monitor service ===")
    ssh(client, "sudo systemctl start traffic-monitor.service")
    time.sleep(3)
    ssh(client, "systemctl is-active traffic-monitor.service")
    print("[restore] Service should be back on port 5000\n")

    print("=" * 60)
    print("Done! Results in: results_pi/")
    print("=" * 60)


if __name__ == "__main__":
    main()
