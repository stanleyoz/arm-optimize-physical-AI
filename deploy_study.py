#!/usr/bin/env python3
"""
Deploy the optimization study to Pi 5 and run it.
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
REMOTE_DIR = "/home/icp/optimize-arm-yolo"


def ssh(client, cmd, timeout=300, echo=True):
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
    print("Optimization Study — Pi 5 Deployment")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=22, username=USER, password=PASSWORD, timeout=15)
    print("[ssh] Connected\n")

    # Stop traffic monitor
    print("=== Stopping traffic-monitor ===")
    ssh(client, "sudo systemctl stop traffic-monitor.service", echo=False)
    print()

    # Copy project
    print("=== Copying project to Pi ===")
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for f in PROJECT_DIR.rglob("*"):
            rel = f.relative_to(PROJECT_DIR)
            parts = rel.parts
            if any(p.startswith(".") or p in ("__pycache__", ".venv", "models", "optimized_models", "results", "results_pi", "venv") for p in parts):
                continue
            if f.is_file() and not f.name.startswith("."):
                tar.add(f, arcname=str(rel))
    tar_buf.seek(0)
    sftp = client.open_sftp()
    with sftp.open("/tmp/project.tar.gz", "wb") as f:
        f.write(tar_buf.read())
    sftp.close()
    ssh(client, f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}")
    ssh(client, f"tar xzf /tmp/project.tar.gz -C {REMOTE_DIR} && rm /tmp/project.tar.gz")
    print("[copy] Done\n")

    # Setup venv
    print("=== Setting up venv ===")
    _, out = ssh(client, f"test -d {REMOTE_DIR}/.venv && echo exists || echo missing", echo=False)
    if "exists" not in out:
        ssh(client, f"cd {REMOTE_DIR} && python3 -m venv .venv", timeout=30)
        ssh(client, f"cd {REMOTE_DIR} && .venv/bin/pip install --upgrade pip setuptools wheel", timeout=120)
    ssh(client, f"cd {REMOTE_DIR} && .venv/bin/pip install -r requirements.txt", timeout=600)
    print()

    # Export + quantize all models
    print("=== Exporting + quantizing models (n, s, m) ===")
    for model in ["yolov8n", "yolov8s", "yolov8m"]:
        print(f"  Model: {model}")
        ssh(client, f"cd {REMOTE_DIR} && .venv/bin/python run.py --quantize-only --model {model}", timeout=600, echo=False)
    print()

    # Run study
    print("=== Running full optimization study ===")
    start = time.time()
    ssh(client, f"cd {REMOTE_DIR} && .venv/bin/python -m study.runner --runs 200", timeout=1800)
    elapsed = time.time() - start
    print(f"  Study complete in {elapsed/60:.1f} min\n")

    # Generate report
    print("=== Generating analysis report ===")
    ssh(client, f"cd {REMOTE_DIR} && .venv/bin/python -m study.analyze", timeout=120)
    print()

    # Fetch results
    print("=== Fetching results ===")
    local_results = PROJECT_DIR / "study_results"
    local_results.mkdir(exist_ok=True)
    sftp = client.open_sftp()
    remote_results = f"{REMOTE_DIR}/study/results"
    try:
        for entry in sftp.listdir_attr(remote_results):
            if entry.filename.startswith("."):
                continue
            sftp.get(f"{remote_results}/{entry.filename}", str(local_results / entry.filename))
            print(f"  {entry.filename} ({entry.st_size // 1024} KB)")
    except FileNotFoundError:
        print(f"  No results found at {remote_results}")
    sftp.close()
    print()

    # Print report preview
    report_path = local_results / "optimization_playbook.md"
    if report_path.exists():
        print("=== Optimization Playbook (preview) ===")
        print(report_path.read_text()[:2000])

    # Restart traffic monitor
    print("=== Restarting traffic-monitor ===")
    ssh(client, "sudo systemctl start traffic-monitor.service", echo=False)
    time.sleep(2)
    ssh(client, "systemctl is-active traffic-monitor.service", echo=False)
    print()

    print("=" * 60)
    print(f"Done! Results in: study_results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
