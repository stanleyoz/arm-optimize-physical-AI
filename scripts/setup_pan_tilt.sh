#!/usr/bin/env bash
# setup_pan_tilt.sh — One-time setup for Physical AI pan-tilt tracker
# Run this ON the Raspberry Pi 5 after cloning the repo.

set -euo pipefail

echo "=== Physical AI Pan-Tilt Tracker Setup ==="

# 1. Enable camera
echo "[1/4] Enabling camera interface..."
sudo raspi-config nonint do_camera 0 2>/dev/null || true
sudo raspi-config nonint do_legacy 0 2>/dev/null || true

# 2. Enable UART for RP2040 communication
echo "[2/4] Enabling UART..."
sudo raspi-config nonint do_serial 2 2>/dev/null || true

# 3. Install system packages for Arduino CLI (firmware flashing)
echo "[3/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-serial \
    minicom \
    arduino-cli \
    2>/dev/null || echo "  (some packages may not be available, continuing)"

# 4. Install Python deps
echo "[4/4] Installing Python dependencies..."
cd "$(dirname "$0")/.."
pip install pyserial 2>/dev/null || pip install --break-system-packages pyserial 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Flash RP2040: arduino-cli compile --upload hardware/pan_tilt_firmware/"
echo "  2. Wire servos per hardware/WIRING.md"
echo "  3. Run: python run.py --track"