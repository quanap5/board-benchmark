#!/bin/bash
# Setup Python virtual environment for YOLO model download and ONNX export.
#
# Usage:
#   bash benchmarkCLI/setup-yolo-env.sh
#   source benchmarkCLI/.venv/bin/activate
#   python3 benchmarkCLI/download-yolo-model.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== YOLO Environment Setup ==="

# Check Python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using: $PY_VERSION ($PYTHON_CMD)"

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Recreate? (y/N): " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "Removed old environment."
    else
        echo "Keeping existing environment."
        source "$VENV_DIR/bin/activate"
        echo "Activated: $VENV_DIR"
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_DIR ..."
$PYTHON_CMD -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies..."
pip install ultralytics onnx onnxruntime numpy --quiet

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Download and convert a YOLO model:"
echo "  python3 $SCRIPT_DIR/download-yolo-model.py"
echo ""
echo "Run benchmark:"
echo "  python3 $SCRIPT_DIR/onnx-benchmark.py $SCRIPT_DIR/models/yolov8n.onnx -n 100"
echo ""
echo "Deactivate when done:"
echo "  deactivate"
