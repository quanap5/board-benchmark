#!/bin/bash
# Setup Python virtual environment for YOLO model download and ONNX export.
#
# Usage:
#   bash benchmarkCLI/setup-yolo-env.sh           # Create or reuse venv
#   bash benchmarkCLI/setup-yolo-env.sh --force    # Recreate from scratch
#
# After setup:
#   source benchmarkCLI/.venv/bin/activate
#   python3 benchmarkCLI/download-yolo-model.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQ_FILE="$SCRIPT_DIR/requirements-dev.txt"
FORCE=false

[ "$1" = "--force" ] && FORCE=true

echo ""
echo "========================================================"
echo "  DEV ENVIRONMENT SETUP"
echo "========================================================"
echo ""

# --- Find Python ---
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

PY_VER=$($PYTHON_CMD --version 2>&1)
echo "  Python: $PY_VER ($PYTHON_CMD)"

# --- Check existing venv ---
if [ -d "$VENV_DIR" ] && [ "$FORCE" = false ]; then
    echo "  Venv:   $VENV_DIR (exists)"

    # Verify venv is healthy
    VENV_PYTHON="$VENV_DIR/bin/python3"
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "  [WARN] Venv broken (python3 not found). Recreating..."
        rm -rf "$VENV_DIR"
    else
        # Verify key packages
        MISSING=""
        for pkg in ultralytics onnx onnxruntime numpy; do
            if ! "$VENV_PYTHON" -c "import $pkg" 2>/dev/null; then
                MISSING="$MISSING $pkg"
            fi
        done

        if [ -z "$MISSING" ]; then
            echo ""
            echo "  [OK] Venv is healthy. All packages installed:"
            "$VENV_PYTHON" -c "
import ultralytics, onnx, onnxruntime, numpy
print(f'         ultralytics  {ultralytics.__version__}')
print(f'         onnx         {onnx.__version__}')
print(f'         onnxruntime  {onnxruntime.__version__}')
print(f'         numpy        {numpy.__version__}')
"
            echo ""
            echo "  Activate with:"
            echo "    source $VENV_DIR/bin/activate"
            echo ""
            exit 0
        else
            echo "  [WARN] Missing packages:$MISSING"
            echo "  Installing missing packages..."
            source "$VENV_DIR/bin/activate"
            pip install --quiet -r "$REQ_FILE"
            echo "  [OK] Packages installed."
            echo ""
            echo "  Activate with:"
            echo "    source $VENV_DIR/bin/activate"
            echo ""
            exit 0
        fi
    fi
fi

# --- Remove old venv if --force ---
if [ -d "$VENV_DIR" ] && [ "$FORCE" = true ]; then
    echo "  Removing old venv..."
    rm -rf "$VENV_DIR"
fi

# --- Create venv ---
echo "  Creating venv at $VENV_DIR ..."
$PYTHON_CMD -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "  Upgrading pip..."
pip install --upgrade pip --quiet

echo "  Installing dev dependencies..."
pip install --quiet -r "$REQ_FILE"

# --- Verify ---
echo ""
echo "--- Verifying installation ---"
VERIFY_OK=true
for pkg in ultralytics onnx onnxruntime numpy; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VER=$(python3 -c "import $pkg; print($pkg.__version__)")
        echo "  [OK] $pkg $VER"
    else
        echo "  [FAIL] $pkg not installed"
        VERIFY_OK=false
    fi
done

echo ""
if [ "$VERIFY_OK" = true ]; then
    echo "========================================================"
    echo "  Setup complete!"
    echo "========================================================"
    echo ""
    echo "  Activate:"
    echo "    source $VENV_DIR/bin/activate"
    echo ""
    echo "  Download model:"
    echo "    python3 $SCRIPT_DIR/download-yolo-model.py"
    echo ""
    echo "  Deactivate:"
    echo "    deactivate"
else
    echo "  [ERROR] Some packages failed to install."
    exit 1
fi
echo ""
