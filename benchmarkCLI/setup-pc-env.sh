#!/bin/bash
# Setup native PC benchmark environment (no Docker).
# Detects CUDA version and installs matching onnxruntime-gpu.
#
# Usage:
#   bash benchmarkCLI/setup-pc-env.sh
#   bash benchmarkCLI/setup-pc-env.sh --force    # Recreate venv

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-pc"
FORCE=false
[ "$1" = "--force" ] && FORCE=true

echo ""
echo "========================================================"
echo "  PC BENCHMARK ENVIRONMENT SETUP"
echo "========================================================"
echo ""

# --- Find Python ---
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    command -v "$cmd" &>/dev/null && PYTHON_CMD="$cmd" && break
done
[ -z "$PYTHON_CMD" ] && echo "[ERROR] Python 3.10+ not found." && exit 1
echo "  Python: $($PYTHON_CMD --version) ($PYTHON_CMD)"

# --- Detect CUDA version ---
echo ""
echo "--- CUDA Detection ---"
CUDA_MAJOR=""
if command -v nvcc &>/dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    CUDA_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
    echo "  nvcc:          $NVCC_VER (toolkit installed)"
else
    echo "  [WARN] nvcc not found. Checking nvidia-smi..."
fi

if command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  nvidia-smi:    CUDA $DRIVER_CUDA (driver max supported)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
    [ -z "$CUDA_MAJOR" ] && CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
else
    echo "  [WARN] nvidia-smi not found."
fi

if [ -z "$CUDA_MAJOR" ]; then
    echo "  [WARN] Could not detect CUDA. Will install CPU-only onnxruntime."
fi
echo "  Detected CUDA major: ${CUDA_MAJOR:-none}"

# --- Check existing venv ---
if [ -d "$VENV_DIR" ] && [ "$FORCE" = false ]; then
    VENV_PYTHON="$VENV_DIR/bin/python3"
    if [ -x "$VENV_PYTHON" ]; then
        echo ""
        echo "--- Checking existing venv ---"
        ALL_OK=true
        for pkg in numpy onnxruntime; do
            "$VENV_PYTHON" -c "import $pkg; print('  [OK] {} {}'.format('$pkg', $pkg.__version__))" 2>/dev/null || {
                echo "  [MISS] $pkg"; ALL_OK=false; }
        done
        if [ "$ALL_OK" = true ]; then
            "$VENV_PYTHON" -c "import onnxruntime as ort; print('  EPs: {}'.format(ort.get_available_providers()))" 2>/dev/null
            echo "  Venv healthy: $VENV_DIR"
            echo "  Activate: source $VENV_DIR/bin/activate"
            exit 0
        fi
    else
        echo "  [WARN] Venv broken. Recreating..."
        rm -rf "$VENV_DIR"
    fi
fi

[ -d "$VENV_DIR" ] && [ "$FORCE" = true ] && rm -rf "$VENV_DIR"

# --- Create venv ---
echo ""
echo "--- Creating venv at $VENV_DIR ---"
$PYTHON_CMD -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

echo "  Installing dependencies from requirements-pc.txt..."
pip install -r "$SCRIPT_DIR/requirements-pc.txt" --quiet

# --- Export model to models-pc (no INT32 fix) ---
MODELS_PC="$SCRIPT_DIR/models-pc"
if [ ! -f "$MODELS_PC/yolov8n.onnx" ]; then
    echo ""
    echo "--- Exporting YOLOv8n to models-pc (no INT64 fix) ---"
    python3 "$SCRIPT_DIR/download-yolo-model.py" --model yolov8n --no-fix-int64 -o "$MODELS_PC"
else
    echo "  [OK] models-pc/yolov8n.onnx already exists"
fi

# --- Install onnxruntime-gpu matching CUDA version ---
echo ""
echo "--- Installing onnxruntime-gpu ---"
ORT_INSTALLED=false
CUDA11_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/"

# Remove any existing onnxruntime to avoid conflicts
pip uninstall onnxruntime-gpu onnxruntime -y --quiet 2>/dev/null
pip cache purge 2>/dev/null

if [ "$CUDA_MAJOR" = "11" ]; then
    echo "  Detected CUDA 11.x. Trying CUDA 11 builds..."
    # Try pinned versions known to ship CUDA 11 builds (newest first)
    for VER in 1.16.3 1.16.0 1.15.1 1.14.1; do
        echo "  Trying onnxruntime-gpu==$VER ..."
        if pip install "onnxruntime-gpu==$VER" --no-cache-dir --extra-index-url "$CUDA11_INDEX" --quiet 2>/dev/null; then
            # Verify it actually loads with CUDA 11
            if python3 -c "import onnxruntime" 2>/dev/null; then
                echo "  [OK] Installed onnxruntime-gpu==$VER"
                ORT_INSTALLED=true
                break
            fi
            pip uninstall onnxruntime-gpu -y --quiet 2>/dev/null
        fi
    done
elif [ "$CUDA_MAJOR" = "12" ] || [ "$CUDA_MAJOR" = "13" ]; then
    echo "  Detected CUDA $CUDA_MAJOR. Installing default (CUDA 12)..."
    pip install onnxruntime-gpu --no-cache-dir --quiet && ORT_INSTALLED=true
fi

if [ "$ORT_INSTALLED" = false ]; then
    echo "  [WARN] onnxruntime-gpu failed. Installing CPU version as fallback..."
    pip install onnxruntime --no-cache-dir --quiet
fi

# --- Optional: TensorRT / trtexec ---
echo ""
echo "--- Optional: TensorRT ---"
python3 -c "import tensorrt; print('  [OK] tensorrt {}'.format(tensorrt.__version__))" 2>/dev/null || \
    echo "  [INFO] tensorrt not found. Optional: install TensorRT SDK for Method 2."

TRTEXEC=$(which trtexec 2>/dev/null || find /usr -name trtexec 2>/dev/null | head -1 || true)
[ -n "$TRTEXEC" ] && echo "  [OK] trtexec: $TRTEXEC" || \
    echo "  [INFO] trtexec not found. Optional: install TensorRT SDK for Method 1."

# --- Verify ---
echo ""
echo "--- Verification ---"
for pkg in numpy onnxruntime; do
    python3 -c "import $pkg; print('  [OK] {} {}'.format('$pkg', $pkg.__version__))" 2>/dev/null || \
        echo "  [FAIL] $pkg"
done
python3 -c "
import onnxruntime as ort
eps = ort.get_available_providers()
print('  EPs: {}'.format(eps))
for ep in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']:
    status = '[OK]' if ep in eps else '[--]'
    print('  {} {}'.format(status, ep))
if 'CUDAExecutionProvider' not in eps and 'TensorrtExecutionProvider' not in eps:
    print('  [WARN] No GPU EP! Check CUDA toolkit: nvcc --version')
" 2>/dev/null

# --- Check model ---
echo ""
MODELS_DIR="$SCRIPT_DIR/models-pc"
ONNX_COUNT=$(find "$MODELS_DIR" -name "*.onnx" 2>/dev/null | wc -l)
[ "$ONNX_COUNT" -gt 0 ] && echo "  Models: $ONNX_COUNT ONNX file(s) in models-pc/" || \
    echo "  [WARN] No ONNX models. Run: python3 benchmarkCLI/download-yolo-model.py --no-fix-int64 -o models-pc"

echo ""
echo "========================================================"
echo "  Setup complete!"
echo "========================================================"
echo ""
echo "  Activate:  source $VENV_DIR/bin/activate"
echo "  Export:    python3 benchmarkCLI/download-yolo-model.py --model yolov8n --no-fix-int64 -o models-pc"
echo "  Benchmark: bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n -n 100"
echo ""
