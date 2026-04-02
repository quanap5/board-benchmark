#!/bin/bash
# Setup benchmark environment on NVIDIA Jetson TX2 NX (JetPack 4.6.x).
# Usage: bash benchmarkCLI/setup-jetson.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
ORT_AVAILABLE=false
ONNX_COUNT=0

echo ""
echo "========================================================"
echo "  JETSON SETUP - Benchmark Environment"
echo "========================================================"

# --- Detect Jetson ---
if [ ! -f /etc/nv_tegra_release ]; then
    echo "[ERROR] Not a Jetson device (/etc/nv_tegra_release missing)."; exit 1
fi
echo ""
echo "  Platform:  $(head -1 /etc/nv_tegra_release)"
echo "  Arch:      $(uname -m)"
L4T_MAJOR=$(grep -oP 'R\K[0-9]+' /etc/nv_tegra_release)
L4T_MINOR=$(grep -oP 'REVISION: \K[0-9.]+' /etc/nv_tegra_release)
echo "  L4T:       R${L4T_MAJOR}.${L4T_MINOR}"

# --- Check CUDA ---
echo ""
if command -v nvcc &>/dev/null; then
    echo "  CUDA:      $(nvcc --version | grep release | sed 's/.*release //;s/,.*//')"
elif [ -d /usr/local/cuda ]; then
    echo "  CUDA:      dir exists, add to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
else
    echo "  [WARN] CUDA not found"
fi

# --- Check TensorRT ---
TRTEXEC=""
for p in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
    [ -x "$p" ] && TRTEXEC="$p" && break
done
if [ -n "$TRTEXEC" ]; then
    echo "  trtexec:   $TRTEXEC"
else
    echo "  [WARN] trtexec not found"
fi

# --- Check cuDNN ---
CUDNN_H="/usr/include/cudnn_version.h"
if [ -f "$CUDNN_H" ]; then
    echo "  cuDNN:     $(grep CUDNN_MAJOR "$CUDNN_H" | awk '{print $3}').$(grep CUDNN_MINOR "$CUDNN_H" | awk '{print $3}').$(grep CUDNN_PATCHLEVEL "$CUDNN_H" | awk '{print $3}')"
fi

# --- Check Python + numpy ---
PYTHON_CMD=""
for cmd in python3 python; do command -v "$cmd" &>/dev/null && PYTHON_CMD="$cmd" && break; done
[ -z "$PYTHON_CMD" ] && echo "[ERROR] Python not found." && exit 1
echo "  Python:    $($PYTHON_CMD --version 2>&1)"
$PYTHON_CMD -m pip install --user --upgrade pip numpy --quiet 2>/dev/null || true

# --- Check onnxruntime ---
echo ""
echo "--- onnxruntime ---"
if $PYTHON_CMD -c "import onnxruntime as ort; print(f'  Version: {ort.__version__}'); print(f'  EPs:     {ort.get_available_providers()}')" 2>/dev/null; then
    ORT_AVAILABLE=true
else
    echo "  [WARN] Not installed. Build from source or use --trtexec-only."
    echo "    git clone --recursive https://github.com/microsoft/onnxruntime"
    echo "    ./build.sh --config Release --build_wheel --parallel \\"
    echo "      --use_cuda --cuda_home /usr/local/cuda \\"
    echo "      --cudnn_home /usr/lib/aarch64-linux-gnu \\"
    echo "      --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu"
fi

# --- Check models ---
echo ""
echo "--- Models ---"
if [ -d "$MODELS_DIR" ]; then
    ONNX_COUNT=$(find "$MODELS_DIR" -name "*.onnx" 2>/dev/null | wc -l)
    [ "$ONNX_COUNT" -gt 0 ] && find "$MODELS_DIR" -name "*.onnx" -exec ls -lh {} \; | awk '{print "  " $5 "  " $9}' \
        || echo "  [WARN] No .onnx files. scp from dev: scp dev:benchmarkCLI/models/*.onnx $MODELS_DIR/"
else
    echo "  [WARN] models/ missing. mkdir -p $MODELS_DIR"
fi

# --- Summary ---
echo ""
echo "========================================================"
echo "  SUMMARY"
echo "========================================================"
[ -n "$TRTEXEC" ] && echo "  [OK]   trtexec" || echo "  [FAIL] trtexec"
[ "$ORT_AVAILABLE" = true ] && echo "  [OK]   onnxruntime" || echo "  [WARN] onnxruntime (trtexec still works)"
[ "$ONNX_COUNT" -gt 0 ] 2>/dev/null && echo "  [OK]   ONNX model(s)" || echo "  [FAIL] No ONNX models"
echo ""
if [ -n "$TRTEXEC" ] && [ "$ONNX_COUNT" -gt 0 ] 2>/dev/null; then
    echo "  Ready! Run: bash benchmarkCLI/run-benchmark-jetson.sh"
else
    echo "  Fix issues above before benchmarking."
fi
echo ""
