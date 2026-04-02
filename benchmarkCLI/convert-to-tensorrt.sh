#!/bin/bash
# Convert ONNX model to TensorRT engine on Jetson.
# The engine is device-specific and not portable across TensorRT versions.
#
# Usage:
#   bash benchmarkCLI/convert-to-tensorrt.sh
#   bash benchmarkCLI/convert-to-tensorrt.sh --model yolov8s --fp16
#   bash benchmarkCLI/convert-to-tensorrt.sh --model yolov8n --int8
#
# Options:
#   --model NAME    Model name (default: yolov8n)
#   --fp16          Enable FP16 precision (default, recommended for Jetson)
#   --fp32          Use FP32 precision
#   --int8          Enable INT8 quantization (needs calibration data)
#   --workspace MB  TensorRT workspace size in MB (default: 1024)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# Defaults
MODEL_NAME="yolov8n"
PRECISION="--fp16"
WORKSPACE=1024

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL_NAME="$2"; shift 2 ;;
        --fp16)      PRECISION="--fp16"; shift ;;
        --fp32)      PRECISION=""; shift ;;
        --int8)      PRECISION="--int8"; shift ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

ONNX_FILE="$MODELS_DIR/$MODEL_NAME.onnx"
ENGINE_FILE="$MODELS_DIR/$MODEL_NAME.engine"

echo ""
echo "========================================================"
echo "  ONNX -> TensorRT Engine Conversion"
echo "========================================================"
echo ""

# --- Find trtexec ---
TRTEXEC=""
for path in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
    if [ -x "$path" ]; then
        TRTEXEC="$path"
        break
    fi
done

if [ -z "$TRTEXEC" ]; then
    echo "[ERROR] trtexec not found."
    echo "        Expected at: /usr/src/tensorrt/bin/trtexec"
    echo "        Is TensorRT installed via JetPack?"
    exit 1
fi

# --- Check ONNX model ---
if [ ! -f "$ONNX_FILE" ]; then
    echo "[ERROR] ONNX model not found: $ONNX_FILE"
    echo "        Copy from dev machine:"
    echo "        scp dev-host:benchmarkCLI/models/$MODEL_NAME.onnx $MODELS_DIR/"
    exit 1
fi

ONNX_SIZE=$(ls -lh "$ONNX_FILE" | awk '{print $5}')
echo "  Input:       $ONNX_FILE ($ONNX_SIZE)"
echo "  Output:      $ENGINE_FILE"
echo "  Precision:   ${PRECISION:-FP32}"
echo "  Workspace:   ${WORKSPACE} MB"
echo "  trtexec:     $TRTEXEC"
echo ""

# --- Convert ---
echo "--- Converting (this may take several minutes on TX2) ---"
echo ""

$TRTEXEC \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    $PRECISION \
    --workspace="$WORKSPACE" \
    2>&1 | tee "$MODELS_DIR/$MODEL_NAME-convert.log"

echo ""

if [ -f "$ENGINE_FILE" ]; then
    ENGINE_SIZE=$(ls -lh "$ENGINE_FILE" | awk '{print $5}')
    echo "========================================================"
    echo "  Conversion complete!"
    echo "  Engine: $ENGINE_FILE ($ENGINE_SIZE)"
    echo ""
    echo "  Benchmark with:"
    echo "    bash benchmarkCLI/run-benchmark-jetson.sh --model $MODEL_NAME"
    echo "========================================================"
else
    echo "[ERROR] Engine file not created. Check log:"
    echo "  $MODELS_DIR/$MODEL_NAME-convert.log"
    exit 1
fi
echo ""
