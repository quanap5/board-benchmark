#!/bin/bash
# Native PC benchmark runner (no Docker).
# Supports onnxruntime (GPU/CPU), trtexec, and TRT Python methods.
#
# Usage:
#   bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n -n 100
#   bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n --ort-only -n 100
#   bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n --trtexec-only -p fp32 fp16 int8
#   bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n --trt-python -p fp16 int8
#   bash benchmarkCLI/run-benchmark-pc.sh --model yolov8n --all -n 100

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models-pc"
VENV_DIR="$SCRIPT_DIR/.venv-pc"

# Colors
if [ -t 1 ]; then
    C_RST="\033[0m"; C_BOLD="\033[1m"; C_DIM="\033[2m"
    C_RED="\033[31m"; C_GRN="\033[32m"; C_YEL="\033[33m"; C_CYA="\033[36m"
else
    C_RST=""; C_BOLD=""; C_DIM=""; C_RED=""; C_GRN=""; C_YEL=""; C_CYA=""
fi
_info()  { echo -e "${C_CYA}${C_BOLD}ℹ️  INFO${C_RST} $1"; }
_ok()    { echo -e "${C_GRN}${C_BOLD}✅ OK  ${C_RST} $1"; }
_warn()  { echo -e "${C_YEL}${C_BOLD}⚠️  WARN${C_RST} $1"; }
_err()   { echo -e "${C_RED}${C_BOLD}❌ ERR ${C_RST} $1"; }
_hdr()   { echo -e "\n${C_CYA}========================================================${C_RST}"; \
           echo -e "  $1 ${C_BOLD}$2${C_RST}"; \
           echo -e "${C_CYA}========================================================${C_RST}"; }

MODEL_NAME="yolov8n"; ITERATIONS=100; WARMUP=10; BATCH_SIZE=1
MODE="auto"; PRECISIONS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL_NAME="$2"; shift 2 ;;
        -n)              ITERATIONS="$2"; shift 2 ;;
        -w)              WARMUP="$2"; shift 2 ;;
        -b)              BATCH_SIZE="$2"; shift 2 ;;
        --ort-only)      MODE="ort"; shift ;;
        --trtexec-only)  MODE="trtexec"; shift ;;
        --trt-python)    MODE="trt-python"; shift ;;
        --all)           MODE="all"; shift ;;
        -p)              shift; while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                             PRECISIONS="$PRECISIONS $1"; shift
                         done ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

ONNX_FILE="$MODELS_DIR/$MODEL_NAME.onnx"

# Activate venv if exists
[ -f "$VENV_DIR/bin/activate" ] && source "$VENV_DIR/bin/activate"

# Detect available methods
HAS_ORT=false; HAS_TRTEXEC=false; HAS_TRT_PYTHON=false
python3 -c "import onnxruntime" 2>/dev/null && HAS_ORT=true
python3 -c "import tensorrt" 2>/dev/null && HAS_TRT_PYTHON=true
command -v trtexec &>/dev/null && HAS_TRTEXEC=true
[ "$HAS_TRTEXEC" = false ] && [ -x /usr/src/tensorrt/bin/trtexec ] && HAS_TRTEXEC=true

# Auto-detect: prefer ORT on PC (easiest install)
if [ "$MODE" = "auto" ]; then
    if [ "$HAS_ORT" = true ]; then MODE="ort"
    elif [ "$HAS_TRT_PYTHON" = true ]; then MODE="trt-python"
    elif [ "$HAS_TRTEXEC" = true ]; then MODE="trtexec"
    else _err "No benchmark backend. Run: bash benchmarkCLI/setup-pc-env.sh"; exit 1
    fi
fi

_hdr "🖥️ " "PC BENCHMARK RUNNER (native, no Docker)"
_info "Model: $MODEL_NAME | Iters: $ITERATIONS | Warmup: $WARMUP | Batch: $BATCH_SIZE"
_info "Mode:  $MODE | ORT: $HAS_ORT | TRT Python: $HAS_TRT_PYTHON | trtexec: $HAS_TRTEXEC"
echo ""

if [ ! -f "$ONNX_FILE" ]; then
    _err "Model not found: $ONNX_FILE"
    _info "Export first: python3 benchmarkCLI/download-yolo-model.py --model $MODEL_NAME"
    exit 1
fi

# --- Hardware info ---
_hdr "🖥️ " "SYSTEM HARDWARE INFO"
python3 "$SCRIPT_DIR/hardware-info.py"
echo ""

# --- Method 3: onnxruntime (default on PC) ---
if [ "$MODE" = "ort" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_ORT" = true ]; then
        # Determine precisions to benchmark
        ORT_PRECS="${PRECISIONS:-fp32}"

        # Export multi-precision models if needed
        PREC_EXPORT_ARGS=""
        for pr in $ORT_PRECS; do PREC_EXPORT_ARGS="$PREC_EXPORT_ARGS $pr"; done
        _hdr "⚡" "EXPORTING MULTI-PRECISION MODELS"
        python3 "$SCRIPT_DIR/export-multi-precision.py" --model "$MODEL_NAME" -o "$MODELS_DIR" -p $PREC_EXPORT_ARGS

        # Benchmark each precision with full EP fallback (TensorRT -> CUDA -> CPU)
        _hdr "🔵" "METHOD 3: ONNXRUNTIME BENCHMARK (TensorRT → CUDA → CPU)"
        for pr in $ORT_PRECS; do
            MODEL_FILE="$MODELS_DIR/${MODEL_NAME}-${pr}.onnx"
            if [ -f "$MODEL_FILE" ]; then
                _info "Precision: ${pr^^} | EP chain: auto (TensorRT → CUDA → CPU)"
                python3 "$SCRIPT_DIR/onnx-benchmark.py" \
                    "$MODEL_FILE" --provider auto -n "$ITERATIONS" -w "$WARMUP" -b "$BATCH_SIZE" --csv
                echo ""
            else
                _warn "Model not found: $MODEL_FILE"
            fi
        done
    else
        _warn "onnxruntime not installed. Run: bash benchmarkCLI/setup-pc-env.sh"
    fi
fi

# --- Method 1: trtexec ---
if [ "$MODE" = "trtexec" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRTEXEC" = true ]; then
        _hdr "🟢" "METHOD 1: TRTEXEC BENCHMARK"
        PREC_ARGS="-p fp16"
        [ -n "$PRECISIONS" ] && PREC_ARGS="-p $PRECISIONS"
        python3 "$SCRIPT_DIR/trtexec-benchmark.py" \
            "$ONNX_FILE" $PREC_ARGS -n "$ITERATIONS" -w "$WARMUP" --save-engine
        echo ""
    else
        _warn "trtexec not found. Install TensorRT SDK."
    fi
fi

# --- Method 2: TRT Python ---
if [ "$MODE" = "trt-python" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRT_PYTHON" = true ]; then
        _hdr "🟣" "METHOD 2: TRT PYTHON BENCHMARK"
        PREC_ARGS=""
        [ -n "$PRECISIONS" ] && PREC_ARGS="-p $PRECISIONS"
        python3 "$SCRIPT_DIR/trt-python-benchmark.py" \
            "$ONNX_FILE" $PREC_ARGS -n "$ITERATIONS" -w "$WARMUP" --save-engine
        echo ""
    else
        _warn "tensorrt Python not found. Install TensorRT SDK."
    fi
fi

echo ""
_ok "PC benchmark complete."
