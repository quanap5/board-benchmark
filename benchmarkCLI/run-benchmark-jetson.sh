#!/bin/bash
# Native benchmark runner for NVIDIA Jetson (no Docker).
# Usage:
#   bash benchmarkCLI/run-benchmark-jetson.sh                         # auto-detect best method
#   bash benchmarkCLI/run-benchmark-jetson.sh --trtexec-only          # trtexec CLI only
#   bash benchmarkCLI/run-benchmark-jetson.sh --trt-python            # TRT Python API only
#   bash benchmarkCLI/run-benchmark-jetson.sh --trt-python -p fp16 int8
#   bash benchmarkCLI/run-benchmark-jetson.sh --all                   # trtexec + TRT Python + ORT

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

MODEL_NAME="yolov8n"; ITERATIONS=100; WARMUP=10; BATCH_SIZE=1
MODE="auto"  # auto, trtexec, trt-python, ort, all
TRT_PYTHON_PRECISIONS=""
CONVERT_FIRST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL_NAME="$2"; shift 2 ;;
        -n)              ITERATIONS="$2"; shift 2 ;;
        -w)              WARMUP="$2"; shift 2 ;;
        -b)              BATCH_SIZE="$2"; shift 2 ;;
        --trtexec-only)  MODE="trtexec"; shift ;;
        --trt-python)    MODE="trt-python"; shift ;;
        --ort-only)      MODE="ort"; shift ;;
        --all)           MODE="all"; shift ;;
        -p)              shift; while [[ $# -gt 0 ]] && [[ ! "$1" == --* ]]; do
                             TRT_PYTHON_PRECISIONS="$TRT_PYTHON_PRECISIONS $1"; shift
                         done ;;
        --convert)       CONVERT_FIRST=true; shift ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

ONNX_FILE="$MODELS_DIR/$MODEL_NAME.onnx"
ENGINE_FILE="$MODELS_DIR/$MODEL_NAME.engine"

# Auto-detect: prefer trt-python if available, fallback to trtexec
HAS_TRT_PYTHON=false
HAS_TRTEXEC=false
HAS_ORT=false
python3 -c "import tensorrt; import pycuda.driver" 2>/dev/null && HAS_TRT_PYTHON=true
[ -x /usr/src/tensorrt/bin/trtexec ] || [ -x /usr/bin/trtexec ] && HAS_TRTEXEC=true
python3 -c "import onnxruntime" 2>/dev/null && HAS_ORT=true

if [ "$MODE" = "auto" ]; then
    if [ "$HAS_TRT_PYTHON" = true ]; then
        MODE="trt-python"
    elif [ "$HAS_TRTEXEC" = true ]; then
        MODE="trtexec"
    elif [ "$HAS_ORT" = true ]; then
        MODE="ort"
    else
        echo "[ERROR] No benchmark backend found. Need tensorrt+pycuda, trtexec, or onnxruntime."
        exit 1
    fi
fi

echo ""
echo "========================================================"
echo "  JETSON BENCHMARK RUNNER (native, no Docker)"
echo "========================================================"
echo "  Model: $MODEL_NAME | Iters: $ITERATIONS | Warmup: $WARMUP | Batch: $BATCH_SIZE"
echo "  Mode:  $MODE | TRT Python: $HAS_TRT_PYTHON | trtexec: $HAS_TRTEXEC | ORT: $HAS_ORT"
echo ""

[ ! -f /etc/nv_tegra_release ] && echo "[WARN] Not a Jetson device." && echo ""

if [ ! -f "$ONNX_FILE" ]; then
    echo "[ERROR] Model not found: $ONNX_FILE"
    echo "  scp dev-host:benchmarkCLI/models/$MODEL_NAME.onnx $MODELS_DIR/"
    exit 1
fi

# --- Hardware info ---
echo "========================================================"
echo "  SYSTEM HARDWARE INFO"
echo "========================================================"
python3 "$SCRIPT_DIR/hardware-info.py"
echo ""

# --- trtexec benchmark ---
TRTEXEC_CSV=""
if [ "$MODE" = "trtexec" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRTEXEC" = true ]; then
        # Convert engine if needed
        if [ ! -f "$ENGINE_FILE" ] || [ "$CONVERT_FIRST" = true ]; then
            bash "$SCRIPT_DIR/convert-to-tensorrt.sh" --model "$MODEL_NAME" --fp16
        fi
        echo "========================================================"
        echo "  TRTEXEC BENCHMARK (TensorRT CLI, FP16)"
        echo "========================================================"
        BENCH_MODEL="$ONNX_FILE"
        [ -f "$ENGINE_FILE" ] && BENCH_MODEL="$ENGINE_FILE"
        TRT_TMPFILE=$(mktemp /tmp/trt-bench.XXXXXX)
        python3 "$SCRIPT_DIR/trtexec-benchmark.py" \
            "$BENCH_MODEL" -n "$ITERATIONS" -w "$WARMUP" --fp16 --csv 2>&1 | tee "$TRT_TMPFILE"
        TRTEXEC_CSV=$(grep "^CSV:" "$TRT_TMPFILE" | tail -1)
        rm -f "$TRT_TMPFILE"
        echo ""
    else
        echo "[WARN] trtexec not found. Skipping."
    fi
fi

# --- TRT Python benchmark (FP32/FP16/INT8) ---
if [ "$MODE" = "trt-python" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRT_PYTHON" = true ]; then
        echo "========================================================"
        echo "  TRT PYTHON BENCHMARK (FP32/FP16/INT8)"
        echo "========================================================"
        PREC_ARGS=""
        [ -n "$TRT_PYTHON_PRECISIONS" ] && PREC_ARGS="-p $TRT_PYTHON_PRECISIONS"
        python3 "$SCRIPT_DIR/trt-python-benchmark.py" \
            "$ONNX_FILE" $PREC_ARGS -n "$ITERATIONS" -w "$WARMUP" --save-engine
        echo ""
    else
        echo "[WARN] tensorrt/pycuda not available. Skipping TRT Python benchmark."
        echo "       Install: pip3 install --user pycuda"
    fi
fi

# --- onnxruntime benchmark ---
ORT_CSV=""
if [ "$MODE" = "ort" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_ORT" = true ]; then
        echo "========================================================"
        echo "  ONNXRUNTIME BENCHMARK (TensorRT EP -> CUDA EP -> CPU)"
        echo "========================================================"
        ORT_TMPFILE=$(mktemp /tmp/ort-bench.XXXXXX)
        python3 "$SCRIPT_DIR/onnx-benchmark.py" \
            "$ONNX_FILE" -p tensorrt -n "$ITERATIONS" -w "$WARMUP" -b "$BATCH_SIZE" --csv 2>&1 | tee "$ORT_TMPFILE"
        ORT_CSV=$(grep "^CSV:" "$ORT_TMPFILE" | tail -1)
        rm -f "$ORT_TMPFILE"
        echo ""
    else
        echo "[WARN] onnxruntime not installed. Skipping."
    fi
fi

# --- Summary (trtexec/ORT modes only, trt-python prints its own) ---
parse_trtexec_csv() {
    [ -z "$1" ] && return
    IFS=',' read -r _ TRT_AVG TRT_MIN TRT_MAX TRT_MEDIAN TRT_P95 TRT_P99 TRT_FPS _ <<< "${1#CSV:}"
}
parse_ort_csv() {
    [ -z "$1" ] && return
    IFS=',' read -r ORT_PROVIDER _ _ _ ORT_AVG ORT_MIN ORT_MAX ORT_MEDIAN ORT_P95 ORT_P99 _ ORT_FPS _ <<< "${1#CSV:}"
}

parse_trtexec_csv "$TRTEXEC_CSV"
parse_ort_csv "$ORT_CSV"

if [ -n "$TRTEXEC_CSV" ] || [ -n "$ORT_CSV" ]; then
    echo "========================================================"
    echo "  JETSON BENCHMARK SUMMARY"
    echo "========================================================"
    echo ""
    echo "  Device:      $(head -1 /etc/nv_tegra_release 2>/dev/null | cut -c1-60 || echo 'Unknown')"
    echo "  Model:       $MODEL_NAME | Iters: $ITERATIONS | Batch: $BATCH_SIZE"
    echo ""

    if [ -n "$TRTEXEC_CSV" ] && [ -n "$ORT_CSV" ]; then
        printf "  %-20s %15s %18s\n" "Metric" "trtexec (FP16)" "ORT ($ORT_PROVIDER)"
        printf "  %-20s %15s %18s\n" "--------------------" "---------------" "------------------"
        for m in "Avg latency:TRT_AVG:ORT_AVG" "Min latency:TRT_MIN:ORT_MIN" "Max latency:TRT_MAX:ORT_MAX" \
                 "Median:TRT_MEDIAN:ORT_MEDIAN" "P95:TRT_P95:ORT_P95" "P99:TRT_P99:ORT_P99"; do
            IFS=':' read -r label v1 v2 <<< "$m"
            [ -z "${!v1}" ] && [ -z "${!v2}" ] && continue
            printf "  %-20s %13s ms %16s ms\n" "$label" "${!v1:-N/A}" "${!v2:-N/A}"
        done
        printf "  %-20s %15s %18s\n" "FPS" "$TRT_FPS" "$ORT_FPS"
    elif [ -n "$TRTEXEC_CSV" ]; then
        printf "  %-20s %15s\n" "Metric" "trtexec (FP16)"
        printf "  %-20s %15s\n" "--------------------" "---------------"
        for m in "Avg:TRT_AVG" "Min:TRT_MIN" "Median:TRT_MEDIAN" "P95:TRT_P95" "P99:TRT_P99"; do
            IFS=':' read -r l v <<< "$m"
            [ -z "${!v}" ] && continue
            printf "  %-20s %13s ms\n" "$l" "${!v}"
        done
        printf "  %-20s %15s\n" "FPS" "$TRT_FPS"
    elif [ -n "$ORT_CSV" ]; then
        printf "  %-20s %18s\n" "Metric" "ORT ($ORT_PROVIDER)"
        printf "  %-20s %18s\n" "--------------------" "------------------"
        for m in "Avg:ORT_AVG" "Min:ORT_MIN" "Median:ORT_MEDIAN" "P95:ORT_P95" "P99:ORT_P99"; do
            IFS=':' read -r l v <<< "$m"
            [ -z "${!v}" ] && continue
            printf "  %-20s %16s ms\n" "$l" "${!v}"
        done
        printf "  %-20s %18s\n" "FPS" "$ORT_FPS"
    fi
fi

echo ""
echo "========================================================="
echo "  Benchmark complete."
echo "========================================================="
