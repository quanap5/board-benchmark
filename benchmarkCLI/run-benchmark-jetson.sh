#!/bin/bash
# Native benchmark runner for NVIDIA Jetson (no Docker).
# Usage:
#   bash benchmarkCLI/run-benchmark-jetson.sh
#   bash benchmarkCLI/run-benchmark-jetson.sh --model yolov8s -n 500
#   bash benchmarkCLI/run-benchmark-jetson.sh --trtexec-only
#   bash benchmarkCLI/run-benchmark-jetson.sh --ort-only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

MODEL_NAME="yolov8n"; ITERATIONS=100; WARMUP=10; BATCH_SIZE=1
RUN_TRTEXEC=true; RUN_ORT=true; CONVERT_FIRST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)         MODEL_NAME="$2"; shift 2 ;;
        -n)              ITERATIONS="$2"; shift 2 ;;
        -w)              WARMUP="$2"; shift 2 ;;
        -b)              BATCH_SIZE="$2"; shift 2 ;;
        --trtexec-only)  RUN_ORT=false; shift ;;
        --ort-only)      RUN_TRTEXEC=false; shift ;;
        --convert)       CONVERT_FIRST=true; shift ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

ONNX_FILE="$MODELS_DIR/$MODEL_NAME.onnx"
ENGINE_FILE="$MODELS_DIR/$MODEL_NAME.engine"

echo ""
echo "========================================================"
echo "  JETSON BENCHMARK RUNNER (native, no Docker)"
echo "========================================================"
echo "  Model: $MODEL_NAME | Iters: $ITERATIONS | Warmup: $WARMUP | Batch: $BATCH_SIZE"
echo ""

[ ! -f /etc/nv_tegra_release ] && echo "[WARN] Not a Jetson device. Results may differ." && echo ""

if [ ! -f "$ONNX_FILE" ]; then
    echo "[ERROR] Model not found: $ONNX_FILE"
    echo "  scp dev-host:benchmarkCLI/models/$MODEL_NAME.onnx $MODELS_DIR/"
    exit 1
fi

# --- Convert to TensorRT engine if needed ---
if [ "$RUN_TRTEXEC" = true ] && [ ! -f "$ENGINE_FILE" ] || [ "$CONVERT_FIRST" = true ]; then
    [ ! -f "$ENGINE_FILE" ] && bash "$SCRIPT_DIR/convert-to-tensorrt.sh" --model "$MODEL_NAME" --fp16
fi
[ -f "$ENGINE_FILE" ] && echo "  TRT engine: $ENGINE_FILE ($(ls -lh "$ENGINE_FILE" | awk '{print $5}'))"
echo ""

# --- Hardware info ---
echo "========================================================"
echo "  SYSTEM HARDWARE INFO"
echo "========================================================"
python3 "$SCRIPT_DIR/hardware-info.py"
echo ""

# --- trtexec benchmark ---
TRTEXEC_CSV=""
if [ "$RUN_TRTEXEC" = true ]; then
    echo "========================================================"
    echo "  TRTEXEC BENCHMARK (TensorRT native, FP16)"
    echo "========================================================"
    BENCH_MODEL="$ONNX_FILE"
    [ -f "$ENGINE_FILE" ] && BENCH_MODEL="$ENGINE_FILE"

    TRTEXEC_RESULT=$(python3 "$SCRIPT_DIR/trtexec-benchmark.py" \
        "$BENCH_MODEL" -n "$ITERATIONS" -w "$WARMUP" --fp16 --csv 2>&1)
    echo "$TRTEXEC_RESULT"
    TRTEXEC_CSV=$(echo "$TRTEXEC_RESULT" | grep "^CSV:" | tail -1)
    echo ""
fi

# --- onnxruntime benchmark ---
ORT_CSV=""
if [ "$RUN_ORT" = true ]; then
    if python3 -c "import onnxruntime" 2>/dev/null; then
        echo "========================================================"
        echo "  ONNXRUNTIME BENCHMARK (TensorRT EP -> CUDA EP -> CPU)"
        echo "========================================================"
        ORT_RESULT=$(python3 "$SCRIPT_DIR/onnx-benchmark.py" \
            "$ONNX_FILE" -p tensorrt -n "$ITERATIONS" -w "$WARMUP" -b "$BATCH_SIZE" --csv 2>&1)
        echo "$ORT_RESULT"
        ORT_CSV=$(echo "$ORT_RESULT" | grep "^CSV:" | tail -1)
        echo ""
    else
        echo "[WARN] onnxruntime not installed. Skipping. Use --trtexec-only or setup-jetson.sh"
        RUN_ORT=false
    fi
fi

# --- Summary ---
# Parse CSV lines into variables
parse_trtexec_csv() {
    [ -z "$1" ] && return
    IFS=',' read -r _ TRT_AVG TRT_MIN TRT_MAX TRT_MEDIAN TRT_P95 TRT_P99 TRT_FPS _ <<< "${1#CSV:}"
}
parse_ort_csv() {
    [ -z "$1" ] && return
    IFS=',' read -r ORT_PROVIDER _ _ _ ORT_AVG ORT_MIN ORT_MAX ORT_MEDIAN ORT_P95 ORT_P99 _ ORT_FPS _ <<< "${1#CSV:}"
}
print_col() { printf "  %-20s %13s ms\n" "$1" "$2"; }

parse_trtexec_csv "$TRTEXEC_CSV"
parse_ort_csv "$ORT_CSV"

echo ""
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
             "Median latency:TRT_MEDIAN:ORT_MEDIAN" "P95 latency:TRT_P95:ORT_P95" "P99 latency:TRT_P99:ORT_P99"; do
        IFS=':' read -r label v1 v2 <<< "$m"
        printf "  %-20s %13s ms %16s ms\n" "$label" "${!v1}" "${!v2}"
    done
    printf "  %-20s %15s %18s\n" "FPS" "$TRT_FPS" "$ORT_FPS"
elif [ -n "$TRTEXEC_CSV" ]; then
    printf "  %-20s %15s\n" "Metric" "trtexec (FP16)"
    printf "  %-20s %15s\n" "--------------------" "---------------"
    for m in "Avg latency:TRT_AVG" "Min latency:TRT_MIN" "Median:TRT_MEDIAN" "P95:TRT_P95" "P99:TRT_P99"; do
        IFS=':' read -r label v1 <<< "$m"; print_col "$label" "${!v1}"
    done
    printf "  %-20s %15s\n" "FPS" "$TRT_FPS"
elif [ -n "$ORT_CSV" ]; then
    printf "  %-20s %18s\n" "Metric" "ORT ($ORT_PROVIDER)"
    printf "  %-20s %18s\n" "--------------------" "------------------"
    for m in "Avg latency:ORT_AVG" "Min latency:ORT_MIN" "Median:ORT_MEDIAN" "P95:ORT_P95" "P99:ORT_P99"; do
        IFS=':' read -r label v1 <<< "$m"; printf "  %-20s %16s ms\n" "$label" "${!v1}"
    done
    printf "  %-20s %18s\n" "FPS" "$ORT_FPS"
fi

echo ""
echo "========================================================"
echo "  Benchmark complete."
echo "========================================================"
echo ""
