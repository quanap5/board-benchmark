#!/bin/bash
# Build Docker images and benchmark ONNX model with EP fallback.
# Usage:
#   bash benchmarkCLI/run-benchmark.sh                              # Docker: GPU + CPU
#   bash benchmarkCLI/run-benchmark.sh --native --model yolov8n     # Native PC (no Docker)
#   bash benchmarkCLI/run-benchmark.sh --cpu-only                   # Docker: CPU only
#   bash benchmarkCLI/run-benchmark.sh --gpu-only                   # Docker: GPU only
#   bash benchmarkCLI/run-benchmark.sh --model yolov8s -n 500       # Docker: custom

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# Auto-detect Jetson -> native benchmark (no Docker on 4GB device)
if [ -f /etc/nv_tegra_release ]; then
    echo "[INFO] Jetson detected. Switching to native benchmark."
    exec bash "$SCRIPT_DIR/run-benchmark-jetson.sh" "$@"
fi

# --native flag -> run PC benchmark without Docker
for arg in "$@"; do
    if [ "$arg" = "--native" ]; then
        # Remove --native from args and forward the rest
        ARGS=()
        for a in "$@"; do [ "$a" != "--native" ] && ARGS+=("$a"); done
        exec bash "$SCRIPT_DIR/run-benchmark-pc.sh" "${ARGS[@]}"
    fi
done

RUN_GPU=true; RUN_CPU=true; BUILD_ONLY=false; SKIP_BUILD=false
MODEL_NAME="yolov8n"; ITERATIONS=100; WARMUP=10; BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)   RUN_GPU=false; shift ;;
        --gpu-only)   RUN_CPU=false; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --model)      MODEL_NAME="$2"; shift 2 ;;
        -n)           ITERATIONS="$2"; shift 2 ;;
        -w)           WARMUP="$2"; shift 2 ;;
        -b)           BATCH_SIZE="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

ONNX_FILE="$MODEL_NAME.onnx"

echo ""
echo "========================================================"
echo "  BENCHMARK RUNNER"
echo "========================================================"
echo "  Model: $MODEL_NAME | Iters: $ITERATIONS | Warmup: $WARMUP | Batch: $BATCH_SIZE"
echo "  GPU: $RUN_GPU | CPU: $RUN_CPU"
echo ""

# --- Verify ONNX model ---
if [ ! -f "$MODELS_DIR/$ONNX_FILE" ]; then
    echo "[ERROR] Model not found: $MODELS_DIR/$ONNX_FILE"
    echo "  Run dev mode first: bash $SCRIPT_DIR/setup-yolo-env.sh && python3 $SCRIPT_DIR/download-yolo-model.py"
    exit 1
fi

# --- Build Docker images ---
if [ "$SKIP_BUILD" = false ]; then
    [ "$RUN_CPU" = true ] && echo "--- Building benchmark-cpu ---" && \
        docker build -t benchmark-cpu -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR" && echo ""
    [ "$RUN_GPU" = true ] && echo "--- Building benchmark-gpu ---" && \
        docker build -t benchmark-gpu -f "$SCRIPT_DIR/Dockerfile.gpu" "$SCRIPT_DIR" && echo ""
fi
[ "$BUILD_ONLY" = true ] && echo "Build complete." && exit 0

# --- Hardware info ---
echo "========================================================"
echo "  SYSTEM HARDWARE INFO"
echo "========================================================"
if [ "$RUN_GPU" = true ]; then
    docker run --rm --pid=host --gpus all -v /proc:/host/proc:ro benchmark-gpu hardware-info.py
else
    docker run --rm --pid=host -v /proc:/host/proc:ro benchmark-cpu hardware-info.py
fi
echo ""

# --- Benchmark helper ---
run_bench() {
    local provider=$1 image=$2 gpu_flag=$3 output_var=$4
    local result
    result=$(docker run --rm -v "$MODELS_DIR:/app/models:ro" $gpu_flag \
        "$image" onnx-benchmark.py /app/models/"$ONNX_FILE" \
        -p "$provider" -n "$ITERATIONS" -w "$WARMUP" -b "$BATCH_SIZE" --csv 2>&1)
    echo "$result"
    eval "$output_var=\"$(echo "$result" | grep '^CSV:' | tail -1)\""
}

GPU_CSV=""; CPU_CSV=""

if [ "$RUN_GPU" = true ]; then
    echo "========================================================" && echo "  GPU BENCHMARK (auto: TensorRT -> CUDA -> CPU)"
    echo "========================================================"
    run_bench "auto" "benchmark-gpu" "--gpus all" GPU_CSV && echo ""
fi
if [ "$RUN_CPU" = true ]; then
    echo "========================================================" && echo "  CPU BENCHMARK"
    echo "========================================================"
    run_bench "cpu" "benchmark-cpu" "" CPU_CSV && echo ""
fi

# --- Parse CSV and print summary ---
parse_csv() {
    local csv=$1 prefix=$2
    [ -z "$csv" ] && return
    local data="${csv#CSV:}"
    IFS=',' read -r p _ _ t avg min max med p95 p99 std fps _ <<< "$data"
    eval "${prefix}_PROVIDER=$p; ${prefix}_AVG=$avg; ${prefix}_MIN=$min; ${prefix}_MAX=$max"
    eval "${prefix}_MEDIAN=$med; ${prefix}_P95=$p95; ${prefix}_P99=$p99; ${prefix}_FPS=$fps; ${prefix}_TOTAL=$t"
}

parse_csv "$GPU_CSV" "GPU"; parse_csv "$CPU_CSV" "CPU"

echo ""
echo "========================================================"
echo "  BENCHMARK SUMMARY"
echo "========================================================"
echo "  Model: $MODEL_NAME | Iters: $ITERATIONS | Batch: $BATCH_SIZE"
echo ""

if [ -n "$GPU_CSV" ] && [ -n "$CPU_CSV" ]; then
    printf "  %-18s %14s %12s\n" "Metric" "GPU ($GPU_PROVIDER)" "CPU"
    printf "  %-18s %14s %12s\n" "------------------" "--------------" "------------"
    for m in "Avg latency:GPU_AVG:CPU_AVG" "Min latency:GPU_MIN:CPU_MIN" "Max latency:GPU_MAX:CPU_MAX" \
             "Median:GPU_MEDIAN:CPU_MEDIAN" "P95:GPU_P95:CPU_P95" "P99:GPU_P99:CPU_P99"; do
        IFS=':' read -r label v1 v2 <<< "$m"
        printf "  %-18s %12s ms %10s ms\n" "$label" "${!v1}" "${!v2}"
    done
    SPEEDUP=$(python3 -c "g=$GPU_AVG;c=$CPU_AVG;print(f'{c/g:.2f}x' if g>0 else 'N/A')" 2>/dev/null || echo "N/A")
    FPS_R=$(python3 -c "g=$GPU_FPS;c=$CPU_FPS;print(f'{g/c:.2f}x' if c>0 else 'N/A')" 2>/dev/null || echo "N/A")
    printf "  %-18s %14s %12s  %s\n" "FPS" "$GPU_FPS" "$CPU_FPS" "$FPS_R"
    echo ""; echo "  Speedup: ${SPEEDUP} (latency) | ${FPS_R} (FPS)"
elif [ -n "$GPU_CSV" ]; then
    for m in "Avg latency:GPU_AVG" "Min:GPU_MIN" "Median:GPU_MEDIAN" "P95:GPU_P95" "P99:GPU_P99"; do
        IFS=':' read -r l v <<< "$m"; printf "  %-18s %12s ms\n" "$l" "${!v}"; done
    printf "  %-18s %14s\n" "FPS" "$GPU_FPS"
elif [ -n "$CPU_CSV" ]; then
    for m in "Avg latency:CPU_AVG" "Min:CPU_MIN" "Median:CPU_MEDIAN" "P95:CPU_P95" "P99:CPU_P99"; do
        IFS=':' read -r l v <<< "$m"; printf "  %-18s %12s ms\n" "$l" "${!v}"; done
    printf "  %-18s %14s\n" "FPS" "$CPU_FPS"
fi

echo ""
echo "========================================================"
echo "  Benchmark complete."
echo "========================================================"
echo ""
