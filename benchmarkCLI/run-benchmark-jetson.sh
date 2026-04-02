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

# --- Colors and emoji helpers ---
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

_hdr "🚀" "JETSON BENCHMARK RUNNER (native, no Docker)"
_info "Model: $MODEL_NAME | Iters: $ITERATIONS | Warmup: $WARMUP | Batch: $BATCH_SIZE"
_info "Mode:  $MODE | TRT Python: $HAS_TRT_PYTHON | trtexec: $HAS_TRTEXEC | ORT: $HAS_ORT"
echo ""

[ ! -f /etc/nv_tegra_release ] && _warn "Not a Jetson device." && echo ""

if [ ! -f "$ONNX_FILE" ]; then
    _err "Model not found: $ONNX_FILE"
    _info "scp dev-host:benchmarkCLI/models/$MODEL_NAME.onnx $MODELS_DIR/"
    exit 1
fi

# --- Hardware info ---
_hdr "🖥️ " "SYSTEM HARDWARE INFO"
python3 "$SCRIPT_DIR/hardware-info.py"
echo ""

# --- trtexec benchmark ---
TRTEXEC_CSV=""
if [ "$MODE" = "trtexec" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRTEXEC" = true ]; then
        _hdr "📊" "TRTEXEC BENCHMARK (FP32/FP16/INT8)"
        PREC_ARGS="-p fp16"
        [ -n "$TRT_PYTHON_PRECISIONS" ] && PREC_ARGS="-p $TRT_PYTHON_PRECISIONS"
        TRT_TMPFILE=$(mktemp /tmp/trt-bench.XXXXXX)
        python3 "$SCRIPT_DIR/trtexec-benchmark.py" \
            "$ONNX_FILE" $PREC_ARGS -n "$ITERATIONS" -w "$WARMUP" --save-engine --csv 2>&1 | tee "$TRT_TMPFILE"
        TRTEXEC_CSV=$(grep "^CSV:" "$TRT_TMPFILE" | tail -1)
        rm -f "$TRT_TMPFILE"
        echo ""
    else
        _warn "trtexec not found. Skipping."
    fi
fi

# --- TRT Python benchmark (FP32/FP16/INT8) ---
if [ "$MODE" = "trt-python" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_TRT_PYTHON" = true ]; then
        _hdr "🐍" "TRT PYTHON BENCHMARK (FP32/FP16/INT8)"
        PREC_ARGS=""
        [ -n "$TRT_PYTHON_PRECISIONS" ] && PREC_ARGS="-p $TRT_PYTHON_PRECISIONS"
        python3 "$SCRIPT_DIR/trt-python-benchmark.py" \
            "$ONNX_FILE" $PREC_ARGS -n "$ITERATIONS" -w "$WARMUP" --save-engine
        echo ""
    else
        _warn "tensorrt/pycuda not available. Skipping TRT Python benchmark."
        _info "Install: pip3 install --user pycuda"
    fi
fi

# --- onnxruntime benchmark ---
ORT_CSV=""
if [ "$MODE" = "ort" ] || [ "$MODE" = "all" ]; then
    if [ "$HAS_ORT" = true ]; then
        _hdr "📊" "ONNXRUNTIME BENCHMARK (TensorRT EP -> CUDA EP -> CPU)"
        ORT_TMPFILE=$(mktemp /tmp/ort-bench.XXXXXX)
        python3 "$SCRIPT_DIR/onnx-benchmark.py" \
            "$ONNX_FILE" -p tensorrt -n "$ITERATIONS" -w "$WARMUP" -b "$BATCH_SIZE" --csv 2>&1 | tee "$ORT_TMPFILE"
        ORT_CSV=$(grep "^CSV:" "$ORT_TMPFILE" | tail -1)
        rm -f "$ORT_TMPFILE"
        echo ""
    else
        _warn "onnxruntime not installed. Skipping."
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
    _hdr "🏁" "JETSON BENCHMARK SUMMARY"
    _info "Device: $(head -1 /etc/nv_tegra_release 2>/dev/null | cut -c1-60 || echo 'Unknown')"
    _info "Model:  $MODEL_NAME | Iters: $ITERATIONS | Batch: $BATCH_SIZE"
    echo ""
    if [ -n "$TRTEXEC_CSV" ] && [ -n "$ORT_CSV" ]; then
        printf "  %-20s %15s %18s\n" "Metric" "trtexec" "ORT ($ORT_PROVIDER)"
        echo -e "  ${C_DIM}$(printf '%.0s-' {1..55})${C_RST}"
        for m in "Avg latency:TRT_AVG:ORT_AVG" "Min latency:TRT_MIN:ORT_MIN" "Max latency:TRT_MAX:ORT_MAX" \
                 "Median:TRT_MEDIAN:ORT_MEDIAN" "P95:TRT_P95:ORT_P95" "P99:TRT_P99:ORT_P99"; do
            IFS=':' read -r label v1 v2 <<< "$m"
            [ -z "${!v1}" ] && [ -z "${!v2}" ] && continue
            printf "  %-20s ${C_BOLD}%11s ms${C_RST} ${C_BOLD}%14s ms${C_RST}\n" "$label" "${!v1:-N/A}" "${!v2:-N/A}"
        done
        printf "  %-20s ${C_GRN}${C_BOLD}%13s${C_RST} ${C_GRN}${C_BOLD}%16s${C_RST}\n" "FPS" "$TRT_FPS" "$ORT_FPS"
    elif [ -n "$TRTEXEC_CSV" ]; then
        printf "  %-20s %15s\n" "Metric" "trtexec"
        echo -e "  ${C_DIM}$(printf '%.0s-' {1..36})${C_RST}"
        for m in "Avg:TRT_AVG" "Min:TRT_MIN" "Median:TRT_MEDIAN" "P95:TRT_P95" "P99:TRT_P99"; do
            IFS=':' read -r l v <<< "$m"; [ -z "${!v}" ] && continue
            printf "  %-20s ${C_BOLD}%11s ms${C_RST}\n" "$l" "${!v}"
        done
        printf "  %-20s ${C_GRN}${C_BOLD}%13s${C_RST}\n" "FPS" "$TRT_FPS"
    elif [ -n "$ORT_CSV" ]; then
        printf "  %-20s %18s\n" "Metric" "ORT ($ORT_PROVIDER)"
        echo -e "  ${C_DIM}$(printf '%.0s-' {1..40})${C_RST}"
        for m in "Avg:ORT_AVG" "Min:ORT_MIN" "Median:ORT_MEDIAN" "P95:ORT_P95" "P99:ORT_P99"; do
            IFS=':' read -r l v <<< "$m"; [ -z "${!v}" ] && continue
            printf "  %-20s ${C_BOLD}%14s ms${C_RST}\n" "$l" "${!v}"
        done
        printf "  %-20s ${C_GRN}${C_BOLD}%16s${C_RST}\n" "FPS" "$ORT_FPS"
    fi
fi

echo ""
_ok "Benchmark complete."
