#!/bin/bash
# Verify Jetson hardware can build onnxruntime v1.11.0 from source.
# Usage: bash benchmarkCLI/verify-ort-build-ready.sh
# Run this BEFORE starting the 2-4 hour build to catch issues early.

set -euo pipefail

PASS=0; WARN=0; FAIL=0
pass() { echo "  [PASS] $1"; ((PASS++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }

echo ""
echo "========================================================"
echo "  PRE-BUILD CHECK: onnxruntime v1.11.0 on Jetson"
echo "========================================================"

# --- 1. Jetson detection ---
echo ""
echo "--- Platform ---"
if [ -f /etc/nv_tegra_release ]; then
    RELEASE=$(head -1 /etc/nv_tegra_release)
    pass "Jetson detected: $RELEASE"
    if echo "$RELEASE" | grep -q "R32"; then
        pass "L4T R32.x (JetPack 4.6.x compatible)"
    else
        warn "L4T version not R32.x — onnxruntime v1.11.0 tested on JetPack 4.6.x"
    fi
else
    fail "Not a Jetson device (/etc/nv_tegra_release missing)"
fi

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    pass "Architecture: aarch64"
else
    fail "Architecture: $ARCH (expected aarch64)"
fi

# --- 2. CPU cores ---
echo ""
echo "--- CPU ---"
CORES=$(nproc 2>/dev/null || echo 0)
if [ "$CORES" -ge 2 ]; then
    pass "CPU cores: $CORES (--parallel 2 safe)"
else
    warn "CPU cores: $CORES (use --parallel 1)"
fi

# --- 3. Memory ---
echo ""
echo "--- Memory ---"
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_GB=$(awk "BEGIN {printf \"%.1f\", $MEM_KB/1024/1024}")
if [ "$MEM_KB" -ge 3500000 ]; then
    pass "RAM: ${MEM_GB} GB"
else
    warn "RAM: ${MEM_GB} GB (low — build may OOM even with swap)"
fi

SWAP_KB=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
SWAP_GB=$(awk "BEGIN {printf \"%.1f\", $SWAP_KB/1024/1024}")
if [ "$SWAP_KB" -ge 3500000 ]; then
    pass "Swap: ${SWAP_GB} GB"
else
    fail "Swap: ${SWAP_GB} GB (need >= 4 GB, run: sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile)"
fi

TOTAL_KB=$((MEM_KB + SWAP_KB))
TOTAL_GB=$(awk "BEGIN {printf \"%.1f\", $TOTAL_KB/1024/1024}")
echo "  [INFO] Total (RAM + swap): ${TOTAL_GB} GB"
if [ "$TOTAL_KB" -lt 7000000 ]; then
    warn "Total RAM + swap < 7 GB — use --parallel 1 to reduce memory usage"
fi

# --- 4. Disk space ---
echo ""
echo "--- Disk ---"
HOME_AVAIL_KB=$(df "$HOME" 2>/dev/null | tail -1 | awk '{print $4}')
HOME_AVAIL_GB=$(awk "BEGIN {printf \"%.1f\", $HOME_AVAIL_KB/1024/1024}")
if [ "$HOME_AVAIL_KB" -ge 5000000 ]; then
    pass "Free disk (HOME): ${HOME_AVAIL_GB} GB"
else
    fail "Free disk (HOME): ${HOME_AVAIL_GB} GB (need >= 5 GB for clone + build)"
fi

# --- 5. CUDA ---
echo ""
echo "--- CUDA ---"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>&1 | grep "release" | sed 's/.*release //;s/,.*//')
    pass "nvcc found: CUDA $CUDA_VER"
    if echo "$CUDA_VER" | grep -q "10.2"; then
        pass "CUDA 10.2 (matches onnxruntime v1.11.0 requirement)"
    else
        warn "CUDA $CUDA_VER — onnxruntime v1.11.0 expects CUDA 10.2"
    fi
elif [ -d /usr/local/cuda ]; then
    warn "CUDA dir exists but nvcc not in PATH — run: export PATH=/usr/local/cuda/bin:\$PATH"
else
    fail "CUDA not found (required for GPU build)"
fi

if [ -f /usr/local/cuda/lib64/libcudart.so ]; then
    pass "libcudart.so found"
else
    warn "libcudart.so not at expected path — check /usr/local/cuda/lib64/"
fi

# --- 6. cuDNN ---
echo ""
echo "--- cuDNN ---"
CUDNN_H="/usr/include/cudnn_version.h"
if [ -f "$CUDNN_H" ]; then
    MAJOR=$(grep "define CUDNN_MAJOR" "$CUDNN_H" | awk '{print $3}')
    MINOR=$(grep "define CUDNN_MINOR" "$CUDNN_H" | awk '{print $3}')
    PATCH=$(grep "define CUDNN_PATCHLEVEL" "$CUDNN_H" | awk '{print $3}')
    pass "cuDNN ${MAJOR}.${MINOR}.${PATCH}"
else
    fail "cuDNN header not found at $CUDNN_H"
fi

if [ -f /usr/lib/aarch64-linux-gnu/libcudnn.so ]; then
    pass "libcudnn.so found"
else
    warn "libcudnn.so not at /usr/lib/aarch64-linux-gnu/ — build may need adjusted --cudnn_home"
fi

# --- 7. TensorRT ---
echo ""
echo "--- TensorRT ---"
if [ -f /usr/lib/aarch64-linux-gnu/libnvinfer.so ]; then
    pass "libnvinfer.so found (TensorRT library)"
    TRT_VER=$(dpkg -l 2>/dev/null | grep "libnvinfer[0-9]" | head -1 | awk '{print $3}' || echo "unknown")
    echo "  [INFO] TensorRT package version: $TRT_VER"
else
    warn "libnvinfer.so not found — TensorrtExecutionProvider won't be available"
fi

if dpkg -l 2>/dev/null | grep -q "libnvinfer-dev"; then
    pass "libnvinfer-dev installed (headers for build)"
else
    fail "libnvinfer-dev not installed — run: sudo apt-get install libnvinfer-dev"
fi

# --- 8. Python ---
echo ""
echo "--- Python ---"
PYTHON_CMD=""
for cmd in python3 python; do command -v "$cmd" &>/dev/null && PYTHON_CMD="$cmd" && break; done
if [ -n "$PYTHON_CMD" ]; then
    PY_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    pass "Python: $PY_VER ($PYTHON_CMD)"
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 6 ] && [ "$PY_MINOR" -le 9 ]; then
        pass "Python version compatible with onnxruntime v1.11.0"
    else
        warn "Python $PY_VER — onnxruntime v1.11.0 best tested with Python 3.6-3.9"
    fi
else
    fail "Python not found"
fi

# python3-dev headers
if dpkg -l 2>/dev/null | grep -q "python3-dev"; then
    pass "python3-dev installed"
else
    fail "python3-dev not installed — run: sudo apt-get install python3-dev"
fi

# --- 9. Build tools ---
echo ""
echo "--- Build Tools ---"
if command -v gcc &>/dev/null; then
    GCC_VER=$(gcc --version | head -1)
    pass "gcc: $GCC_VER"
else
    fail "gcc not found — run: sudo apt-get install build-essential"
fi

if command -v g++ &>/dev/null; then
    pass "g++ found"
else
    fail "g++ not found — run: sudo apt-get install build-essential"
fi

if command -v cmake &>/dev/null; then
    CMAKE_VER=$(cmake --version | head -1 | awk '{print $3}')
    CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
    if [ "$CMAKE_MAJOR" -ge 3 ] && [ "$CMAKE_MINOR" -ge 13 ]; then
        pass "cmake $CMAKE_VER (>= 3.13)"
    else
        fail "cmake $CMAKE_VER (need >= 3.13, run: pip3 install --user cmake --upgrade)"
    fi
else
    fail "cmake not found — run: sudo apt-get install cmake"
fi

if command -v protoc &>/dev/null; then
    pass "protoc found: $(protoc --version)"
else
    fail "protoc not found — run: sudo apt-get install protobuf-compiler"
fi

if command -v git &>/dev/null; then
    pass "git found"
else
    fail "git not found — run: sudo apt-get install git"
fi

# --- 10. numpy check ---
echo ""
echo "--- Python Packages ---"
if [ -n "$PYTHON_CMD" ]; then
    if $PYTHON_CMD -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null; then
        # Check if apt-installed (avoids "Illegal instruction" on Tegra)
        if dpkg -l 2>/dev/null | grep -q "python3-numpy"; then
            pass "numpy installed via apt (safe on Tegra)"
        else
            warn "numpy installed via pip — may cause 'Illegal instruction' on Tegra. Fix: pip3 uninstall numpy && sudo apt-get install python3-numpy"
        fi
    else
        warn "numpy not installed — run: sudo apt-get install python3-numpy"
    fi

    if $PYTHON_CMD -c "import pip" 2>/dev/null; then
        pass "pip available"
    else
        fail "pip not available — run: sudo apt-get install python3-pip"
    fi

    if $PYTHON_CMD -c "import wheel" 2>/dev/null; then
        pass "wheel available"
    else
        warn "wheel not installed — run: pip3 install --user wheel"
    fi

    if $PYTHON_CMD -c "import setuptools" 2>/dev/null; then
        pass "setuptools available"
    else
        warn "setuptools not installed — run: pip3 install --user setuptools"
    fi
fi

# --- 11. tmux/screen ---
echo ""
echo "--- Session Manager ---"
if command -v tmux &>/dev/null; then
    pass "tmux available (recommended for long build)"
elif command -v screen &>/dev/null; then
    pass "screen available"
else
    warn "Neither tmux nor screen found — SSH disconnect will kill build. Install: sudo apt-get install tmux"
fi

# --- 12. Check if already installed ---
echo ""
echo "--- Existing onnxruntime ---"
if [ -n "$PYTHON_CMD" ]; then
    if $PYTHON_CMD -c "import onnxruntime as ort; print('Version:', ort.__version__); print('EPs:', ort.get_available_providers())" 2>/dev/null; then
        warn "onnxruntime already installed — rebuild will overwrite"
    else
        echo "  [INFO] onnxruntime not installed (expected — that's why we're building)"
    fi
fi

# --- Summary ---
echo ""
echo "========================================================"
echo "  RESULTS"
echo "========================================================"
echo "  PASS: $PASS   WARN: $WARN   FAIL: $FAIL"
echo ""

if [ "$FAIL" -eq 0 ]; then
    if [ "$WARN" -eq 0 ]; then
        echo "  ✅ ALL CLEAR — ready to build onnxruntime!"
    else
        echo "  ⚠️  READY with warnings — review warnings above, build should work."
    fi
    echo ""
    echo "  Next steps:"
    echo "    tmux new -s ort-build"
    echo "    cd ~ && git clone --recursive --branch v1.11.0 https://github.com/microsoft/onnxruntime"
    echo "    cd onnxruntime"
    echo "    ./build.sh --config Release --build_wheel --parallel 2 \\"
    echo "      --use_cuda --cuda_home /usr/local/cuda \\"
    echo "      --cudnn_home /usr/lib/aarch64-linux-gnu \\"
    echo "      --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu \\"
    echo "      --skip_tests"
else
    echo "  ❌ NOT READY — fix $FAIL issue(s) above before building."
fi
echo ""
