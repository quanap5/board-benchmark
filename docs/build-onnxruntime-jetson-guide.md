# Build onnxruntime-gpu on Jetson TX2 NX

Step-by-step guide to build onnxruntime v1.11.0 from source on Jetson TX2 NX with JetPack 4.6.x.

**Time required:** 2-4 hours (mostly unattended compile)
**Disk required:** ~5 GB free
**Risk to system:** None (everything in home directory)

---

## Prerequisites

| Component | Required version | Check command |
|-----------|-----------------|---------------|
| JetPack | 4.6.x (L4T R32) | `cat /etc/nv_tegra_release` |
| CUDA | 10.2 | `nvcc --version` |
| TensorRT | 8.2.x | `dpkg -l \| grep tensorrt` |
| cuDNN | 8.2.x | `dpkg -l \| grep cudnn` |
| Python | 3.6 | `python3 --version` |
| Free disk | ~5 GB | `df -h ~` |

---

## Step 0: Verify readiness

Run the pre-build checker to catch issues before the long build:

```bash
bash benchmarkCLI/verify-ort-build-ready.sh
```

Fix any `[FAIL]` items before proceeding. If all pass, continue.

---

## Step 1: Add swap (2 min)

The build **will be killed by OOM** without extra swap on a 4GB device.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify — should show ~4 GB swap
free -h
```

---

## Step 2: Install build dependencies (5 min)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake libprotobuf-dev protobuf-compiler \
    python3-dev python3-pip python3-setuptools python3-wheel
```

Check cmake version (need >= 3.13):

```bash
cmake --version
```

If too old:

```bash
pip3 install --user cmake --upgrade
export PATH=$HOME/.local/bin:$PATH
cmake --version
```

---

## Step 3: Fix numpy (1 min)

Pip-installed numpy causes "Illegal instruction" on Tegra. Use the apt version:

```bash
pip3 uninstall numpy -y 2>/dev/null
sudo apt-get install -y python3-numpy
python3 -c "import numpy; print('numpy', numpy.__version__)"
```

---

## Step 4: Set CUDA environment (1 min)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

Make it permanent:

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## Step 5: Start tmux + clone (10 min)

Use tmux so SSH disconnect won't kill the build:

```bash
sudo apt-get install -y tmux
tmux new -s ort-build
```

Clone v1.11.0 (last version supporting CUDA 10.2 + Python 3.6):

```bash
cd ~
git clone --recursive --branch v1.11.0 https://github.com/microsoft/onnxruntime
cd onnxruntime
```

This downloads ~2 GB. Takes ~10 minutes on a decent connection.

---

## Step 6: Build (2-4 hours)

This is the long step. It runs unattended.

```bash
./build.sh \
    --config Release \
    --build_wheel \
    --parallel 2 \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --use_tensorrt \
    --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --skip_tests
```

### Build flags explained

| Flag | Purpose |
|------|---------|
| `--config Release` | Optimized build (not debug) |
| `--build_wheel` | Create pip-installable `.whl` file |
| `--parallel 2` | Use 2 threads (safe for 4GB + 4GB swap) |
| `--use_cuda` | Enable CUDAExecutionProvider |
| `--cuda_home` | Path to CUDA toolkit |
| `--cudnn_home` | Path to cuDNN libraries |
| `--use_tensorrt` | Enable TensorrtExecutionProvider |
| `--tensorrt_home` | Path to TensorRT libraries |
| `--skip_tests` | Don't run tests after build (saves ~30 min) |

### Detach tmux and come back later

Detach: `Ctrl+B` then `D`

Re-attach later to check progress:

```bash
tmux attach -t ort-build
```

If the build gets killed (OOM), reduce to `--parallel 1` and retry.

---

## Step 7: Install the wheel (1 min)

```bash
pip3 install --user build/Linux/Release/dist/onnxruntime_gpu-*.whl
```

---

## Step 8: Verify (1 min)

```bash
python3 -c "
import onnxruntime as ort
print('Version:', ort.__version__)
print('EPs:', ort.get_available_providers())
print('TensorRT EP:', 'TensorrtExecutionProvider' in ort.get_available_providers())
print('CUDA EP:', 'CUDAExecutionProvider' in ort.get_available_providers())
"
```

Expected output:

```
Version: 1.11.0
EPs: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
TensorRT EP: True
CUDA EP: True
```

If only `CPUExecutionProvider` shows, the CUDA/TRT paths were wrong during build. See [Troubleshooting](#build-succeeds-but-only-cpuexecutionprovider) below.

---

## Step 9: Run benchmark (5 min)

```bash
bash benchmarkCLI/run-benchmark-jetson.sh --model yolov8n --ort-only -n 100
```

Or all methods together:

```bash
bash benchmarkCLI/run-benchmark-jetson.sh --model yolov8n --all -n 100
```

---

## Step 10: Cleanup (optional)

```bash
# Remove source + build files (~5 GB freed)
rm -rf ~/onnxruntime

# Remove swap
sudo swapoff /swapfile
sudo rm /swapfile

# The installed wheel stays in ~/.local/ (~50 MB)
```

---

## Quick Reference

| Step | Time | Command to verify success |
|------|------|--------------------------|
| 0. Verify | 1 min | `bash benchmarkCLI/verify-ort-build-ready.sh` — no FAIL |
| 1. Swap | 2 min | `free -h` shows ~4 GB swap |
| 2. Deps | 5 min | `cmake --version` >= 3.13 |
| 3. Numpy | 1 min | `python3 -c "import numpy"` no error |
| 4. CUDA | 1 min | `nvcc --version` shows 10.2 |
| 5. Clone | 10 min | `ls ~/onnxruntime/build.sh` exists |
| 6. **Build** | **2-4 hr** | No error at end of output |
| 7. Install | 1 min | pip succeeds |
| 8. Verify | 1 min | 3 EPs listed |
| 9. Benchmark | 5 min | Results printed |

---

## Troubleshooting

### Build fails with OOM (Killed)

```bash
# Check if swap is active
free -h

# If no swap, add it (Step 1)
# If already have swap, reduce parallel:
./build.sh ... --parallel 1
```

### cmake version too old

```
CMake Error at CMakeLists.txt: cmake_minimum_required(VERSION 3.13)
```

Fix:

```bash
pip3 install --user cmake --upgrade
export PATH=$HOME/.local/bin:$PATH
cmake --version
# Then re-run build.sh
```

### protobuf version mismatch

```
error: protobuf version mismatch
```

Fix:

```bash
sudo apt-get install --reinstall libprotobuf-dev protobuf-compiler
pip3 install --user protobuf==3.19.6
```

### Build succeeds but import fails

```
ImportError: libcudart.so.10.2: cannot open shared object file
```

Fix:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python3 -c "import onnxruntime"
```

### Build succeeds but only CPUExecutionProvider

```
EPs: ['CPUExecutionProvider']
```

TensorRT or CUDA EP not linked. Rebuild with correct paths:

```bash
# Verify paths exist
ls /usr/local/cuda/lib64/libcudart.so
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so

# If libnvinfer.so is elsewhere:
find / -name "libnvinfer.so*" 2>/dev/null
# Use that path as --tensorrt_home
```

### Want to uninstall completely

```bash
pip3 uninstall onnxruntime-gpu -y
rm -rf ~/onnxruntime
sudo swapoff /swapfile && sudo rm /swapfile
```

System is back to original state. Nothing in `/usr` or JetPack was modified.

---

## What you get after building

| Execution Provider | Available | Performance |
|-------------------|-----------|-------------|
| TensorrtExecutionProvider | Yes | ~23 ms (same TRT engine) |
| CUDAExecutionProvider | Yes | ~25 ms (cuDNN, no TRT optimization) |
| CPUExecutionProvider | Yes | ~200 ms (ARM CPU, very slow) |

The benchmark script (`onnx-benchmark.py`) auto-selects the best EP via fallback chain: TensorRT -> CUDA -> CPU.

---

## Related

- [Jetson Workflow Guide](jetson-workflow-guide.md) -- all three benchmark methods explained
- [Project Summary Report](project-summary-report.md) -- method comparison
