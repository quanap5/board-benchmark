![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17%2B-005CED?logo=onnx&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-8.2%2B-76B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20|%2012.4-76B900?logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-x86__64%20|%20aarch64-informational)
![Jetson](https://img.shields.io/badge/Jetson-TX2%20NX-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/github/license/quanap5/board-benchmark)
![Last Commit](https://img.shields.io/github/last-commit/quanap5/board-benchmark)
![Repo Size](https://img.shields.io/github/repo-size/quanap5/board-benchmark)

# Board Benchmark

Benchmark YOLO model inference across execution providers (TensorRT, CUDA, CPU) on desktop GPUs and NVIDIA Jetson edge devices.

## How It Works

The project separates model preparation from inference benchmarking:

- **Dev mode** (local machine) -- downloads YOLO `.pt` weights, exports to `.onnx`
- **Desktop runtime** (Docker) -- benchmarks `.onnx` with `onnxruntime` in minimal containers
- **Jetson runtime** (native) -- benchmarks on-device with `trtexec` and optional `onnxruntime`, no Docker

```
 DEV (local)                 DESKTOP (Docker)            JETSON (native)
+--------------------+      +---------------------+     +---------------------+
| ultralytics        |      | onnxruntime + numpy |     | JetPack TensorRT    |
| .pt -> .onnx       |      | NO ultralytics      |     | trtexec (built-in)  |
+---------+----------+      +----------+----------+     +----------+----------+
          |                            |                            |
          v                            v                            v
    models/yolov8n.onnx -----> mount (read-only)           scp to device
```

## Quick Start

### 1. Export model (dev machine)

```bash
bash benchmarkCLI/setup-yolo-env.sh
source benchmarkCLI/.venv/bin/activate
python3 benchmarkCLI/download-yolo-model.py --model yolov8n
```

### 2a. Benchmark on desktop (Docker)

```bash
bash benchmarkCLI/run-benchmark.sh
```

Builds Docker images, runs GPU (TensorRT -> CUDA -> CPU fallback) then CPU benchmark, prints comparison.

```bash
# GPU only
bash benchmarkCLI/run-benchmark.sh --gpu-only

# CPU only
bash benchmarkCLI/run-benchmark.sh --cpu-only

# Custom model and iterations
bash benchmarkCLI/run-benchmark.sh --model yolov8s -n 500 -b 4

# Skip rebuild on repeated runs
bash benchmarkCLI/run-benchmark.sh --skip-build -n 1000
```

### 2b. Benchmark on Jetson TX2 NX (native, no Docker)

```bash
# Copy model to Jetson
scp benchmarkCLI/models/yolov8n.onnx jetson:benchmarkCLI/models/

# On Jetson: validate environment
bash benchmarkCLI/setup-jetson.sh

# Run benchmark (auto-converts to TensorRT engine on first run)
bash benchmarkCLI/run-benchmark-jetson.sh

# trtexec only (no onnxruntime needed)
bash benchmarkCLI/run-benchmark-jetson.sh --trtexec-only
```

`run-benchmark.sh` auto-detects Jetson and redirects to the native runner.

## Example Output

### GPU vs CPU comparison

```
========================================================
  BENCHMARK SUMMARY
========================================================
  Model: yolov8n | Iters: 100 | Batch: 1

  Metric             GPU (TensorRT)          CPU
  ------------------ -------------- ------------
  Avg latency            0.812 ms    24.344 ms
  Min latency            0.701 ms    20.749 ms
  Median                 0.787 ms    24.347 ms
  P95                    1.002 ms    27.857 ms
  P99                    1.105 ms     28.73 ms
  FPS                      1231.5        41.07  29.98x

  Speedup: 29.98x (latency) | 29.98x (FPS)
```

## CLI Options

### run-benchmark.sh (desktop)

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu-only` | off | GPU benchmark only |
| `--cpu-only` | off | CPU benchmark only |
| `--model NAME` | `yolov8n` | YOLO model name |
| `-n NUM` | `100` | Benchmark iterations |
| `-w NUM` | `10` | Warmup iterations |
| `-b NUM` | `1` | Batch size |
| `--skip-build` | off | Reuse cached Docker images |
| `--build-only` | off | Build images only |

### run-benchmark-jetson.sh

| Option | Default | Description |
|--------|---------|-------------|
| `--trtexec-only` | off | trtexec benchmark only |
| `--ort-only` | off | onnxruntime benchmark only |
| `--convert` | off | Force ONNX to TensorRT conversion |
| `--model NAME` | `yolov8n` | YOLO model name |
| `-n NUM` | `100` | Benchmark iterations |
| `-w NUM` | `10` | Warmup iterations |
| `-b NUM` | `1` | Batch size |

### download-yolo-model.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model` / `-m` | `yolov8n` | Model (yolov8n/s/m/l/x, yolo11n/s/m/l/x) |
| `--imgsz` / `-s` | `640` | Input image size |
| `--opset` | `17` | ONNX opset version |
| `--output-dir` / `-o` | `models` | Output directory |
| `--half` | off | FP16 export |

## Docker Images

| Image | Base | Size | Use |
|-------|------|------|-----|
| `benchmark-cpu` | `python:3.12-slim` | ~150 MB | CPU-only benchmark |
| `benchmark-gpu` | `nvcr.io/nvidia/tensorrt:24.07-py3` | ~8 GB | TensorRT + CUDA benchmark |

GPU image requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Execution Provider Fallback

| `--provider` | EP chain |
|--------------|----------|
| `auto` (default) | TensorRT -> CUDA -> CPU |
| `tensorrt` | TensorRT -> CUDA -> CPU |
| `cuda` | CUDA -> CPU |
| `cpu` | CPU only |

Unavailable providers are silently skipped. Active EP is reported in output.

## Jetson Notes

- **Supported**: Jetson TX2 NX with JetPack 4.6.x (CUDA 10.2, TensorRT 8.2.1)
- **No Docker**: 4 GB shared RAM makes container overhead significant
- **Two backends**: `trtexec` (always available) and `onnxruntime` (build from source)
- **`.engine` files are device-specific** -- not portable across TensorRT versions
- **onnxruntime**: no pre-built aarch64 wheel; see `setup-jetson.sh` for build instructions

## File Structure

```
benchmarkCLI/
+-- Dockerfile                  # CPU runtime image
+-- Dockerfile.gpu              # GPU runtime image (TensorRT base)
+-- requirements.txt            # CPU runtime: numpy, onnxruntime
+-- requirements-gpu.txt        # GPU runtime: numpy, onnxruntime-gpu
+-- requirements-dev.txt        # Dev: adds ultralytics, onnx
+-- setup-yolo-env.sh           # Dev: create venv
+-- download-yolo-model.py      # Dev: download .pt, export .onnx
+-- setup-jetson.sh             # Jetson: validate JetPack environment
+-- convert-to-tensorrt.sh      # Jetson: ONNX -> TensorRT .engine
+-- onnx-benchmark.py           # Benchmark via onnxruntime (all platforms)
+-- trtexec-benchmark.py        # Benchmark via trtexec (Jetson)
+-- hardware-info.py            # System hardware report
+-- run-benchmark.sh            # Desktop: Docker build + benchmark
+-- run-benchmark-jetson.sh     # Jetson: native benchmark
+-- models/                     # Model files (git-ignored)
```

## Output Metrics

| Metric | Description |
|--------|-------------|
| Avg / Min / Max latency | Inference time per iteration (ms) |
| Median / P95 / P99 | Latency percentiles (ms) |
| Std deviation | Latency consistency (ms) |
| FPS | Frames per second |
| Throughput | Samples per second |

## Prerequisites

| Target | Requirements |
|--------|-------------|
| Dev | Python 3.10+, internet |
| Desktop | Docker, NVIDIA Container Toolkit (GPU) |
| Jetson | JetPack 4.6.x, ONNX model from dev machine |
