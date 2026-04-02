#!/usr/bin/env python3
"""Benchmark TensorRT inference with FP32/FP16/INT8 using Python API.

Uses CUDA events for accurate GPU timing. No trtexec CLI needed.
Requires: tensorrt, pycuda, numpy (all pre-installed in JetPack 4.6.x).
"""

import argparse
import os
import sys

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as e:
    print("[ERROR] Missing: {}".format(e))
    print("  tensorrt and pycuda ship with JetPack. Check your install.")
    sys.exit(1)

# Import engine builder from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
builder_mod = import_module("trt-engine-builder")
build_engine = builder_mod.build_engine
save_engine = builder_mod.save_engine


def run_benchmark(engine, warmup, iterations):
    """Benchmark with CUDA events for accurate GPU timing."""
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate device buffers
    bindings = []
    device_buffers = []
    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        buf = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        device_buffers.append(buf)
        bindings.append(int(buf))

    # Copy random input to GPU
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            data = np.random.randn(*shape).astype(dtype)
            cuda.memcpy_htod_async(device_buffers[i], data, stream)
    stream.synchronize()

    # Warmup
    print("[INFO] Warmup ({} runs)...".format(warmup))
    for _ in range(warmup):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # Benchmark with CUDA events
    print("[INFO] Benchmarking ({} runs)...".format(iterations))
    latencies = []
    for _ in range(iterations):
        start = cuda.Event()
        end = cuda.Event()
        start.record(stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        end.record(stream)
        end.synchronize()
        latencies.append(start.time_till(end))

    for buf in device_buffers:
        buf.free()
    return latencies


def compute_stats(latencies):
    """Compute benchmark statistics from latency samples (ms)."""
    arr = np.array(latencies)
    return {
        "avg": round(float(np.mean(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "std": round(float(np.std(arr)), 3),
        "fps": round(len(arr) / (arr.sum() / 1000.0), 2),
    }


def print_results(stats, precision, model_path, iterations):
    """Print formatted benchmark results."""
    print("\n" + "=" * 50)
    print("  BENCHMARK RESULTS [{}]".format(precision))
    print("=" * 50)
    print("  Model:             {}".format(model_path))
    print("  Iterations:        {}".format(iterations))
    for label, key, unit in [
        ("Avg latency", "avg", "ms"), ("Min latency", "min", "ms"),
        ("Max latency", "max", "ms"), ("Median latency", "median", "ms"),
        ("P95 latency", "p95", "ms"), ("P99 latency", "p99", "ms"),
        ("Std deviation", "std", "ms"), ("FPS", "fps", ""),
    ]:
        sfx = " {}".format(unit) if unit else ""
        print("  {:<20} {}{}".format(label, stats[key], sfx))
    print("=" * 50)


def print_comparison(all_results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("  PRECISION COMPARISON")
    print("=" * 60)
    header = "  {:<16}".format("Metric")
    for prec, _ in all_results:
        header += " {:>12}".format(prec)
    print(header)
    print("  " + "-" * (16 + 13 * len(all_results)))

    for label, key, unit in [
        ("Avg latency", "avg", "ms"), ("Min latency", "min", "ms"),
        ("Median", "median", "ms"), ("P95", "p95", "ms"),
        ("P99", "p99", "ms"), ("Std dev", "std", "ms"), ("FPS", "fps", ""),
    ]:
        row = "  {:<16}".format(label)
        for _, stats in all_results:
            val = "{}{}".format(stats[key], " " + unit if unit else "")
            row += " {:>12}".format(val)
        print(row)

    # Speedup vs first precision
    base_prec, base_stats = all_results[0]
    if len(all_results) > 1:
        print("")
        for prec, stats in all_results[1:]:
            sp = round(base_stats["avg"] / stats["avg"], 2) if stats["avg"] > 0 else 0
            print("  {} vs {} speedup: {}x".format(prec, base_prec, sp))
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(
        description="TRT Python API benchmark (FP32/FP16/INT8)")
    p.add_argument("model", help="Path to ONNX model")
    p.add_argument("-p", "--precision", nargs="+", default=["fp32", "fp16", "int8"],
                   choices=["fp32", "fp16", "int8"],
                   help="Precision(s) to benchmark (default: all three)")
    p.add_argument("-n", "--iterations", type=int, default=100)
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("--workspace", type=int, default=1024, help="Workspace MB")
    p.add_argument("--save-engine", action="store_true", help="Save .engine files")
    args = p.parse_args()

    print("\n  TRT PYTHON BENCHMARK")
    print("  Model:      {}".format(args.model))
    print("  Precisions: {}".format(", ".join(pr.upper() for pr in args.precision)))
    print("  Iterations: {}  Warmup: {}".format(args.iterations, args.warmup))

    all_results = []
    for prec in args.precision:
        print("\n--- Building {} engine ---".format(prec.upper()))
        engine = build_engine(args.model, prec, args.workspace)
        if not engine:
            print("[WARN] Skipping {} (build failed)".format(prec.upper()))
            continue
        if args.save_engine:
            base = os.path.splitext(args.model)[0]
            save_engine(engine, "{}-{}.engine".format(base, prec))
        latencies = run_benchmark(engine, args.warmup, args.iterations)
        stats = compute_stats(latencies)
        print_results(stats, prec.upper(), args.model, args.iterations)
        all_results.append((prec.upper(), stats))
        del engine

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
