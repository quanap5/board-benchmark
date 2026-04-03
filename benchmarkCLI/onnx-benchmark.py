#!/usr/bin/env python3
"""Benchmark ONNX model inference: FPS, latency, throughput."""

import argparse
import time
import sys

import numpy as np
import onnxruntime as ort

PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
    "auto": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
}


def create_session(model_path, provider):
    """Create ONNX session with EP fallback chain. Handles EP load failures."""
    requested = PROVIDER_MAP.get(provider, [provider])
    available = ort.get_available_providers()
    providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]

    print(f"[INFO] Requested EPs: {requested}")
    print(f"[INFO] Available EPs: {available}")
    print(f"[INFO] Using EPs:     {providers}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Try full EP list first, fallback one-by-one if EP crashes
    for i in range(len(providers)):
        try_eps = providers[i:]
        try:
            session = ort.InferenceSession(model_path, opts, providers=try_eps)
            print(f"[INFO] Active EPs:    {session.get_providers()}")
            return session
        except Exception as e:
            failed = try_eps[0]
            print(f"[WARN] {failed} failed: {e}")
            print(f"[INFO] Falling back, skipping {failed}...")

    # Last resort: CPU only
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    print(f"[INFO] Active EPs:    {session.get_providers()}")
    return session


def build_dummy_inputs(session, batch_size):
    """Build random input tensors matching model spec."""
    type_map = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(int64)": np.int64, "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_,
    }
    inputs = {}
    for inp in session.get_inputs():
        shape = [batch_size if (isinstance(d, str) or d is None) else d for d in inp.shape]
        if shape and shape[0] != batch_size:
            shape[0] = batch_size
        inputs[inp.name] = np.random.randn(*shape).astype(type_map.get(inp.type, np.float32))
    return inputs


def run_benchmark(session, inputs, warmup, iterations):
    """Run inference benchmark and collect latency samples."""
    names = [o.name for o in session.get_outputs()]
    print(f"\n  Warming up ({warmup} runs)...")
    for _ in range(warmup):
        session.run(names, inputs)

    print(f"  Benchmarking ({iterations} runs)...")
    latencies = []
    t_start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(names, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies, time.perf_counter() - t_start


def compute_stats(latencies, total_time, batch_size):
    """Compute benchmark statistics."""
    arr = np.array(latencies)
    n = len(arr)
    return {
        "iterations": n, "batch_size": batch_size,
        "total_time_s": round(total_time, 3),
        "avg_ms": round(float(np.mean(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "fps": round(n * batch_size / total_time, 2),
        "throughput_ips": round(n * batch_size / total_time, 2),
    }


def print_results(session, stats, active_ep):
    """Print model info and benchmark results."""
    print("\n  MODEL INFO\n" + "=" * 50)
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name:<30} {inp.shape} ({inp.type})")
    for out in session.get_outputs():
        print(f"  Output: {out.name:<30} {out.shape} ({out.type})")

    print("\n" + "=" * 50 + "\n  BENCHMARK RESULTS\n" + "=" * 50)
    print(f"  Active EP:           {active_ep}")
    for label, key, unit in [
        ("Iterations", "iterations", ""), ("Batch size", "batch_size", ""),
        ("Total time", "total_time_s", "s"), ("", "", ""),
        ("Avg latency", "avg_ms", "ms"), ("Min latency", "min_ms", "ms"),
        ("Max latency", "max_ms", "ms"), ("Median latency", "median_ms", "ms"),
        ("P95 latency", "p95_ms", "ms"), ("P99 latency", "p99_ms", "ms"),
        ("Std deviation", "std_ms", "ms"), ("", "", ""),
        ("FPS", "fps", ""), ("Throughput", "throughput_ips", "samples/s"),
    ]:
        if not label:
            print()
        else:
            val = stats[key]
            suffix = f" {unit}" if unit else ""
            print(f"  {label:<20} {val}{suffix}")
    print("=" * 50)


def parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Benchmark ONNX model inference")
    p.add_argument("model", help="Path to ONNX model file")
    p.add_argument("--provider", "-p", default="auto",
                   choices=["cpu", "cuda", "tensorrt", "auto"],
                   help="EP (default: auto = TensorRT->CUDA->CPU)")
    p.add_argument("--iterations", "-n", type=int, default=100)
    p.add_argument("--warmup", "-w", type=int, default=10)
    p.add_argument("--batch-size", "-b", type=int, default=1)
    p.add_argument("--csv", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n  ONNX MODEL BENCHMARK (🔵 Method 3: onnxruntime)")
    print(f"  Model: {args.model} | Provider: {args.provider}")
    print(f"  Iters: {args.iterations} | Warmup: {args.warmup} | Batch: {args.batch_size}")

    try:
        session = create_session(args.model, args.provider)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    active_ep = session.get_providers()[0] if session.get_providers() else "Unknown"
    inputs = build_dummy_inputs(session, args.batch_size)
    latencies, total_time = run_benchmark(session, inputs, args.warmup, args.iterations)
    stats = compute_stats(latencies, total_time, args.batch_size)
    print_results(session, stats, active_ep)

    if args.csv:
        s = stats
        print(f"CSV:{active_ep},{s['iterations']},{s['batch_size']},"
              f"{s['total_time_s']},{s['avg_ms']},{s['min_ms']},"
              f"{s['max_ms']},{s['median_ms']},{s['p95_ms']},"
              f"{s['p99_ms']},{s['std_ms']},{s['fps']},{s['throughput_ips']}")


if __name__ == "__main__":
    main()
