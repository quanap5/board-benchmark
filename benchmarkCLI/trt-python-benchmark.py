#!/usr/bin/env python3
"""Benchmark TensorRT inference with FP32/FP16/INT8 using Python API.

Uses CUDA events for accurate GPU timing. No pycuda needed.
Each precision runs in a subprocess for clean GPU memory.
Requires: tensorrt (JetPack), numpy, libcudart.so (JetPack).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
log = import_module("log-utils")


def _run_single(model, precision, iterations, warmup, workspace, save_engine):
    """Run benchmark for one precision (called in subprocess via --_single)."""
    import tensorrt as trt
    cuda = import_module("cuda-utils")
    builder_mod = import_module("trt-engine-builder")

    base_name = os.path.splitext(model)[0]
    cached = "{}-{}.engine".format(base_name, precision)
    engine = None

    if os.path.isfile(cached):
        log.ok("Loading cached engine: {}".format(cached))
        engine = builder_mod.load_engine(cached)
    else:
        log.step("Building {} engine (no cache)...".format(precision.upper()))
        engine = builder_mod.build_engine(model, precision, workspace)
        if engine and save_engine:
            builder_mod.save_engine(engine, cached)

    if not engine:
        log.error("Engine build/load failed for {}".format(precision.upper()))
        sys.exit(1)

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate buffers
    bindings, device_buffers = [], []
    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        buf = cuda.DeviceBuffer(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        device_buffers.append(buf)
        bindings.append(int(buf))

    # Copy random input
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            cuda.memcpy_htod_async(device_buffers[i],
                                   np.random.randn(*shape).astype(dtype), stream)
    stream.synchronize()

    # Warmup
    log.wait("Warmup ({} runs)...".format(warmup))
    for _ in range(warmup):
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle.value)
    stream.synchronize()

    # Benchmark
    log.info("Benchmarking ({} runs)...".format(iterations))
    latencies = []
    for _ in range(iterations):
        start, end = cuda.Event(), cuda.Event()
        start.record(stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle.value)
        end.record(stream)
        end.synchronize()
        latencies.append(end.time_since(start))

    for buf in device_buffers:
        buf.free()

    # Compute stats
    arr = np.array(latencies)
    stats = {
        "avg": round(float(np.mean(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "median": round(float(np.median(arr)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "std": round(float(np.std(arr)), 3),
        "fps": round(len(arr) / (arr.sum() / 1000.0), 2),
    }

    log.header("RESULTS [{}]".format(precision.upper()))
    for l, k, u in [("Avg", "avg", "ms"), ("Min", "min", "ms"), ("Max", "max", "ms"),
                     ("Median", "median", "ms"), ("P95", "p95", "ms"), ("P99", "p99", "ms"),
                     ("Std", "std", "ms"), ("FPS", "fps", "")]:
        log.metric(l, stats[k], u)
    print("RESULT_JSON:{}".format(json.dumps({"precision": precision.upper(), "stats": stats})))


def print_comparison(all_results, model_path, iterations):
    """Print final comparison table with method info."""
    model_name = os.path.basename(model_path).replace(".onnx", "")
    n = len(all_results)
    log.header("FINAL COMPARISON — {} ({} iters)".format(model_name.upper(), iterations))
    log.info("Method: TRT Python API + CUDA events (🟣)")

    log.table_header("Metric", [prec for prec, _ in all_results])
    log.divider(n)
    for label, key, unit in [("Avg latency", "avg", "ms"), ("Min latency", "min", "ms"),
                              ("Max latency", "max", "ms"), ("Median", "median", "ms"),
                              ("P95", "p95", "ms"), ("P99", "p99", "ms"), ("Std dev", "std", "ms")]:
        log.table_metric(label, [s[key] for _, s in all_results], unit)
    log.divider(n)
    log.table_fps([s["fps"] for _, s in all_results])

    base_prec, base = all_results[0]
    if n > 1:
        print("")
        for prec, s in all_results[1:]:
            if s["avg"] > 0:
                sp = round(base["avg"] / s["avg"], 2)
                fp = round(s["fps"] / base["fps"], 2) if base["fps"] > 0 else 0
                log.speed("{} vs {}: {}x latency | {}x FPS".format(prec, base_prec, sp, fp))


def main():
    p = argparse.ArgumentParser(description="TRT Python benchmark")
    p.add_argument("model", help="ONNX model path")
    p.add_argument("-p", "--precision", nargs="+", default=["fp32", "fp16", "int8"],
                   choices=["fp32", "fp16", "int8"])
    p.add_argument("-n", "--iterations", type=int, default=100)
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("--workspace", type=int, default=1024)
    p.add_argument("--save-engine", action="store_true")
    p.add_argument("--_single", help=argparse.SUPPRESS)
    args = p.parse_args()

    # Internal: run single precision in subprocess
    if args._single:
        _run_single(args.model, args._single, args.iterations, args.warmup,
                    args.workspace, args.save_engine)
        return

    log.header("TRT PYTHON BENCHMARK (🟣 Method 2)")
    log.metric("Model", args.model)
    log.metric("Precisions", ", ".join(pr.upper() for pr in args.precision))
    log.info("Each precision runs in separate process (clean GPU memory)")

    all_results = []
    script = os.path.abspath(__file__)

    for prec in args.precision:
        log.step("{} — spawning subprocess".format(prec.upper()))
        cmd = [sys.executable, script, args.model, "--_single", prec,
               "-n", str(args.iterations), "-w", str(args.warmup),
               "--workspace", str(args.workspace)]
        if args.save_engine:
            cmd.append("--save-engine")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1)
        result_json = None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.startswith("RESULT_JSON:"):
                result_json = line.strip()[len("RESULT_JSON:"):]
        proc.wait()

        if result_json:
            data = json.loads(result_json)
            all_results.append((data["precision"], data["stats"]))
        else:
            log.warn("{} failed (process exited {})".format(prec.upper(), proc.returncode))
        print("")

    if len(all_results) > 1:
        print_comparison(all_results, args.model, args.iterations)
    log.ok("Benchmark complete.")


if __name__ == "__main__":
    main()
