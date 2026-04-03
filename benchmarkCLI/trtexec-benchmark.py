#!/usr/bin/env python3
"""Benchmark with trtexec CLI across FP32/FP16/INT8 precisions.

Each precision runs in a subprocess for clean GPU memory.
Supports TensorRT 8.2 (JetPack 4.6) and newer output formats.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
log = import_module("log-utils")

METHOD = "trtexec CLI"


def find_trtexec():
    """Find trtexec binary on the system."""
    for path in ["/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "trtexec")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _run_single(model, precision, iterations, warmup, save_engine):
    """Run trtexec for one precision (called via --_single)."""
    trtexec = find_trtexec()
    if not trtexec:
        log.error("trtexec not found."); sys.exit(1)

    base_name = os.path.splitext(model)[0]
    cached = "{}-{}.engine".format(base_name, precision)

    # Build command
    cmd = [trtexec, "--iterations={}".format(iterations), "--warmUp={}".format(warmup * 1000)]
    if os.path.isfile(cached):
        log.ok("Using cached engine: {}".format(cached))
        cmd.append("--loadEngine={}".format(cached))
    else:
        log.wait("Building {} engine from ONNX...".format(precision.upper()))
        cmd.append("--onnx={}".format(model))
        if save_engine:
            cmd.append("--saveEngine={}".format(cached))

    if precision == "fp16": cmd.append("--fp16")
    elif precision == "int8": cmd.append("--int8")

    log.info("Running: {}".format(" ".join(cmd)))

    # Stream output
    captured = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush(); captured.append(line)
    proc.wait()
    output = "".join(captured)

    # Parse
    stats = {}
    patterns = {"min_ms": r"min\s*[:=]\s*([\d.]+)\s*ms", "max_ms": r"max\s*[:=]\s*([\d.]+)\s*ms",
                "avg_ms": r"mean\s*[:=]\s*([\d.]+)\s*ms", "median_ms": r"median\s*[:=]\s*([\d.]+)\s*ms",
                "p99_ms": r"percentile\(99%\)\s*[:=]\s*([\d.]+)\s*ms"}

    search = output
    for bn in ["GPU Compute", "Host Latency"]:
        block, inside = "", False
        for line in output.split("\n"):
            if bn in line: inside = True
            if inside:
                block += line + "\n"
                if ("percentile" in line.lower() or not line.strip()) and block.count("\n") > 1:
                    inside = False
        if block: search = block; break

    for key, pat in patterns.items():
        m = re.search(pat, search, re.IGNORECASE)
        if m: stats[key] = round(float(m.group(1)), 3)

    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", output, re.IGNORECASE)
    if m: stats["fps"] = round(float(m.group(1)), 2)
    elif "avg_ms" in stats and stats["avg_ms"] > 0:
        stats["fps"] = round(1000.0 / stats["avg_ms"], 2)

    if not stats:
        log.error("Could not parse output for {}".format(precision.upper())); sys.exit(1)

    log.header("RESULTS [{}]".format(precision.upper()))
    for l, k, u in [("Avg", "avg_ms", "ms"), ("Min", "min_ms", "ms"), ("Max", "max_ms", "ms"),
                     ("Median", "median_ms", "ms"), ("P99", "p99_ms", "ms"), ("FPS", "fps", "")]:
        log.metric(l, stats.get(k, "N/A"), u)

    print("RESULT_JSON:{}".format(json.dumps({"precision": precision.upper(), "stats": stats})))


def print_comparison(all_results, model_path, iterations):
    """Print final comparison table with method info."""
    model_name = os.path.basename(model_path).replace(".onnx", "").replace(".engine", "")
    n = len(all_results)
    log.header("FINAL COMPARISON — {} ({} iters)".format(model_name.upper(), iterations))
    log.info("Method: {} (🟢)".format(METHOD))

    log.table_header("Metric", [prec for prec, _ in all_results])
    log.divider(n)
    for label, key, unit in [("Avg latency", "avg_ms", "ms"), ("Min latency", "min_ms", "ms"),
                              ("Max latency", "max_ms", "ms"), ("Median", "median_ms", "ms"),
                              ("P99", "p99_ms", "ms")]:
        log.table_metric(label, [s.get(key, "N/A") for _, s in all_results], unit)
    log.divider(n)
    log.table_fps([s.get("fps", "N/A") for _, s in all_results])

    base_prec, base = all_results[0]
    if n > 1 and "avg_ms" in base:
        print("")
        for prec, s in all_results[1:]:
            if "avg_ms" in s and s["avg_ms"] > 0:
                sp = round(base["avg_ms"] / s["avg_ms"], 2)
                fp = round(s["fps"] / base["fps"], 2) if base.get("fps", 0) > 0 else 0
                log.speed("{} vs {}: {}x latency | {}x FPS".format(prec, base_prec, sp, fp))


def main():
    p = argparse.ArgumentParser(description="trtexec benchmark (FP32/FP16/INT8)")
    p.add_argument("model", help="ONNX model or .engine file")
    p.add_argument("-p", "--precision", nargs="+", default=["fp16"],
                   choices=["fp32", "fp16", "int8"])
    p.add_argument("-n", "--iterations", type=int, default=100)
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("--save-engine", action="store_true")
    p.add_argument("--csv", action="store_true", default=False)
    p.add_argument("--_single", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._single:
        _run_single(args.model, args._single, args.iterations, args.warmup, args.save_engine)
        return

    log.header("TRTEXEC BENCHMARK (🟢 Method 1)")
    log.metric("Model", args.model)
    log.metric("Precisions", ", ".join(pr.upper() for pr in args.precision))
    log.info("Each precision runs in separate process (clean GPU memory)")

    all_results = []
    script = os.path.abspath(__file__)
    for prec in args.precision:
        log.step("{} — spawning subprocess".format(prec.upper()))
        cmd = [sys.executable, script, args.model, "--_single", prec,
               "-n", str(args.iterations), "-w", str(args.warmup)]
        if args.save_engine: cmd.append("--save-engine")

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1)
        result_json = None
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            if line.startswith("RESULT_JSON:"): result_json = line.strip()[12:]
        proc.wait()

        if result_json:
            data = json.loads(result_json)
            all_results.append((data["precision"], data["stats"]))
        else:
            log.warn("{} failed (exit {})".format(prec.upper(), proc.returncode))
        print("")

    if len(all_results) > 1:
        print_comparison(all_results, args.model, args.iterations)
    log.ok("Benchmark complete.")


if __name__ == "__main__":
    main()
