#!/usr/bin/env python3
"""Parse trtexec benchmark output into structured results."""

import argparse
import os
import re
import subprocess
import sys


def find_trtexec():
    """Find trtexec binary on the system."""
    for path in ["/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    # Fallback: search PATH
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "trtexec")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def run_trtexec(model_path, iterations, warmup, fp16):
    """Run trtexec and stream output in real-time."""
    trtexec = find_trtexec()
    if not trtexec:
        print("[ERROR] trtexec not found.")
        sys.exit(1)

    cmd = [
        trtexec,
        "--iterations={}".format(iterations),
        "--warmUp={}".format(warmup * 1000),
    ]

    if model_path.endswith(".engine"):
        cmd.append("--loadEngine={}".format(model_path))
    else:
        cmd.append("--onnx={}".format(model_path))

    if fp16:
        cmd.append("--fp16")

    print("[INFO] Running: {}".format(" ".join(cmd)))
    print("")

    # Stream output line-by-line so user sees progress
    captured = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()

    return "".join(captured)


def parse_output(output):
    """Parse trtexec timing output into stats dict.

    Supports both TensorRT 8.2 (JetPack 4.6) and newer formats.
    """
    stats = {}

    # Patterns for both TRT 8.2 (JetPack 4.6) and newer formats
    # TRT 8.2: "min = 20.57 ms, max = 22.1 ms, mean = 21.3 ms, median = 20.9 ms, percentile(99%) = 21.8 ms"
    # TRT 8.5+: "min: 20.57 ms  max: 22.1 ms  mean: 21.3 ms  median: 20.9 ms  percentile(99%): 21.8 ms"
    patterns = {
        "min_ms": r"min\s*[:=]\s*([\d.]+)\s*ms",
        "max_ms": r"max\s*[:=]\s*([\d.]+)\s*ms",
        "avg_ms": r"mean\s*[:=]\s*([\d.]+)\s*ms",
        "median_ms": r"median\s*[:=]\s*([\d.]+)\s*ms",
        "p99_ms": r"percentile\(99%\)\s*[:=]\s*([\d.]+)\s*ms",
    }

    # Find GPU Compute or Host Latency block, or search entire output
    search_text = output
    for block_name in ["GPU Compute", "Host Latency", "Total Host"]:
        block = ""
        in_block = False
        for line in output.split("\n"):
            if block_name in line:
                in_block = True
            if in_block:
                block += line + "\n"
                if "percentile" in line.lower() or line.strip() == "":
                    if block.count("\n") > 1:
                        in_block = False
        if block:
            search_text = block
            break

    for key, pattern in patterns.items():
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            stats[key] = round(float(match.group(1)), 3)

    # TRT 8.2 only reports one percentile (default 99%). P95 not available.
    # Use P99 as P95 approximation if P95 is missing.
    if "p99_ms" in stats and "p95_ms" not in stats:
        stats["p95_ms"] = stats.get("median_ms", stats.get("p99_ms"))

    # Extract throughput
    match = re.search(r"Throughput:\s*([\d.]+)\s*qps", output, re.IGNORECASE)
    if match:
        stats["fps"] = round(float(match.group(1)), 2)
        stats["throughput_ips"] = stats["fps"]
    elif "avg_ms" in stats and stats["avg_ms"] > 0:
        stats["fps"] = round(1000.0 / stats["avg_ms"], 2)
        stats["throughput_ips"] = stats["fps"]

    return stats


def print_results(stats, model_path):
    """Print formatted results."""
    print("\n" + "=" * 50)
    print("  TRTEXEC BENCHMARK RESULTS\n" + "=" * 50)
    print("  Model:               {}".format(model_path))
    print("  Backend:             TensorRT (native)\n")
    for label, key, unit in [
        ("Avg latency", "avg_ms", "ms"), ("Min latency", "min_ms", "ms"),
        ("Max latency", "max_ms", "ms"), ("Median latency", "median_ms", "ms"),
        ("P95 latency", "p95_ms", "ms"), ("P99 latency", "p99_ms", "ms"),
        ("FPS", "fps", ""), ("Throughput", "throughput_ips", "samples/s"),
    ]:
        val = stats.get(key, "N/A")
        suffix = " {}".format(unit) if unit and val != "N/A" else ""
        print("  {:<20} {}{}".format(label, val, suffix))
    print("=" * 50)


def print_csv_line(stats):
    """Print CSV line for scripting."""
    keys = ["avg_ms", "min_ms", "max_ms", "median_ms", "p95_ms", "p99_ms", "fps", "throughput_ips"]
    print("CSV:trtexec,{}".format(",".join(str(stats.get(k, "")) for k in keys)))


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Benchmark with trtexec")
    p.add_argument("model", help="Path to ONNX or TensorRT engine file")
    p.add_argument("--iterations", "-n", type=int, default=100)
    p.add_argument("--warmup", "-w", type=int, default=10)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--fp32", action="store_true", default=False)
    p.add_argument("--csv", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    fp16 = args.fp16 and not args.fp32

    print("")
    print("  TRTEXEC BENCHMARK")
    print("  Model:      {}".format(args.model))
    print("  Iterations: {}".format(args.iterations))
    print("  Warmup:     {}".format(args.warmup))
    print("  Precision:  {}".format("FP16" if fp16 else "FP32"))

    output = run_trtexec(args.model, args.iterations, args.warmup, fp16)

    stats = parse_output(output)
    if not stats:
        print("[ERROR] Could not parse trtexec output.")
        sys.exit(1)

    print_results(stats, args.model)

    if args.csv:
        print_csv_line(stats)


if __name__ == "__main__":
    main()
