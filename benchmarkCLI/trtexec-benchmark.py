#!/usr/bin/env python3
"""Parse trtexec benchmark output into structured results."""

import argparse
import re
import subprocess
import sys


def find_trtexec():
    """Find trtexec binary on the system."""
    import shutil
    for path in ["/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"]:
        if shutil.which(path) or __import__("os").path.isfile(path):
            return path
    return shutil.which("trtexec")


def run_trtexec(model_path, iterations, warmup, fp16):
    """Run trtexec and capture output."""
    trtexec = find_trtexec()
    if not trtexec:
        print("[ERROR] trtexec not found.")
        sys.exit(1)

    cmd = [
        trtexec,
        f"--iterations={iterations}",
        f"--warmUp={warmup * 1000}",
        "--noDataTransfers",
    ]

    if model_path.endswith(".engine"):
        cmd.append(f"--loadEngine={model_path}")
    else:
        cmd.append(f"--onnx={model_path}")

    if fp16:
        cmd.append("--fp16")

    print(f"[INFO] Running: {' '.join(cmd)}")
    print("")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output = result.stdout + result.stderr
    return output


def parse_output(output):
    """Parse trtexec timing output into stats dict."""
    stats = {}

    # Match patterns like: "mean = 1.234 ms"
    patterns = {
        "min_ms": r"min\s*[:=]\s*([\d.]+)\s*ms",
        "max_ms": r"max\s*[:=]\s*([\d.]+)\s*ms",
        "avg_ms": r"mean\s*[:=]\s*([\d.]+)\s*ms",
        "median_ms": r"median\s*[:=]\s*([\d.]+)\s*ms",
        "p99_ms": r"percentile\(99%\)\s*[:=]\s*([\d.]+)\s*ms",
        "p95_ms": r"percentile\(95%\)\s*[:=]\s*([\d.]+)\s*ms",
    }

    # Find the GPU Compute block
    compute_block = ""
    lines = output.split("\n")
    in_compute = False
    for line in lines:
        if "GPU Compute Time" in line or "GPU Compute" in line:
            in_compute = True
        if in_compute:
            compute_block += line + "\n"
            if "percentile" in line.lower() and "99" in line:
                in_compute = False

    search_text = compute_block if compute_block else output

    for key, pattern in patterns.items():
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            stats[key] = round(float(match.group(1)), 3)

    # Extract throughput
    throughput_match = re.search(
        r"Throughput:\s*([\d.]+)\s*qps", output, re.IGNORECASE
    )
    if throughput_match:
        stats["fps"] = round(float(throughput_match.group(1)), 2)
        stats["throughput_ips"] = stats["fps"]
    elif "avg_ms" in stats and stats["avg_ms"] > 0:
        stats["fps"] = round(1000.0 / stats["avg_ms"], 2)
        stats["throughput_ips"] = stats["fps"]

    return stats


def print_results(stats, model_path):
    """Print formatted results."""
    print("")
    print("=" * 50)
    print("  TRTEXEC BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Model:               {model_path}")
    print(f"  Backend:             TensorRT (native)")
    print()

    fields = [
        ("Avg latency", "avg_ms", "ms"),
        ("Min latency", "min_ms", "ms"),
        ("Max latency", "max_ms", "ms"),
        ("Median latency", "median_ms", "ms"),
        ("P95 latency", "p95_ms", "ms"),
        ("P99 latency", "p99_ms", "ms"),
        ("FPS", "fps", ""),
        ("Throughput", "throughput_ips", "samples/s"),
    ]

    for label, key, unit in fields:
        val = stats.get(key, "N/A")
        suffix = f" {unit}" if unit and val != "N/A" else ""
        print(f"  {label:<20} {val}{suffix}")

    print("=" * 50)


def print_csv_line(stats):
    """Print CSV line for scripting."""
    keys = [
        "avg_ms", "min_ms", "max_ms", "median_ms",
        "p95_ms", "p99_ms", "fps", "throughput_ips",
    ]
    values = [str(stats.get(k, "")) for k in keys]
    print(f"CSV:trtexec,{','.join(values)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark model with trtexec (no onnxruntime needed)"
    )
    parser.add_argument(
        "model", help="Path to ONNX or TensorRT engine file"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=100,
        help="Benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=10,
        help="Warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use FP16 precision (default: True)"
    )
    parser.add_argument(
        "--fp32", action="store_true", default=False,
        help="Use FP32 precision"
    )
    parser.add_argument(
        "--csv", action="store_true", default=False,
        help="Append CSV summary line"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fp16 = args.fp16 and not args.fp32

    print("")
    print("  TRTEXEC BENCHMARK")
    print(f"  Model:      {args.model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Precision:  {'FP16' if fp16 else 'FP32'}")

    output = run_trtexec(args.model, args.iterations, args.warmup, fp16)
    print(output)

    stats = parse_output(output)
    if not stats:
        print("[ERROR] Could not parse trtexec output.")
        sys.exit(1)

    print_results(stats, args.model)

    if args.csv:
        print_csv_line(stats)


if __name__ == "__main__":
    main()
