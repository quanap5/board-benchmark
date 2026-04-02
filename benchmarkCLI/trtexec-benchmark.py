#!/usr/bin/env python3
"""Benchmark with trtexec CLI across FP32/FP16/INT8 precisions.

Runs trtexec for each precision, parses output, prints comparison.
Supports TensorRT 8.2 (JetPack 4.6) and newer output formats.
"""

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
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, "trtexec")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def run_trtexec(model_path, precision, iterations, warmup, save_engine=None):
    """Run trtexec and stream output in real-time."""
    trtexec = find_trtexec()
    if not trtexec:
        print("[ERROR] trtexec not found.")
        return ""

    cmd = [trtexec, "--iterations={}".format(iterations),
           "--warmUp={}".format(warmup * 1000)]

    if model_path.endswith(".engine"):
        cmd.append("--loadEngine={}".format(model_path))
    else:
        cmd.append("--onnx={}".format(model_path))

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")

    if save_engine:
        cmd.append("--saveEngine={}".format(save_engine))

    print("[INFO] Running [{}]: {}".format(precision.upper(), " ".join(cmd)))
    print("")

    captured = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    return "".join(captured)


def parse_output(output):
    """Parse trtexec timing output into stats dict."""
    stats = {}
    patterns = {
        "min_ms": r"min\s*[:=]\s*([\d.]+)\s*ms",
        "max_ms": r"max\s*[:=]\s*([\d.]+)\s*ms",
        "avg_ms": r"mean\s*[:=]\s*([\d.]+)\s*ms",
        "median_ms": r"median\s*[:=]\s*([\d.]+)\s*ms",
        "p99_ms": r"percentile\(99%\)\s*[:=]\s*([\d.]+)\s*ms",
    }

    # Find timing block
    search_text = output
    for block_name in ["GPU Compute", "Host Latency", "Total Host"]:
        block, in_block = "", False
        for line in output.split("\n"):
            if block_name in line:
                in_block = True
            if in_block:
                block += line + "\n"
                if ("percentile" in line.lower() or not line.strip()) and block.count("\n") > 1:
                    in_block = False
        if block:
            search_text = block
            break

    for key, pattern in patterns.items():
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            stats[key] = round(float(match.group(1)), 3)

    match = re.search(r"Throughput:\s*([\d.]+)\s*qps", output, re.IGNORECASE)
    if match:
        stats["fps"] = round(float(match.group(1)), 2)
    elif "avg_ms" in stats and stats["avg_ms"] > 0:
        stats["fps"] = round(1000.0 / stats["avg_ms"], 2)

    return stats


def print_results(stats, precision, model_path):
    """Print single precision results."""
    print("\n" + "=" * 50)
    print("  TRTEXEC RESULTS [{}]".format(precision.upper()))
    print("=" * 50)
    print("  Model:             {}".format(model_path))
    for label, key, unit in [
        ("Avg latency", "avg_ms", "ms"), ("Min latency", "min_ms", "ms"),
        ("Max latency", "max_ms", "ms"), ("Median", "median_ms", "ms"),
        ("P99 latency", "p99_ms", "ms"), ("FPS", "fps", ""),
    ]:
        val = stats.get(key, "N/A")
        sfx = " {}".format(unit) if unit and val != "N/A" else ""
        print("  {:<20} {}{}".format(label, val, sfx))
    print("=" * 50)


def print_comparison(all_results):
    """Print side-by-side precision comparison table."""
    print("\n" + "=" * 60 + "\n  TRTEXEC PRECISION COMPARISON\n" + "=" * 60)
    hdr = "  {:<16}".format("Metric")
    for prec, _ in all_results:
        hdr += " {:>12}".format(prec)
    print(hdr + "\n  " + "-" * (16 + 13 * len(all_results)))
    for label, key, unit in [("Avg latency", "avg_ms", "ms"), ("Min latency", "min_ms", "ms"),
                              ("Median", "median_ms", "ms"), ("P99", "p99_ms", "ms"), ("FPS", "fps", "")]:
        row = "  {:<16}".format(label)
        for _, s in all_results:
            v = s.get(key, "N/A")
            row += " {:>12}".format("{}{}".format(v, " " + unit if unit and v != "N/A" else ""))
        print(row)
    base_prec, base = all_results[0]
    if len(all_results) > 1 and "avg_ms" in base:
        print("")
        for prec, s in all_results[1:]:
            if "avg_ms" in s and s["avg_ms"] > 0:
                print("  {} vs {} speedup: {}x".format(prec, base_prec, round(base["avg_ms"] / s["avg_ms"], 2)))
    print("=" * 60)


def print_csv(precision, stats):
    """Print CSV line."""
    keys = ["avg_ms", "min_ms", "max_ms", "median_ms", "p99_ms", "fps"]
    print("CSV:{},{}".format(precision, ",".join(str(stats.get(k, "")) for k in keys)))


def main():
    p = argparse.ArgumentParser(description="trtexec benchmark (FP32/FP16/INT8)")
    p.add_argument("model", help="Path to ONNX model or .engine file")
    p.add_argument("-p", "--precision", nargs="+", default=["fp16"],
                   choices=["fp32", "fp16", "int8"],
                   help="Precision(s) to benchmark (default: fp16)")
    p.add_argument("-n", "--iterations", type=int, default=100)
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("--save-engine", action="store_true", help="Save .engine files")
    p.add_argument("--csv", action="store_true", default=False)
    args = p.parse_args()

    print("\n  TRTEXEC BENCHMARK")
    print("  Model:      {}".format(args.model))
    print("  Precisions: {}".format(", ".join(pr.upper() for pr in args.precision)))
    print("  Iterations: {}  Warmup: {}".format(args.iterations, args.warmup))

    all_results = []
    base_name = os.path.splitext(args.model)[0]

    for prec in args.precision:
        print("\n--- {} ---".format(prec.upper()))

        # Check for cached engine first
        cached = "{}-{}.engine".format(base_name, prec)
        if os.path.isfile(cached):
            print("[INFO] Using cached engine: {}".format(cached))
            model_input = cached
            save_path = None
        else:
            model_input = args.model
            save_path = cached if args.save_engine else None

        output = run_trtexec(model_input, prec, args.iterations, args.warmup, save_path)
        stats = parse_output(output)
        if stats:
            print_results(stats, prec, args.model)
            all_results.append((prec.upper(), stats))
            if args.csv:
                print_csv(prec.upper(), stats)
        else:
            print("[WARN] Failed to parse {} results.".format(prec.upper()))

    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
