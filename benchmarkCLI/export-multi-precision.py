#!/usr/bin/env python3
"""Export YOLO model in FP32, FP16, INT8 with INT64->INT32 fix for TensorRT.

FP32: default ONNX export + INT32 fix
FP16: ultralytics --half export + INT32 fix
INT8: onnxruntime dynamic quantization from FP32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fix_int64(onnx_path):
    """Apply INT64->INT32 fix only if needed. Skip if already clean."""
    count = _count_int64(onnx_path)
    if count == 0:
        print("[OK] Already INT64-free (skip fix)")
        return
    print("[STEP] Found {} INT64 tensors. Fixing...".format(count))
    from importlib import import_module
    download_mod = import_module("download-yolo-model")
    download_mod.fix_int64_to_int32(onnx_path)
    remaining = _count_int64(onnx_path)
    if remaining > 0:
        print("[WARN] {} INT64 tensors still remain after fix".format(remaining))
    else:
        print("[OK] Fixed. Model is INT64-free (TensorRT ready)")


def _count_int64(onnx_path):
    """Count remaining INT64 tensors in ONNX model."""
    import onnx
    from onnx import TensorProto
    model = onnx.load(onnx_path)
    count = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            count += 1
    for t in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if t.type.tensor_type.elem_type == TensorProto.INT64:
            count += 1
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.INT64:
                    count += 1
        elif node.op_type in ("Constant", "ConstantOfShape"):
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.INT64:
                    count += 1
    return count


def _export_onnx(model_name, output_dir, precision, half=False):
    """Export ONNX model and apply INT32 fix."""
    from ultralytics import YOLO
    tag = "fp16" if half else "fp32"
    dst = os.path.join(output_dir, "{}-{}.onnx".format(model_name, tag))
    if os.path.exists(dst):
        print("[OK] {} already exists".format(dst))
        return dst
    print("[STEP] Exporting {} ({})...".format(model_name, tag.upper()))
    model = YOLO(model_name)
    model.export(format="onnx", imgsz=640, opset=17, simplify=True, half=half)
    src = "{}.onnx".format(model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.rename(src, dst)
    pt_src = "{}.pt".format(model_name)
    pt_dst = os.path.join(output_dir, pt_src)
    if os.path.exists(pt_src) and os.path.abspath(pt_src) != os.path.abspath(pt_dst):
        os.rename(pt_src, pt_dst)
    # Fix INT64 for TensorRT
    print("[STEP] Fixing INT64 -> INT32 for TensorRT...")
    _fix_int64(dst)
    print("[OK] Saved: {} ({})".format(dst, _size(dst)))
    return dst


def quantize_int8(fp32_path, output_dir, model_name):
    """Quantize FP32 model to INT8 using onnxruntime."""
    dst = os.path.join(output_dir, "{}-int8.onnx".format(model_name))
    if os.path.exists(dst):
        print("[OK] {} already exists".format(dst))
        return dst
    print("[STEP] Quantizing to INT8 (dynamic quantization)...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[WARN] onnxruntime.quantization not available.")
        return None
    quantize_dynamic(fp32_path, dst, weight_type=QuantType.QInt8)
    print("[OK] Saved: {} ({})".format(dst, _size(dst)))
    return dst


def _size(path):
    """Human-readable file size."""
    n = os.path.getsize(path)
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return "{:.1f} {}".format(n, u)
        n /= 1024.0
    return "{:.1f} TB".format(n)


def main():
    p = argparse.ArgumentParser(description="Export FP32/FP16/INT8 models (INT32 fixed)")
    p.add_argument("--model", "-m", default="yolov8n")
    p.add_argument("--output-dir", "-o", default="models-pc")
    p.add_argument("-p", "--precision", nargs="+", default=["fp32", "fp16", "int8"],
                   choices=["fp32", "fp16", "int8"])
    args = p.parse_args()

    print("\n  MULTI-PRECISION EXPORT (with INT32 fix)")
    print("  Model:      {}".format(args.model))
    print("  Output:     {}".format(args.output_dir))
    print("  Precisions: {}".format(", ".join(args.precision)))
    print("")

    paths = {}
    for prec in args.precision:
        if prec == "fp32":
            paths["fp32"] = _export_onnx(args.model, args.output_dir, prec, half=False)
        elif prec == "fp16":
            paths["fp16"] = _export_onnx(args.model, args.output_dir, prec, half=True)
        elif prec == "int8":
            fp32 = paths.get("fp32") or _export_onnx(args.model, args.output_dir, "fp32")
            paths["fp32"] = fp32
            paths["int8"] = quantize_int8(fp32, args.output_dir, args.model)
        print("")

    print("  EXPORTED FILES:")
    for prec, path in paths.items():
        if path:
            print("    {}: {} ({})".format(prec.upper(), path, _size(path)))
    print("")
    print("  Benchmark:")
    print("    bash benchmarkCLI/run-benchmark-pc.sh --model {} -p {} -n 100".format(
        args.model, " ".join(paths.keys())))


if __name__ == "__main__":
    main()
