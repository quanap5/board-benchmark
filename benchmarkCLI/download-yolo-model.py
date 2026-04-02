#!/usr/bin/env python3
"""Download YOLO model from Ultralytics, export to ONNX, fix INT64 for TensorRT."""

import argparse
import os


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download YOLO model and export to ONNX"
    )
    parser.add_argument(
        "--model", "-m", default="yolov8n",
        help="YOLO model name (default: yolov8n). "
             "Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, "
             "yolo11n, yolo11s, yolo11m, yolo11l, yolo11x"
    )
    parser.add_argument(
        "--imgsz", "-s", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="models",
        help="Output directory for exported model (default: models)"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="Simplify ONNX model (default: True)"
    )
    parser.add_argument(
        "--half", action="store_true", default=False,
        help="Export with FP16 half precision (default: False)"
    )
    parser.add_argument(
        "--no-fix-int64", action="store_true", default=False,
        help="Skip INT64->INT32 conversion (default: convert for TensorRT)"
    )
    return parser.parse_args()


def fix_int64_to_int32(onnx_path):
    """Convert INT64 tensors to INT32 for TensorRT compatibility.

    TensorRT (especially 8.x on Jetson) does not natively support INT64.
    Ultralytics exports shape/index operations as INT64 which causes
    warnings or failures on TensorRT. This converts them to INT32.
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    model = onnx.load(onnx_path)
    fixed = 0

    # Fix initializers (constant weights/tensors)
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.INT64:
            data = numpy_helper.to_array(initializer)
            new_data = data.astype("int32")
            new_tensor = numpy_helper.from_array(new_data, name=initializer.name)
            initializer.CopyFrom(new_tensor)
            fixed += 1

    # Fix graph inputs
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.INT64:
            inp.type.tensor_type.elem_type = TensorProto.INT32
            fixed += 1

    # Fix graph outputs
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.INT64:
            out.type.tensor_type.elem_type = TensorProto.INT32
            fixed += 1

    # Fix node attributes (Constant nodes with INT64 values)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.t.data_type == TensorProto.INT64:
                    data = numpy_helper.to_array(attr.t)
                    new_tensor = numpy_helper.from_array(data.astype("int32"))
                    attr.t.CopyFrom(new_tensor)
                    fixed += 1

    if fixed > 0:
        onnx.save(model, onnx_path)
        print(f"  Fixed {fixed} INT64 -> INT32 tensors for TensorRT")
    else:
        print("  No INT64 tensors found (model already compatible)")


def main():
    args = parse_args()

    from ultralytics import YOLO

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading {args.model}...")
    model = YOLO(args.model)

    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset})...")
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
    )

    # Move files to output directory
    src_pt = f"{args.model}.pt"
    src_onnx = f"{args.model}.onnx"
    dst_pt = os.path.join(args.output_dir, src_pt)
    dst_onnx = os.path.join(args.output_dir, src_onnx)

    if os.path.exists(src_pt) and os.path.abspath(src_pt) != os.path.abspath(dst_pt):
        os.rename(src_pt, dst_pt)
    if os.path.exists(src_onnx) and os.path.abspath(src_onnx) != os.path.abspath(dst_onnx):
        os.rename(src_onnx, dst_onnx)

    # Fix INT64 for TensorRT compatibility
    if not args.no_fix_int64:
        print("Fixing INT64 tensors for TensorRT compatibility...")
        fix_int64_to_int32(dst_onnx)

    print(f"\nDone! Files saved to {args.output_dir}/")
    print(f"  PyTorch: {dst_pt}")
    print(f"  ONNX:    {dst_onnx}")
    print(f"\nBenchmark with:")
    print(f"  python3 onnx-benchmark.py {dst_onnx} -p cpu -n 100")


if __name__ == "__main__":
    main()
