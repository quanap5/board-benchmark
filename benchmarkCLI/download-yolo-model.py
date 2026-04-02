#!/usr/bin/env python3
"""Download YOLO model from Ultralytics and export to ONNX format."""

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
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    os.makedirs(args.output_dir, exist_ok=True)

    pt_name = f"{args.model}.pt"
    pt_path = os.path.join(args.output_dir, pt_name)

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

    print(f"\nDone! Files saved to {args.output_dir}/")
    print(f"  PyTorch: {dst_pt}")
    print(f"  ONNX:    {dst_onnx}")
    print(f"\nBenchmark with:")
    print(f"  python3 onnx-benchmark.py {dst_onnx} -p cpu -n 100")


if __name__ == "__main__":
    main()
