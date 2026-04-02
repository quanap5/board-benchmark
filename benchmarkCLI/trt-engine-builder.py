#!/usr/bin/env python3
"""Build TensorRT engine from ONNX with FP32/FP16/INT8 precision.

Uses tensorrt + ctypes CUDA bindings. No pycuda needed.
"""

import os
import sys

import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("[ERROR] tensorrt not found. Is JetPack installed?")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
cuda = import_module("cuda-utils")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class DummyCalibrator(trt.IInt8EntropyCalibrator2):
    """Random-data calibrator for INT8 benchmarking (not for accuracy)."""

    def __init__(self, network, num_batches=50):
        super(DummyCalibrator, self).__init__()
        inp = network.get_input(0)
        self.shape = [d if d > 0 else 1 for d in inp.shape]
        self.num_batches = num_batches
        self.batch_idx = 0
        self.device_input = cuda.DeviceBuffer(int(np.prod(self.shape) * 4))

    def get_batch_size(self):
        return self.shape[0]

    def get_batch(self, names):
        if self.batch_idx >= self.num_batches:
            return None
        data = np.random.randn(*self.shape).astype(np.float32)
        cuda.memcpy_htod(self.device_input, data)
        self.batch_idx += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


def build_engine(onnx_path, precision="fp16", workspace_mb=1024):
    """Build TensorRT engine from ONNX with specified precision."""
    builder = trt.Builder(TRT_LOGGER)
    flag = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(1 << flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("[ERROR] {}".format(parser.get_error(i)))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 enabled")
        else:
            print("[WARN] FP16 not supported. Using FP32.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = DummyCalibrator(network)
            print("[INFO] INT8 enabled (FP16 fallback for unsupported layers)")
        else:
            print("[WARN] INT8 not supported. Using FP32.")

    print("[INFO] Building engine ({})...".format(precision.upper()))
    engine = builder.build_engine(network, config)
    if not engine:
        print("[ERROR] Engine build failed.")
    return engine


def load_engine(engine_path):
    """Load serialized TensorRT engine."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def save_engine(engine, path):
    """Save serialized engine to disk."""
    with open(path, "wb") as f:
        f.write(engine.serialize())
    print("[INFO] Engine saved: {} ({})".format(
        path, _human_size(os.path.getsize(path))))


def _human_size(nbytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return "{:.1f} {}".format(nbytes, unit)
        nbytes /= 1024.0
    return "{:.1f} TB".format(nbytes)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build TRT engine from ONNX")
    p.add_argument("onnx", help="Path to ONNX model")
    p.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    p.add_argument("-o", "--output", help="Output engine path")
    p.add_argument("--workspace", type=int, default=1024)
    args = p.parse_args()

    engine = build_engine(args.onnx, args.precision, args.workspace)
    if engine:
        out = args.output or "{}-{}.engine".format(
            os.path.splitext(args.onnx)[0], args.precision)
        save_engine(engine, out)
