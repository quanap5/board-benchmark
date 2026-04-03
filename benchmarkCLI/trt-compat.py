#!/usr/bin/env python3
"""TensorRT version compatibility layer. Supports TRT 8.x and 10.x APIs."""

import tensorrt as trt
import numpy as np

TRT_MAJOR = int(trt.__version__.split(".")[0])


def get_io_info(engine):
    """Get list of (name, shape, dtype, is_input) for all engine I/O tensors."""
    ios = []
    if TRT_MAJOR >= 10:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ios.append({"name": name, "shape": tuple(shape), "dtype": dtype, "is_input": is_input})
    else:
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i) if hasattr(engine, "get_binding_name") else str(i)
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            is_input = engine.binding_is_input(i)
            ios.append({"name": name, "shape": tuple(shape), "dtype": dtype, "is_input": is_input})
    return ios


def setup_bindings(context, io_info, device_buffers, stream_handle):
    """Set tensor addresses for TRT 10.x, or return bindings list for TRT 8.x."""
    if TRT_MAJOR >= 10:
        for info, buf in zip(io_info, device_buffers):
            context.set_tensor_address(info["name"], int(buf))
        return None  # TRT 10 doesn't use bindings list
    else:
        return [int(buf) for buf in device_buffers]


def execute_async(context, bindings, stream_handle):
    """Run inference compatible with TRT 8.x and 10.x."""
    if TRT_MAJOR >= 10:
        context.execute_async_v3(stream_handle)
    else:
        context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)


def create_network(builder):
    """Create network with correct flags for TRT version."""
    if TRT_MAJOR >= 10:
        return builder.create_network()
    else:
        flag = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return builder.create_network(1 << flag)


def set_workspace(config, size_bytes):
    """Set workspace size compatible with TRT 8.x and 10.x."""
    if TRT_MAJOR >= 10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, size_bytes)
    else:
        config.max_workspace_size = size_bytes


def build_engine_compat(builder, network, config):
    """Build engine compatible with TRT 8.x and 10.x."""
    if TRT_MAJOR >= 10:
        serialized = builder.build_serialized_network(network, config)
        if not serialized:
            return None
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        return runtime.deserialize_cuda_engine(serialized)
    else:
        return builder.build_engine(network, config)
