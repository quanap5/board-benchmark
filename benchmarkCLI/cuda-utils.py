#!/usr/bin/env python3
"""Minimal CUDA runtime bindings via ctypes. No pycuda needed.

Provides GPU memory allocation, host-device transfers, streams,
and CUDA event timing using only libcudart.so (ships with JetPack).
"""

import ctypes
import ctypes.util
import numpy as np
import sys


def _load_cudart():
    """Load CUDA runtime library."""
    for name in ["libcudart.so", "libcudart.so.10.2", "libcudart.so.11.0",
                 "libcudart.so.12.0"]:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    path = ctypes.util.find_library("cudart")
    if path:
        return ctypes.cdll.LoadLibrary(path)
    print("[ERROR] libcudart.so not found. Is CUDA installed?")
    sys.exit(1)


_cudart = _load_cudart()

# CUDA constants
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def _check(status, msg="CUDA error"):
    """Check CUDA return status."""
    if status != 0:
        raise RuntimeError("{}: code {}".format(msg, status))


class DeviceBuffer(object):
    """GPU memory buffer allocated via cudaMalloc."""

    def __init__(self, nbytes):
        self.ptr = ctypes.c_void_p()
        self.nbytes = nbytes
        _check(_cudart.cudaMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(nbytes)),
               "cudaMalloc")

    def __int__(self):
        return self.ptr.value or 0

    def free(self):
        if self.ptr.value:
            _cudart.cudaFree(self.ptr)
            self.ptr = ctypes.c_void_p()


class Stream(object):
    """CUDA stream."""

    def __init__(self):
        self.handle = ctypes.c_void_p()
        _check(_cudart.cudaStreamCreate(ctypes.byref(self.handle)),
               "cudaStreamCreate")

    def synchronize(self):
        _check(_cudart.cudaStreamSynchronize(self.handle),
               "cudaStreamSynchronize")

    def __del__(self):
        if self.handle.value:
            _cudart.cudaStreamDestroy(self.handle)


class Event(object):
    """CUDA event for GPU timing."""

    def __init__(self):
        self.handle = ctypes.c_void_p()
        _check(_cudart.cudaEventCreate(ctypes.byref(self.handle)),
               "cudaEventCreate")

    def record(self, stream):
        _check(_cudart.cudaEventRecord(self.handle, stream.handle),
               "cudaEventRecord")

    def synchronize(self):
        _check(_cudart.cudaEventSynchronize(self.handle),
               "cudaEventSynchronize")

    def time_since(self, start_event):
        """Return elapsed time in ms from start_event to this event."""
        elapsed = ctypes.c_float()
        _check(_cudart.cudaEventElapsedTime(
            ctypes.byref(elapsed), start_event.handle, self.handle),
            "cudaEventElapsedTime")
        return float(elapsed.value)

    def __del__(self):
        if self.handle.value:
            _cudart.cudaEventDestroy(self.handle)


def memcpy_htod(device_buf, host_array):
    """Copy numpy array from host to device."""
    nbytes = host_array.nbytes
    src = host_array.ctypes.data_as(ctypes.c_void_p)
    _check(_cudart.cudaMemcpy(device_buf.ptr, src,
                               ctypes.c_size_t(nbytes), cudaMemcpyHostToDevice),
           "cudaMemcpy H2D")


def memcpy_htod_async(device_buf, host_array, stream):
    """Async copy numpy array from host to device."""
    nbytes = host_array.nbytes
    src = host_array.ctypes.data_as(ctypes.c_void_p)
    _check(_cudart.cudaMemcpyAsync(device_buf.ptr, src,
                                    ctypes.c_size_t(nbytes),
                                    cudaMemcpyHostToDevice, stream.handle),
           "cudaMemcpyAsync H2D")


def memcpy_dtoh(host_array, device_buf):
    """Copy from device to host numpy array."""
    nbytes = host_array.nbytes
    dst = host_array.ctypes.data_as(ctypes.c_void_p)
    _check(_cudart.cudaMemcpy(dst, device_buf.ptr,
                               ctypes.c_size_t(nbytes), cudaMemcpyDeviceToHost),
           "cudaMemcpy D2H")
