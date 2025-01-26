import ctypes
from ctypes import POINTER, c_size_t, c_uint8

import cupy as cp
import cupy
import numpy as np
import pycusz_connector as _connector  # nanobind module
from pycusz_connector import psz_predtype as Predictor
from cupy.cuda.stream import Stream

from pycusz_connector import *

CompressedPointer = ctypes.POINTER(ctypes.c_uint8)


def create_resource_manager(
    dtype: _connector.psz_dtype, xyz: tuple, stream: Stream
) -> _connector.psz_resource:
    if len(xyz) != 3:
        raise ValueError("Dimensions must be a tuple of three elements")
    x, y, z = xyz
    if x < 0 or y < 0 or z < 0:
        raise ValueError("Invalid dimensions")
    if x * y * z == 1:
        raise ValueError("Invalid dimensions")

    stream_ptr = ctypes.c_void_p(stream.ptr).value
    return _connector.PYCONNECTOR_create_resource_manager(dtype, x, y, z, stream_ptr)


def create_resource_manager_from_header(header: _connector.psz_header, stream: Stream):
    stream_ptr = ctypes.c_void_p(stream.ptr).value
    return _connector.create_resource_manager_from_header(header, stream_ptr)


def compress_float(
    manager: _connector.psz_resource,
    runtime_config: _connector.psz_runtime_config,
    d_input: cupy.ndarray,
) -> tuple[_connector.psz_header, cupy.ndarray, int]:

    d_data_ptr = ctypes.cast(d_input.data.ptr, ctypes.c_void_p).value
    header = _connector.psz_header()
    d_internal_compressed = ctypes.POINTER(ctypes.c_uint8)()
    compressed_size: ctypes.c_size_t

    _connector.PYCONNECTOR_compress_float(
        manager,
        runtime_config,
        d_data_ptr,
        ctypes.addressof(header),
        ctypes.addressof(d_internal_compressed),
        ctypes.addressof(compressed_size),
    )

    d_internal_compressed_array = cp.zeros(compressed_size.value, dtype=cp.uint8)
    ctypes.memmove(
        d_internal_compressed_array.data.ptr,
        d_internal_compressed,
        compressed_size.value,
    )
    return header, d_internal_compressed_array, compressed_size.value


def decompress_float(
    manager: _connector.psz_resource, d_input: cupy.ndarray, compressed_size: int
) -> :
    pass
