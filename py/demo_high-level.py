import ctypes
import cupy as cp
import numpy as np
import os

import pycusz_connector
import pycusz
from pycusz_connector import psz_predtype as Predictor

from cupy.cuda.stream import Stream


def f4demo_compress_v2(predictor, len3, d_uncomp, stream: Stream):

    manager = pycusz.create_resource_manager(pycusz.F4, len3, stream)
    header = pycusz.psz_header()
    d_internal_compressed = ctypes.POINTER(ctypes.c_uint8)()
    compressed_size = ctypes.c_size_t()

    runtime_config = pycusz_connector.psz_runtime_config()
    runtime_config.predictor = predictor
    runtime_config.hist = pycusz_connector.DEFAULT_HISTOGRAM
    runtime_config.codec1 = pycusz_connector.DEFAULT_CODEC
    runtime_config.mode = pycusz_connector.Abs
    runtime_config.eb = 1e-3

    d_data_ptr = ctypes.cast(d_uncomp.data.ptr, ctypes.c_void_p).value

    pycusz_connector.PYCONNECTOR_compress_float(
        manager,
        runtime_config,
        d_data_ptr,
        header,
        ctypes.addressof(d_internal_compressed),
        ctypes.addressof(compressed_size),
    )

    d_compressed = cp.cuda.memory.alloc(compressed_size.value)
    cp.cuda.runtime.memcpy(
        int(d_compressed.ptr),
        int(ctypes.cast(d_internal_compressed, ctypes.c_void_p).value),
        compressed_size.value,
        cp.cuda.runtime.memcpyDeviceToDevice,
    )
    pycusz_connector.PYCONNECTOR_release_resource(manager)
    return d_compressed, compressed_size.value, header


def f4demo_decompress_v2(header, d_compressed, stream):
    stream_ptr = ctypes.c_void_p(stream.ptr).value
    manager = pycusz_connector.PYCONNECTOR_create_resource_manager_from_header(
        header, stream_ptr
    )
    d_decompressed = cp.empty_like(d_compressed)  # Assume same size
    pycusz_connector.PYCONNECTOR_decompress_float(
        manager,
        d_compressed.data.ptr,  # Pass device pointer
        d_compressed.nbytes,
        d_decompressed.data.ptr,  # Pass device pointer for decompressed output
    )
    pycusz_connector.release_resource(manager)
    return d_decompressed


if __name__ == "__main__":
    predictor = pycusz_connector.Lorenzo
    len3 = (3600, 1800, 1)
    stream = cp.cuda.Stream()

    h_uncomp = np.fromfile(os.environ["CESM"], dtype="f4")
    d_uncomp = cp.asarray(h_uncomp)

    d_compressed, compressed_size, header = f4demo_compress_v2(
        predictor, len3, d_uncomp, stream
    )

    print(f"Compression complete. Compressed size: {compressed_size} bytes")

    # Run decompression
    d_decompressed = f4demo_decompress_v2(header, d_compressed, stream)

    print("Decompression complete.")
