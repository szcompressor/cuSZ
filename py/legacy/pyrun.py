from psz import *
import numpy as np
import cupy as cp
import os
import sys


dtype = psz_dtype.F4
uncomp_len = psz_len3(384, 352, 16)
uncomp_bytes = 384 * 352 * 16 * 4
predictor = psz_predtype.Lorenzo
radius = 512
codec = psz_codectype.Huffman

psz_instance = pSZ("../build/libcusz.so")
psz_instance.compressor = psz_instance.psz.capi_psz_create__experimental(
    dtype, uncomp_len, predictor, radius, codec
)

header = psz_header()
record_c = psz_instance.psz.capi_psz_make_timerecord()
record_x = psz_instance.psz.capi_psz_make_timerecord()

stream = cp.cuda.Stream()
stream_ptr = ctypes.c_void_p(stream.ptr)

d_compressed = POINTER(ctypes.c_uint8)()
comp_bytes = c_size_t()


if len(sys.argv) != 2:
    print("Usage: python pyrun.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]


for i in range(10):
    h_uncomp = np.fromfile(file_path, dtype="f4")
    d_uncomp = cp.asarray(h_uncomp)
    d_decompressed = cp.empty_like(d_uncomp)

    compress_status = psz_instance.psz.capi_psz_compress__experimental(
        psz_instance.compressor,
        d_uncomp.data.ptr,
        uncomp_len,
        3.0,
        psz_mode.Abs,
        ctypes.byref(d_compressed),
        ctypes.byref(comp_bytes),
        ctypes.byref(header),
        record_c,
        stream_ptr,
    )

    print(f"{i:05d}, CR: {uncomp_bytes / comp_bytes.value}")
    psz_instance.psz.capi_psz_release(psz_instance.compressor)
