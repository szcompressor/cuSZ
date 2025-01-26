import sys
import ctypes
import numpy as np
from psz_types import *


class pSZ:
    def __init__(self, libpath=None):
        if libpath is None:
            libpath = {
                "linux": "libcusz.so",
                "win32": "libcusz.dll",
            }.get(sys.platform, "libcusz.so")

        self.psz = ctypes.cdll.LoadLibrary(libpath)

        self.psz.capi_psz_create__experimental.argtypes = [
            psz_dtype,
            psz_len3,
            psz_predtype,
            c_int,
            psz_codectype,
        ]
        self.psz.capi_psz_create__experimental.restype = c_void_p

        self.psz.capi_psz_create_from_header__experimental.argtypes = [
            POINTER(psz_header)
        ]
        self.psz.capi_psz_create_from_header__experimental.restype = c_void_p

        self.psz.capi_psz_release__experimental.argtypes = [c_void_p]
        self.psz.capi_psz_release__experimental.restype = psz_error_status

        self.psz.capi_psz_compress__experimental.argtypes = [
            c_void_p,
            c_void_p,
            psz_len3,
            c_double,
            psz_mode,
            POINTER(POINTER(ctypes.c_uint8)),
            POINTER(c_size_t),
            POINTER(psz_header),
            c_void_p,
            c_void_p,
        ]
        self.psz.capi_psz_compress__experimental.restype = psz_error_status

        # self.psz.capi_psz_decompress.argtypes = [
        #     POINTER(psz_compressor),
        #     POINTER(ctypes.c_uint8),
        #     c_size_t,
        #     c_void_p,
        #     psz_len3,
        #     c_void_p,
        #     c_void_p,
        # ]
        # self.psz.capi_psz_decompress.restype = psz_error_status

        self.psz.capi_psz_make_timerecord.argtypes = []
        self.psz.capi_psz_make_timerecord.restype = c_void_p

        self.compressor = None

    # def __del__(self):
    #     if self.compressor:
    #         self.psz.capi_psz_release(self.compressor)
