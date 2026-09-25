"""cusz.h for Python: compress_init/decompress_init return a Compressor/Decompressor."""

import math
import warnings

import cupy as cp

from . import _psz_comp
from ._psz_comp import (
    DEFAULT_CODEC,
    DEFAULT_PREDICTOR,
    Codec,
    DataSummary,
    Header,
    Predictor,
    Quality,
)

__all__ = [
    "Codec",
    "Compressor",
    "DataSummary",
    "Decompressor",
    "Header",
    "Pipeline",
    "Predictor",
    "Quality",
    "DEFAULT_CODEC",
    "DEFAULT_PREDICTOR",
    "compress_init",
    "compress_init_2stage",
    "compress_init_3stage",
    "decompress_init",
    "assess_quality_float",
    "assess_quality_double",
    "print_concise_quality",
    "review_compression",
    "review_decompression",
    "header_from_archive",
]

_DTYPE = {
    cp.dtype(cp.float32): _psz_comp.Dtype.F4,
    cp.dtype(cp.float64): _psz_comp.Dtype.F8,
}
_SUFFIX = {cp.dtype(cp.float32): "float", cp.dtype(cp.float64): "double"}


def _xyz(shape):
    # psz dims are x-fastest; a C-contiguous cupy array's fastest axis is its last.
    x, y, z = (list(reversed([int(s) for s in shape])) + [1, 1, 1])[:3]
    return x, y, z


def _shape_of(header):
    x, y, z = header.len
    return (x,) if y == 1 and z == 1 else (y, x) if z == 1 else (z, y, x)


def _check(stat, what):
    if stat != 0:
        raise RuntimeError(f"{what} failed: {_psz_comp.psz_error_string(stat)}")


def Pipeline(spec):
    """psz_ppl from the CLI's --pipeline syntax."""
    err, ppl = _psz_comp.pszctx_pipeline_from_name(spec)
    if err:
        raise ValueError(err)
    return ppl


def _pipeline(p):
    if isinstance(p, str):
        return Pipeline(p)
    return p


def _device(a, dtype):
    return cp.ascontiguousarray(cp.asarray(a, dtype=dtype))


class Compressor:
    """psz_ctx and its stream, from compress_init; free or `with` releases it."""

    __slots__ = ("_ptr", "_stream", "shape", "dtype", "header", "nstage")

    def __init__(self, ptr, stream, shape, dtype, header=None, nstage=None):
        self._ptr, self._stream = ptr, stream
        self.shape, self.dtype, self.header = tuple(shape), cp.dtype(dtype), header
        self.nstage = nstage

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()

    def _ptr_for(self, decompress):
        if not self._ptr:
            raise ValueError("compressor already freed")
        if isinstance(self, Decompressor) != decompress:
            raise ValueError(
                "compress_* needs a Compressor, decompress_* a Decompressor"
            )
        return self._ptr

    def _fn(self, name):
        return getattr(_psz_comp, f"psz_{name}_{_SUFFIX[self.dtype]}")

    def _in(self, data):
        data = _device(data, self.dtype)
        if data.size != math.prod(self.shape):
            raise ValueError(
                f"data has {data.size} elements; the compressor is {self.shape}"
            )
        return data

    def compress_extrema(self, data):
        """psz_compress_extrema_* -> DataSummary; a relative eb times its rng is the absolute eb."""
        ptr = self._ptr_for(decompress=False)
        with self._stream:
            data = self._in(data)
            summary = self._fn("compress_extrema")(ptr, int(data.data.ptr))
        _check(_psz_comp.psz_last_error(), "extrema")
        return summary

    def compress_process(self, pipeline, data, eb):
        """psz_compress_process_*, absolute eb; pipeline is a Pipeline or its CLI string."""
        ptr = self._ptr_for(decompress=False)
        pipeline = _pipeline(pipeline)
        if self.nstage == 2 and pipeline.codec2 != Codec.CodecNull:
            warnings.warn(
                "2-stage compressor: pass 2 dropped; use compress_init_3stage",
                RuntimeWarning,
                stacklevel=2,
            )
        with self._stream:
            data = self._in(data)
            stat = self._fn("compress_process")(
                ptr, pipeline, float(eb), int(data.data.ptr)
            )
        _check(stat, "compress")

    def compress_archive(self):
        """psz_compress_archive -> (header, archive), archive a cupy uint8 copy."""
        stat, header, d_archive, nbytes = _psz_comp.psz_compress_archive(
            self._ptr_for(decompress=False)
        )
        _check(stat, "archive")
        with self._stream:
            archive = cp.empty(nbytes, dtype=cp.uint8)
        cp.cuda.runtime.memcpy(
            archive.data.ptr, d_archive, nbytes, cp.cuda.runtime.memcpyDeviceToDevice
        )
        return header, archive

    def compress_reset(self):
        """psz_compress_reset; call before a rerun or another pipeline."""
        _check(_psz_comp.psz_compress_reset(self._ptr_for(decompress=False)), "reset")

    def decompress_process(self, archive, out=None):
        """psz_decompress_process_* -> data of self.shape and self.dtype."""
        ptr = self._ptr_for(decompress=True)
        with self._stream:
            archive = _device(archive, cp.uint8)
            if out is None:
                out = cp.empty(self.shape, dtype=self.dtype)
            elif (
                out.dtype != self.dtype
                or not out.flags.c_contiguous
                or out.size != math.prod(self.shape)
            ):
                raise ValueError(
                    f"out must be C-contiguous {self.dtype.name} of {self.shape}"
                )
            stat = self._fn("decompress_process")(
                ptr, int(archive.data.ptr), int(archive.size), int(out.data.ptr)
            )
        _check(stat, "decompress")
        return out

    def decompress_reset(self):
        """psz_decompress_reset; call before decoding again."""
        _check(_psz_comp.psz_decompress_reset(self._ptr_for(decompress=True)), "reset")

    def free(self):
        """psz_free; a second call does nothing."""
        if self._ptr:
            _psz_comp.psz_free(self._ptr)
            self._ptr = 0


class Decompressor(Compressor):
    """psz_ctx and its stream, from decompress_init; a Compressor in all but name."""

    __slots__ = ()


def _compress_init(init, nstage, shape, dtype, stream):
    if cp.dtype(dtype) not in _DTYPE:
        raise ValueError(f"no such dtype: {cp.dtype(dtype)}; have float32, float64")
    stream = stream if stream is not None else cp.cuda.get_current_stream()
    x, y, z = _xyz(shape)
    ptr = init(_DTYPE[cp.dtype(dtype)], x, y, z, int(stream.ptr))
    if not ptr:
        _check(_psz_comp.psz_last_error(), init.__name__)
    return Compressor(ptr, stream, shape, dtype, nstage=nstage)


def compress_init(shape, dtype=cp.float32, stream=None):
    """psz_compress_init -> Compressor for predictor + codec 1 pipelines."""
    return _compress_init(_psz_comp.psz_compress_init, 2, shape, dtype, stream)


compress_init_2stage = compress_init


def compress_init_3stage(shape, dtype=cp.float32, stream=None):
    """psz_compress_init_3stage -> Compressor for pipelines with a pass 2 as well."""
    return _compress_init(_psz_comp.psz_compress_init_3stage, 3, shape, dtype, stream)


def decompress_init(header, stream=None):
    """psz_decompress_init -> Decompressor for archives with this header."""
    stream = stream if stream is not None else cp.cuda.get_current_stream()
    ptr = _psz_comp.psz_decompress_init(header, int(stream.ptr))
    if not ptr:
        _check(_psz_comp.psz_last_error(), "psz_decompress_init")
    dtype = next(k for k, v in _DTYPE.items() if v == header.dtype)
    return Decompressor(ptr, stream, _shape_of(header), dtype, header)


def assess_quality_float(reconst, origin):
    """psz_assess_quality_float -> Quality."""
    reconst = _device(reconst, cp.float32)
    origin = _device(origin, cp.float32)
    if reconst.size != origin.size:
        raise ValueError(f"length mismatch: {reconst.size} vs {origin.size}")
    stat, quality = _psz_comp.psz_assess_quality_float(
        int(reconst.data.ptr), int(origin.data.ptr), int(origin.size)
    )
    _check(stat, "assess_quality")
    return quality


def assess_quality_double(reconst, origin):
    """psz_assess_quality_double -> Quality."""
    reconst = _device(reconst, cp.float64)
    origin = _device(origin, cp.float64)
    if reconst.size != origin.size:
        raise ValueError(f"length mismatch: {reconst.size} vs {origin.size}")
    stat, quality = _psz_comp.psz_assess_quality_double(
        int(reconst.data.ptr), int(origin.data.ptr), int(origin.size)
    )
    _check(stat, "assess_quality")
    return quality


def print_concise_quality(header, quality, comp_bytes):
    """psz_print_concise_quality; prints from C, to the process stdout."""
    _psz_comp.psz_print_concise_quality(header, quality, int(comp_bytes))


def review_compression(header, verbose=False):
    """psz_review_compression / _verbose; prints from C, to the process stdout."""
    (
        _psz_comp.psz_review_compression_verbose
        if verbose
        else _psz_comp.psz_review_compression
    )(header)


def review_decompression(header, verbose=False):
    """psz_review_decompression / _verbose."""
    (
        _psz_comp.psz_review_decompression_verbose
        if verbose
        else _psz_comp.psz_review_decompression
    )(header)


def header_from_archive(archive):
    """Not in cusz.h: the Header at the front of an archive."""
    archive = _device(archive, cp.uint8)
    return _psz_comp.header_from_archive(int(archive.data.ptr))
