"""Round-trip + ratio smoke test for the cusz.h C API pybinding."""

import ctypes
import os
import sys
import warnings

import numpy as np

import cupy as cp

from psz import (
    header_from_archive,
    assess_quality_double,
    assess_quality_float,
    compress_init,
    compress_init_2stage,
    compress_init_3stage,
    decompress_init,
    Pipeline,
    print_concise_quality,
)


def _smooth(shape):
    grids = [np.linspace(0, 6 * np.pi, n, dtype=np.float32) for n in shape]
    mesh = np.meshgrid(*grids, indexing="ij")
    f = np.zeros(shape, dtype=np.float32)
    for g in mesh:
        f = f + np.sin(g)
    return np.ascontiguousarray(f, dtype=np.float32)


def run(shape, eb, mode, predictor, codec1, dtype=cp.float32):
    assess = assess_quality_double if dtype == cp.float64 else assess_quality_float
    d = cp.asarray(_smooth(shape), dtype=dtype)

    with compress_init(shape, dtype) as c:
        abs_eb = eb
        if mode == "rel":
            s = c.compress_extrema(d)
            if not (s.min == float(cp.min(d)) and s.max == float(cp.max(d)) and s.rng > 0):
                print(f"[FAIL] summary {s.min} {s.max} {s.rng} vs {cp.min(d)} {cp.max(d)}")
                return False
            abs_eb = eb * s.rng
        c.compress_process(f"{predictor},{codec1}", d, abs_eb)
        header, archive = c.compress_archive()

    with decompress_init(header) as dc:
        xr = dc.decompress_process(archive)

    ratio = d.nbytes / archive.nbytes
    q = assess(xr, d)
    rng = float(cp.max(d) - cp.min(d))
    abs_eb = eb * rng if mode == "rel" else eb
    ok = q.max_err_abs <= abs_eb * (1 + 1e-2) and xr.shape == d.shape and xr.dtype == d.dtype
    tag = "OK " if ok else "FAIL"
    print(f"[{tag}] {cp.dtype(dtype).name:7s} {predictor:8s} {codec1:10s} {mode} shape={shape} "
          f"eb={eb:.0e} ratio={ratio:6.1f}x maxerr={q.max_err_abs:.3e} bound={abs_eb:.3e}")
    return ok


def rejects_spline_f8(shape):
    with compress_init(shape, cp.float64) as c:
        try:
            c.compress_process(Pipeline("spl,hfr-pbkc"), cp.zeros(shape, dtype=cp.float64), 1e-3)
            ok = False
        except RuntimeError as e:
            ok = "unsupported data type" in str(e)
    print(f"[{'OK ' if ok else 'FAIL'}] float64 spl rejected")
    return ok


def reads_header_from_archive(shape):
    d = cp.asarray(_smooth(shape))
    with compress_init(shape) as c:
        c.compress_process("lrz,hf", d, 1e-2)
        header, archive = c.compress_archive()
    read_back = header_from_archive(archive.get())
    with decompress_init(read_back) as dc:
        xr = dc.decompress_process(archive.get())
    ok = read_back.len == header.len and read_back.dtype == header.dtype and xr.shape == d.shape
    ok = ok and float(cp.max(cp.abs(xr - d))) <= 1e-2 * (1 + 1e-2)
    print(f"[{'OK ' if ok else 'FAIL'}] header read back from a host copy of the archive")
    return ok


def parses_like_cli():
    def fields(spec):
        p = Pipeline(spec)
        return p.predictor, p.codec1, p.codec2

    same = [("lrz..", "lrz,_"), ("lrz,..", "lrz,_"), ("spl,hf..", "spl,hf"),
            ("_,_", "lrz,hfr-pbkc"), ("preset:hicr", "spl,hf,lc-rtr"), ("preset:fzg", "lrz-zz,fzg")]
    ok = all(fields(a) == fields(b) for a, b in same)
    errors = {"lrz": "--pipeline takes p1,c1[,c2] or preset:<name>",
              "lrz,hf,_": "no default pass 2; name lc-bitr or lc-rtr",
              "lrz,hfr-v3": "hfr-v3 is not selectable; use hfr-v4",
              "preset:hicr..": 'a preset already names every stage; drop the ".."'}
    for spec, message in errors.items():
        try:
            Pipeline(spec)
            ok = False
        except ValueError as e:
            ok = ok and str(e) == message
    print(f"[{'OK ' if ok else 'FAIL'}] Pipeline parses exactly as the CLI's --pipeline")
    return ok


def three_stage(shape):
    d = cp.asarray(_smooth(shape))
    with compress_init_3stage(shape) as c:
        c.compress_process("lrz,hf,lc-rtr", d, 1e-2)
        header, archive = c.compress_archive()
    with decompress_init(header) as dc:
        xr = dc.decompress_process(archive)
    ok = float(cp.max(cp.abs(xr - d))) <= 1e-2 * (1 + 1e-2)
    with compress_init_2stage(shape) as c:
        c.compress_process("lrz,hf", d, 1e-2)
        _, two = c.compress_archive()
        c.compress_reset()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c.compress_process("lrz,hf,lc-rtr", d, 1e-2)
        header, archive = c.compress_archive()
    with decompress_init(header) as dc:
        xr = dc.decompress_process(archive)
    warned = any(issubclass(w.category, RuntimeWarning) for w in caught)
    ok = ok and warned and archive.nbytes == two.nbytes
    ok = ok and float(cp.max(cp.abs(xr - d))) <= 1e-2 * (1 + 1e-2)
    print(f"[{'OK ' if ok else 'FAIL'}] 3-stage round trip; 2-stage compressor warns, drops pass 2")
    return ok


def prints_concise_quality(shape):
    d = cp.asarray(_smooth(shape))
    with compress_init(shape) as c:
        c.compress_process("lrz,hf", d, 1e-2)
        header, archive = c.compress_archive()
    with decompress_init(header) as dc:
        xr = dc.decompress_process(archive)
    sys.stdout.flush()
    r, w = os.pipe()
    saved = os.dup(1)
    os.dup2(w, 1)
    try:
        print_concise_quality(header, assess_quality_float(xr, d), archive.nbytes)
        ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(w)
        os.close(saved)
    line = os.read(r, 4096).decode()
    os.close(r)
    ok = f"CR={d.nbytes / archive.nbytes:.2f}\t" in line
    print(f"[{'OK ' if ok else 'FAIL'}] concise quality line: {line.strip()}")
    return ok


def rejects_freed_compressor(shape):
    c = compress_init(shape)
    c.free()
    c.free()
    try:
        c.compress_process("lrz,hf", cp.zeros(shape, dtype=cp.float32), 1e-3)
        ok = False
    except ValueError:
        ok = True
    print(f"[{'OK ' if ok else 'FAIL'}] freed compressor rejected")
    return ok


def rejects_wrong_side(shape):
    d = cp.asarray(_smooth(shape))
    refused = []
    with compress_init(shape) as c:
        c.compress_process("lrz,hf", d, 1e-2)
        header, archive = c.compress_archive()
        try:
            c.decompress_process(archive)
        except ValueError:
            refused.append("decompress on a Compressor")
    with decompress_init(header) as dc:
        try:
            dc.compress_process("lrz,hf", d, 1e-2)
        except ValueError:
            refused.append("compress on a Decompressor")
    ok = len(refused) == 2
    print(f"[{'OK ' if ok else 'FAIL'}] wrong side rejected: {refused}")
    return ok


def main():
    ok = True
    for predictor, codec1 in (("lrz", "hfr-pbkc"), ("spl", "hfr-pbkc"), ("lrz", "hf")):
        for mode, eb in (("abs", 1e-2), ("rel", 1e-3)):
            ok = run((64, 64, 64), eb, mode, predictor, codec1) and ok
    for predictor, codec1 in (("lrz", "hfr-pbkc"), ("lrz", "hf")):
        for mode, eb in (("abs", 1e-2), ("rel", 1e-3)):
            ok = run((64, 64, 64), eb, mode, predictor, codec1, dtype=cp.float64) and ok
    ok = run((96, 128), 1e-3, "rel", "lrz", "hf") and ok
    ok = rejects_spline_f8((64, 64, 64)) and ok
    ok = reads_header_from_archive((64, 64, 64)) and ok
    ok = parses_like_cli() and ok
    ok = three_stage((64, 64, 64)) and ok
    ok = prints_concise_quality((64, 64, 64)) and ok
    ok = rejects_freed_compressor((64, 64, 64)) and ok
    ok = rejects_wrong_side((64, 64, 64)) and ok
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
