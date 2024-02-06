# 24-02-04

import numpy as np
import zstandard as zstd
from _lrz import *
from _errlog import *


def byte_shuffle(fname, interpret_size, dtype="f4", detailed_bytes=False):
    nbyte = 4 if dtype == "f4" else 8

    fp = np.fromfile(fname, dtype=dtype)
    u1_view = fp.view(np.uint8)

    kwargs = {}
    cor = zstd.ZstdCompressor()

    # compress directly, d
    res0 = len(cor.compress(u1_view))  # (0)

    # compress (1) bs(d) and (2) lrz(bs(d))
    bs, bslrz = 0, 0
    for i in range(nbyte):
        ith_byte = u1_view.reshape(-1, nbyte)[:, i].flatten()

        bs += len(cor.compress(ith_byte))  # (1)

        u1_nd = ith_byte.reshape(interpret_size)
        u1_prd_delta = np.zeros_like(u1_nd)
        PszApprox.succeeding_lorenzo_predict(u1_nd, u1_prd_delta)

        bslrz += len(cor.compress(u1_prd_delta.flatten()))  # (2)

    # compress lrz(reinterp<ux>(d))
    ux_nd = fp.view("u4" if nbyte == 4 else "u8").reshape(interpret_size)
    ux_prd_delta = np.zeros_like(ux_nd)
    PszApprox.succeeding_lorenzo_predict(ux_nd, ux_prd_delta)
    lrz_res0 = len(cor.compress(ux_prd_delta.flatten()))  # (3)

    # compress bs(lrz(reinterp<ux>(d)))
    lrzbs = 0
    lrz_u1_data = ux_prd_delta.flatten().view("u1")
    for i in range(nbyte):
        ith_delta_byte = lrz_u1_data.reshape(-1, nbyte)[:, i].flatten()
        lrzbs += len(cor.compress(ith_delta_byte))  # (4)

    bytes = len(fp) * nbyte
    cr_fp = bytes / res0
    cr_bs = bytes / bs
    cr_lrz = bytes / lrz_res0
    cr_bslrz = bytes / bslrz
    cr_lrzbs = bytes / lrzbs

    print(
        f"{fname}" f"\tCR (fp,bs,bslrz,lrz,lrzbs):",
        f"{cr_fp:.3f}",
        f"{cr_bs:.3f}",
        f"{cr_bslrz:.3f}",
        f"{cr_lrz:.3f}",
        f"{cr_lrzbs:.3f}",
        sep="\t",
    )
