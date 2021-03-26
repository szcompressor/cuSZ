#ifndef PACK_HH
#define PACK_HH

/**
 * @file pack.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-01-22
 * (created) 2020-09-25, (rev.1) 2021-01-22
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */
#include "argparse.hh"
#include "utils/io.hh"

enum DataType { kF32, kF64 };

typedef struct MetadataPackage {
    UInteger4 dim4, stride4, nblk4;
    int      ndim;
    size_t   len;
    double   eb;

    int M, MxM;  // padded M
    int nnz;

    size_t total_bits, total_uInt, huff_meta_size;

    int quant_byte;
    int huff_byte;
    int huffman_chunk;

    bool skip_huffman_enc;
    bool nvcomp_in_use;

    DataType dtype;  // TODO f64 support

    int cb_bytesize;       // TODO for single binary
    int h_bytesize;        // TODO for single binary
    int h_meta_bytesize;   // TODO for single binary
    int outlier_bytesize;  // TODO for single binary

} metadata_pack;

void PackMetadata(argpack* ap, metadata_pack* mp, int& nnz)
{
    mp->dim4    = ap->dim4;
    mp->stride4 = ap->stride4;
    mp->nblk4   = ap->nblk4;
    mp->ndim    = ap->ndim;
    mp->eb      = ap->eb;
    mp->len     = ap->len;

    mp->nnz = nnz;

    if (ap->dtype == "f32") mp->dtype = DataType::kF32;
    if (ap->dtype == "f64") mp->dtype = DataType::kF64;

    mp->quant_byte       = ap->quant_byte;
    mp->huff_byte        = ap->huff_byte;
    mp->huffman_chunk    = ap->huffman_chunk;
    mp->skip_huffman_enc = ap->szwf.skip_huffman_enc;
}

void UnpackMetadata(argpack* ap, metadata_pack* mp, int& nnz)
{
    ap->dim4    = mp->dim4;
    ap->stride4 = mp->stride4;
    ap->nblk4   = mp->nblk4;
    ap->ndim    = mp->ndim;
    ap->eb      = mp->eb;
    ap->len     = mp->len;

    nnz = mp->nnz;

    if (mp->dtype == DataType::kF32) ap->dtype = "f32";
    if (mp->dtype == DataType::kF64) ap->dtype = "f64";

    ap->quant_byte            = mp->quant_byte;
    ap->huff_byte             = mp->huff_byte;
    ap->huffman_chunk         = mp->huffman_chunk;
    ap->szwf.skip_huffman_enc = mp->skip_huffman_enc;
}

#endif
