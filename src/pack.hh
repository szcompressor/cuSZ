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
    int       ndim;
    size_t    len;
    double    eb;

    int M, MxM;  // padded M
    int nnz;

    size_t num_bits, num_uints, revbook_nbyte;

    int quant_byte;
    int huff_byte;
    int huffman_chunk;

    bool skip_huffman;
    bool nvcomp_in_use;

    DataType dtype;  // TODO f64 support

    int cb_bytesize;       // TODO for single binary
    int h_bytesize;        // TODO for single binary
    int h_meta_bytesize;   // TODO for single binary
    int outlier_bytesize;  // TODO for single binary

} cusz_header;

#endif
