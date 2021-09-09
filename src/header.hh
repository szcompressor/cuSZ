#ifndef HEADER_HH
#define HEADER_HH

/**
 * @file header.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-01-22
 * (created) 2020-09-25, (rev.1) 2021-01-22 (rev.2) 2021-09-08
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "argparse.hh"

enum DataType { kF32, kF64 };

typedef struct alignas(128) cuszHEADER {
    uint32_t x, y, z, w, ndim;
    uint32_t data_len;
    double   eb;

    int M, MxM;  // padded M
    int nnz;

    size_t   num_bits, num_uints;
    uint16_t revbook_nbyte;
    uint8_t  quant_byte, huff_byte;
    uint32_t huffman_chunk;

    bool skip_huffman;

    DataType dtype;  // TODO f64 support

    struct {
        uint32_t huffman_revbook;
        uint32_t huffman_bitstream;
        uint32_t huffman_uints_entries;
        uint32_t outlier;
    } nbytes;

} cusz_header;

#endif
