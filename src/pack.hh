#ifndef PACK_HH
#define PACK_HH

/**
 * @file pack.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-25
 *
 * Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 *
 */

#include "argparse.hh"
#include "constants.hh"
#include "io.hh"

enum DataType { kF32, kF64 };

typedef struct MetadataPackage {
    size_t dim_array[16];
    double eb_array[4];
    int    M, MxM;  // padded M
    int    nnz;

    size_t total_bits, total_uInt, huff_meta_size;

    int quant_rep;
    int huffman_rep;
    int huffman_chunk;

    DataType dtype;  // TODO f64 support

    int cb_bytesize;       // TODO for single binary
    int h_bytesize;        // TODO for single binary
    int h_meta_bytesize;   // TODO for single binary
    int outlier_bytesize;  // TODO for single binary

} metadata_pack;

void PackMetadata(argpack* ap, metadata_pack* mp, int& nnz, size_t* dim_array, double* eb_array)
{
    memcpy(mp->dim_array, dim_array, sizeof(size_t) * 16);
    memcpy(mp->eb_array, eb_array, sizeof(double) * 4);

    mp->nnz = nnz;

    if (ap->dtype == "f32") mp->dtype = DataType::kF32;
    if (ap->dtype == "f64") mp->dtype = DataType::kF64;

    mp->quant_rep     = ap->quant_rep;
    mp->huffman_rep   = ap->huffman_rep;
    mp->huffman_chunk = ap->huffman_chunk;

    // TODO type trait
}

void UnpackMetadata(argpack* ap, metadata_pack* mp, int& nnz, size_t* dim_array, double* eb_array)
{
    memcpy(dim_array, mp->dim_array, sizeof(size_t) * 16);
    memcpy(eb_array, mp->eb_array, sizeof(double) * 4);

    nnz = mp->nnz;

    if (mp->dtype == DataType::kF32) ap->dtype = "f32";
    if (mp->dtype == DataType::kF64) ap->dtype = "f64";

    ap->quant_rep     = mp->quant_rep;
    ap->huffman_rep   = mp->huffman_rep;
    ap->huffman_chunk = mp->huffman_chunk;
}

#endif