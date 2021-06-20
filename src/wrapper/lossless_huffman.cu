/**
 * @file lossless_huffman.cu
 * @author Jiannan Tian
 * @brief A high-level Huffman wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-17
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include "../kernel/huffman_codec.h"
#include "lossless_huffman.h"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

namespace {

auto get_npart = [](auto size, auto subsize) {
    static_assert(
        std::numeric_limits<decltype(size)>::is_integer and std::numeric_limits<decltype(subsize)>::is_integer,
        "[get_npart] must be plain interger types.");
    return (size + subsize - 1) / subsize;
};

template <typename T>
size_t get_nbyte(T*, size_t len)
{
    return sizeof(T) * len;
}

// TODO add __restrict__ to each
template <typename Huff, typename MetadataT = size_t>
inline void process_huffman_metadata(
    MetadataT*   d_seg_bits,     // on-device space
    MetadataT*   h_seg_bits,     // on-host space & archive
    MetadataT*   h_seg_uints,    // on-host space
    MetadataT*   h_seg_entries,  // on-host space & archive
    unsigned int nchunk,
    size_t&      num_bits,
    size_t&      num_uints)
{
    constexpr auto bitlen = sizeof(Huff) * 8;
    cudaMemcpy(h_seg_bits, d_seg_bits, nchunk * sizeof(MetadataT), D2H);
    memcpy(h_seg_uints, h_seg_bits, nchunk * sizeof(MetadataT));
    for_each(h_seg_uints, h_seg_uints + nchunk, [&](MetadataT& i) { i = (i + bitlen - 1) / bitlen; });
    memcpy(h_seg_entries + 1, h_seg_uints, (nchunk - 1) * sizeof(MetadataT));
    for (auto i = 1; i < nchunk; i++) h_seg_entries[i] += h_seg_entries[i - 1];  // inclusive scan
    num_bits  = std::accumulate(h_seg_bits, h_seg_bits + nchunk, (size_t)0);
    num_uints = std::accumulate(h_seg_uints, h_seg_uints + nchunk, (size_t)0);
}

template <typename Huff, typename MetadataT = size_t>
__global__ void concatenate_huffman_segments(
    Huff* __restrict__ input_dn,
    Huff* __restrict__ output_sp,
    MetadataT* seg_entries,
    MetadataT* seg_uints,
    MetadataT  chunk_size)
{
    auto len      = seg_uints[blockIdx.x];
    auto sp_entry = seg_entries[blockIdx.x];
    auto dn_entry = chunk_size * blockIdx.x;
    auto n        = (len + blockDim.x - 1) / blockDim.x;

    for (auto i = 0; i < n; i++) {
        auto _tid = threadIdx.x + i * blockDim.x;
        if (_tid < len) *(output_sp + sp_entry + _tid) = *(input_dn + dn_entry + _tid);
        __syncthreads();
    }
}

}  // namespace

/********************************************************************************
 * high-level API
 ********************************************************************************/
#define ENC_CTX HuffmanEncodingDescriptor<Input, Huff, MetadataT>

template <typename Input, typename Huff, typename MetadataT, bool NSYMBOL_RESTRICT>
void compress_huffman_encode(ENC_CTX* ctx, Input* d_input, size_t len, int chunk_size)
{
    constexpr auto ENC_SEQ = 4;
    static_assert(
        (std::is_floating_point<Input>::value and NSYMBOL_RESTRICT) == false,
        "[compress_huffman_encode] floating-point input cannot work with symbol number restricted workflow.");
    if CONSTEXPR (NSYMBOL_RESTRICT == false) {  // TODO
        throw std::runtime_error("[compress_huffman_encode] branch(NSYMBOL_RESTRICT) not implemented.");
    }

    auto nchunk = get_npart(len, chunk_size);

    /********************************************************************************
     * encoding in a fixed-length space
     ********************************************************************************/
    {
        auto dim_block = 256;
        auto dim_grid  = get_npart(len, dim_block);
        cusz::EncodeFixedLen_cub                                       //
            <Input, Huff, ENC_SEQ><<<dim_grid, dim_block / ENC_SEQ>>>  //
            (d_input, ctx->space.fixed_len, len, ctx->space.d_book);
        cudaDeviceSynchronize();
    }
    /********************************************************************************
     * deflate
     ********************************************************************************/
    {
        auto dim_block = 256;
        auto dim_grid  = get_npart(nchunk, dim_block);
        cusz::Deflate<Huff><<<dim_grid, dim_block>>>(ctx->space.fixed_len, len, ctx->space.d_seg_bits, chunk_size);
        cudaDeviceSynchronize();
    }
    /********************************************************************************
     * process metadata
     ********************************************************************************/
    {
        process_huffman_metadata<Huff>(
            ctx->space.d_seg_bits, ctx->space.h_seg_bits, ctx->space.h_seg_uints, ctx->space.h_seg_entries, nchunk,
            ctx->num_bits, ctx->num_uints);

        cudaMemcpy(
            ctx->space.d_seg_bits, ctx->space.h_seg_bits,  //
            get_nbyte(ctx->space.h_seg_uints, ctx->nchunk), H2D);
        cudaMemcpy(
            ctx->space.d_seg_entries, ctx->space.h_seg_entries,  //
            get_nbyte(ctx->space.h_seg_entries, ctx->nchunk), H2D);
    }
    /********************************************************************************
     * concatenate segments
     ********************************************************************************/
    {
        concatenate_huffman_segments<<<nchunk, 128>>>(
            ctx->space.fixed_len, ctx->space.d_bitstream, ctx->space.d_seg_entries, ctx->space.d_seg_uints, chunk_size);
        cudaDeviceSynchronize();

        cudaMemcpy(
            ctx->space.h_bitstream, ctx->space.d_bitstream,  //
            get_nbyte(ctx->ctx->space.h_bitstream, ctx->num_uints), D2H);
    }
    /* EOF */
}

#define INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(Input, Huff, MetadataT) \
    template <>                                                     \
    void compress_huffman_encode<Input, Huff, MetadataT>(           \
        HuffmanEncodingDescriptor<Input, Huff, MetadataT>*, Input*, size_t, int);

INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint8_t, uint32_t, size_t)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint8_t, uint32_t, unsigned int)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint8_t, uint64_t, size_t)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint8_t, uint64_t, unsigned int)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint16_t, uint32_t, size_t)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint16_t, uint32_t, unsigned int)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint16_t, uint64_t, size_t)
INSTANTIATE_COMPRESS_HUFFMAN_ENCODE(uint16_t, uint64_t, unsigned int)

/****************************************************************************************************/

#define DEC_CTX HuffmanDecodingDescriptor<Output, Huff, MetadataT>

template <typename Output, typename Huff, typename MetadataT, bool NSYMBOL_RESTRICT>
void decompress_huffman_decode(DEC_CTX* ctx, Output* d_output, size_t len, int chunk_size)
{
    static_assert(
        (std::is_floating_point<Output>::value and NSYMBOL_RESTRICT) == false,
        "[compress_huffman_encode] floating-point input cannot work with symbol number restricted workflow.");
    if CONSTEXPR (NSYMBOL_RESTRICT == false) {  // TODO
        throw std::runtime_error("[compress_huffman_encode] branch(NSYMBOL_RESTRICT) not implemented.");
    }

    auto nchunk = get_npart(len, chunk_size);
    // clang-format off
    cudaMemcpy(ctx->space.d_bitstream,   ctx->space.h_bitstream,   get_nbyte(ctx->space.h_bitstream,   ctx->num_uints),   H2D);
    cudaMemcpy(ctx->space.d_seg_bits,    ctx->space.h_seg_bits,    get_nbyte(ctx->space.h_seg_bits,    ctx->nchunk),      H2D);
    cudaMemcpy(ctx->space.d_seg_entries, ctx->space.h_seg_entries, get_nbyte(ctx->space.h_seg_entries, ctx->nchunk),      H2D);
    cudaMemcpy(ctx->space.d_revbook,     ctx->space.h_revbook,     get_nbyte(ctx->space.h_revbook,     ctx->len.revbook), H2D);
    // clang-format on
    {
        auto dim_block = 256;  // the same as deflate
        auto dim_grid  = get_npart(nchunk, dim_block);
        huffman_decode_kernel<<<dim_grid, dim_block, ctx->len.revbook>>>(
            ctx->space.d_bitstream, ctx->space.d_seg_entries, ctx->space.d_seg_uints, d_output, chunk_size, nchunk,
            ctx->space.d_revbook, ctx->len.revbook);
        cudaDeviceSynchronize();
    }
    /* EOF */
}

#define INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(Output, Huff, MetadataT) \
    template <>                                                        \
    void decompress_huffman_decode<Output, Huff, MetadataT>(           \
        HuffmanDecodingDescriptor<Output, Huff, MetadataT>*, Output*, size_t, int);

INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint8_t, uint32_t, size_t)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint8_t, uint32_t, unsigned int)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint8_t, uint64_t, size_t)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint8_t, uint64_t, unsigned int)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint16_t, uint32_t, size_t)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint16_t, uint32_t, unsigned int)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint16_t, uint64_t, size_t)
INSTANTIATE_DECOMPRESS_HUFFMAN_DECODE(uint16_t, uint64_t, unsigned int)
